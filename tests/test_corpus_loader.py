from __future__ import annotations

from pathlib import Path
from typing import Any

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import yaml

from scorpus.loaders.corpus_loader import Corpus, load_corpus


N_GENES = 8

DATASETS = (
    {"dataset_id": "mock_00", "dataset_index": 0, "global_start": 0, "cell_count": 4},
    {"dataset_id": "mock_01", "dataset_index": 1, "global_start": 4, "cell_count": 5},
)


def _obs_frame(dataset_id: str, dataset_index: int, global_start: int, n_cells: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "global_row_index": np.arange(global_start, global_start + n_cells, dtype=np.int64),
            "cell_id": [f"{dataset_id}_cell_{idx}" for idx in range(n_cells)],
            "dataset_id": [dataset_id] * n_cells,
            "dataset_index": np.full(n_cells, dataset_index, dtype=np.int32),
            "local_row_index": np.arange(n_cells, dtype=np.int64),
            "perturb_label": ["ctrl" if idx % 2 == 0 else "treat" for idx in range(n_cells)],
            "perturb_type": ["CRISPR"] * n_cells,
            "dose": [None] * n_cells,
            "dose_unit": [None] * n_cells,
            "timepoint": [None] * n_cells,
            "timepoint_unit": [None] * n_cells,
            "cell_context": ["K562"] * n_cells,
            "cell_line_or_type": ["K562"] * n_cells,
            "species": ["Homo sapiens"] * n_cells,
            "tissue": ["bone marrow"] * n_cells,
            "assay": ["Perturb-seq"] * n_cells,
            "condition": ["mock"] * n_cells,
            "batch_id": [f"batch_{dataset_index}"] * n_cells,
            "donor_id": ["donor_0"] * n_cells,
            "sex": ["NA"] * n_cells,
            "disease_state": ["healthy"] * n_cells,
        }
    )


def _var_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "origin_index": np.arange(N_GENES, dtype=np.int32),
            "gene_id": [f"ENSG{i:05d}" for i in range(N_GENES)],
            "canonical_gene_id": [f"GENE{i:05d}" for i in range(N_GENES)],
            "global_id": np.arange(N_GENES, dtype=np.int32),
        }
    )


def _expression_rows(n_cells: int, *, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for _ in range(n_cells):
        genes = np.sort(rng.choice(N_GENES, size=3, replace=False).astype(np.int32))
        counts = rng.integers(1, 5, size=3, dtype=np.int32)
        rows.append(
            {
                "expressed_gene_indices": genes.tolist(),
                "expression_counts": counts.tolist(),
            }
        )
    return rows


def _write_lance_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "expressed_gene_indices": pa.array(
                [row["expressed_gene_indices"] for row in rows],
                type=pa.list_(pa.int32()),
            ),
            "expression_counts": pa.array(
                [row["expression_counts"] for row in rows],
                type=pa.list_(pa.int32()),
            ),
        }
    )
    lance.write_dataset(table, str(path), mode="overwrite")


def _write_corpus_index(root: Path, *, topology: str, backend: str = "lance") -> None:
    doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "global_metadata": {"backend": backend, "topology": topology},
        "datasets": [
            {
                "dataset_id": item["dataset_id"],
                "join_mode": "create_new" if idx == 0 else "append_routed",
                "dataset_index": item["dataset_index"],
                "cell_count": item["cell_count"],
                "global_start": item["global_start"],
                "global_end": item["global_start"] + item["cell_count"],
            }
            for idx, item in enumerate(DATASETS)
        ],
    }
    with open(root / "corpus-index.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(doc, handle)


def _write_metadata(root: Path, *, topology: str) -> None:
    for item in DATASETS:
        if topology == "aggregate":
            meta_root = root / "meta" / item["dataset_id"] / "canonical_meta"
        else:
            meta_root = root / item["dataset_id"] / "meta" / "canonical_meta"
        meta_root.mkdir(parents=True, exist_ok=True)
        _obs_frame(
            item["dataset_id"],
            item["dataset_index"],
            item["global_start"],
            item["cell_count"],
        ).write_parquet(meta_root / "canonical-obs.parquet")
        _var_frame().write_parquet(meta_root / "canonical-var.parquet")


def _build_aggregate_lance_corpus(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_corpus_index(root, topology="aggregate")
    _write_metadata(root, topology="aggregate")
    rows: list[dict[str, Any]] = []
    for item in DATASETS:
        rows.extend(_expression_rows(item["cell_count"], seed=100 + item["dataset_index"]))
    _write_lance_rows(root / "matrix" / "aggregated-cells.lance", rows)


def _build_federated_lance_corpus(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_corpus_index(root, topology="federated")
    _write_metadata(root, topology="federated")
    for item in DATASETS:
        rows = _expression_rows(item["cell_count"], seed=200 + item["dataset_index"])
        _write_lance_rows(
            root / item["dataset_id"] / "matrix" / f"{item['dataset_id']}.lance",
            rows,
        )


def test_load_corpus_builds_components_without_runtime_loader_state(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)

    corpus = load_corpus(tmp_path)

    assert isinstance(corpus, Corpus)
    assert corpus.topology == "aggregate"
    assert corpus.backend == "lance"
    assert len(corpus.metadata_index) == 9
    assert corpus.dataset_index_by_id == {"mock_00": 0, "mock_01": 1}
    assert corpus.feature_registry.global_vocab_size == N_GENES
    assert not hasattr(corpus, "loader")
    assert not hasattr(corpus, "dataset")
    assert not hasattr(corpus, "read_expression")
    assert not hasattr(corpus, "set_sampler")
    assert not hasattr(corpus, "select_obs_indices")


def test_to_anndata_exports_whole_dataset_as_csr(tmp_path: Path) -> None:
    import scipy.sparse as sp

    _build_aggregate_lance_corpus(tmp_path)
    corpus = load_corpus(tmp_path)

    adata = corpus.to_anndata(dataset_id="mock_00")

    assert adata.shape == (4, N_GENES)
    assert sp.isspmatrix_csr(adata.X)
    assert adata.X.dtype == np.float32
    assert set(adata.obs["dataset_id"]) == {"mock_00"}
    assert adata.var["canonical_gene_id"].to_list() == [f"GENE{i:05d}" for i in range(N_GENES)]
    assert adata.var["hvg_rank"].to_list() == [0] * N_GENES
    assert not adata.var["highly_variable"].any()


def test_to_anndata_exports_selected_global_rows_with_hvg_metadata(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    pl.DataFrame(
        {
            "origin_index": np.arange(N_GENES, dtype=np.int64),
            "hvg_rank": np.asarray([4, 1, 7, 2, 8, 3, 6, 5], dtype=np.int32),
            "selected_at_default_n_hvg": np.asarray(
                [False, True, False, True, False, True, False, False],
                dtype=bool,
            ),
        }
    ).write_parquet(tmp_path / "meta" / "mock_00" / "hvg.parquet")
    corpus = load_corpus(tmp_path)

    selected = corpus.to_anndata(global_row_indices=[3, 1, 3])
    whole = corpus.to_anndata(dataset_id="mock_00")

    assert selected.shape == (3, N_GENES)
    assert selected.obs["global_row_index"].to_list() == [3, 1, 3]
    assert selected.obs["dataset_id"].to_list() == ["mock_00"] * 3
    assert (selected.X != whole.X[[3, 1, 3]]).nnz == 0
    assert selected.var["gene_id"].to_list() == [f"ENSG{i:05d}" for i in range(N_GENES)]
    assert selected.var["hvg_rank"].to_list() == [4, 1, 7, 2, 8, 3, 6, 5]
    assert selected.var["highly_variable"].to_list() == [False, True, False, True, False, True, False, False]


def test_to_anndata_selected_rows_rejects_multiple_datasets(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    corpus = load_corpus(tmp_path)

    with pytest.raises(ValueError, match="same dataset"):
        corpus.to_anndata(global_row_indices=[0, 4])


def test_to_anndata_selected_rows_rejects_dataset_mismatch(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    corpus = load_corpus(tmp_path)

    with pytest.raises(ValueError, match="not requested dataset"):
        corpus.to_anndata(dataset_id="mock_01", global_row_indices=[0, 1])


def test_to_anndata_selected_rows_rejects_empty_indices(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    corpus = load_corpus(tmp_path)

    with pytest.raises(ValueError, match="must not be empty"):
        corpus.to_anndata(global_row_indices=[])


def test_to_anndata_multi_dataset_requires_matching_feature_axis(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    var_path = tmp_path / "meta" / "mock_01" / "canonical_meta" / "canonical-var.parquet"
    var_df = pl.read_parquet(var_path).with_columns(
        pl.Series("canonical_gene_id", [f"OTHER{i:05d}" for i in range(N_GENES)])
    )
    var_df.write_parquet(var_path)
    corpus = load_corpus(tmp_path)

    with pytest.raises(ValueError, match="same ordered canonical_gene_id"):
        corpus.to_anndata(dataset_id=["mock_00", "mock_01"])


def _make_var_df_mismatch(tmp_path: Path) -> None:
    """Modify mock_01's var to have a partially overlapping gene set with mock_00."""
    var_path = tmp_path / "meta" / "mock_01" / "canonical_meta" / "canonical-var.parquet"
    var_df = pl.read_parquet(var_path).with_columns(
        pl.Series(
            "canonical_gene_id",
            [f"GENE{i:05d}" for i in range(2, N_GENES + 2)],
        )
    )
    var_df.write_parquet(var_path)


def test_to_anndata_var_join_inner_intersection(tmp_path: Path) -> None:
    import scipy.sparse as sp

    _build_aggregate_lance_corpus(tmp_path)
    _make_var_df_mismatch(tmp_path)
    corpus = load_corpus(tmp_path)

    with pytest.warns(UserWarning, match="intersection genes"):
        adata = corpus.to_anndata(dataset_id=["mock_00", "mock_01"], var_join="inner")

    expected_genes = [f"GENE{i:05d}" for i in range(2, 8)]  # overlap: GENE00002..GENE00007
    assert adata.shape == (9, 6)
    assert adata.var["canonical_gene_id"].tolist() == expected_genes
    assert sp.isspmatrix_csr(adata.X)
    assert adata.X.dtype == np.float32
    assert set(adata.obs["dataset_id"]) == {"mock_00", "mock_01"}


def test_to_anndata_var_join_inner_single_dataset_no_change(tmp_path: Path) -> None:
    """Single dataset with var_join='inner' should behave same as exact."""
    _build_aggregate_lance_corpus(tmp_path)
    corpus = load_corpus(tmp_path)
    adata = corpus.to_anndata(dataset_id="mock_00", var_join="inner")
    assert adata.shape == (4, N_GENES)
    assert adata.var["canonical_gene_id"].tolist() == [f"GENE{i:05d}" for i in range(N_GENES)]


def test_to_anndata_var_join_exact_still_errors_on_mismatch(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    _make_var_df_mismatch(tmp_path)
    corpus = load_corpus(tmp_path)

    with pytest.raises(ValueError, match="same ordered canonical_gene_id"):
        corpus.to_anndata(dataset_id=["mock_00", "mock_01"], var_join="exact")


def test_to_anndata_var_join_inner_no_common_genes(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    var_path = tmp_path / "meta" / "mock_01" / "canonical_meta" / "canonical-var.parquet"
    var_df = pl.read_parquet(var_path).with_columns(
        pl.Series("canonical_gene_id", [f"OTHER{i:05d}" for i in range(N_GENES)])
    )
    var_df.write_parquet(var_path)
    corpus = load_corpus(tmp_path)

    with pytest.raises(ValueError, match="no common genes"):
        corpus.to_anndata(dataset_id=["mock_00", "mock_01"], var_join="inner")


def test_to_anndata_lazy_var_join_inner_intersection(tmp_path: Path) -> None:
    import dask.array as da
    import scipy.sparse as sp

    _build_aggregate_lance_corpus(tmp_path)
    _make_var_df_mismatch(tmp_path)
    corpus = load_corpus(tmp_path)

    with pytest.warns(UserWarning, match="intersection genes"):
        adata = corpus.to_anndata_lazy(
            dataset_id=["mock_00", "mock_01"], chunk_rows=2, var_join="inner"
        )

    expected_genes = [f"GENE{i:05d}" for i in range(2, 8)]
    assert adata.shape == (9, 6)
    assert adata.var["canonical_gene_id"].tolist() == expected_genes
    assert isinstance(adata.X, da.Array)

    computed = adata.X.compute()
    assert sp.isspmatrix_csr(computed)
    assert computed.shape == (9, 6)
    assert computed.dtype == np.float32


def test_to_anndata_lazy_builds_dask_sparse_x(tmp_path: Path) -> None:
    import dask.array as da
    import scipy.sparse as sp

    _build_aggregate_lance_corpus(tmp_path)
    corpus = load_corpus(tmp_path)

    adata = corpus.to_anndata_lazy(dataset_id="mock_00", chunk_rows=2)

    assert adata.shape == (4, N_GENES)
    assert isinstance(adata.X, da.Array)
    assert adata.X.chunks[0] == (2, 2)
    computed = adata.X.compute()
    assert sp.isspmatrix_csr(computed)
    assert computed.shape == (4, N_GENES)
    assert computed.dtype == np.float32


def test_to_anndata_lazy_rejects_unknown_device(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    corpus = load_corpus(tmp_path)

    with pytest.raises(ValueError, match="device must be"):
        corpus.to_anndata_lazy(dataset_id="mock_00", device="gpu")
    with pytest.raises(ValueError):
        corpus.to_anndata_lazy(dataset_id="mock_00", device="cuda:x")


def test_expression_reader_and_take_metadata_use_global_rows(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    corpus = load_corpus(tmp_path)

    expression = corpus.expression_reader.read_expression_flat([0, 4, 8])
    metadata = corpus.take_metadata([0, 4, 8], columns=["dataset_id", "dataset_index"])

    np.testing.assert_array_equal(expression.global_row_index, [0, 4, 8])
    assert expression.batch_size == 3
    assert metadata["dataset_id"] == ("mock_00", "mock_01", "mock_01")
    np.testing.assert_array_equal(metadata["dataset_index"], [0, 1, 1])


def test_load_corpus_federated_reader_uses_dataset_files(tmp_path: Path) -> None:
    _build_federated_lance_corpus(tmp_path)
    corpus = load_corpus(tmp_path)

    batch = corpus.expression_reader.read_expression_flat([0, 4, 8])

    assert corpus.topology == "federated"
    np.testing.assert_array_equal(batch.global_row_index, [0, 4, 8])
    assert batch.batch_size == 3


def test_load_corpus_rejects_unknown_backend(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    index_path = tmp_path / "corpus-index.yaml"
    with open(index_path, encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    doc["global_metadata"]["backend"] = "unknown_backend"
    with open(index_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(doc, handle)

    with pytest.raises(ValueError, match="backend"):
        load_corpus(tmp_path)
