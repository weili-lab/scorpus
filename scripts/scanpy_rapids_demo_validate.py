from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import dask.array as da
import scanpy as sc

from scorpus.loaders import load_corpus


DATASET_IDS = ["marson_d2_rest", "xorion_hct116_dual_guide"]
OBS_COLUMNS = ["perturb_label", "condition", "cell_context", "batch_id"]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _type_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__name__}"


def _capture_warnings(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    return result, [str(item.message) for item in caught]


def _run_cpu_scanpy(adata) -> dict[str, object]:
    _log("Running CPU Scanpy smoke on lazy combined AnnData")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars))
    subset = adata[:100, :100].to_memory()
    return {
        "status": "success",
        "steps": ["normalize_total", "log1p", "highly_variable_genes", "to_memory_subset"],
        "highly_variable_genes": int(adata.var["highly_variable"].sum()),
        "subset_shape": [int(subset.n_obs), int(subset.n_vars)],
        "subset_nnz": int(subset.X.nnz),
        "subset_x_type": _type_name(subset.X),
    }


def _run_rapids(corpus_root: Path, require_rapids: bool) -> dict[str, object]:
    try:
        import rapids_singlecell as rsc
        import torch
    except Exception as exc:
        if require_rapids:
            raise
        return {"status": "blocked", "reason": f"RAPIDS import failed: {exc}"}

    if not torch.cuda.is_available():
        if require_rapids:
            raise RuntimeError("torch.cuda.is_available() is False")
        return {"status": "blocked", "reason": "CUDA is not available in this environment"}

    _log("Running RAPIDS smoke on GPU AnnData")
    corpus = load_corpus(corpus_root, extra_metadata_columns=OBS_COLUMNS)
    adata_gpu, warning_messages = _capture_warnings(
        lambda: corpus.to_anndata(
            dataset_id=DATASET_IDS,
            obs_columns=OBS_COLUMNS,
            var_join="inner",
        )
    )
    rsc.get.anndata_to_GPU(adata_gpu)
    rsc.pp.normalize_total(adata_gpu, target_sum=1e4)
    rsc.pp.log1p(adata_gpu)
    rsc.pp.highly_variable_genes(adata_gpu, n_top_genes=min(2000, adata_gpu.n_vars))
    return {
        "status": "success",
        "device_name": torch.cuda.get_device_name(0),
        "shape": [int(adata_gpu.n_obs), int(adata_gpu.n_vars)],
        "x_type": _type_name(adata_gpu.X),
        "highly_variable_genes": int(adata_gpu.var["highly_variable"].sum()),
        "warnings": warning_messages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--skip-rapids", action="store_true")
    parser.add_argument("--require-rapids", action="store_true")
    args = parser.parse_args()

    corpus_root = Path(args.corpus_root).resolve()
    summary_path = Path(args.summary_json).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    _log(f"Loading corpus from {corpus_root}")
    corpus = load_corpus(corpus_root, extra_metadata_columns=OBS_COLUMNS)
    summary: dict[str, object] = {
        "timestamp": _timestamp(),
        "corpus_root": str(corpus_root),
        "dataset_ids": list(DATASET_IDS),
        "obs_columns": list(OBS_COLUMNS),
        "chunk_rows": int(args.chunk_rows),
        "corpus": {
            "dataset_ids": list(corpus.dataset_ids),
            "n_rows": int(len(corpus.metadata_index)),
        },
    }

    single, single_warnings = _capture_warnings(
        lambda: corpus.to_anndata_lazy(
            dataset_id="marson_d2_rest",
            obs_columns=OBS_COLUMNS,
            chunk_rows=args.chunk_rows,
        )
    )
    summary["single_dataset"] = {
        "dataset_id": "marson_d2_rest",
        "shape": [int(single.n_obs), int(single.n_vars)],
        "x_is_dask": isinstance(single.X, da.Array),
        "x_type": _type_name(single.X),
        "warnings": single_warnings,
    }

    combined, combined_warnings = _capture_warnings(
        lambda: corpus.to_anndata_lazy(
            dataset_id=DATASET_IDS,
            obs_columns=OBS_COLUMNS,
            chunk_rows=args.chunk_rows,
            var_join="inner",
        )
    )
    probe = combined.X[:64, :64].compute()
    summary["cross_dataset"] = {
        "shape": [int(combined.n_obs), int(combined.n_vars)],
        "x_is_dask": isinstance(combined.X, da.Array),
        "x_type": _type_name(combined.X),
        "chunk_rows": [int(value) for value in combined.X.chunks[0]],
        "chunk_cols": [int(value) for value in combined.X.chunks[1]],
        "n_intersection_genes": int(combined.n_vars),
        "obs_columns_present": [column for column in OBS_COLUMNS if column in combined.obs.columns],
        "probe_block_shape": list(probe.shape),
        "probe_block_nnz": int(probe.nnz),
        "probe_block_type": _type_name(probe),
        "warnings": combined_warnings,
    }

    summary["cpu_scanpy"] = _run_cpu_scanpy(combined)

    if args.skip_rapids:
        summary["rapids"] = {"status": "not-run", "reason": "Skipped by flag"}
    else:
        summary["rapids"] = _run_rapids(corpus_root, require_rapids=args.require_rapids)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    _log(f"Wrote validation summary to {summary_path}")


if __name__ == "__main__":
    main()
