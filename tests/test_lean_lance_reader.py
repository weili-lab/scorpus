"""Lean Lance reads (one sorted take per table) return the same batches as row-by-row reads."""

import shutil
from dataclasses import replace

import anndata as ad
import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import yaml
from scipy import sparse

from scorpus import concat, from_h5ad, load_corpus
from scorpus.loaders.expression import AggregateLanceReader, DatasetEntry, FederatedLanceReader, LanceDatasetEntry
from scorpus.loaders import expression

TWO_COLUMNS = ["expressed_gene_indices", "expression_counts"]


def random_rows(rng, n_rows, counts_dtype, special_counts=()):
    """Sparse rows (sorted gene indices, counts), about one in ten empty."""
    genes_per_row, counts_per_row = [], []
    for _ in range(n_rows):
        n_genes = 0 if rng.random() < 0.1 else int(rng.integers(1, 40))
        genes = np.sort(rng.choice(36_601, n_genes, replace=False)).astype(np.int32)
        if counts_dtype == np.float32:
            counts = (rng.random(n_genes) * 100).astype(np.float32)
        else:
            counts = rng.integers(1, 50, n_genes).astype(counts_dtype)
        genes_per_row.append(genes)
        counts_per_row.append(counts)
    # Put the special values in the first nonempty rows.
    nonempty = [i for i, genes in enumerate(genes_per_row) if len(genes)]
    for row, value in zip(nonempty, special_counts):
        counts_per_row[row][0] = value
    return genes_per_row, counts_per_row


def write_table(path, genes_per_row, counts_per_row, packed=False, rows_per_file=1000):
    """Write rows as a two-column or packed Lance table, in small files and row groups."""
    lengths = [len(genes) for genes in genes_per_row]
    offsets = pa.array(np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32))
    genes = np.concatenate(genes_per_row).astype(np.int32)
    counts = np.concatenate(counts_per_row)
    columns = {"global_row_index": pa.array(np.arange(len(lengths), dtype=np.int64))}
    if packed:
        count_bits = counts.view(np.uint32) if counts.dtype == np.float32 else counts
        values = (count_bits.astype(np.int64) << 16) | genes.astype(np.int64)
        columns["packed"] = pa.ListArray.from_arrays(offsets, pa.array(values))
    else:
        columns["expressed_gene_indices"] = pa.ListArray.from_arrays(offsets, pa.array(genes))
        columns["expression_counts"] = pa.ListArray.from_arrays(offsets, pa.array(counts))
    lance.write_dataset(pa.table(columns), path, max_rows_per_file=rows_per_file, max_rows_per_group=128)


def read_row_by_row(paths, starts, indices):
    """Reference: one take of one row at a time, joined in batch order."""
    genes, counts, lengths = [], [], []
    for index in indices:
        table = int(np.searchsorted(starts, index, side="right")) - 1
        row = lance.dataset(paths[table]).take([index - starts[table]], columns=TWO_COLUMNS).to_pylist()[0]
        genes.extend(row["expressed_gene_indices"])
        counts.extend(row["expression_counts"])
        lengths.append(len(row["expressed_gene_indices"]))
    return np.concatenate([[0], np.cumsum(lengths)]), np.array(genes), np.array(counts)


def assert_same_batch(actual, expected):
    np.testing.assert_array_equal(actual.global_row_index, expected.global_row_index)
    np.testing.assert_array_equal(actual.row_offsets, expected.row_offsets)
    np.testing.assert_array_equal(actual.expressed_gene_indices, expected.expressed_gene_indices)
    np.testing.assert_array_equal(actual.expression_counts, expected.expression_counts)
    assert actual.row_offsets.dtype == np.int64
    assert actual.expressed_gene_indices.dtype == np.int32
    assert actual.expression_counts.dtype == expected.expression_counts.dtype


def test_lean_read_equals_row_by_row_reference(tmp_path):
    rng = np.random.default_rng(0)
    sizes = [3000, 500, 2000]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    paths = [str(tmp_path / f"table_{i}.lance") for i in range(len(sizes))]
    for path, size in zip(paths, sizes):
        write_table(path, *random_rows(rng, size, np.int32))
    reader = FederatedLanceReader([
        LanceDatasetEntry(f"table_{i}", int(start), int(start + size), path)
        for i, (path, start, size) in enumerate(zip(paths, starts, sizes))
    ])
    n_total = sum(sizes)
    random_order = rng.choice(n_total, 400, replace=False)
    batches = [
        [4_000],
        list(range(2_900, 3_600)),  # across a table boundary, in order
        random_order.tolist(),  # all tables, any order
        list(random_order[::-1]) + list(random_order[:50]),  # duplicates
        [5, 5, 5, 3_200, 5],
        list(range(100, 900, 3)),  # sorted, distinct rows of one table: the batch is the read block itself
        list(range(600, 640)) + list(range(10, 30)) + list(range(20, 25)) + [3_100, 3_101],  # runs, any order
    ]
    for indices in batches:
        batch = reader.read_expression_flat(indices)
        offsets, genes, counts = read_row_by_row(paths, starts, indices)
        np.testing.assert_array_equal(batch.global_row_index, np.array(indices, dtype=np.int64))
        np.testing.assert_array_equal(batch.row_offsets, offsets)
        np.testing.assert_array_equal(batch.expressed_gene_indices, genes)
        np.testing.assert_array_equal(batch.expression_counts, counts)
        assert batch.row_offsets.dtype == np.int64
        assert batch.expressed_gene_indices.dtype == np.int32
        assert batch.expression_counts.dtype == np.int32
    # An aggregate reader reads one table, whatever its routing entries.
    aggregate = AggregateLanceReader(paths[0], [DatasetEntry("a", 0, 1000), DatasetEntry("b", 1000, 3000)])
    indices = rng.integers(0, 3000, 300).tolist()
    batch = aggregate.read_expression_flat(indices)
    for actual, expected in zip([batch.row_offsets, batch.expressed_gene_indices, batch.expression_counts],
                                read_row_by_row(paths, starts, indices)):
        np.testing.assert_array_equal(actual, expected)
    assert reader.read_expression_flat([]).batch_size == 0
    with pytest.raises(IndexError):
        reader.read_expression_flat([0, n_total])
    with pytest.raises(IndexError):
        reader.read_expression_flat([-1])


def test_one_take_of_65536_rows_equals_chunked_takes(tmp_path):
    rng = np.random.default_rng(1)
    path = str(tmp_path / "large.lance")
    write_table(path, *random_rows(rng, 70_000, np.int32), rows_per_file=20_000)
    rows = np.sort(rng.choice(70_000, 65_536, replace=False))
    dataset = lance.dataset(path)
    one_take = dataset.take(rows, columns=TWO_COLUMNS)
    chunked = pa.concat_tables([dataset.take(rows[i:i + 2048], columns=TWO_COLUMNS) for i in range(0, len(rows), 2048)])
    for column in TWO_COLUMNS:
        assert one_take.column(column).combine_chunks().equals(chunked.column(column).combine_chunks())
    batch = FederatedLanceReader([LanceDatasetEntry("large", 0, 70_000, path)]).read_expression_flat(rows)
    genes = chunked.column("expressed_gene_indices").combine_chunks()
    np.testing.assert_array_equal(batch.row_offsets, genes.offsets.to_numpy())
    np.testing.assert_array_equal(batch.expressed_gene_indices, genes.flatten().to_numpy())
    np.testing.assert_array_equal(batch.expression_counts,
                                  chunked.column("expression_counts").combine_chunks().flatten().to_numpy())


@pytest.mark.parametrize(
    ("counts_dtype", "special_counts"),
    [
        ("int32", [65_535, 65_536, 70_000, 1_000_000, 2**31 - 1]),
        ("float32", [0.5, 65_536.25, 1e-30, 3.4e38, 123_456.78]),
    ],
)
def test_packed_table_reads_like_two_column(tmp_path, counts_dtype, special_counts):
    rng = np.random.default_rng(2)
    rows = random_rows(rng, 2500, np.dtype(counts_dtype), special_counts)
    write_table(str(tmp_path / "two_column.lance"), *rows)
    write_table(str(tmp_path / "packed.lance"), *rows, packed=True)
    two_column = FederatedLanceReader([LanceDatasetEntry("t", 0, 2500, tmp_path / "two_column.lance")])
    packed = FederatedLanceReader([LanceDatasetEntry("t", 0, 2500, tmp_path / "packed.lance", "packed", counts_dtype)])
    random_order = list(range(20)) + rng.integers(0, 2500, 1000).tolist()  # includes the special values
    for indices in [random_order, sorted(set(random_order))]:  # copied by ranges; one block in stored order
        expected = two_column.read_expression_flat(indices)
        assert expected.expression_counts.dtype == np.dtype(counts_dtype)
        assert set(np.asarray(special_counts, dtype=counts_dtype)) <= set(expected.expression_counts)
        assert_same_batch(packed.read_expression_flat(indices), expected)


def test_packed_standalone_corpus_reads_like_two_column_in_a_view(tmp_path):
    rng = np.random.default_rng(3)
    counts = sparse.random(60, 30, density=0.3, random_state=4, format="csr", dtype=np.float64)
    counts.data = np.ceil(counts.data * 100_000)  # above 65,535
    obs = pd.DataFrame(index=[f"c{i}" for i in range(60)])
    var = pd.DataFrame(index=[f"g{j}" for j in range(30)])
    ad.AnnData(counts, obs=obs, var=var).write_h5ad(tmp_path / "input.h5ad")
    from_h5ad(tmp_path / "input.h5ad", tmp_path / "two_column", chunk_rows=25)

    # Packed copy: same metadata, expression rewritten packed, layout recorded in corpus.yaml.
    shutil.copytree(tmp_path / "two_column", tmp_path / "packed")
    shutil.rmtree(tmp_path / "packed" / "expression.lance")
    rows = lance.dataset(tmp_path / "two_column" / "expression.lance").to_table().to_pylist()
    manifest = yaml.safe_load((tmp_path / "packed" / "corpus.yaml").read_text())
    write_table(
        str(tmp_path / "packed" / "expression.lance"),
        [np.array(row["expressed_gene_indices"], dtype=np.int32) for row in rows],
        [np.array(row["expression_counts"], dtype=manifest["dtype"]) for row in rows],
        packed=True, rows_per_file=25,
    )
    manifest["expression_layout"] = "packed"
    (tmp_path / "packed" / "corpus.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    view = concat({"two_column": load_corpus(tmp_path / "two_column"), "packed": load_corpus(tmp_path / "packed")})
    local_rows = rng.integers(0, 60, 200)
    expected = view.expression_reader.read_expression_flat(local_rows)
    actual = view.expression_reader.read_expression_flat(local_rows + 60)
    # Same cells from the packed member: equal except for their global row indices.
    assert_same_batch(actual, replace(expected, global_row_index=local_rows + 60))
    assert expected.expression_counts.max() > 65_535


def test_tables_with_different_counts_dtypes_promote_like_concatenation(tmp_path):
    rng = np.random.default_rng(5)
    write_table(str(tmp_path / "int32.lance"), *random_rows(rng, 300, np.int32))
    write_table(str(tmp_path / "float32.lance"), *random_rows(rng, 200, np.float32))
    reader = FederatedLanceReader([
        LanceDatasetEntry("int32", 0, 300, tmp_path / "int32.lance"),
        LanceDatasetEntry("float32", 300, 500, tmp_path / "float32.lance"),
    ])
    indices = rng.integers(0, 500, 400).tolist()
    batch = reader.read_expression_flat(indices)
    # Row-by-row reads joined with np.concatenate, as the old reader did: int32 and float32 give float64.
    counts_per_row = []
    for index in indices:
        table, row = ("int32.lance", index) if index < 300 else ("float32.lance", index - 300)
        counts_per_row.append(lance.dataset(tmp_path / table).take([row], columns=TWO_COLUMNS)
                              .column("expression_counts").combine_chunks().flatten().to_numpy())
    expected_counts = np.concatenate(counts_per_row)
    assert expected_counts.dtype == np.float64
    assert batch.expression_counts.dtype == np.float64
    np.testing.assert_array_equal(batch.expression_counts, expected_counts)


@pytest.mark.parametrize("max_gap", [0, 1, 4, 10**9])
def test_max_gap_keeps_batches_and_reads_gaps(tmp_path, monkeypatch, max_gap):
    monkeypatch.setattr(expression, "_TAKE_PIECE_ROWS", 2)  # several takes per table
    rng = np.random.default_rng(6)
    sizes = [1000, 300, 700]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    entries = []
    for i, (start, size) in enumerate(zip(starts, sizes)):
        write_table(str(tmp_path / f"table_{i}.lance"), *random_rows(rng, size, np.int32), rows_per_file=250)
        entries.append(LanceDatasetEntry(f"table_{i}", int(start), int(start + size), tmp_path / f"table_{i}.lance"))
    reference = FederatedLanceReader(entries)
    reader = FederatedLanceReader(entries)
    reader.max_gap = max_gap
    rows_taken = []
    take_rows = reader._take_rows

    def recording_take_rows(table, rows):
        rows_taken.append(len(rows))
        return take_rows(table, rows)

    monkeypatch.setattr(reader, "_take_rows", recording_take_rows)
    random_order = rng.choice(2000, 300, replace=False)
    batches = [
        random_order.tolist(),  # all tables, any order
        list(random_order[::-1]) + list(random_order[:40]),  # duplicates
        list(range(100, 900, 3)),  # sorted, distinct rows of one table: no-copy path without filling
        list(range(950, 1100, 2)) + [5, 5, 1999],  # gaps across a table boundary are never filled
    ]
    for indices in batches:
        rows_taken.clear()
        assert_same_batch(reader.read_expression_flat(indices), reference.read_expression_flat(indices))
        # Rows read: the distinct requested rows plus every gap of at most max_gap rows inside a table.
        expected = 0
        for start, size in zip(starts, sizes):
            rows = np.unique([i for i in indices if start <= i < start + size])
            gaps = np.diff(rows) - 1
            expected += len(rows) + gaps[gaps <= max_gap].sum()
        assert sum(rows_taken) == expected
