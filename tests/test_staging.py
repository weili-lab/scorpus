from collections import namedtuple

import anndata as ad
import pandas as pd
import pytest
from scipy import sparse

from scorpus import from_h5ad, staging


def write_corpus(tmp_path, name="tiny"):
    values = [[1, 0, 2], [0, 0, 0], [3, 1, 0], [0, 5, 0]]
    obs = pd.DataFrame({"condition": ["ctrl", "p", "p", "ctrl"]}, index=[f"c{i}" for i in range(4)])
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    path = tmp_path / f"{name}.h5ad"
    ad.AnnData(sparse.csr_matrix(values), obs=obs, var=var).write_h5ad(path)
    from_h5ad(path, tmp_path / "nfs" / f"{name}.corpus", chunk_rows=2)
    return tmp_path / "nfs" / f"{name}.corpus"


def test_stage_round_trips(tmp_path, monkeypatch):
    # Small ranges so every file is copied in several pieces.
    monkeypatch.setattr(staging, "RANGE_BYTES", 100)
    roots = [write_corpus(tmp_path, "a"), write_corpus(tmp_path, "b")]
    (tmp_path / "nvme").mkdir()
    staged_roots = staging.stage(roots, tmp_path / "nvme")
    assert staged_roots == [tmp_path / "nvme" / "a.corpus", tmp_path / "nvme" / "b.corpus"]
    for root, staged_root in zip(roots, staged_roots):
        source_files = staging._files(root)
        assert [p.relative_to(root) for p in source_files] == [p.relative_to(staged_root) for p in staging._files(staged_root)]
        for source in source_files:
            assert (staged_root / source.relative_to(root)).read_bytes() == source.read_bytes()


def test_stage_refuses_when_it_does_not_fit(tmp_path, monkeypatch):
    root = write_corpus(tmp_path)
    (tmp_path / "nvme").mkdir()
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(staging.shutil, "disk_usage", lambda path: Usage(10, 9, 1))
    with pytest.raises(OSError, match="Staging needs"):
        staging.stage([root], tmp_path / "nvme")
    assert list((tmp_path / "nvme").iterdir()) == []


def test_staged_deletes_copies_on_exit(tmp_path):
    root = write_corpus(tmp_path)
    (tmp_path / "nvme").mkdir()
    with staging.staged([root], tmp_path / "nvme") as staged_roots:
        assert (staged_roots[0] / "corpus.yaml").is_file()
    assert not staged_roots[0].exists()
