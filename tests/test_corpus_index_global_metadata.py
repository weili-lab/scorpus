"""update_corpus_index writes global-metadata.yaml."""

from __future__ import annotations

from pathlib import Path

from scorpus.materializers import update_corpus_index
from scorpus.materializers.models import DatasetJoinRecord

# ---------------------------------------------------------------------------
# update_corpus_index global metadata
# ---------------------------------------------------------------------------


class TestUpdateCorpusIndexGlobalMetadata:
    def test_new_corpus_writes_global_metadata(self, tmp_path: Path):
        """Creating a new corpus index writes global-metadata.yaml."""
        corpus_index = tmp_path / "corpus-index.yaml"
        dataset_record = DatasetJoinRecord(
            dataset_id="ds1",
            join_mode="create_new",
            manifest_path="meta/materialization-manifest.yaml",
            cell_count=100,
            global_start=0,
            global_end=100,
        )
        update_corpus_index(
            corpus_index_path=corpus_index,
            new_dataset_record=dataset_record,
            corpus_id="test-v0",
            backend="lance",
            topology="aggregate",
        )
        # Verify global-metadata.yaml was written
        global_meta_path = tmp_path / "global-metadata.yaml"
        assert global_meta_path.exists()
        import yaml

        meta = yaml.safe_load(global_meta_path.read_text())
        assert meta["backend"] == "lance"
        assert meta["topology"] == "aggregate"

