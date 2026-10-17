# pertTF integration

The paired loader, the gene token vocabulary and the gene sampling that turns a cell into
a fixed-length gene sequence live in pertTF: `perttf.model.corpus_adapter`. Which cells are
paired is described with `scorpus.pairing` (see [`pairing.md`](pairing.md)).

Use `perttf.model.corpus_data.produce_corpus_datasets` for corpus-native training
and validation, including a shared unified-encoder perturbation mapping.
The pertTF worktree documents this in `demos/tutorials/CORPUS_TRAINING.md` and
`demos/tutorials/CORPUS_PAIRED_LOADER.md`.

Data preparation and temporary composition are documented here in
[`composable_corpora.md`](composable_corpora.md).
