# pertTF integration

The paired loader, perturbation sampler, and model-specific vocabulary adapter
now live in pertTF: `perttf.model.corpus_adapter`.

Use `perttf.model.corpus_data.produce_corpus_datasets` for corpus-native training
and validation, including a shared unified-encoder perturbation mapping.
The pertTF worktree documents this in `demos/tutorials/CORPUS_TRAINING.md` and
`demos/tutorials/CORPUS_PAIRED_LOADER.md`.

Data preparation and temporary composition are documented here in
[`composable_corpora.md`](composable_corpora.md).
