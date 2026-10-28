# Backend Note

The only storage backend is `lance`.

The maintained corpus topologies are:

- `aggregate`: one Lance table for all datasets, `matrix/aggregated-cells.lance`
- `federated`: one Lance table per dataset, `<dataset_id>/matrix/<dataset_id>.lance`

Use aggregate topology for the normal create, append, canonicalize, and load
path. Use federated topology when whole-dataset isolation or recomposition is
more important than one shared aggregate matrix object.
