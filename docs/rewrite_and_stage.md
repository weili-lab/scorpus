# Rewriting and staging corpora

`reorder_corpus` writes a standalone corpus again with its cells in a new order. With
`pairing.write_order`, every group becomes one contiguous block, shuffled within, so
`Sampler(..., run_length=r)` reads runs of neighbouring rows. `write_order(groups, seed,
keep=rows)` keeps only the given rows (for example the cells passing your own QC filter):
the new corpus holds only those cells.

`scorpus.stage` / `scorpus.staged` copy corpora to faster storage first (node-local NVMe,
or RAM under `/dev/shm`).

```python
from scorpus import load_corpus, reorder_corpus, staged
from scorpus.pairing import make_groups, write_order

corpus = load_corpus("source.corpus")
order, blocks = write_order(make_groups(corpus.obs, ["perturbation", "condition"]), seed=0, keep=kept_rows)
note = {"by": ["perturbation", "condition"], "seed": 0, "source": "source.corpus"}

with staged(["source.corpus"], "/scratch1/me/job") as [local_source]:
    reorder_corpus(local_source, "/scratch1/me/job/new.corpus", order, note, workers=8)
```

## Choosing a route

Every route gives the same corpus, row for row. Timings, memory and temporary disk for
each route have not been measured yet.

| Route | Call | Needs |
|---|---|---|
| NVMe staging | `staged([...], "/scratch1/...")`, then `reorder_corpus(..., workers=8)` | local disk for the source and the new corpus |
| RAM staging (opt-in) | `staged([...], "/dev/shm/...")`, then as above | job memory for the source copy, the workers' readers and the new corpus's page cache |
| Bins | `reorder_corpus(..., workers=8, method="bins", temp_dir=...)` | temporary disk for the bins, zstd-compressed by default (`bin_compression="zstd"`); reads the source sequentially. Measured on Replogle and Marson: bins about 4x smaller (Replogle 104 to 26 GB; Marson 1.15 TB to 275 GB), peak temp disk 2.3-2.8x lower, write 1.30-1.43x faster on Replogle and 2.14x on Marson (cross-node), outputs identical. Cost: more CPU (7.5-11 busy cores of 16 vs 3.4-4.6) and on Marson 343 GB summed process RSS in pass 1 at 16 workers (use a 384G job or fewer workers). lz4 was worse than zstd on size and time. `bin_compression=None` gives uncompressed bins (about 8 bytes per nonzero value) |
| Plain network storage | `reorder_corpus(..., workers=8)` on the source where it is | nothing extra; every row is a separate request |

- **RAM:** `stage` refuses a `/dev/shm` copy that would not fit in the job's memory limit;
  use `staged` so the copies are deleted, since `/dev/shm` belongs to the node, not the job.
- **Workers:** the new order is cut into chunks of at most `reorder.FRAGMENT_ROWS` rows, at
  least one per worker; every chunk is one Lance fragment, the same for every route.
- **Bins:** see `method="bins"` and `temp_dir` in `reorder_corpus`.
