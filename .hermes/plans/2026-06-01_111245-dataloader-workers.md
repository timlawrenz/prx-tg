---
goal: Configure get_production_dataloader with pin_memory=True and 8-12 num_workers
---

# Plan: Configure Production DataLoader for Optimal Throughput

## Current Context / Assumptions
- The training loop expects `get_production_dataloader` (in `production/data.py`) to yield batches.
- The configuration file explicitly specifies `num_workers: 4`, `prefetch_factor: 2`, and `pin_memory: true` under `data`.
- However, the actual dataloader implementations (`WebDataset` via `BucketAwareDataLoader` or `StratumDataset` via `get_stratum_dataloader`) do **not** use `torch.utils.data.DataLoader`. Instead, they directly implement `__iter__` or return an iterable `wds.WebDataset` that is iterated over synchronously.
- Because they are evaluated synchronously in the main thread (without a wrapping PyTorch DataLoader), they completely ignore the config's `num_workers` and `pin_memory` settings. This creates a severe CPU bottleneck during high-throughput execution (like BF16 Tensor Cores or compiled models).

## Proposed Approach
To enable parallel background loading and memory pinning, we need to wrap the base dataset generators (`BucketAwareDataLoader` and `StratumDataset`) inside standard `torch.utils.data.DataLoader` instances. 

Since WebDataset natively supports PyTorch DataLoaders and `StratumDataset` behaves as an IterableDataset (it implements `__iter__`), we can safely pass both into a PyTorch DataLoader configured with the settings from `config.data`.

## Step-by-step plan

1. **Modify `get_production_dataloader` in `production/data.py`**
   - Import `torch.utils.data.DataLoader`.
   - Extract `num_workers`, `pin_memory`, and `prefetch_factor` from `config.data` (defaulting to 8, True, and 2 respectively if missing or invalid).
   - *For the Stratum source:* Update `get_stratum_dataloader` to return the `StratumDataset` wrapped in a `DataLoader`.
   - *For the WebDataset source:* Wrap the returned `BucketAwareDataLoader` inside a PyTorch `DataLoader`.

2. **Handle IterableDataset quirks**
   - Because both custom dataloaders act like PyTorch `IterableDataset`s returning pre-collated batches, the wrapper `DataLoader` must have `batch_size=None`. This tells PyTorch not to attempt to re-batch the outputs.
   - For `StratumDataset`, multiple workers might yield the exact same samples if we don't implement worker-aware sharding. We must update `StratumDataset.__iter__` to split `self._dirs` across workers using `torch.utils.data.get_worker_info()`.

3. **Update `StratumDataset.__iter__` (`production/data_stratum.py`)**
   - Add worker-aware chunking:
     ```python
     worker_info = torch.utils.data.get_worker_info()
     if worker_info is None:
         worker_dirs = list(self._dirs)
     else:
         per_worker = int(math.ceil(len(self._dirs) / float(worker_info.num_workers)))
         worker_id = worker_info.id
         worker_dirs = self._dirs[worker_id * per_worker:(worker_id + 1) * per_worker]
     ```

4. **Verify Configuration**
   - Check `experiments/arm_i_fp8/config.yaml` to ensure `num_workers` is set to 8 or 12.

## Files likely to change
- `production/data.py`
- `production/data_stratum.py`

## Tests / Validation
- Run a short 50-step sanity check `python -m production.train_production --config experiments/arm_i_fp8/config.yaml`.
- Verify using `top` or `ps aux` that multiple Python worker processes spawn.
- Monitor throughput to confirm iter/sec increases due to asynchronous prefetching.

## Risks, tradeoffs, and open questions
- **BucketAwareDataLoader thread-safety:** `BucketAwareDataLoader` relies on iterating over multiple `WebDataset` instances. Wrapping it in a multi-worker `DataLoader` might cause identical sampling across workers unless `wds.WebDataset` is correctly configured for worker splitting. WebDataset usually handles this internally via its nodes/shards logic if `wds.split_by_worker` is in the pipeline. We need to ensure WebDataset doesn't duplicate data.
- **Stratum NAS bottleneck:** If 12 workers simultaneously hammer the Stratum NAS via SMB/NFS to read hundreds of small `.npy` files per second, the network or disk IO might become the new bottleneck, potentially causing hangs or timeouts. We should monitor IO wait (`wa` in `top`).