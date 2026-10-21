# Remote GPU runner

Everything that needs a GPU runs on a Kaggle Tesla T4 through `remote/kaggle.py`. The laptop that
records the data has no NVIDIA GPU, so the runner ships the code (by git SHA) and the archive (as a
private Kaggle dataset) to the card, runs one job and pulls the results back into `results/<run_id>/`.

## How a job runs

1. `python -m remote.kaggle run --job <name> ...` composes `remote/.push/<run_id>/run.py` from the
   bootstrap in `remote/base.py` and the job body in `remote/jobs/<name>.py`, writes
   `kernel-metadata.json` (script kernel, private, GPU on, internet on, `machine_shape: NvidiaTeslaT4`,
   the dataset attached) and pushes it with `kaggle kernels push`.
2. The bootstrap's first action is
   `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`. Any card whose name does not
   contain `T4` makes the job exit non-zero; the runner re-pushes up to three times.
3. It records the environment (`env.json`: GPU properties from `cudaGetDeviceProperties`, driver,
   CUDA runtime, nvcc, Nsight Systems, g++, CPU model and cores, clocks, temperature, power),
   installs Nsight Systems (`cuda-nsight-systems-12-8`) when the image lacks it, clones this
   repository and checks out the pinned SHA.
4. The job body runs (`bench_all` runs `python -m bench.run_all` on the attached archive after checking
   its content hash and source). The bootstrap writes `manifest.json` with git SHA, archive hash,
   source, GPU name, driver, CUDA version, clocks before and after, CPU model and core count.
5. The runner polls `kaggle kernels status`, downloads the output bundle and the log, and unpacks it
   into `results/<run_id>/` (plus a flattened `kernel.txt`).

## Quota

Kaggle gives about 30 GPU hours a week. `remote/quota.json` is a ledger of every run (pushed, running,
finished, minutes on the GPU) and the `kaggle quota` report; the runner refuses to push a job that would
take the estimated total above 20 GPU hours. Jobs are kept under 45 minutes.

## Commands

```bash
python -m remote.kaggle probe                       # environment probe on a T4
python -m remote.kaggle dataset --archive data/archive --slug exchange-engine-archive --wait-ready
python -m remote.kaggle run --job bench_all --dataset tanushbhatia/exchange-engine-archive \
    --source real --archive-hash <content_hash from data/archive/archive.json>
python -m remote.kaggle run --job bench_quick --dataset tanushbhatia/exchange-engine-archive-synth --source synthetic
python -m remote.kaggle quota
python -m bench.reproduce --source real             # build, upload, run, pull, render, check
```

The Kaggle token lives in `~/.kaggle/access_token` and is read only by the `kaggle` executable. It is
never printed, logged, uploaded or committed, and the runner scrubs anything token-shaped from the
output it keeps.

## GitHub Actions

`.github/workflows/gpu.yml` is a manual (`workflow_dispatch`) workflow that runs the same pipeline from
GitHub. It needs a repository secret:

1. On Kaggle, Settings > API > create a new token and copy it.
2. On GitHub, open the repository's Settings > Secrets and variables > Actions > New repository secret.
3. Name it `KAGGLE_API_TOKEN` and paste the token.

Without the secret the workflow exits early with a notice. The raw recording is not in the repository,
so the real-data run reuses the archive version already uploaded to Kaggle.

## Colab fallback

`remote/colab_exchange_engine.ipynb` is self-contained: it stops unless the runtime has a T4, clones the
`gpu-analytics` branch, downloads the Kaggle dataset (after you upload your own token) or generates a
synthetic archive, runs `python -m bench.run_all --quick` and zips `results/` for download.

## SSH backend (documented, not implemented)

Nsight Compute needs access to the GPU performance counters, which Kaggle's container denies
(`ERR_NVGPUCTRPERM`, see `DECISIONS.md`). A rented cloud virtual machine with a T4, where the profiling
permission can be enabled (`NVreg_RestrictProfilingToAdminUsers=0`, or running as root), would add a
third backend behind the same interface:

1. `rsync -az --exclude data/raw ./ vm:exchange-engine/` and `rsync data/archive vm:exchange-engine/data/`;
2. `ssh vm 'cd exchange-engine && python -m bench.run_all --archive data/archive --out results/<run_id> --source real'`,
   or `ncu --set full -o profiles/<kernel> python bench/bench_layout.py --layout soa ...` for kernel-level
   counters (achieved occupancy, L2 hit rate, DRAM throughput per kernel);
3. `rsync vm:exchange-engine/results/<run_id>/ results/<run_id>/` and the same manifest checks.

It is not built because Nsight Systems answers every question this project asks (kernel time, overlap,
transfer share) and the Kaggle allowance is free.
