# Example: an empirical experimental pipeline with hydra

## 1) Objective
This repository is a teaching template that demonstrates how to build, run, and compare empirical ML experiments using Hydra. It shows students how to keep configs, code, and results organized so ideas can be tested quickly and reproduced later.

## 2) Motivation
Structuring experiments up front reduces “glue code”, makes every run reproducible, and keeps baselines and new ideas directly comparable. The same layout scales from a laptop to large parallel executions (joblib locally or Slurm on HPC) without refactoring. A clean layout also helps collaborators (and future you) understand what was run, with which settings, and where the results live.

## 3) Ask yourself first
- What exact question am I answering, and which metrics define success?
- Which datasets, models, and hyperparameters will vary?
- What baselines will I compare against?
- How will I ensure repeatability (seeds, logged configs, versioned data/code)?
- Where will results be stored and how will they be aggregated?
- What compute/launcher do I need (local, joblib, Slurm, GPU)?

## 4) Repository structure (what goes where)
- `config/` — Hydra configs split by domain (model, data, training, launcher, experiments). Add new models by creating a new YAML under `config/model/`, e.g., `net_bn.yaml` pointing to a class in `modules/models/`.
- `modules/` — Python source. Put new model code in `modules/models/`, datasets in `modules/datasets/`, training loops in `modules/training/`, shared utilities in `modules/utils/`.
- `runs/` — CLI entrypoints (tasks). Keep each logical task here (e.g., `train.py`, `report.py`).
- `data/` — downloaded datasets or artifacts.
- `run_all_tasks.*` — convenience scripts to chain tasks.
- `env_setup/` — environment files (Dockerfile, requirements).
- Outputs — Hydra writes under `outputs/...`; trained models and `result.json` per run go under `models/...` (see `config/path/relative.yaml`).

If you add a new model: implement it in `modules/models/my_model.py`, expose it via `_target_` in a new `config/model/my_model.yaml`, then reference it on the CLI with `model=my_model`.

## 5) Designing the pipeline
**Why structure experiments first?**  
- Removes ambiguity: goals, metrics, and success criteria are written down before coding.  
- Reproducibility by default: every run records the exact config snapshot Hydra used.  
- Faster iteration: you can sweep parameters or models with one CLI call instead of editing code.  
- Comparability: baselines and new ideas share the same data/metrics pipeline.  

**What to consider**  
- **Baselines:** start with `modules/models/simple_net.py` (`config/model/net2.yaml`) and compare to the BatchNorm variant `modules/models/simple_net_bn.py` (`config/model/net_bn.yaml`).  
- **Repeatability:** set `seed` in `config/train_model.yaml`; Hydra stores the resolved config per run.  
- **Ease of use & readability:** prefer small, composable YAML files; override via CLI instead of editing Python.  

## 6) Configs and instantiation
- A config is a declarative YAML describing how to build an object. Hydra injects `_target_` to map config → Python class.  
- Example models:  
  - `config/model/net2.yaml`
    ```yaml
    object:
      _target_: modules.models.simple_net.Net
      num_layers: 2
      latent_dim: 128
    ```
  - `config/model/net_bn.yaml`
    ```yaml
    object:
      _target_: modules.models.simple_net_bn.NetBN
      num_layers: 2
      latent_dim: 128
      dropout: 0.3
    ```
- Instantiation happens inside `runs/train.py`:
  ```python
  train_loader = hydra.utils.instantiate(cfg.data.dataloaders.train)
  test_loader = hydra.utils.instantiate(cfg.data.dataloaders.test)
  model = hydra.utils.instantiate(cfg.model.object).to(device)
  ```
- Equivalent manual Python (without Hydra) for the BatchNorm variant:
  ```python
  from modules.models.simple_net_bn import NetBN

  model = NetBN(num_layers=2, latent_dim=128, dropout=0.3)
  model = model.to(device)
  ```

## 7) Anatomy of a single run
- A **run** = one merged config group (model + data + training + path + optional launcher).  
- `runs/train.py` sequence: seed setup → instantiate dataloaders → instantiate model → train/evaluate → save checkpoint + `result.json`.  
- Outputs go to `${path.base_path}/outputs/...` and `${path.base_path_models}/...` (see `config/path/relative.yaml`).  

## 8) Running tasks (env + single run + overrides)
- Create environment (choose one):
  ```bash
  # Conda
  conda create --prefix ./.venv python=3.11
  conda activate ./.venv
  pip install -r env_setup/requirements.txt

  # venv
  python -m venv .venv
  source .venv/bin/activate      # Linux/Mac
  .venv\Scripts\activate         # Windows
  pip install -r env_setup/requirements.txt
  ```
- Default run:
  ```bash
  python runs/train.py
  ```
- Override on the fly:
  ```bash
  python runs/train.py model=net_bn training.epochs=3 seed=123
  ```
- Change layers without touching code:
  ```bash
  python runs/train.py model.object.num_layers=5
  ```

## 9) Experiments as sweeps
- Multirun example:
  ```bash
  python runs/train.py --multirun model=net2,net_bn seed=0,1,2 training.epochs=2
  ```
- Preset sweep (`config/experiment/sweep_models.yaml`):
  ```bash
  python runs/train.py +experiment=sweep_models
  ```
  (Edit the YAML to add `net_bn` if you want it included.)
- Example sweep YAML (what it does):
  ```yaml
  # config/experiment/sweep_models.yaml
  defaults:
    - override /training: basic

  hydra:
    mode: MULTIRUN
    sweeper:
      params:
        model: net2,net5,net7   # try three model depths
        seed: range(0,5)        # run seeds 0–4 for robustness
        training.epochs: 3      # fix epochs for all runs
  ```
  Runs 3 models × 5 seeds = 15 jobs, each with 3 epochs, storing one config/output folder per job.

## 10) Launchers (local and HPC)
- Hydra swaps launch backends via the `launcher` config group. Select with `+launcher=...`.
- Local parallel jobs: `+launcher=joblib` splits multirun work across CPU cores.
- Slurm CPU: `+launcher=slurm`
- Slurm GPU example (`config/launcher/slurmgpu.yaml`):
  ```yaml
  defaults:
    - override /hydra/launcher: submitit_slurm

  hydra:
    callbacks:
      log_job_return:
        _target_: hydra.experimental.callbacks.LogJobReturnCallback
    launcher:
      setup:
        - "module load Python/3.10.4 2>&1"
        - "module load CUDA/12.6.3 2>&1"
        - ". .venv/bin/activate"
        - "nvidia-smi"
        - "python -m torch.utils.collect_env"
      submitit_folder: ${hydra.sweep.dir}/.submitit/%j
      cpus_per_task: 20         # CPU cores per job
      gpus_per_node: 1          # request one GPU
      gres: "gpu:1"             # Slurm gres string
      tasks_per_node: 1
      array_parallelism: 50     # how many array jobs run in parallel
      timeout_min: 30           # walltime per job
  ```
- What it does: running `python runs/train.py --multirun +launcher=slurmgpu` submits a Slurm array via SubmitIt; each job loads modules, activates the venv, collects env info, and trains on one GPU. Logs/config snapshots stay under `outputs/...` and `.submitit/...`.

## 11) End-to-end experimentation flow
1) Sweep over models:  
   ```bash
   python runs/train.py --multirun model=net2,net_bn seed=0,1,2 training.epochs=2
   ```  
   Hydra/SubmitIt fans out jobs; each run writes its merged config, checkpoint, and `result.json` into its own output folder.
2) Generate the report:  
   ```bash
   python runs/report.py base_dir=./models
   ```  
   The reporter uses helper aggregator utilities (`modules/utils/aggregator.py`) to load every `result.json`, normalize to a DataFrame, compute mean/±std, and emit `results_table.csv` plus `results_plot.png`.
3) One-click reproducibility: `./run_all_tasks.sh` (Linux/Mac) or `run_all_tasks.bat` (Windows) chain the same sweep-and-report steps, so anyone can rerun the complete pipeline end-to-end—locally or at cluster scale—without editing code.

## 12) Aggregating and comparing results
- After runs finish, collect metrics into tables/plots:
  ```bash
  python runs/report.py base_dir=./models
  ```
- Output: `results_table.csv`, `results_plot.png` in `reports/...`, showing means and ±std across models/seeds so you can pick winners (e.g., `simple_net` vs `simple_net_bn`).

## 13) Naming conventions for outputs
- Default naming in `config/train_model.yaml`: `${data.name}_${model.name}_${training.name}_${seed}` ensures one folder per parameter group.
- To further disambiguate, append short suffixes via CLI: `suffix=_try1` or override `name=mnist_net_bn_lr1e-3_s123`.
- Keep names deterministic (include model, key hyperparams, and seed) so aggregations cleanly group runs; avoid spaces or ambiguous labels.

## 14) Portability with Docker
- Build the image (uses `env_setup/Dockerfile`):
  ```bash
  docker build -t example-pipeline ./env_setup
  ```
- Run with project mounted:
  ```bash
  docker run --rm -it -v $(pwd):/workspace -w /workspace example-pipeline bash
  ```
- Why: identical environment on laptop, server, or CI; GPU pass-through works with `--gpus all` when the host has NVIDIA runtime.

## 15) Moving data/results with rclone
- Configure remote once:
  ```bash
  rclone config
  ```
- Sync datasets up/down (progress + parallel transfers):
  ```bash
  rclone sync ./data/datasets remote:bucket/path -P --transfers=8
  ```
- Mirror results to cloud for backup/sharing:
  ```bash
  rclone sync ./models remote:bucket/experiments/models -P
  rclone sync ./outputs remote:bucket/experiments/outputs -P
  ```
- Tip: keep large artifacts out of git; use rclone to stage them where your Slurm jobs can read them.

## 16) References and further reading
- Hydra documentation: https://hydra.cc/
- SubmitIt (Hydra launcher backend): https://github.com/facebookincubator/submitit
- Reproducibility in ML: https://www.nature.com/articles/s42256-019-0035-4
- Ten Simple Rules for Reproducible Research: https://doi.org/10.1371/journal.pcbi.1003285
- Experiment tracking tools overview: https://neptune.ai/blog/ml-experiment-tracking-tools
- Docker docs: https://docs.docker.com/
- rclone docs: https://rclone.org/
