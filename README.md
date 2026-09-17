# Example v2.0

A starting point for empirical thesis work: define an experiment, check its
implementation, repeat it across seeds, and turn results into tables and figures.
The example compares small MNIST classifiers; replace them with your research.

**Question -> configuration -> execution -> saved evidence -> comparison.**

## 0. From research question to experiment

1. **State what you want to study and why.** Define the goal: test a hypothesis,
   understand a behavior, or improve performance. Specify what you will measure
   and under which conditions.

   **Example study:** we investigate whether increasing a classifier's hidden
   linear depth from two to five layers improves MNIST digit classification under
   a fixed training protocol. The two-layer model is the baseline; our hypothesis
   is that the five-layer model achieves higher mean held-out accuracy across
   training seeds.

   Both models use the same convolutional feature-extractor design and hidden
   width, with weights learned independently. Additional layers increase parameter
   count and computation. The study therefore measures the effect of this
   architectural change; it does not isolate depth at equal capacity or compute.

2. **Define the problem mathematically.** Identify the entities, functions, and
   processes. State the assumptions, what training optimizes, and how performance
   will be assessed.

   **Example formulation:** the training dataset is
   $D_{\mathrm{train}}=\{(x_i,y_i)\}_{i=1}^{n}$, where each normalized image
   $x_i\in\mathbb{R}^{1\times28\times28}$ has a label $y_i\in\{0,\ldots,9\}$.
   A network $f_{\theta,d}$ with parameters $\theta$ and depth $d\in\{2,5\}$ maps
   an image to ten logits. Softmax converts these scores into class probabilities:
   $p_{\theta,d}(x)=\operatorname{softmax}(f_{\theta,d}(x))$.

   The training objective is mean cross-entropy:

   $$\mathcal{L}(\theta;d)=-\frac{1}{n}\sum_{i=1}^{n}\log p_{\theta,d}(x_i)_{y_i}.$$

   Minibatch SGD with momentum updates the parameters. During training, dropout
   makes the network stochastic: updates sample dropout masks as well as batches,
   approximating minimization of the expected training loss over those masks.
   Optimization need not reach a global minimum. Validation loss controls the
   learning-rate schedule and selects the checkpoint.

   Evaluation disables dropout. The primary metric is accuracy on a held-out set $D$:

   $$\operatorname{Acc}(\theta,d;D)=\frac{1}{|D|}\sum_{(x,y)\in D}\mathbf{1}\!\left[\arg\max_k f_{\theta,d}(x)_k=y\right].$$

   Generalization claims assume that evaluation data represent the target
   population. Validation supports development; the separate test set is reserved
   for evaluating the finalized protocol.

3. **Translate the definition into code and interfaces.** Map each entity and
   process to a component. Define the shapes, types, and meanings of inputs and
   outputs so implementations can be exchanged consistently.

   **Example implementation:**

   - **Data $D$: `modules/datasets/`:** load and normalize images, define the split,
     and supply floating-point image batches `(batch, 1, 28, 28)` with integer
     labels `(batch,)`.
   - **Function $f_{\theta,d}$: `modules/models/`:** map images to raw logits
     `(batch, 10)`. Both depths expose the same interface.
   - **Learning and evaluation: `modules/training/`:** compute cross-entropy,
     update parameters, measure validation loss and accuracy, and save the selected
     checkpoint. Cross-entropy accepts logits directly and handles their conversion
     internally.
   - **Experimental choices: `config/`:** specify depth, data settings, seed,
     optimizer, and training budget without changing the implementation.
   - **Execution: `runs/train.py`:** connect these components. `modules/utils/seeds.py`
     configures randomness; `modules/utils/hydraqol.py` records the configuration
     and run status alongside the results.

4. **Turn the protocol into repeatable experimental runs.** At this point, an
   experimental run is **a configuration plus a defined sequence of steps**.
   The configuration states what varies and what stays fixed; the shared procedure
   produces comparable metrics. A study repeats that procedure across settings
   and seeds, then brings the results into a common report.

   **Example sequence:** `runs/prepare.py` downloads the data once. For each configuration,
   `runs/train.py` sets the seed, builds the data split and model, trains and validates,
   selects the checkpoint with lowest validation loss, and saves its metrics.
   Configurations, split indices, checkpoints, and results remain together under
   `data/`. After the runs, `runs/report.py` compares validation accuracy and loss across
   settings. `sweep.sh` and `sweep.bat` express this preparation, execution,
   and reporting sequence as repeatable commands.

   **Example comparison:** `config/experiment/sweep_models.yaml` defines six runs:
   two depths, each with seeds 0, 1, and 2. Defaults keep the 50,000/10,000
   training/validation split, preprocessing, batch size, optimizer settings,
   scheduler rule, and three-epoch budget fixed. Equal epochs mean equal data
   exposure, not equal compute; the validation-driven learning rates may differ.

   The report gives means and standard deviations across seeds. These describe
   initial training variability on one split; they do not establish statistical
   significance or robustness across datasets. Once the protocol is fixed, use
   `runs/evaluate.py` for final test evaluation. Broader claims require more seeds and
   relevant settings, such as data sizes or training budgets, assessed with the
   same metric and a clearly specified protocol.

The outcome should explain **what matters for the research goal, under which
conditions, and how consistently**.

## 1. Run one experiment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from
this repository's root. uv manages Python 3.13 and the local environment.
The default dependencies include CPU PyTorch; no extra selection is needed.
Python 3.12 is also allowed. Python 3.14 currently needs prerelease Hydra/OmegaConf:
stable Hydra 1.3 fails during argument parsing
([upstream issue](https://github.com/hydra-ecosystem/hydra/issues/3121)).
The project stays on stable dependencies until that support is released.

```bash
uv sync --locked
uv run runs/prepare.py
uv run runs/train.py +experiment=quick
```

`prepare` downloads MNIST once. `quick` trains for one epoch on small subsets,
so you can inspect the entire workflow before spending compute. It is not a benchmark.
Each execution prints its artifact directory under `data/outputs/quick/train/`.

## 2. Follow the components

| Location | Responsibility |
|---|---|
| `runs/prepare.py`, `runs/train.py`, `runs/report.py`, `runs/evaluate.py` | Direct entry points connecting configuration, computation and results. |
| `modules/datasets/`, `models/`, `training/` | Implement the scientific computation. |
| `modules/utils/hydraqol.py` | Handle run directories, records and Hydra configuration helpers. |
| `modules/utils/seeds.py` | Configure randomness and execution settings. |
| `config/` | Describe data, models, training, experiments and launchers. |
| `sweep.sh`, `sweep.bat` | Repeat the preparation, training sweep, and reporting workflow. |
| `data/` | Store datasets and **all generated results**, including `outputs/`. Ignored by Git. |
| `pyproject.toml`, `uv.lock` | Declare dependencies and record resolved versions. |

`modules` is an ordinary importable package. You can rename it or use `src/thesis`
(or another name); update imports, YAML `_target_` paths, package configuration,
and any moved Hydra config paths accordingly. Packaging supports local imports;
this repository does not need to become a published library.

## 3. Change the experiment, not the implementation

[Hydra](https://hydra.cc/docs/1.3/intro/) combines YAML groups and command-line overrides.
For example, `config/model/net.yaml` points `_target_` to `modules.models.simple_net.Net`;
its remaining `object` fields become constructor arguments.

```bash
uv run runs/train.py --cfg job --resolve
uv run runs/train.py study=baseline training.epochs=3 seed=1
uv run runs/train.py study=batchnorm model=net_bn
```

The model returns logits; training applies cross-entropy. A fixed 10,000-image
validation split controls learning-rate changes and checkpoint selection. The test
set is used only by `evaluate`. Decide your question, metric, baseline, split and
compute budget before changing these choices.

Keep functions focused, annotate useful interfaces, and document shapes, assumptions
and scientific choices. Use short numbered comments for meaningful steps and cite
methods where implemented. Configuration values belong in YAML, not copied docstrings.

## 4. Repeat and compare

`config/experiment/sweep_models.yaml` specifies two depths x three seeds = six runs.
Use a new `study` name for each sweep; configurations are fixed except for the chosen
variables. The split seed stays fixed while the training seed varies.

```bash
uv run runs/train.py +experiment=sweep_models study=depth_01
uv run runs/report.py study=depth_01
```

Reports appear in `data/outputs/depth_01/report/`: `runs.csv`, numeric `summary.csv`
and `comparison.png`. Identical data/model/training configurations form a group;
the training seed varies within it. Standard deviation is blank for one observation.
The report skips unfinished runs and prints their names. Check the completed count
against the six planned runs and compare the `runs` and distinct `seeds` columns.
Repeating a completed training configuration skips it by default. Use a new study
when changing code, environment or datasets; keep them fixed within a comparison.

The root Bash/Batch scripts show the same commands in order. Pass a new study name:

```bash
bash sweep.sh depth_02      # Bash
sweep.bat depth_02         # Windows Command Prompt
```

After selecting a configuration using validation, evaluate its checkpoints on the
held-out test set. Replace the path below with a completed training directory:

```bash
uv run runs/evaluate.py study=final run_dir=data/outputs/depth_01/train/RUN_ID
```

Choose the seed set before evaluation; do not select the best seed using test scores.
For a multi-seed final result, evaluate every selected seed and retain all results.

## 5. Understand the saved evidence

A training directory contains resolved `config.yaml`, exact `split.json`, per-epoch
`history.csv`, selected-checkpoint `metrics.json`, `weights.pt` and `run_info.json`.
The run record includes commands, timing, state, task result, Git state and package
versions; `uv.lock` is copied when available. Logs live in the study's `logs/` directory.
`weights.pt` contains model weights, not the optimizer state needed to resume training.

`@run_decorator` below `@hydra.main` owns run handling in `modules/utils/hydraqol.py`.
YAML defines `save_dir`; tasks use `Path(cfg.save_dir)` and return their results normally.
Training directory names hash the data/model/training settings and seed.

| Mode | Behavior for the configured `save_dir` |
|---|---|
| `base` (default) | Skip completed runs; run new or incomplete ones. |
| `check` | Report status without running the task. |
| `clean` | Delete incomplete runs and stop; retain completed ones. |
| `force` | Delete the existing run and execute again. |

```bash
uv run runs/train.py +experiment=quick mode=check
uv run runs/train.py +experiment=quick mode=force
```

`max_retries=0` and `retry_delay=5` control optional single-process retries. A retry
restarts the task, not its checkpoint. Errors are saved per rank/attempt and re-raised;
interrupted runs can remain `running`. Set `wrapper_quiet=true` to reduce lifecycle logs.
Only directories containing a run configuration can be deleted by `clean`/`force`.

For older workflows, `venv_force=true venv=.venv` selects an existing interpreter;
`venv` also accepts a parent directory containing `.venv`. Keep only Hydra and the
wrapper imports at module scope so switching happens before project dependencies load.
Each sweep job is passed separately as resolved configuration. Across interpreters,
results are transferred as JSON (other objects become strings). Normal uv usage needs
no forced environment. Standalone callers of `venv_force_check` receive `(switched, result)`.

The file is self-contained apart from Hydra/OmegaConf, with optional PyTorch rank
coordination. Its resolver names are kept consistent for later reuse across projects:

| Resolvers | Use |
|---|---|
| `default`, `default_if_missing`, `math`, `ceildiv` | Null fallback and arithmetic; `math` includes `min`/`max`. |
| `config_hash`, `percentagestring` | Stable configuration IDs and percentage labels. |
| `concat`, `grid_range` | Combine lists and construct nested Cartesian grids. |
| `cpu_count`, `mp_start_method`, `device_to_backend` | Resource and platform settings. |
| `n_patches_overlap`, `n_patches_padded` | Patch counts with truncated or padded context. |

Examples: `${math:*,4,128}`, `${ceildiv:129,64}`, `${default:${value},1}`.
`register_resolvers()` is idempotent; the first registered implementation wins. Use the
same module version when combining projects. `cpu_count` respects affinity/Slurm limits;
`mp_start_method` follows Python's platform default (choose `spawn` explicitly for CUDA).

Migration: retain `cfg.save_dir` and `run_info.json`; use top-level retry settings
(nested `retry` is still accepted when top-level values are absent). `clean` follows
FMDS/SynthSeries deletion-only behavior. Readers accept the earlier example's `run.json`
and TSFoundation's success records. GPU-memory tracking is intentionally separate.

Deterministic operations are enabled by default, but identical seeds do not guarantee
identical results across hardware/software; see [PyTorch's guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).
Commit code/configs and `uv.lock` before experiments. Archive datasets and study outputs
separately: `data/` is ignored, and Git metadata is not a source backup.

## 6. Scale the same experiment

For two local CPU jobs at a time:

```bash
uv run --extra parallel runs/train.py +experiment=sweep_models study=local_01 +launcher=joblib
```

For an NVIDIA GPU, change the `pytorch` index URL in `pyproject.toml` from `/cpu`
to `/cu126` once, then regenerate the lockfile and environment. This selects the
installed build; `training.device=cuda` selects where computation runs. CUDA builds
are for Linux/Windows; macOS uses PyPI. Commit the dependency choice for your study:

```bash
uv lock
uv sync --locked
uv run runs/prepare.py
uv run runs/train.py +experiment=quick study=gpu_01 training.device=cuda
```

On Slurm, prepare the environment/data on a permitted node first. Keep the checkout,
environment and `data/` on storage visible to workers. Adapt account, partition,
CPU/memory/time and GPU requests to your site; load site-required modules before
launching. Start with one job, then submit the sweep:

```bash
uv run --extra hpc runs/train.py +experiment=sweep_models study=hpc_01 +launcher=slurmgpu hydra.launcher.partition=YOUR_PARTITION hydra.launcher.account=YOUR_ACCOUNT
```

`+launcher=slurm` is the CPU variant. Each job runs the same task; launchers change
execution resources. Bound array concurrency, CPU threads and data-loader workers.
Workers never download data. CUDA requests fail if no GPU is available.
The `parallel` and `hpc` extras install optional launchers; keep them enabled in
commands that use those launchers.

For multi-rank tasks, initialize the PyTorch process group and select each rank's CUDA
device **before** entering the decorated function; keep the group alive until it returns.
The wrapper broadcasts lifecycle decisions, writes shared records on global rank zero,
and gathers task failures. All ranks must participate; collective hangs/process crashes
remain the launcher's responsibility. Use launcher restarts instead of wrapper retries
for distributed jobs. Select the environment before process-group initialization.
This example's training loop itself is single-process; independent Slurm sweep jobs
need no process group. See [PyTorch distributed setup](https://docs.pytorch.org/docs/stable/distributed).


## 7. Optional container

The root Dockerfile contains CUDA runtime, uv and minimal system tools. Python and
project dependencies are installed with uv when used. GPU drivers and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
belong on the host. Select and lock the `/cu126` PyTorch source as above before
using this GPU container. Linux shell example:

```bash
docker build -t thesis-example .
docker run --rm -it --gpus all -v "$PWD:/workspace" thesis-example
# Inside the container:
uv sync --locked
uv run runs/prepare.py
uv run runs/train.py +experiment=quick training.device=cuda
```

The container environment lives in `/opt/venv`, separate from the host environment.
Mounted `data/` persists after exit. For dependency changes use `uv add`/`uv lock`,
review and commit the updated lockfile. PyTorch sources follow
[uv's PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Adapt for your thesis

Replace the question and example model/data, preserve explicit experimental choices,
and keep the path from command to reported result visible. Use `runs/` for executable experiment steps
and root Bash/Batch scripts for study recipes; move reused logic into `modules/`. Prefer direct steps and short numbered
comments over dispatchers and generic validation layers. Inspect small runs before scaling.
Optional formatting: `uv run ruff check .` and `uv run ruff format .`.

[MIT License](LICENSE).
