
# Example Repository: Workflow

This repository is a template for structuring (empirical) machine-learning projects, especially for bachelor and master thesis work. It shows how to organize experiments, manage configurations, run evaluations, and generate results in a clean and reproducible way.

---

## 1. What is this repository?

This project shows **an example workflow for experimentation**, organized into repeatable steps. It helps you run experiments, compare results, and generate reports in a systematic way.

### 1.1 Why this workflow matters

- Research often needs many experiments. A clear pipeline keeps things organized.  
- You can change settings like model type or training epochs without touching the main code.  
- You can repeat experiments later, or share them with others, which is important for reproducibility.  
- It shows an easy way to use high-performance-computing (HPC) resources without major code changes.  
- Because every run is tracked, results can be explored or reproduced by other people—even months later.  
- A good structure makes it simpler to extend your work in the future.  

### 1.2 Before starting

Before writing any code **define your problem precisely, both semantically and mathematically**. Clear definitions save time and make later experiments meaningful.

| Step | What to write down | Why it matters |
|------|-------------------|----------------|
| **1. Formal problem statement** | Describe the task in plain language *and* in formal terms. <br>Example: “Given an image \(x \in \mathbb{R}^{H\times W\times 3}\), predict a label \(y \in \{0,1,\dots,K-1\}\).” | Removes ambiguity and gives direction. |
| **2. Research question / hypothesis** | Specify exactly **what you are testing**. <br>Is it a new loss, a data-augmentation trick, or a training schedule? | Keeps experiments focused; prevents scope creep. |
| **3. Baselines and comparisons** | List existing methods you will compare against and justify the choice. | Makes results easier to interpret and publish. |
| **4. Evaluation metrics** | Pick metrics that match the goal (e.g., accuracy, F1, MAE, BLEU). Define how each metric is calculated. | A metric defined up front prevents “metric shopping” after the fact. |
| **5. Success criteria** | State the test or comparisons that will show if there is an improvement. | Allows objective and significantconclusions. |

Spend time on this step; a precise target makes every later script simpler to write and review.

### 1.3 Workflow Overview

When you turn a research question into code, follow these steps:

1. **Describe one experimental run.**  
   A run is the smallest unit that produces a set of results (for example: train ➜ evaluate ➜ save metrics).

2. **List parameters you want to vary.**  
   Think of models, hyper-parameters, datasets, data splits, or random seeds.

3. **Group runs into an experiment sweep.**  
   Running the same code many times with different parameters lets you build fair comparisons and collect statistics.

4. **Automate runs with a task manager.**  
   Here we use **Hydra**, which was built at Facebook AI Research to launch Python functions from the command line while reading YAML configuration files.  
   - Each run gets its own config snapshot.  
   - You can override any parameter from the shell.  
   - Multirun mode lets you queue many jobs at once.  

5. **Aggregate results and create a report.**  
   A separate script loads all run outputs, computes summary tables/plots, and writes a PDF or notebook that can be copied straight into a thesis.

---

## 2. Repository Structure

```bash
.
├── modules/              # Code for training, evaluation, and utilities
│   ├── training/
│   ├── utils/
├── data/                 # Data, model checkpoints, and generated reports
│   ├── datasets/
│   ├── models/
│   └── reports/
├── config/               # Hydra configuration files
│   ├── train_model.yaml
│   ├── report.yaml
├── runs/                 # Scripts to run experiments and workflows
│   ├── train.py
│   ├── report.py
│   └── run_all_tasks.*
└── env_setup/            # Environment setup files
    ├── Dockerfile
    └── requirements.txt
```

### Why this structure is important

- Separates code, data, and configuration.  
- Makes debugging and extension easier.  
- Helps other people understand your work quickly.  

---

## 3. How to Run the Code

### 3.1 Set up your Python environment

A well-defined environment makes the code portable to different machines (local PC, lab server, or HPC cluster).  
All required packages are listed in `./env_setup/requirements.txt`.

Use either Conda or Virtualenv to create an isolated environment.

**Using Conda**

```bash
conda create --prefix ./.venv python=3.10.4
conda activate ./.venv
pip install -r ./env_setup/requirements.txt
```

**Using Virtualenv**

```bash
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows
pip install -r ./env_setup/requirements.txt
```

### Why this matters

- Keeps dependencies under control.  
- Prevents conflicts with other Python projects.  

### 3.2 Run experiments

In this repository, a **task** is any shell call that triggers one logical step, such as a single training run or a report-generation job.

Run single tasks:

```bash
python runs/train.py      # one training run
python runs/report.py     # aggregate previous runs and build a report
```

Run the entire pipeline:

```bash
# Linux/Mac
./run_all_tasks.sh

# Windows
run_all_tasks.bat
```

### Why this matters

- Lets you test one step at a time or run everything in one command.  
- Saves time when launching many experiments.  

---

## 4. Hydra Basics

[Hydra](https://hydra.cc/) lets you launch Python functions from the command line and manage configuration files.

- Every YAML file in `./config/` defines default settings for one part of the project (model, data, training, launcher).  
- You can override any field directly in the shell with `key=value`.  
- **Multirun** mode runs many configurations back-to-back and stores each result in its own folder.

### 4.1 Basic Run

```bash
python runs/train.py        # uses defaults in config/
```

Pass command-line overrides:

```bash
python runs/train.py training.epochs=10 model=net5
```

### 4.2 Run multiple experiments

```bash
python runs/train.py --multirun model=net2,net5 training.epochs=2,5
```

Hydra creates one sub-folder per setting.

### 4.3 Run predefined experiments

```bash
python runs/train.py +experiment=sweep_models
```

The file `./config/experiment/sweep_models.yaml` lists all overrides for this sweep.

### 4.4 Use launchers for parallel runs

Launchers let you run many jobs in parallel on one machine or an HPC cluster.  
Launcher configs live in `./config/launcher/`.  
See the Hydra launcher docs: <https://hydra.cc/docs/advanced/launcher_plugins/>.

```bash
# Local CPU parallelism with joblib
python runs/train.py --multirun +launcher=joblib

# Slurm cluster
python runs/train.py --multirun +launcher=slurm

# Slurm with GPUs
python runs/train.py --multirun +launcher=slurmgpu
```

### Why this matters

- You can explore many settings with a single command.  
- Hydra records every config, so you know exactly what produced each result.  
- The same code you were running in your machine, can be run in a cluster with minimal changes or coding rabbit-holes (mostly).

---

## 5. Tools

Besides Python and Hydra, two tools are worth adding to your workflow.

### 5.1 Docker

Docker packages your code, environment, and dependencies into one container, so it runs the same on any operating system.

```bash
docker build -t example ./env_setup
docker run -d --rm --name example --gpus all -v $(pwd):/home/example example bash
```

### Why this matters

- Handy for sharing your work or moving it to a server.  
- Removes “works on my machine” problems.  

### 5.2 Rclone for syncing data

`rclone` moves large datasets between your workstation and remote storage (e.g., S3, Google Drive).

```bash
rclone config
rclone sync ./data/datasets remote:bucket/path -P --transfers=8
```

### Why this matters

- Keeps local disks clean and backed up.  
- Speeds up transfers to HPC clusters.  

---

## 6. Using this Template

This project is meant to help students organize and run machine-learning experiments. What's important is the abstract idea of organizing your thoughts and your code, not this specific implementation of it.

---

## 7. References and Further Reading

- [Hydra Documentation](https://hydra.cc/)  
- [Reproducibility in Machine Learning](https://www.nature.com/articles/s42256-019-0035-4)  
- [Ten Simple Rules for Reproducible Research](https://doi.org/10.1371/journal.pcbi.1003285)  
- [ML Experiment Tracking Tools](https://neptune.ai/blog/ml-experiment-tracking-tools)  
- [Hydra Launcher Plugins](https://hydra.cc/docs/advanced/launcher_plugins/)  
- [Docker Official Docs](https://docs.docker.com/)  
- [Rclone Documentation](https://rclone.org/)  

---

The exact folder names and tools can change, but the key idea stays the same: **a clear, automated workflow makes large-scale experimentation faster, easier to debug, and easier for others to reproduce.**
