# Example Repository

Welcome to the example repository! This guide will help you understand the structure of the repository, set up your environment, and run the code locally, on a high-performance cluster (HPC), or on the GPU server using docker. Additionally, it covers how to manage data with Rclone.

## Structure of the Repository
- **modules**: Contains the core code for what you want to do. e.g. training and testing models.
- **data**: Directory to store datasets and related files.
  - `config/`: Configuration files for Hydra.
    - `train_model.yaml`: (e.g) Default configuration for training.
- **env setup**: Environment setup files.
  - `Dockerfile`: Docker configuration for containerized environment.
  - `requirements.txt`: Python dependencies.
- **runs**: Contains scripts for experimental runs, create datasets, train models, analyze them, etc.
  - `train.py`: Main script for training the model.

## Setting Up the Virtual Environment

### Using Conda

```bash
# Create a new conda environment with Python 3.10.4 and place it in the ./.venv directory
conda create --prefix ./.venv python=3.10.4

# Activate the newly created conda environment
conda activate ./.venv

# Install all required Python packages as listed in the requirements.txt file
pip install -r ./env_setup/requirements.txt

# Export the list of installed packages to a new requirements file
pip freeze > ./env_setup/requirements2.txt

# Deactivate the conda environment
conda deactivate
```

### Using Virtualenv

```bash
# If you are using a high-performance computing cluster (HPC), consider loading a specific Python module from the beginning
module load Python/3.10.4

# Create a new virtual environment named .venv
python -m venv .venv

# Activate the newly created virtual environment
source .venv/bin/activate

# Install all required Python packages as listed in the requirements.txt file
pip install -r ./env_setup/requirements.txt

# Export the list of installed packages to a new requirements file
pip freeze > ./env_setup/requirements2.txt

# Deactivate the virtual environment
deactivate
```

## Running Code Locally

### Single Run

```bash
# Display help message with all available options and arguments
python runs/train.py --help

# Execute the script with default configuration settings
python runs/train.py
```

### Manual run

```bash
# Execute the script with specific arguments, changing the number of epochs to 2 and the seed to 7
python runs/train.py training.epochs=2 training.seed=7
```

### Sweep with Hydra

```bash
# Execute multiple runs with different model sizes using Hydra's multirun feature
# This command will run the script for each combination of the specified values
python runs/train.py --multirun training.epochs=2 model=net2,net5,net7

# Execute multiple runs as defined in a configuration file
python runs/train.py +experiment=sweep_models_seeds
```

### Launchers

```bash
# Execute multiple runs with Hydra's joblib launcher
# This will run the script for each combination of the specified values using joblib for parallel execution
python runs/train.py --multirun training.epochs=2 model=net2,net5,net7 +launcher=joblib

# Or use Hydra's slurm launcher for running on a Slurm-based cluster
python runs/train.py --multirun training.epochs=2 model=net2,net5,net7 +launcher=slurm

# Or use Slurm with GPU support, running the script with multiple seed values
python runs/train.py --multirun training.epochs=2 training.seed=0,1,2,3,4 +launcher=slurmgpu
```

## Run Code with Docker (GPU Server)

Docker allows you to execute your code on different machines with the same environment, ensuring consistent results. This is particularly useful for avoiding stochastic issues and differences between Windows and Linux.

### Build and Launch Docker Container

```bash
# Build a Docker image from the Dockerfile located in the env_setup directory
docker build -t andresfp14/xaicu118 ./env_setup

# (Optional) Push the built image to a Docker repository for public access
docker push andresfp14/xaicu118

# Examples of how to launch the Docker container in Windows

# Run the container in detached mode, remove it after exiting, name it xaicu118, use all GPUs, map ports, and mount the current directory
docker run -d --rm --name xaicu118 --gpus all -p 8888:8888 -p 6007:6007 -v %cd%:/home/example andresfp14/xaicu118 bash

# Examples of how to launch the Docker container in Linux

# Run the container in detached mode, remove it after exiting, name it xaicu118, allocate 50G of shared memory, use all GPUs, map ports, and mount the current directory
docker run -d --rm --name xaicu118 --shm-size 50G --gpus all -p 8888:8888 -p 6007:6007 -v $(pwd):/home/example andresfp14/xaicu118 bash

# Run the container in detached and interactive mode, remove it after exiting, name it xai_1, allocate 50G of shared memory, use the first GPU device, and mount specified directories
docker run -idt --rm --name xai_1 --shm-size 50G --gpus '"device=0:0"' -v ~/data/datasets:/home/example/data/datasets -v $(pwd):/home/example andresfp14/xaicu118 bash

```

## Moving Data Around with Rclone

Rclone is a command-line program to manage files on cloud storage. It is useful for transferring large datasets to and from remote servers.

### Installing Rclone

Follow the instructions on the [Rclone website](https://rclone.org/install/) to install Rclone on your system.

### Configuring Rclone

Run the following command to configure Rclone with your cloud storage provider:

```bash
# Configure Rclone with your cloud storage credentials and settings
rclone config
```

### Using Rclone

#### Copying Data to Remote Storage

```bash
# Copy data from a local directory to a remote storage bucket
rclone copy ./data remote:bucket/path
```

#### Copying Data from Remote Storage

```bash
# Copy data from a remote storage bucket to a local directory
rclone copy remote:bucket/path ./data
```

#### Sync Data to Remote Storage

```bash
# sync from local to remote
rclone sync ./data/datasets merkur:axai/data/datasets -P --transfers=8
```

This setup ensures that you can efficiently manage your project environment, run your code in different scenarios, and handle data transfers seamlessly. For more details, refer to the [repository](https://github.com/andresfp14/example).
