
## 1) Build and launch your docker container (optional)

Docker will allow you to execute your code in different machines (with docker) and have the same behavior. This is specially important if you have stochastic issues and different results between windows and linux computers. To avoid these issues, and to have control over the full environment (including gpu drivers) where our code work, we use docker.

The docker image that we are going to use is the one on 'env_setup/Dockerfile'.

```bash
# build image
docker build -t andresfp14/xaicu118 ./env_setup

# push image to docker repo (if you want to make it available in general)
docker push andresfp14/xaicu118

# Examples of how to launch it in windows
docker run -it --rm --name xaicu118 --gpus all -p 8888:8888 -p 6007:6007 -v %cd%:/home/example andresfp14/xaicu118
docker run -d --rm --name xaicu118 --gpus all -p 8888:8888 -p 6007:6007 -v %cd%:/home/example andresfp14/xaicu118 bash

# Examples of how to launch it in linux
docker run -itd --rm --name xaicu118 --shm-size 5G --gpus all -p 8888:8888 -p 6007:6007 -v $(pwd):/home/example andresfp14/xaicu118 bash
docker run -idt --rm --name xai_1 --shm-size 5G --gpus '"device=0:0"' -v ~/data/datasets:/home/example/data/datasets -v $(pwd):/home/example andresfp14/xaicu118
docker run -idt --rm --name xai_2 --shm-size 5G --gpus '"device=0:0"' -v $(pwd):/home/example andresfp14/xaicu118

```

## 2) Build and activate your virtual environment

Our virtual environment will be the collection of libraries that this project requires, and the versions of each library that are required.
In general, this is defined in the file 'env/requirements.txt'.

```bash
###############################
# with conda
###############################
# create environment
conda create --prefix ./.venv python=3.11
# activate environment
conda activate ./.venv
# install requirements
pip install -r ./env_setup/requirements.txt
# export environment (if you want to update it)
pip freeze > ./env_setup/requirements2.txt
# deactivate virtual environment
conda deactivate

###############################
# with virtualenv
###############################
# creates a virtualenv
python -m venv .venv
# activates the virtualenv
source .venv/bin/activate
. .venv/bin/activate
# install requirements
pip install -r ./env_setup/requirements.txt
# export environment (if you want to update it)
pip freeze > ./env_setup/requirements2.txt
# deactivate virtual environment
deactivate


# if you are using the HPC, consider:
module load Python/3.10.4
```

## 3) Run code

Now, with the environment setup, we can run the needed code from the base directory. We recommend using the "fire" library to avoid argparsers and maintain cleaner code.

```bash
###############################
# Getting help
###############################
python 01_train_model.py --help

###############################
# Executing with default arguments
###############################
python 01_train_model.py

###############################
# Executing and changing an argument
###############################
python 01_train_model.py training.epochs=2 training.seed=7

###############################
# Executing multiple runs with different model sizes using Hydra's multirun feature
###############################
python 01_train_model.py --multirun training.epochs=2 model.num_layers=1,2,3

###############################
# Executing multiple runs with launchers
###############################
python 01_train_model.py --multirun training.epochs=2 model.num_layers=1,2,3 +launcher=joblib

# or 

python 01_train_model.py --multirun training.epochs=2 model.num_layers=1,2,3 +launcher=slurm

or

python 01_train_model.py --multirun training.epochs=2 training.seed=0,1,2,3,4 +launcher=slurmgpu

# or 

python 01_train_model.py --multirun +experiment=sweep_models_lr +launcher=slurm

```