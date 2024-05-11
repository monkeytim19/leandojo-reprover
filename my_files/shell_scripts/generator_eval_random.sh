#!/bin/bash
#SBATCH --gres=gpu:teslaa40:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tcwong
source ReProver/bin/activate
export PYTHONPATH=/vol/bitbucket/tcwong/individual_project/leandojo-reprover

# TODO
export GITHUB_ACCESS_TOKEN=TODO

. /vol/cuda/12.2.0/setup.sh

# export CUDA_HOME=/vol/cuda/12.2.0
# export PATH=${CUDA_HOME}/bin:${PATH}
# export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export PATH=$(echo $PATH | tr ':' '\n' | grep -v "/homes/tcwong/.elan/bin" | tr '\n' ':')
export PATH="$PATH:/vol/bitbucket/tcwong/individual_project/leandojo-reprover/.elan/bin"

export CACHE_DIR=/vol/bitbucket/tcwong/individual_project/leandojo-reprover/.cache/lean_dojo

python scripts/trace_repos.py

python prover/evaluate.py --data-path data/leandojo_benchmark_4/random/ --ckpt_path my_files/saved_checkpoints/generator_random.ckpt --split test --num-workers 2 --num-gpus 1
