# BasicGNNProject
This repository is forked from [BasicGNNProject](https://github.com/ocatias/BasicGNNProject), and modified.
Note that the original tests from [BasicGNNProject](https://github.com/ocatias/BasicGNNProject) can be unsuccessful.

## Setup 
```
$ pwd 
xxx/BasicGNNProject
$ export PYTHONPATH=$PYTHONPATH:$PATH # Let `$PATH` be the path to where this repository is stored (i.e. the result of running `pwd`).
$ pyenv global 3.10.11
$ python -m venv bgnn_env
$ source bgnn_env/bin/activate
$ pip install --upgrade pip
$ pip list 
Package    Version
---------- -------
pip        24.2
setuptools 65.5.0
$ pip install -r requirements.txt
```

For VSCode users, you can open this repo with wilt.code-workspace after installing the following extensions:
- ms-python.black-formatter
- ms-python.flake8
- ms-python.isort
- ms-python.mypy-type-checker

## Run experiments
Note that only the following datasets are supported. However, it is quite straightforward to extend to other datasets.
- ogbg-mollipo
- Mutagenicity
- ENZYMES

### 1. Train GNN with train/eval/test data
The whole dataset is split into k_fold subsets, and then one of the subsets is further split into an eval/test dataset.
E.g.) Training GCN on Mutagenicity. The dataset is split into 5 subsets, and the 2nd subset (0-index) is used for eval/test datasets.
```
python Exp/run_model.py --model GCN --dataset Mutagenicity --scheduler None --epochs 100 --lr 0.001 --batch_size 32 --k_fold 5 --test_fold 2 --seed 0 --emb_dim 64 --pooling sum --num_mp_layers 3
```
### 2. Train GNN with only train data
E.g.) Training GCN on all data in Mutagenicity.
```
python Exp/run_model.py --model GCN --dataset Mutagenicity --scheduler None --train_with_all_data --epochs 100 --lr 0.001 --batch_size 32 --seed 0 --emb_dim 64 --pooling sum --num_mp_layers 3
```

### 3. Calculate the structural alignment
Computing RMSE in Definition 3 of our paper. Note that you should train GNN with the train/eval/test split before.
```
python Paper/rmse.py --dataset Mutagenicity --seed 0 --kfold 5
```

### 4. Calculate the functional alignment
Computing $ALI_k$ in Definition 4 of our paper. Note that you should train GNN with the train/eval/test split before.
```
python Paper/alignment.py --dataset Mutagenicity --seed 0 --kfold 5
```

### 5. Reproduce figures 
```
python Paper/figures.py
```