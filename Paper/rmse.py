import argparse
import json
import os
from argparse import Namespace

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from ogb.graphproppred import PygGraphPropPredDataset  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from torch_geometric.datasets import ZINC, TUDataset  # type: ignore

from Exp.preparation import get_model, load_dataset
from Exp.run_model import set_seed
from Exp.training_loop_functions import compute_embeddings
from Misc.config import config


def calc_rmse_wo_outliers(x1: torch.Tensor, y1: torch.Tensor) -> tuple:
    x1 /= np.max(x1)
    y1 /= np.max(y1)
    c1 = np.sum(x1 * y1) / np.sum(x1**2)
    diff = np.abs(y1 - c1 * x1)
    sorted_indices = np.argsort(diff)
    n_outliers = len(sorted_indices) // 100
    x = x1[sorted_indices[:-n_outliers]]
    y = y1[sorted_indices[:-n_outliers]]
    x /= np.max(x)
    y /= np.max(y)
    coeff = np.sum(x * y) / np.sum(x**2)
    return x, y, coeff, np.sqrt(np.mean((y - coeff * x) ** 2))


def rmse_dmpnn_dfunc(
    dataset_name: str,
    kfold: int,
    model: str,
    layer: int,
    emb_dim: int,
    pooling: str,
    metrics: list[str],
    seed: int,
    epoch: str,
):

    dirpath = os.path.join(
        os.path.dirname(__file__),
        "../Results",
        "split",
        dataset_name,
        model,
        f"l={layer}_p={pooling}_d={emb_dim}",
    )
    if os.path.exists(os.path.join(dirpath, f"neighbor_{epoch}.json")):
        with open(os.path.join(dirpath, f"neighbor_{epoch}.json")) as f:
            stats = json.load(f)
    else:
        stats = {}

    if dataset_name in ["Mutagenicity", "ENZYMES"]:
        dataset = TUDataset(
            root=os.path.join(
                os.path.dirname(__file__),
                "../Data/Datasets",
                dataset_name,
                "Compose([])",
            ),
            name=dataset_name,
        )
    elif dataset_name == "ogbg-mollipo":
        dataset = PygGraphPropPredDataset(
            root=os.path.join(
                os.path.dirname(__file__),
                "../Data/Datasets",
                "ogbg-mollipo",
                "Compose([])",
            ),
            name="ogbg-mollipo",
        )
    else:
        raise ValueError("Invalid dataset name")
    n_samples = len(dataset)
    indices = np.random.RandomState(seed=seed).permutation(n_samples)
    keep_train = np.zeros((len(metrics), 4))

    train_indices = indices[n_samples // kfold :]
    for midx, metric in enumerate(metrics):
        dist_mat = torch.load(
            os.path.join(dirpath, "fold0", f"dist_{metric}_{epoch}.pt")
        )
        dist_y = dist_mat[
            train_indices[
                np.random.RandomState(seed=1).randint(0, len(train_indices), 1000)
            ],
            train_indices[
                np.random.RandomState(seed=2).randint(0, len(train_indices), 1000)
            ],
        ]  # (1000, )
        # dist_y = dist_y[np.random.RandomState(seed=3).permutation(1000)]  # Threshold!
        for didx, path in enumerate(
            [
                "fold0_GED_t=30.pt",
                f"fold0_TMD_d={layer+1}.pt",
                f"fold0_WWL_d={layer+1}.pt",
                f"fold0_WLOA_d={layer+1}.pt",
            ]
        ):
            dist_x = torch.load(
                os.path.join(os.path.dirname(__file__), "../Dis_mx", dataset_name, path)
            )
            _, _, _, rmse = calc_rmse_wo_outliers(
                dist_x.flatten().numpy(), dist_y.flatten().numpy()
            )
            keep_train[midx, didx] = rmse
    # save
    if "rmse" not in stats:
        stats["rmse"] = {}
    for midx, metric in enumerate(metrics):
        if metric not in stats["rmse"]:
            stats["rmse"][metric] = {}
        for didx, d in enumerate(["GED", "TMD", "WWL", "WLOA"]):
            if d not in stats["rmse"][metric]:
                stats["rmse"][metric][d] = {}
            stats["rmse"][metric][d] = keep_train[midx, didx].item()
    with open(os.path.join(dirpath, f"neighbor_{epoch}.json"), "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "MUTAG",
            "Mutagenicity",
            "NCI1",
            "ENZYMES",
            "ogbg-mollipo",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kfold", type=int, default=5)

    args = parser.parse_args()
    print(args)

    for model in ["GCN", "GIN"]:
        for layer in [1, 2, 3, 4]:
            for emb_dim in [32, 64, 128]:
                for pooling in ["mean", "sum"]:
                    for epoch in ["init", "best"]:
                        print(
                            f"model: {model}, layer: {layer}, emb_dim: {emb_dim}, pooling: {pooling}, epoch: {epoch}"
                        )
                        rmse_dmpnn_dfunc(
                            args.dataset,
                            args.kfold,
                            model=model,
                            layer=layer,
                            emb_dim=emb_dim,
                            pooling=pooling,
                            metrics=["l1", "l2"],
                            seed=args.seed,
                            epoch=epoch,
                        )
