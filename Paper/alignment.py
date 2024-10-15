import argparse
import json
import os
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from torch_geometric.datasets import ZINC, TUDataset  # type: ignore

from Exp.preparation import get_model, load_dataset
from Exp.run_model import set_seed
from Exp.training_loop_functions import compute_embeddings
from Misc.config import config


def neighbors_correspondence_TUDataset(
    dataset_name: str,
    kfold: int,
    model: str,
    layer: int,
    emb_dim: int,
    pooling: str,
    metrics: list[str],
    ks: list[int],
    seed: int,
    epoch: str,
):
    ks.sort()

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

    dataset = TUDataset(
        root=os.path.join(
            os.path.dirname(__file__), "../Data/Datasets", dataset_name, "Compose([])"
        ),
        name=dataset_name,
    )
    n_samples = len(dataset)
    indices = np.random.RandomState(seed=seed).permutation(n_samples)
    keep_train = torch.zeros((len(metrics), len(ks), kfold))
    keep_test = torch.zeros((len(metrics), len(ks), kfold))
    for fold in range(kfold):
        train_indices = np.concatenate(
            (
                indices[: (fold * n_samples) // kfold],
                indices[(fold + 1) * n_samples // kfold :],
            )
        )
        test_indices = indices[
            ((2 * fold + 1) * n_samples)
            // (2 * kfold) : (fold + 1)
            * n_samples
            // kfold
        ]
        # list of all classes in dataset
        dataset_y = torch.tensor([g.y for g in dataset])
        classes = torch.unique(dataset_y)
        train_expectation = torch.zeros(len(classes))
        test_expectation = torch.zeros(len(classes))
        for cidx, c in enumerate(classes):
            train_expectation[cidx] = (
                torch.sum(dataset_y[train_indices] == c).item() - 1
            ) / (len(train_indices) - 1)
            test_expectation[cidx] = (
                torch.sum(dataset_y[test_indices] == c).item() - 1
            ) / (len(test_indices) - 1)
        for midx, metric in enumerate(metrics):
            dist_mat = torch.load(
                os.path.join(dirpath, f"fold{fold}", f"dist_{metric}_{epoch}.pt")
            )
            # set diagonal to inf
            dist_mat.fill_diagonal_(float("inf"))
            # train
            dist_mat_train = dist_mat[train_indices][:, train_indices]
            _, train_indices_sorted = torch.sort(dist_mat_train, dim=1)
            anchor_class = torch.zeros(len(train_indices))
            neighbor_classes = torch.zeros(len(train_indices), max(ks))
            for anchor in range(len(train_indices)):
                anchor_class[anchor] = dataset[train_indices[anchor]].y
                for k in range(max(ks)):
                    neighbor_classes[anchor][k] = dataset[
                        train_indices[train_indices_sorted[anchor][k]]
                    ].y
            corresp = neighbor_classes == anchor_class.unsqueeze(1)
            for kidx, k in enumerate(ks):
                keep_train[midx, kidx, fold] = (
                    torch.mean(
                        corresp[:, :k].sum(dim=1) / k
                        - torch.tensor(
                            [train_expectation[g.y] for g in dataset[train_indices]]
                        )
                    ).item()
                    * (len(train_indices) - 1)
                    / (len(train_indices) - k - 1)
                )
            # test
            dist_mat_test = dist_mat[test_indices][:, test_indices]
            _, test_indices_sorted = torch.sort(dist_mat_test, dim=1)
            anchor_class = torch.zeros(len(test_indices))
            neighbor_classes = torch.zeros(len(test_indices), max(ks))
            for anchor in range(len(test_indices)):
                anchor_class[anchor] = dataset[test_indices[anchor]].y
                for k in range(max(ks)):
                    neighbor_classes[anchor][k] = dataset[
                        test_indices[test_indices_sorted[anchor][k]]
                    ].y
            corresp = neighbor_classes == anchor_class.unsqueeze(1)
            for kidx, k in enumerate(ks):
                keep_test[midx, kidx, fold] = (
                    torch.mean(
                        corresp[:, :k].sum(dim=1) / k
                        - torch.tensor(
                            [test_expectation[g.y] for g in dataset[test_indices]]
                        )
                    ).item()
                    * (len(test_indices) - 1)
                    / (len(test_indices) - k - 1)
                )

    # average over kfold
    if "train" not in stats:
        stats["train"] = {}
    if "test" not in stats:
        stats["test"] = {}
    for midx, metric in enumerate(metrics):
        if metric not in stats["train"]:
            stats["train"][metric] = {}
        if metric not in stats["test"]:
            stats["test"][metric] = {}
        for kidx, k in enumerate(ks):
            if k not in stats["train"][metric]:
                stats["train"][metric][k] = {}
                stats["test"][metric][k] = {}
            stats["train"][metric][k]["mean"] = keep_train[midx, kidx].mean().item()
            stats["train"][metric][k]["std"] = keep_train[midx, kidx].std().item()
            stats["test"][metric][k]["mean"] = keep_test[midx, kidx].mean().item()
            stats["test"][metric][k]["std"] = keep_test[midx, kidx].std().item()
    with open(os.path.join(dirpath, f"neighbor_{epoch}.json"), "w") as f:
        json.dump(stats, f)


def neighbors_correspondence_Lipo(
    kfold: int,
    model: str,
    layer: int,
    emb_dim: int,
    pooling: str,
    metrics: list[str],
    ks: list[int],
    seed: int,
    epoch: str,
):
    ks.sort()

    dirpath = os.path.join(
        os.path.dirname(__file__),
        "../Results",
        "split",
        "ogbg-mollipo",
        model,
        f"l={layer}_p={pooling}_d={emb_dim}",
    )
    if os.path.exists(os.path.join(dirpath, f"neighbor_{epoch}.json")):
        with open(os.path.join(dirpath, f"neighbor_{epoch}.json")) as f:
            stats = json.load(f)
    else:
        stats = {}

    dataset = PygGraphPropPredDataset(
        root=os.path.join(
            os.path.dirname(__file__), "../Data/Datasets", "ogbg-mollipo", "Compose([])"
        ),
        name="ogbg-mollipo",
    )
    n_samples = len(dataset)
    indices = np.random.RandomState(seed=seed).permutation(n_samples)
    keep_train = torch.zeros((len(metrics), len(ks), kfold))
    keep_test = torch.zeros((len(metrics), len(ks), kfold))
    for fold in range(kfold):
        train_indices = np.concatenate(
            (
                indices[: (fold * n_samples) // kfold],
                indices[(fold + 1) * n_samples // kfold :],
            )
        )
        test_indices = indices[
            ((2 * fold + 1) * n_samples)
            // (2 * kfold) : (fold + 1)
            * n_samples
            // kfold
        ]
        train_dataset = dataset[train_indices]
        test_dataset = dataset[test_indices]
        for midx, metric in enumerate(metrics):
            dist_mat = torch.load(
                os.path.join(dirpath, f"fold{fold}", f"dist_{metric}_{epoch}.pt")
            )
            # set diagonal to inf
            dist_mat.fill_diagonal_(float("inf"))
            # train
            dist_mat_train = dist_mat[train_indices][:, train_indices]
            _, train_indices_sorted = torch.sort(dist_mat_train, dim=1)
            anchor_y = torch.zeros(len(train_indices))
            neighbor_y = torch.zeros(len(train_indices), max(ks))
            ymax, ymin = -float("inf"), float("inf")
            for anchor in range(len(train_indices)):
                anchor_y[anchor] = train_dataset[anchor].y
                ymax = max(ymax, anchor_y[anchor].item())
                ymin = min(ymin, anchor_y[anchor].item())
                for k in range(max(ks)):
                    neighbor_y[anchor][k] = train_dataset[
                        train_indices_sorted[anchor][k]
                    ].y
            for kidx, k in enumerate(ks):
                keep_train[midx, kidx, fold] = (
                    torch.mean(
                        (
                            1
                            - torch.abs(anchor_y.reshape(-1, 1) - neighbor_y[:, :k])
                            / (ymax - ymin)
                        ).sum(dim=1)
                        / k
                        - torch.sum(
                            (
                                1
                                - torch.abs(
                                    anchor_y.reshape(-1, 1) - anchor_y.reshape(1, -1)
                                )
                                / (ymax - ymin)
                            ),
                            dim=1,
                        )
                        / (len(anchor_y) - 1)
                    ).item()
                    * (len(train_indices) - 1)
                    / (len(train_indices) - k - 1)
                )
            # test
            dist_mat_test = dist_mat[test_indices][:, test_indices]
            _, test_indices_sorted = torch.sort(dist_mat_test, dim=1)
            ymax, ymin = -float("inf"), float("inf")
            for anchor in range(len(test_indices)):
                anchor_y[anchor] = test_dataset[anchor].y
                ymax = max(ymax, anchor_y[anchor].item())
                ymin = min(ymin, anchor_y[anchor].item())
                for k in range(max(ks)):
                    neighbor_y[anchor][k] = test_dataset[
                        test_indices_sorted[anchor][k]
                    ].y
            for kidx, k in enumerate(ks):
                keep_test[midx, kidx] = (
                    torch.mean(
                        (
                            1
                            - torch.abs(anchor_y.reshape(-1, 1) - neighbor_y[:, :k])
                            / (ymax - ymin)
                        ).sum(dim=1)
                        / k
                        - torch.sum(
                            (
                                1
                                - torch.abs(
                                    anchor_y.reshape(-1, 1) - anchor_y.reshape(1, -1)
                                )
                                / (ymax - ymin)
                            ),
                            dim=1,
                        )
                        / (len(anchor_y) - 1)
                    ).item()
                    * (len(test_indices) - 1)
                    / (len(test_indices) - k - 1)
                )

    # average over kfold
    if "train" not in stats:
        stats["train"] = {}
    if "test" not in stats:
        stats["test"] = {}
    for midx, metric in enumerate(metrics):
        if metric not in stats["train"]:
            stats["train"][metric] = {}
        if metric not in stats["test"]:
            stats["test"][metric] = {}
        for kidx, k in enumerate(ks):
            if k not in stats["train"][metric]:
                stats["train"][metric][k] = {}
                stats["test"][metric][k] = {}
            stats["train"][metric][k]["mean"] = keep_train[midx, kidx].mean().item()
            stats["train"][metric][k]["std"] = keep_train[midx, kidx].std().item()
            stats["test"][metric][k]["mean"] = keep_test[midx, kidx].mean().item()
            stats["test"][metric][k]["std"] = keep_test[midx, kidx].std().item()
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
                        if args.dataset == "ogbg-mollipo":
                            neighbors_correspondence_Lipo(
                                args.kfold,
                                model=model,
                                layer=layer,
                                emb_dim=emb_dim,
                                pooling=pooling,
                                metrics=["l1", "l2"],
                                ks=[1, 5, 10, 20],
                                seed=args.seed,
                                epoch=epoch,
                            )
                        else:
                            neighbors_correspondence_TUDataset(
                                args.dataset,
                                args.kfold,
                                model=model,
                                layer=layer,
                                emb_dim=emb_dim,
                                pooling=pooling,
                                metrics=["l1", "l2"],
                                ks=(
                                    [1, 5, 10]
                                    if args.dataset in ["MUTAG"]
                                    else [1, 5, 10, 20]
                                ),
                                seed=args.seed,
                                epoch=epoch,
                            )
