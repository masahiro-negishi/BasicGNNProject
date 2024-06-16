import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.datasets import TUDataset  # type: ignore

KFOLD = 5
EPOCHS = 100
SEED = 0


def visualize_loss(
    dataset: str,
    model: str,
    layers: list[int],
    emb_dims: list[int],
    poolings: list[str],
) -> None:
    fig, axes = plt.subplots(
        len(emb_dims) * len(poolings),
        len(layers),
        figsize=(4 * len(layers), 4 * len(emb_dims) * len(poolings)),
    )
    fig.suptitle(f"dataset: {dataset}, model: {model}")
    for i, emb_dim in enumerate(emb_dims):
        for j, pooling in enumerate(poolings):
            for k, layer in enumerate(layers):
                dirpath = os.path.join(
                    os.path.dirname(__file__),
                    "../Results",
                    dataset,
                    model,
                    f"l={layer}_p={pooling}_d={emb_dim}",
                )
                train_losses = np.zeros((KFOLD, EPOCHS))
                test_losses = np.zeros((KFOLD, EPOCHS))
                for fold in range(KFOLD):
                    with open(
                        os.path.join(dirpath, f"fold{fold}", "results.json")
                    ) as f:
                        log = json.load(f)
                        train_losses[fold] = log["details_train"]["total_loss"]
                        test_losses[fold] = log["details_val"]["total_loss"]
                # plot mean and std
                axes[i * len(poolings) + j, k].plot(
                    np.mean(train_losses, axis=0), label="train loss"
                )
                axes[i * len(poolings) + j, k].fill_between(
                    np.arange(EPOCHS),
                    np.mean(train_losses, axis=0) - np.std(train_losses, axis=0),
                    np.mean(train_losses, axis=0) + np.std(train_losses, axis=0),
                    alpha=0.3,
                )
                axes[i * len(poolings) + j, k].plot(
                    np.mean(test_losses, axis=0), label="test loss"
                )
                axes[i * len(poolings) + j, k].fill_between(
                    np.arange(EPOCHS),
                    np.mean(test_losses, axis=0) - np.std(test_losses, axis=0),
                    np.mean(test_losses, axis=0) + np.std(test_losses, axis=0),
                    alpha=0.3,
                )
                # set labels
                if k == 0:
                    axes[i * len(poolings) + j, k].set_ylabel(
                        f"{emb_dims[i]}, {poolings[j]}", size="large"
                    )
                if i * len(poolings) + j == len(emb_dims) * len(poolings) - 1:
                    axes[i * len(poolings) + j, k].set_xlabel(
                        f"{layers[k]}", size="large"
                    )
    fig.supxlabel("Number of layers", size="xx-large")
    fig.supylabel("Embdim, Pooling", size="xx-large")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    os.makedirs(
        os.path.join(
            os.path.dirname(__file__),
            "../Evals",
            dataset,
        ),
        exist_ok=True,
    )
    plt.savefig(
        os.path.join(
            os.path.dirname(__file__),
            "../Evals",
            dataset,
            f"{model}_loss.png",
        ),
    )


def neighbors_correspondence(
    dataset: str,
    model: str,
    layer: int,
    emb_dim: int,
    pooling: str,
    metrics: list[str],
    ks: list[int],
):
    ks.sort()

    dirpath = os.path.join(
        os.path.dirname(__file__),
        "../Results",
        dataset,
        model,
        f"l={layer}_p={pooling}_d={emb_dim}",
    )
    if os.path.exists(os.path.join(dirpath, "neighbor.json")):
        with open(os.path.join(dirpath, "neighbor.json")) as f:
            stats = json.load(f)
    else:
        stats = {}

    dataset = TUDataset(
        root=os.path.join(
            os.path.dirname(__file__), "../Data/Datasets", dataset, "Compose([])"
        ),
        name=dataset,
    )
    n_samples = len(dataset)
    indices = np.random.RandomState(seed=SEED).permutation(n_samples)
    keep_train = torch.zeros((len(metrics), len(ks), KFOLD))
    keep_test = torch.zeros((len(metrics), len(ks), KFOLD))
    for fold in range(KFOLD):
        train_indices = np.concatenate(
            (
                indices[: (fold * n_samples) // KFOLD],
                indices[(fold + 1) * n_samples // KFOLD :],
            )
        )
        test_indices = indices[
            (fold * n_samples) // KFOLD : (fold + 1) * n_samples // KFOLD
        ]
        for midx, metric in enumerate(metrics):
            dist_mat = torch.load(
                os.path.join(dirpath, f"fold{fold}", f"dist_{metric}.pt")
            )
            # train
            dist_mat_train = dist_mat[train_indices][:, train_indices]
            _, train_indices_sorted = torch.sort(dist_mat_train, dim=1)
            anchor_class = torch.zeros(len(train_indices))
            neighbor_classes = torch.zeros(len(train_indices), max(ks))
            for anchor in range(len(train_indices)):
                anchor_class[anchor] = dataset[train_indices[anchor]].y
                for k in range(max(ks)):
                    neighbor_classes[anchor][k] = dataset[
                        train_indices[
                            train_indices_sorted[anchor][k + 1]
                        ]  # not regard itself as neighbor
                    ].y
            corresp = neighbor_classes == anchor_class.unsqueeze(1)
            for kidx, k in enumerate(ks):
                keep_train[midx, kidx, fold] = torch.mean(
                    corresp[:, :k].sum(dim=1) / k
                ).item()
            # test
            dist_mat_test = dist_mat[test_indices][:, test_indices]
            _, test_indices_sorted = torch.sort(dist_mat_test, dim=1)
            anchor_class = torch.zeros(len(test_indices))
            neighbor_classes = torch.zeros(len(test_indices), max(ks))
            for anchor in range(len(test_indices)):
                anchor_class[anchor] = dataset[test_indices[anchor]].y
                for k in range(max(ks)):
                    neighbor_classes[anchor][k] = dataset[
                        test_indices[
                            test_indices_sorted[anchor][k + 1]
                        ]  # not regard itself as neighbor
                    ].y
            corresp = neighbor_classes == anchor_class.unsqueeze(1)
            for kidx, k in enumerate(ks):
                keep_test[midx, kidx, fold] = torch.mean(
                    corresp[:, :k].sum(dim=1) / k
                ).item()

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
    with open(os.path.join(dirpath, "neighbor.json"), "w") as f:
        json.dump(stats, f)


def neighbor_acc_plot(
    dataset: str,
    layers: list[int],
    emb_dims: list[int],
    poolings: list[str],
    metrics: list[str],
    ks: list[int],
):
    neighbors = np.zeros(
        (2, len(metrics), len(ks), 3, len(layers), len(emb_dims), len(poolings))
    )
    for midx, model in enumerate(["GAT", "GCN", "GIN"]):
        for lidx, layer in enumerate(layers):
            for eidx, emb_dim in enumerate(emb_dims):
                for pidx, pooling in enumerate(poolings):
                    dirpath = os.path.join(
                        os.path.dirname(__file__),
                        "../Results",
                        dataset,
                        model,
                        f"l={layer}_p={pooling}_d={emb_dim}",
                    )
                    with open(os.path.join(dirpath, "neighbor.json")) as f:
                        stats = json.load(f)
                    for tidx, target in enumerate(["train", "test"]):
                        for meidx, metric in enumerate(metrics):
                            for kidx, k in enumerate(ks):
                                neighbors[
                                    tidx,
                                    meidx,
                                    kidx,
                                    midx,
                                    lidx,
                                    eidx,
                                    pidx,
                                ] = stats[target][metric][str(k)]["mean"]

    accs = np.zeros((2, 3, len(layers), len(emb_dims), len(poolings)))
    for tidx, target in enumerate(["train", "test"]):
        for midx, model in enumerate(["GAT", "GCN", "GIN"]):
            for lidx, layer in enumerate(layers):
                for eidx, emb_dim in enumerate(emb_dims):
                    for pidx, pooling in enumerate(poolings):
                        dirpath = os.path.join(
                            os.path.dirname(__file__),
                            "../Results",
                            dataset,
                            model,
                            f"l={layer}_p={pooling}_d={emb_dim}",
                        )
                        for fold in range(KFOLD):
                            with open(
                                os.path.join(dirpath, f"fold{fold}", "results.json")
                            ) as f:
                                log = json.load(f)
                                accs[tidx, midx, lidx, eidx, pidx] += log[
                                    f"details_{target}"
                                ]["accuracy"][-1]
    accs /= KFOLD

    for color, cands in [
        ("model", ["GAT", "GCN", "GIN"]),
        ("layer", layers),
        ("emb_dim", emb_dims),
        ("pooling", poolings),
    ]:
        fig, axes = plt.subplots(
            len(ks),
            2 * len(metrics),
            figsize=(4 * 2 * len(metrics), 4 * len(ks)),
        )
        fig.suptitle(f"dataset: {dataset}")
        for tidx, target in enumerate(["train", "test"]):
            for meidx, metric in enumerate(metrics):
                for kidx, k in enumerate(ks):
                    for cidx, cand in enumerate(cands):
                        if color == "model":
                            axes[kidx, tidx * len(metrics) + meidx].scatter(
                                neighbors[tidx, meidx, kidx, cidx].flatten(),
                                accs[tidx, cidx].flatten(),
                                label=f"{cand}",
                            )
                        elif color == "layer":
                            axes[kidx, tidx * len(metrics) + meidx].scatter(
                                neighbors[tidx, meidx, kidx, :, cidx].flatten(),
                                accs[tidx, :, cidx].flatten(),
                                label=f"{cand}",
                            )
                        elif color == "emb_dim":
                            axes[kidx, tidx * len(metrics) + meidx].scatter(
                                neighbors[tidx, meidx, kidx, :, :, cidx].flatten(),
                                accs[tidx, :, :, cidx].flatten(),
                                label=f"{cand}",
                            )
                        elif color == "pooling":
                            axes[kidx, tidx * len(metrics) + meidx].scatter(
                                neighbors[tidx, meidx, kidx, :, :, :, cidx].flatten(),
                                accs[tidx, :, :, :, cidx].flatten(),
                                label=f"{cand}",
                            )
                    if tidx * len(metrics) + meidx == 0:
                        axes[kidx, tidx * len(metrics) + meidx].set_ylabel(
                            f"{k}", size="large"
                        )
                    if kidx == len(ks) - 1:
                        axes[kidx, tidx * len(metrics) + meidx].set_xlabel(
                            f"{target}, {metric}", size="large"
                        )
        fig.supxlabel("Neighbor probability", size="xx-large")
        fig.supylabel("Accuracy", size="xx-large")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        os.makedirs(
            os.path.join(
                os.path.dirname(__file__),
                "../Evals",
                dataset,
            ),
            exist_ok=True,
        )
        plt.savefig(
            os.path.join(
                os.path.dirname(__file__),
                "../Evals",
                dataset,
                f"scatter_{color}.png",
            ),
        )


if __name__ == "__main__":
    args = sys.argv
    if args[1] == "visualize_loss":
        for model in ["GAT", "GCN", "GIN"]:
            visualize_loss(
                dataset="Mutagenicity",
                model=model,
                layers=[1, 2, 3, 4],
                emb_dims=[32, 64, 128],
                poolings=["mean", "sum"],
            )
    elif args[1] == "neighbors_correspondence":
        for model in ["GAT", "GCN", "GIN"]:
            for layer in [1, 2, 3, 4]:
                for emb_dim in [32, 64, 128]:
                    for pooling in ["mean", "sum"]:
                        print(
                            f"model: {model}, layer: {layer}, emb_dim: {emb_dim}, pooling: {pooling}"
                        )
                        neighbors_correspondence(
                            dataset="Mutagenicity",
                            model=model,
                            layer=layer,
                            emb_dim=emb_dim,
                            pooling=pooling,
                            metrics=["l1", "l2"],
                            ks=[1, 5, 10, 20],
                        )
    elif args[1] == "neighbor_acc_plot":
        neighbor_acc_plot(
            dataset="Mutagenicity",
            layers=[1, 2, 3, 4],
            emb_dims=[32, 64, 128],
            poolings=["mean", "sum"],
            metrics=["l1", "l2"],
            ks=[1, 5, 10, 20],
        )
    else:
        raise ValueError("Invalid command")
