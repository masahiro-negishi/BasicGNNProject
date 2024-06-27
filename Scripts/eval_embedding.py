import argparse
import json
import os
import sys
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch_geometric.datasets import ZINC, TUDataset  # type: ignore

from Exp.preparation import get_model, load_dataset
from Exp.run_model import set_seed
from Exp.training_loop_functions import compute_embeddings
from Misc.config import config

SEED = 0


def visualize_loss(
    dataset: str,
    kfold: int,
    epochs: int,
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
                    "split",
                    dataset,
                    model,
                    f"l={layer}_p={pooling}_d={emb_dim}",
                )
                train_losses = np.zeros((kfold, epochs))
                test_losses = np.zeros((kfold, epochs))
                for fold in range(kfold):
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
                    np.arange(epochs),
                    np.mean(train_losses, axis=0) - np.std(train_losses, axis=0),
                    np.mean(train_losses, axis=0) + np.std(train_losses, axis=0),
                    alpha=0.3,
                )
                axes[i * len(poolings) + j, k].plot(
                    np.mean(test_losses, axis=0), label="test loss"
                )
                axes[i * len(poolings) + j, k].fill_between(
                    np.arange(epochs),
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


def neighbors_correspondence_TUDataset(
    dataset_name: str,
    kfold: int,
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
        "split",
        dataset_name,
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
            os.path.dirname(__file__), "../Data/Datasets", dataset_name, "Compose([])"
        ),
        name=dataset_name,
    )
    n_samples = len(dataset)
    indices = np.random.RandomState(seed=SEED).permutation(n_samples)
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
                os.path.join(dirpath, f"fold{fold}", f"dist_{metric}_best.pt")
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
                    - torch.tensor(
                        [train_expectation[g.y] for g in dataset[train_indices]]
                    )
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
                    - torch.tensor(
                        [test_expectation[g.y] for g in dataset[test_indices]]
                    )
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


def neighbors_correspondence_ZINC(
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
        "split",
        "ZINC",
        model,
        f"l={layer}_p={pooling}_d={emb_dim}",
    )
    if os.path.exists(os.path.join(dirpath, "neighbor.json")):
        with open(os.path.join(dirpath, "neighbor.json")) as f:
            stats = json.load(f)
    else:
        stats = {}

    root = os.path.join(
        os.path.dirname(__file__), "../Data/Datasets", "ZINC", "Compose([])"
    )
    train_dataset = ZINC(root=root, subset=True, split="train")
    test_dataset = ZINC(root=root, subset=True, split="test")
    keep_train = torch.zeros((len(metrics), len(ks)))
    keep_test = torch.zeros((len(metrics), len(ks)))
    for midx, metric in enumerate(metrics):
        dist_mat = torch.load(os.path.join(dirpath, f"fold0", f"dist_{metric}_best.pt"))
        # train
        dist_mat_train = dist_mat[:10000][:, :10000]
        _, train_indices_sorted = torch.sort(dist_mat_train, dim=1)
        anchor_y = torch.zeros(10000)
        neighbor_y = torch.zeros(10000, max(ks))
        for anchor in range(10000):
            anchor_y[anchor] = train_dataset[anchor].y
            for k in range(max(ks)):
                neighbor_y[anchor][k] = train_dataset[
                    train_indices_sorted[anchor][k + 1]  # not regard itself as neighbor
                ].y
        for kidx, k in enumerate(ks):
            keep_train[midx, kidx] = torch.mean(
                torch.abs(neighbor_y[:, :k].sum(dim=1) / k - anchor_y)
                - torch.sum(
                    torch.abs(anchor_y.reshape(-1, 1) - anchor_y.reshape(1, -1)), dim=1
                )
                / (len(anchor_y) - 1)
            ).item()
        # test
        dist_mat_test = dist_mat[11000:][:, 11000:]
        _, test_indices_sorted = torch.sort(dist_mat_test, dim=1)
        anchor_y = torch.zeros(1000)
        neighbor_y = torch.zeros(1000, max(ks))
        for anchor in range(1000):
            anchor_y[anchor] = test_dataset[anchor].y
            for k in range(max(ks)):
                neighbor_y[anchor][k] = test_dataset[
                    test_indices_sorted[anchor][k + 1]  # not regard itself as neighbor
                ].y
        for kidx, k in enumerate(ks):
            keep_test[midx, kidx] = torch.mean(
                torch.abs(neighbor_y[:, :k].sum(dim=1) / k - anchor_y)
                - torch.sum(
                    torch.abs(anchor_y.reshape(-1, 1) - anchor_y.reshape(1, -1)), dim=1
                )
                / (len(anchor_y) - 1)
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
            stats["train"][metric][k]["mean"] = keep_train[midx, kidx].item()
            stats["train"][metric][k]["std"] = 0
            stats["test"][metric][k]["mean"] = keep_test[midx, kidx].item()
            stats["test"][metric][k]["std"] = 0
    with open(os.path.join(dirpath, "neighbor.json"), "w") as f:
        json.dump(stats, f)


def neighbor_acc_plot(
    dataset: str,
    kfold: int,
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
                        "split",
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

    mets = np.zeros((2, 3, len(layers), len(emb_dims), len(poolings)))
    for tidx, target in enumerate(["train", "test"]):
        for midx, model in enumerate(["GAT", "GCN", "GIN"]):
            for lidx, layer in enumerate(layers):
                for eidx, emb_dim in enumerate(emb_dims):
                    for pidx, pooling in enumerate(poolings):
                        dirpath = os.path.join(
                            os.path.dirname(__file__),
                            "../Results",
                            "split",
                            dataset,
                            model,
                            f"l={layer}_p={pooling}_d={emb_dim}",
                        )
                        for fold in range(kfold):
                            with open(
                                os.path.join(dirpath, f"fold{fold}", "results.json")
                            ) as f:
                                log = json.load(f)
                                mets[tidx, midx, lidx, eidx, pidx] += log[
                                    f"details_{target}"
                                ]["accuracy" if dataset != "ZINC" else "mae"][-1]
    mets /= kfold

    for color, cands in [
        ("model", ["GAT", "GCN", "GIN"]),
        ("layer", layers),
        ("emb_dim", emb_dims),
        ("pooling", poolings),
    ]:
        fig, axes = plt.subplots(
            len(ks),
            2 * (len(metrics) + 1),
            figsize=(4 * 2 * (len(metrics) + 1), 4 * len(ks)),
        )
        fig.suptitle(f"dataset: {dataset}")
        for meidx, metric in enumerate(metrics):
            for tidx, (xidx, yidx) in enumerate([(0, 0), (1, 1), (0, 1)]):
                for kidx, k in enumerate(ks):
                    for cidx, cand in enumerate(cands):
                        if color == "model":
                            axes[kidx, meidx * 3 + tidx].scatter(
                                neighbors[xidx, meidx, kidx, cidx].flatten(),
                                mets[yidx, cidx].flatten(),
                                label=f"{cand}",
                            )
                        elif color == "layer":
                            axes[kidx, meidx * 3 + tidx].scatter(
                                neighbors[xidx, meidx, kidx, :, cidx].flatten(),
                                mets[yidx, :, cidx].flatten(),
                                label=f"{cand}",
                            )
                        elif color == "emb_dim":
                            axes[kidx, meidx * 3 + tidx].scatter(
                                neighbors[xidx, meidx, kidx, :, :, cidx].flatten(),
                                mets[yidx, :, :, cidx].flatten(),
                                label=f"{cand}",
                            )
                        elif color == "pooling":
                            axes[kidx, meidx * 3 + tidx].scatter(
                                neighbors[xidx, meidx, kidx, :, :, :, cidx].flatten(),
                                mets[yidx, :, :, :, cidx].flatten(),
                                label=f"{cand}",
                            )
                    if meidx * 3 + tidx == 0:
                        axes[kidx, meidx * 3 + tidx].set_ylabel(f"{k}", size="large")
                    if kidx == len(ks) - 1:
                        axes[kidx, meidx * 3 + tidx].set_xlabel(
                            f"{'train' if xidx == 0 else 'test'}/{'train' if yidx == 0 else 'test'}, {metric}",
                            size="large",
                        )
        fig.supxlabel(
            "Neighbor probability" if dataset != "ZINC" else "Neighbor difference",
            size="xx-large",
        )
        fig.supylabel("Accuracy" if dataset != "ZINC" else "MAE", size="xx-large")
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


def tSNE_embedding(dataset: str, modeldir: str):
    with open(os.path.join(modeldir, "info.json")) as f:
        info_dict = json.load(f)
    info = Namespace(**info_dict)
    info.train_with_all_data = False
    set_seed(info.seed)
    _, train_loader, _, test_loader = load_dataset(info, config)
    num_classes, num_vertex_features = (
        train_loader.dataset.num_classes,
        train_loader.dataset.num_node_features,
    )
    if info.dataset.lower() == "zinc":
        num_classes = 1
    try:
        num_tasks = train_loader.dataset.num_tasks
    except:
        num_tasks = 1
    model = get_model(info, num_classes, num_vertex_features, num_tasks)
    model.to(info.device)
    model.load_state_dict(
        torch.load(os.path.join(modeldir, f"fold{info.test_fold}", "model_best.pt"))
    )
    # Embedding
    train_embedding = torch.zeros((len(train_loader.dataset), info.emb_dim))
    train_y = torch.zeros(len(train_loader.dataset))
    for i, batch in enumerate(train_loader):
        batch = batch.to(info.device)
        train_embedding[i * info.batch_size : (i + 1) * info.batch_size] = (
            compute_embeddings(batch, model, info.device)
        )
        train_y[i * info.batch_size : (i + 1) * info.batch_size] = batch.y
    test_embedding = torch.zeros((len(test_loader.dataset), info.emb_dim))
    test_y = torch.zeros(len(test_loader.dataset))
    for i, batch in enumerate(test_loader):
        batch = batch.to(info.device)
        test_embedding[i * info.batch_size : (i + 1) * info.batch_size] = (
            compute_embeddings(batch, model, info.device)
        )
        test_y[i * info.batch_size : (i + 1) * info.batch_size] = batch.y
    # tSNE
    train_tsne = TSNE(
        n_components=2, perplexity=10 if dataset == "MUTAG" else 30
    ).fit_transform(train_embedding.cpu().detach())
    test_tsne = TSNE(
        n_components=2, perplexity=10 if dataset == "MUTAG" else 30
    ).fit_transform(test_embedding.cpu().detach())
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for i, (tsne, y) in enumerate([(train_tsne, train_y), (test_tsne, test_y)]):
        axes[i].scatter(
            tsne[:500, 0],
            tsne[:500, 1],
            c=y[:500],
        )
        axes[i].set_title(f"{'Train' if i == 0 else 'Test'}")
    fig.suptitle(f"dataset: {dataset}, model: {info.model}")
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
            f"tsne_{info.model}_l={info.num_mp_layers}_p={info.pooling}_d={info.emb_dim}.png",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--command",
        type=str,
        choices=[
            "visualize_loss",
            "neighbors_correspondence",
            "neighbor_acc_plot",
            "tSNE_embedding",
        ],
    )
    parser.add_argument(
        "--dataset", type=str, choices=["MUTAG", "Mutagenicity", "NCI1", "ZINC"]
    )
    # For tSNE_embedding
    parser.add_argument(
        "--model", type=str, choices=["GAT", "GCN", "GIN"], required=False
    )
    parser.add_argument("--l", type=int, required=False)
    parser.add_argument("--p", type=str, required=False)
    parser.add_argument("--d", type=int, required=False)
    args = parser.parse_args()
    if args.dataset == "MUTAG":
        args.kfold = 5
        args.epochs = 30
    elif args.dataset in ["Mutagenicity", "NCI1"]:
        args.kfold = 5
        args.epochs = 100
    elif args.dataset == "ZINC":
        args.kfold = 1
        args.epochs = 100
    print(args)
    if args.command == "visualize_loss":
        for model in ["GAT", "GCN", "GIN"]:
            visualize_loss(
                args.dataset,
                args.kfold,
                args.epochs,
                model=model,
                layers=[1, 2, 3, 4],
                emb_dims=[32, 64, 128],
                poolings=["mean", "sum"],
            )
    elif args.command == "neighbors_correspondence":
        for model in ["GAT", "GCN", "GIN"]:
            for layer in [1, 2, 3, 4]:
                for emb_dim in [32, 64, 128]:
                    for pooling in ["mean", "sum"]:
                        print(
                            f"model: {model}, layer: {layer}, emb_dim: {emb_dim}, pooling: {pooling}"
                        )
                        if args.dataset == "ZINC":
                            neighbors_correspondence_ZINC(
                                model=model,
                                layer=layer,
                                emb_dim=emb_dim,
                                pooling=pooling,
                                metrics=["l1", "l2"],
                                ks=(
                                    [1, 5, 10]
                                    if args.dataset == "MUTAG"
                                    else [1, 5, 10, 20]
                                ),
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
                                    if args.dataset == "MUTAG"
                                    else [1, 5, 10, 20]
                                ),
                            )
    elif args.command == "neighbor_acc_plot":
        neighbor_acc_plot(
            args.dataset,
            args.kfold,
            layers=[1, 2, 3, 4],
            emb_dims=[32, 64, 128],
            poolings=["mean", "sum"],
            metrics=["l1", "l2"],
            ks=[1, 5, 10] if args.dataset == "MUTAG" else [1, 5, 10, 20],
        )
    elif args.command == "tSNE_embedding":
        tSNE_embedding(
            args.dataset,
            os.path.join(
                os.path.dirname(__file__),
                "../Results",
                "split",
                args.dataset,
                args.model,
                f"l={args.l}_p={args.p}_d={args.d}",
            ),
        )
