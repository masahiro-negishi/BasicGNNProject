import argparse
import json
import os
from argparse import Namespace

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import torch  # type: ignore
from ogb.graphproppred import PygGraphPropPredDataset  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from torch_geometric.datasets import ZINC, TUDataset  # type: ignore

from Exp.preparation import get_model, load_dataset
from Exp.run_model import set_seed
from Exp.training_loop_functions import compute_embeddings
from Misc.config import config

SEED = 0


def figure_alignment_dmpnn_dfunc(hue: str):
    # Data
    dic: dict = {}
    for columns in [
        "dataset",
        "weight",
        "model",
        "layer",
        "emb_dim",
        "pooling",
        "k",
        "split",
        "metric",
    ]:
        dic[columns] = []
    for dataset in ["Mutagenicity", "ENZYMES", "ogbg-mollipo"]:
        for model in ["GCN", "GIN"]:
            for layer in [1, 2, 3, 4]:
                for emb_dim in [32, 64, 128]:
                    for pooling in ["mean", "sum"]:
                        dirpath = os.path.join(
                            os.path.dirname(__file__),
                            "../Results",
                            "split",
                            dataset,
                            model,
                            f"l={layer}_p={pooling}_d={emb_dim}",
                        )
                        for epoch in ["init", "best"]:
                            with open(
                                os.path.join(dirpath, f"neighbor_{epoch}.json")
                            ) as f:
                                stats = json.load(f)
                            for k in [1, 5, 10, 20]:
                                for split in ["train", "test"]:
                                    dic["dataset"].append(
                                        dataset
                                        if dataset != "ogbg-mollipo"
                                        else "Lipophilicity"
                                    )
                                    dic["weight"].append(
                                        "trained" if epoch == "best" else "untrained"
                                    )
                                    dic["model"].append(model)
                                    dic["layer"].append(layer)
                                    dic["emb_dim"].append(emb_dim)
                                    dic["pooling"].append(pooling)
                                    dic["k"].append(k)
                                    dic["split"].append(split)
                                    dic["metric"].append(
                                        stats[split]["l2"][str(k)]["mean"]
                                    )
    df = pd.DataFrame(dic)

    # Plot
    sns.set_context(
        "paper",
        rc={
            "axes.titlesize": 30,
            "axes.labelsize": 30,
        },
    )
    p = sns.catplot(
        data=df[df["split"] == "train"],
        x="k",
        y="metric",
        col="dataset",
        hue=hue,
        kind="violin",
        sharey=False,
        alpha=0.75,
        split=True,
        inner="quart",
    )
    p.set_axis_labels(r"$k$", "ALI" + r"$_k$", fontsize=30)
    p.tick_params(labelsize=20)
    p.set_titles(
        "{col_name}",
    )
    sns.move_legend(
        p,
        "center left",
        ncol=1,
        title=None,
        fontsize=25,
        bbox_to_anchor=(1, 0.5),
        handlelength=0.8,
        handletextpad=0.2,
        borderaxespad=0.0,
        borderpad=0.01,
    )
    plt.tight_layout()
    p.savefig(
        os.path.join(
            os.path.dirname(__file__), "../Paper", "alignment_dmpnn_dfunc.pdf"
        ),
        bbox_inches="tight",
        pad_inches=0.05,
    )


def mk_acc_plot(
    kfold: int = 5,
    layers: list[int] = [1, 2, 3, 4],
    emb_dims: list[int] = [32, 64, 128],
    poolings: list[str] = ["mean", "sum"],
    metrics: list[str] = ["l1", "l2"],
    ks: list[int] = [1, 5, 10, 20],
):
    neighbors = np.zeros(
        (3, 2, len(metrics), len(ks), 2, len(layers), len(emb_dims), len(poolings))
    )
    for didx, dataset in enumerate(["Mutagenicity", "ENZYMES", "ogbg-mollipo"]):
        for midx, model in enumerate(["GCN", "GIN"]):
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
                        with open(os.path.join(dirpath, "neighbor_best.json")) as f:
                            stats = json.load(f)
                        for tidx, target in enumerate(["train", "test"]):
                            for meidx, metric in enumerate(metrics):
                                for kidx, k in enumerate(ks):
                                    neighbors[
                                        didx,
                                        tidx,
                                        meidx,
                                        kidx,
                                        midx,
                                        lidx,
                                        eidx,
                                        pidx,
                                    ] = stats[target][metric][str(k)]["mean"]

    mets = np.zeros((3, 2, 2, len(layers), len(emb_dims), len(poolings)))
    for didx, dataset in enumerate(["Mutagenicity", "ENZYMES", "ogbg-mollipo"]):
        for tidx, target in enumerate(["train", "test"]):
            for midx, model in enumerate(["GCN", "GIN"]):
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
                                    mets[didx, tidx, midx, lidx, eidx, pidx] += log[
                                        f"details_{target}"
                                    ][
                                        (
                                            "mae"
                                            if dataset == "ZINC"
                                            else (
                                                "rmse (ogb)"
                                                if dataset == "ogbg-mollipo"
                                                else "accuracy"
                                            )
                                        )
                                    ][
                                        -1
                                    ]
    mets /= kfold

    fig, axes = plt.subplots(
        2, 3, figsize=(4 * 3, 4 * 2), gridspec_kw={"wspace": 0.30, "hspace": 0.30}
    )
    for didx, dataset in enumerate(["Mutagenicity", "ENZYMES", "ogbg-mollipo"]):
        for tidx, target in enumerate(["train", "test"]):
            x = neighbors[didx, 0, 0, 1].flatten()
            y = mets[didx, tidx].flatten()
            a, b = np.polyfit(x, y, 1)
            axes[tidx, didx].scatter(x, y, alpha=0.75)
            axes[tidx, didx].plot(
                np.linspace(np.min(x), np.max(x), 100),
                a * np.linspace(np.min(x), np.max(x), 100) + b,
                color="red",
            )
            axes[tidx, didx].text(
                s=f"corr: {np.corrcoef(x, y)[0, 1]:.2f}",
                x=0.7,
                y=0.4,
                transform=axes[tidx, didx].transAxes,
            )
            if tidx == 0:
                axes[tidx, didx].set_title(
                    dataset if dataset != "ogbg-mollipo" else "Lipophilicity",
                    size="xx-large",
                    pad=30,
                )
            ylabel = "RMSE" if dataset == "ogbg-mollipo" else "ACC"
            axes[tidx, didx].set_xlabel("ALI" + r"$_5$", size="large")
            axes[tidx, didx].set_ylabel(f"{target} {ylabel}", size="large")
    os.makedirs(
        os.path.join(
            os.path.dirname(__file__),
            "../Paper",
        ),
        exist_ok=True,
    )
    plt.savefig(
        os.path.join(
            os.path.dirname(__file__),
            "../Paper",
            f"alignment_acc.pdf",
        ),
    )


def figure_rmse_dmpnn_dstruc(hue: str):
    # Data
    dic: dict = {}
    for columns in [
        "dataset",
        "weight",
        "model",
        "layer",
        "emb_dim",
        "dstruc",
        "metric",
    ]:
        dic[columns] = []
    for dataset in ["Mutagenicity", "ENZYMES", "ogbg-mollipo"]:
        for model in ["GCN", "GIN"]:
            for layer in [1, 2, 3, 4]:
                for emb_dim in [32, 64, 128]:
                    for pooling in ["mean", "sum"]:
                        dirpath = os.path.join(
                            os.path.dirname(__file__),
                            "../Results",
                            "split",
                            dataset,
                            model,
                            f"l={layer}_p={pooling}_d={emb_dim}",
                        )
                        for epoch in ["init", "best"]:
                            with open(
                                os.path.join(dirpath, f"neighbor_{epoch}.json")
                            ) as f:
                                stats = json.load(f)
                            for dstruc in ["GED", "TMD", "WLOA", "WWL"]:
                                dic["dataset"].append(
                                    dataset
                                    if dataset != "ogbg-mollipo"
                                    else "Lipophilicity"
                                )
                                trained = "trained" if epoch == "best" else "untrained"
                                dic["weight"].append(f"{trained}/{pooling}")
                                dic["model"].append(model)
                                dic["layer"].append(layer)
                                dic["emb_dim"].append(emb_dim)
                                dic["dstruc"].append(dstruc)
                                dic["metric"].append(stats["rmse"]["l2"][dstruc])
    df = pd.DataFrame(dic)

    # Plot
    sns.set_context(
        "paper",
        rc={
            "axes.titlesize": 30,
            "axes.labelsize": 30,
        },
    )
    p = sns.catplot(
        data=df,
        x="dstruc",
        y="metric",
        col="dataset",
        hue=hue,
        kind="violin",
        sharey=True,
        alpha=0.75,
        split=True,
        inner="quart",
    )
    p.set_axis_labels("", "RMSE", fontsize=30)
    p.set_xticklabels(
        [
            r"$d_\mathrm{GED}$",
            r"$d_\mathrm{TMD}$",
            r"$d_\mathrm{WLOA}$",
            r"$d_\mathrm{WWL}$",
        ],
        fontsize=30,
    )
    p.tick_params(labelsize=20)
    p.set_titles("{col_name}")
    sns.move_legend(
        p,
        "center left",
        ncol=1,
        title=None,
        fontsize=25,
        bbox_to_anchor=(1, 0.5),
        handlelength=0.8,
        handletextpad=0.2,
        borderaxespad=0.0,
        borderpad=0.01,
    )
    plt.tight_layout()
    p.savefig(
        os.path.join(os.path.dirname(__file__), "../Paper", "rmse_dmpnn_dstruc.pdf"),
        bbox_inches="tight",
        pad_inches=0.05,
    )


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["text.usetex"] = True
    figure_alignment_dmpnn_dfunc(hue="weight")
    mk_acc_plot()
    figure_rmse_dmpnn_dstruc(hue="weight")
