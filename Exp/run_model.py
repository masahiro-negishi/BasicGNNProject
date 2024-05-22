"""
Trains and evaluates a model a single time for given hyperparameters.
"""

import json
import os
import random
import time

import numpy as np
import torch

from Exp.parser import parse_args
from Exp.preparation import (
    get_loss,
    get_model,
    get_optimizer_scheduler,
    get_prediction_type,
    load_dataset,
)
from Exp.training_loop_functions import compute_embeddings, eval, step_scheduler, train
from Misc.config import config
from Misc.tracking import get_tracker
from Misc.utils import list_of_dictionary_to_dictionary_of_lists


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def track_epoch(tracker, epoch, metric_name, train_result, val_result, test_result, lr):
    tracker.log(
        {
            "Epoch": epoch,
            "Train/Loss": train_result["total_loss"],
            "Val/Loss": val_result["total_loss"],
            f"Val/{metric_name}": val_result[metric_name],
            "Test/Loss": test_result["total_loss"],
            f"Test/{metric_name}": test_result[metric_name],
            "LearningRate": lr,
        }
    )


def print_progress(
    train_loss, val_loss, test_loss, metric_name, val_metric, test_metric
):
    print(f"\tTRAIN\t loss: {train_loss:6.4f}")
    print(f"\tVAL\t loss: {val_loss:6.4f}\t  {metric_name}: {val_metric:10.4f}")
    print(f"\tTEST\t loss: {test_loss:6.4f}\t  {metric_name}: {test_metric:10.4f}")


def main(args):
    print(args)
    device = args.device
    use_tracking = args.use_tracking
    dataset_name = args.dataset

    set_seed(args.seed)
    full_loader, train_loader, val_loader, test_loader = load_dataset(args, config)
    num_classes, num_vertex_features = (
        train_loader.dataset.num_classes,
        train_loader.dataset.num_node_features,
    )
    prediction_type = get_prediction_type(args.dataset.lower())

    if (
        ("qm9" in dataset_name.lower() and "_" in dataset_name.lower())
        or (
            args.dataset.lower() in ["zinc", "zinc_full"]
            or "ogb" in args.dataset.lower()
        )
        or (args.dataset.lower() == "pcqm-contact")
    ):
        num_classes = 1

    try:
        num_tasks = train_loader.dataset.num_tasks
    except:
        num_tasks = 1

    print(f"#Features: {num_vertex_features}")
    print(f"#Classes: {num_classes}")
    print(f"#Tasks: {num_tasks}")

    model = get_model(args, num_classes, num_vertex_features, num_tasks)
    print(model)
    nr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device)
    optimizer, scheduler = get_optimizer_scheduler(model, args)
    use_scheduler_with_early_stopping = args.scheduler == "ReduceLROnPlateau"
    if use_scheduler_with_early_stopping:
        print("Using a scheduler that supports early stopping")

    loss_dict = get_loss(dataset_name)
    loss_fct = loss_dict["loss"]
    eval_name = loss_dict["metric"]
    metric_method = loss_dict["metric_method"]

    tracker = None
    if use_tracking:
        tracker = get_tracker(config.tracker, args, config.project)

    print("Begin training.\n")
    time_start = time.time()
    train_results, val_results, test_results = [], [], []
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}")
        train_result = train(
            model,
            device,
            train_loader,
            optimizer,
            loss_fct,
            eval_name,
            tracker,
            metric_method=metric_method,
            prediction_type=prediction_type,
        )
        val_result = eval(
            model,
            device,
            val_loader,
            loss_fct,
            eval_name,
            metric_method=metric_method,
            prediction_type=prediction_type,
        )
        test_result = eval(
            model,
            device,
            test_loader,
            loss_fct,
            eval_name,
            metric_method=metric_method,
            prediction_type=prediction_type,
        )

        train_results.append(train_result)
        val_results.append(val_result)
        test_results.append(test_result)

        print_progress(
            train_result["total_loss"],
            val_result["total_loss"],
            test_result["total_loss"],
            eval_name,
            val_result[eval_name],
            test_result[eval_name],
        )

        if use_tracking:
            track_epoch(
                tracker,
                epoch,
                eval_name,
                train_result,
                val_result,
                test_result,
                optimizer.param_groups[0]["lr"],
            )

        step_scheduler(scheduler, args.scheduler, val_result["total_loss"])

        # EXIT CONDITIONS
        if (
            use_scheduler_with_early_stopping
            and optimizer.param_groups[0]["lr"] < args.scheduler_min_lr
        ):
            print("\nLR reached minimum: exiting.")
            break

        runtime = (time.time() - time_start) / 3600
        if args.max_time > 0 and runtime > args.max_time:
            print("\nMaximum training time reached: exiting.")
            break

    # Final result
    train_results = list_of_dictionary_to_dictionary_of_lists(train_results)
    val_results = list_of_dictionary_to_dictionary_of_lists(val_results)
    test_result = list_of_dictionary_to_dictionary_of_lists(test_results)

    if eval_name in ["mae", "rmse (ogb)"]:
        best_val_epoch = np.argmin(val_results[eval_name])
        mode = "min"
    else:
        best_val_epoch = np.argmax(val_results[eval_name])
        mode = "max"

    loss_train, loss_val, loss_test = (
        train_results["total_loss"][best_val_epoch],
        val_results["total_loss"][best_val_epoch],
        test_result["total_loss"][best_val_epoch],
    )
    result_val, result_test = (
        val_results[eval_name][best_val_epoch],
        test_result[eval_name][best_val_epoch],
    )

    print("\n\nFinal Result:")
    print(f"\tRuntime: {runtime:.2f}h")
    print(f"\tBest epoch {best_val_epoch} / {args.epochs}")
    print_progress(loss_train, loss_val, loss_test, eval_name, result_val, result_test)

    if use_tracking:
        tracker.log(
            {
                "Final/Train/Loss": loss_train,
                "Final/Val/Loss": loss_val,
                f"Final/Val/{eval_name}": result_val,
                "Final/Test/Loss": loss_test,
                f"Final/Test/{eval_name}": result_test,
            }
        )

        tracker.finish()

    # Save results
    if args.save_rslt:
        path = os.path.join(
            config.RESULTS_PATH,
            args.dataset,
            args.model,
            str(args.num_mp_layers),
            f"fold{args.test_fold}",
        )
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))
        with open(os.path.join(path, "results.json"), "w") as f:
            json.dump(
                {"train": train_results, "val": val_results, "test": test_result}, f
            )
        with open(
            os.path.join(
                config.RESULTS_PATH,
                args.dataset,
                args.model,
                str(args.num_mp_layers),
                "info.json",
            ),
            "w",
        ) as f:
            json.dump(args.__dict__, f)
    if args.save_dist:
        model.eval()
        embeddings = torch.cat(
            [
                compute_embeddings(batch, model, device).detach()
                for batch in full_loader
            ],
            dim=0,
        )
        # L1/L2 distance
        dist_1 = torch.cdist(embeddings, embeddings, p=1)
        dist_2 = torch.cdist(embeddings, embeddings, p=2)
        torch.save(dist_1, os.path.join(path, "dist_1.pt"))
        torch.save(dist_2, os.path.join(path, "dist_2.pt"))

    return {
        "mode": mode,
        "loss_train": loss_train,
        "loss_val": loss_val,
        "loss_test": loss_test,
        "val": result_val,
        "test": result_test,
        "runtime_hours": runtime,
        "epochs": epoch,
        "best_val_epoch": int(best_val_epoch),
        "parameters": nr_parameters,
        "details_train": train_results,
        "details_val": val_results,
        "details_test": test_results,
    }


def run(passed_args=None):
    args = parse_args(passed_args)
    return main(args)


if __name__ == "__main__":
    run()
