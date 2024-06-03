import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import lr_scheduler

from src.data.cl_datasets import get_cl_dataloaders
from src.models.factory import get_model
from src.utils.cl_metric_accumulator import CLResults
from src.utils.optimizer import get_optimizer
from src.utils.summary_writer import SummaryWriter
from src.utils.train_evaluate_utils import evaluate, train_single_epoch
from src.utils.utils import seed_everything


def train_single_task(exp_dict, model, cur_train_dataloader, cur_valid_dataloader, test_dataloaders, optimizer, loss_fn,
                      t_idx, is_hp_search, least_acc, metrics, summary_writer, add_epochwise_metrics):
    cur_test_dataloader = test_dataloaders[-1]

    restore_best_weight = "restore_best_weight" in exp_dict and exp_dict["restore_best_weight"]
    use_valid_dataset = restore_best_weight and (cur_valid_dataloader is not None)

    best_acc = 0
    best_model_param_dict = None

    for epoch in range(exp_dict["n_epochs_per_task"]):
        train_single_epoch(
            model, cur_train_dataloader, optimizer, loss_fn, task_idx=t_idx, use_fixed_mask=False,
            is_hp_search=is_hp_search)

        if not add_epochwise_metrics and (epoch % 5 == 0 and epoch > 3):
            acc = evaluate(model, cur_test_dataloader, task_idx=t_idx)["test_acc"]
            print(f"Epoch: {epoch} Current task Test Acc : {acc:.4f}")
            if is_hp_search:
                assert acc > least_acc

        if use_valid_dataset:
            valid_acc = evaluate(model, cur_valid_dataloader, task_idx=t_idx)["test_acc"]
            print(f"Epoch: {epoch} Current task Valid Acc : {valid_acc:.4f}")
            if is_hp_search:
                assert valid_acc > least_acc

            if use_valid_dataset:
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_model_param_dict = model.state_dict()

        # if not is_hp_search:
        #     summary_writer.log_model_params(t_idx, epoch, model)
        #     if add_epochwise_metrics:
        #         evaluate_test_dataloaders(model, test_dataloaders, metrics, t_idx, run_independent=False,
        #                                   use_fixed_mask=False, add_epochwise_metrics=True)
        #     summary_writer.log_learning_rate(optimizer)

    if use_valid_dataset:
        model.load_state_dict(best_model_param_dict)
        print("Best model selected!")
        print(f"Best valid acc: {best_acc:.4f}")
        print(f"Best test acc: {evaluate(model, cur_test_dataloader, task_idx=t_idx)['test_acc']:.4f}")

    if model.use_fixed_mask:
        model.eval()
        model.add_task_mask(task_idx=t_idx)
        model.train()
        print("Fine tuning started!")

        for epoch in range(exp_dict["n_epochs_per_task_finetune"]):
            if not ("no_lr_reduction" in exp_dict):
                scheduler = lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 0.1)
                scheduler.step()

            train_single_epoch(
                model, cur_train_dataloader, optimizer, loss_fn, task_idx=t_idx, use_fixed_mask=True,
                is_hp_search=is_hp_search)

            if use_valid_dataset:
                valid_acc = evaluate(model, cur_valid_dataloader, task_idx=t_idx, use_fixed_mask=True)["test_acc"]
                print(f"Epoch: {epoch} Current task Valid Acc : {valid_acc:.4f}")
                if is_hp_search:
                    assert valid_acc > least_acc

                if use_valid_dataset:
                    if valid_acc > best_acc:
                        best_acc = valid_acc
                        best_model_param_dict = model.state_dict()

            if not add_epochwise_metrics and (epoch % 13 == 0 and epoch > 3):
                acc = evaluate(model, cur_test_dataloader, task_idx=t_idx, use_fixed_mask=True)["test_acc"]
                print(f"Epoch: {epoch} Current task Test Acc : {acc:.4f}")
                assert acc > least_acc

            # if not is_hp_search and add_epochwise_metrics:
            #     evaluate_test_dataloaders(model, test_dataloaders, metrics, t_idx, run_independent=False,
            #                               use_fixed_mask=True, add_epochwise_metrics=True)
            #     summary_writer.log_learning_rate(optimizer)

        if use_valid_dataset:
            model.load_state_dict(best_model_param_dict)
            print("Best model selected!")
            print(f"Best valid acc: {best_acc:.4f}")
            print(
                f"Best test acc: {evaluate(model, cur_test_dataloader, task_idx=t_idx, use_fixed_mask=True)['test_acc']:.4f}")


def evaluate_test_dataloaders(model, test_dataloaders, metrics, task_idx=None, run_independent=False,
                              use_fixed_mask=False, add_epochwise_metrics=False):
    test_dataloaders_ = [test_dataloaders[-1]] if run_independent else test_dataloaders

    tasks_test_accs = []
    for t_idx_, test_dataloader in enumerate(test_dataloaders_):
        if model.use_fixed_mask:
            if use_fixed_mask:
                use_fixed_mask_ = True
            else:
                use_fixed_mask_ = (t_idx_ < task_idx)
        else:
            use_fixed_mask_ = False

        acc = evaluate(
            model, test_dataloader, task_idx=task_idx if run_independent else t_idx_,
            use_fixed_mask=use_fixed_mask_
        )["test_acc"]

        tasks_test_accs.append(acc)

    print("Test acc:", tasks_test_accs)
    metrics.add_result(tasks_test_accs, add_to_taskwise_metrics=not add_epochwise_metrics)
    if not add_epochwise_metrics:
        print("\n", "\n", f"Accuracies: {tasks_test_accs}")


def cl_train(
        exp_dict,
        model, get_dataloader, loss_fn, optimizer_args,
        is_hp_search=True, least_acc=50.,
        model_args=None, run_task_independent=False, add_epochwise_metrics=False):
    test_dataloaders = []
    metrics = CLResults(exp_dict["experiment_name"])
    n_layers = model.n_layers if hasattr(model, "n_layers") else model.truncation_level
    summary_writer = SummaryWriter(n_layers, exp_dict["n_tasks"])


    # if not is_hp_search:
    #     fig1, ax1 = plt.subplots(1, exp_dict["n_tasks"], figsize=[exp_dict["n_tasks"] * 1.25, 5])
    #     ax1[0].set_ylabel("Input Dimension", fontsize=20)
    #     fig2, ax2 = plt.subplots(1, exp_dict["n_tasks"], figsize=[exp_dict["n_tasks"] * 1.25, 5])
    #     network_usages = []

    for t_idx in range(exp_dict["n_tasks"]):
        optimizer = get_optimizer(model, optimizer_args, t_idx)

        cur_train_dataloader, cur_valid_dataloader, cur_test_dataloader = get_dataloader(t_idx)
        test_dataloaders.append(cur_test_dataloader)

        train_single_task(exp_dict, model, cur_train_dataloader, cur_valid_dataloader, test_dataloaders, optimizer,
                          loss_fn, t_idx,
                          is_hp_search, least_acc, metrics, summary_writer, add_epochwise_metrics)

        evaluate_test_dataloaders(model, test_dataloaders, metrics, t_idx, run_task_independent,
                                  use_fixed_mask=model.use_fixed_mask)

        model.reset_for_new_task(task_id=t_idx)

        # if not is_hp_search:
        #     summary_writer.log_test_structure_params(model, is_final_task=(t_idx == exp_dict["n_tasks"] - 1))

    return metrics, summary_writer


def train(exp_dict, model_args, optimizer_args, is_hp_search=True, least_acc=50., run_task_independent=False,
          add_epochwise_metrics=False):
    seed_everything(exp_dict["seed"])

    get_dataloader, n_classes_per_task = get_cl_dataloaders(exp_dict, exp_dict["batch_size"])
    model_args["out_feature_dim"] = n_classes_per_task

    model = get_model(exp_dict, model_args) if not run_task_independent else None
    print(model)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    cl_metrics, summary_writer = cl_train(
        exp_dict, model, get_dataloader, loss_fn, optimizer_args,
        is_hp_search=is_hp_search, least_acc=least_acc, model_args=model_args,
        run_task_independent=run_task_independent, add_epochwise_metrics=add_epochwise_metrics
    )

    print(cl_metrics)
    return cl_metrics, summary_writer
