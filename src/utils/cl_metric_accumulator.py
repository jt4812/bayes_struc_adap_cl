import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

LOG_INDEPENDENT_FOLDER = "runs_independent"


def read_bi_metrics_from_file(experiment_name):
    file_path = os.path.join(f"{experiment_name}-independent.pkl")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            independent_run_metrics = pickle.load(f)
        return independent_run_metrics.task_wise_metrics
    return None


class CLResults:
    def __init__(self, experiment_name=None):
        self.taskwise_metrics = []
        self.epochwise_metrics = []
        self.experiment_name = experiment_name
        self.b_i = read_bi_metrics_from_file(experiment_name)  # independent model predictions

    def add_result(self, task_metric_ls, add_to_taskwise_metrics=True):
        if add_to_taskwise_metrics:
            self.taskwise_metrics.append(np.array(task_metric_ls))
        else:
            self.epochwise_metrics.append(np.array(task_metric_ls))

    @property
    def mean_metrics_across_training_task(self):
        return [np.mean(metrics) for metrics in self.taskwise_metrics]

    @property
    def final_metrics_across_task(self):
        return self.taskwise_metrics[-1]

    @property
    def final_max_min_metrics(self):
        return max(self.final_metrics_across_task), min(self.final_metrics_across_task)

    @property
    def final_mean_metric(self):
        return np.mean(self.final_metrics_across_task)

    def align_taskwise_metrics(self):
        """ Aligns the taskwise metrics """
        n_tasks = len(self.taskwise_metrics)
        taskwise_metrics = [[] for _ in range(n_tasks)]
        for training_task_idx in range(n_tasks):
            metric_ls = self.taskwise_metrics[training_task_idx]
            for task_idx, metric in enumerate(metric_ls):
                taskwise_metrics[task_idx].append(metric)

        return taskwise_metrics

    @property
    def backward_transfer_metrics_for_task(self):
        metric_ii = np.array([metric_[idx] for idx, metric_ in enumerate(self.taskwise_metrics)])
        metric_Ti = self.final_metrics_across_task
        return (metric_Ti - metric_ii)[:len(metric_Ti) - 1]

    @property
    def mean_backward_transfer_metric(self):
        return np.mean(self.backward_transfer_metrics_for_task)

    @property
    def mean_forward_transfer_metric(self):
        if self.b_i is not None:
            metric_ii = np.array([metric_[idx] for idx, metric_ in enumerate(self.taskwise_metrics)])
            return np.mean(metric_ii - self.b_i)
        else:
            return None

    def jsonify_taskwise_metrics(self):
        return {task_idx: taskwise_metrics_ for task_idx, taskwise_metrics_ in enumerate(self.taskwise_metrics)}

    def __repr__(self):
        return f"""
        final_mean_metric = {self.final_mean_metric}
        final_max_min_metric = {self.final_max_min_metrics}
        final_metrics_across_task = {self.final_metrics_across_task}
        mean_metrics_across_training = {self.mean_metrics_across_training_task}
        """

    def plot_save_metrics_across_tasks(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(self.mean_metrics_across_training_task)
        fig.savefig(save_path)

    def save(self, save_dir, experiment_name, independent=False):
        filename = f"cl_metrics{'-independent' if independent else ''}.pkl"
        save_path = os.path.join(save_dir, experiment_name, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
