import numpy as np
import pandas as pd
from os.path import join


class MetricsHolder:
    def __init__(self, learner):

        self.metrics = [learner.loss] + learner.metrics
        self.metrics_names = [metric.__name__ for metric in self.metrics]

        self.main_metric_name = learner.main_metric.__name__
        self.main_metric_best_score = 0
        self.main_metric_curr_score = 0
        self.main_metric_improved = True

        self.batches_history = {metric_name: [] for metric_name in self.metrics_names}
        self.epochs_history = {metric_name: [] for metric_name in self.metrics_names}

        self.epochs_df = None

    def batch_ended_update(self, pr, gt):
        for metric, metric_name in zip(self.metrics, self.batches_history.keys()):
            self.batches_history[metric_name].append(metric(pr, gt).detach().item())

    def epoch_ended_update(self):
        for metric_name in self.metrics_names:
            self.epochs_history[metric_name].append(np.mean(self.batches_history[metric_name]))
            self.batches_history[metric_name] = []

        self.update_main_metric_best_score()
        self.create_history_df()

    def update_main_metric_best_score(self):
        curr_score = self.epochs_history[self.main_metric_name][-1].item()
        self.main_metric_curr_score = curr_score

        if curr_score > self.main_metric_best_score:
            self.main_metric_improved = True
            self.main_metric_best_score = curr_score
        else:
            self.main_metric_improved = False

    def create_history_df(self):

        history_df = pd.DataFrame(self.epochs_history)
        self.epochs_df = pd.concat([update_history_df(history_df, metric) for metric in list(history_df.columns)],
                                   axis=0)

    def save_history(self, filename, dir_path):
        d = pd.DataFrame(self.epochs_history).reset_index().rename(columns={'index': 'epoch'})
        d.to_csv(join(dir_path, filename), index=False)


def update_history_df(history_df, metric):
    d = history_df[[metric]].reset_index().rename(columns={'index': 'epoch', metric: 'score'})
    d['metric'] = metric
    d = d[['epoch', 'metric', 'score']]

    return d

