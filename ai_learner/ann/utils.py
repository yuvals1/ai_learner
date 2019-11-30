import torch
from os.path import join
import os
import shutil


def save_torch_state_dict(model, path):
    torch.save(model.state_dict(), path)


def change_learner_model_dir_name(learner):

    best_score_str = str(learner.validation_phase.metrics_holder.main_metric_best_score)[:6]
    model_dir_new_name = '_'.join([learner.model_name, best_score_str])
    model_dir_new_path = join(learner.model_dir_root, model_dir_new_name)
    model_dir_old_path = learner.model_dir_path

    if not os.path.exists(model_dir_new_path):
        shutil.move(model_dir_old_path, model_dir_new_path)
    learner.model_dir_path = model_dir_new_path
