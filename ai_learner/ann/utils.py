import torch
from os.path import join
import os
import shutil


def save_torch_state_dict(model, path):
    torch.save(model.state_dict(), path)


