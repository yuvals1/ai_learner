import os
import re
import shutil


def create_dir_and_clear_if_exists(base_dir, new_dir_name):
    folder_path = os.path.join(base_dir, new_dir_name)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)
    return folder_path


def create_dir_if_not_exist(new_dir_path):
    if not os.path.exists(new_dir_path):
        os.mkdir(new_dir_path)


def to_snake_case(string):
    """Converts CamelCase string into snake_case."""

    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def get_class_name(obj):
    return obj.__class__.__name__

