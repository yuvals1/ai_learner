from ..utils.utils import create_dir_and_clear_if_exists
from os.path import join
import pickle


class Learner:
    def __init__(self, model):
        self.model = model

        # for saving config
        self.model_name = None
        self.model_dir_path = None
        self.model_dir_root = None

        # for inference config
        self.inference_dir_path = None

        ##
        self.wrong_analysis = None

    def train(self, *args):
        raise NotImplementedError

    def infer(self, *args):
        raise NotImplementedError

    def config_for_saving(self, model_name, model_dir_root='saved_models'):
        self.model_name = model_name
        self.model_dir_root = model_dir_root
        self.model_dir_path = create_dir_and_clear_if_exists(self.model_dir_root, model_name)

    def save(self):
        path = join(self.model_dir_path, 'learner.pkl')
        with open(path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

