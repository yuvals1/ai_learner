from .utils import create_dir_and_clear_if_exists
from .callbacks import CallbacksGroup, MetricsCB, SimpleProgressBar, SaveModel
from os.path import join
import pickle


class Learner:
    def __init__(self, model, loss, optimizer, metrics):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = [loss] + metrics
        self.main_metric = metrics[0]

        # phases
        self.training_phase = None
        self.validation_phase = None
        self.inference_phase = None

        # for training config
        self.callbacks = None
        self.data_dir = None

        # for saving config
        self.model_name = None
        self.model_path = None
        self.model_dir_path = None
        self.model_dir_root = None

        # for inference config
        self.inference_dir_path = None
        self.task = None

        ##
        self.wrong_analysis = None

    def train(self):
        raise NotImplementedError

    def infer(self):
        raise NotImplementedError

    def train_and_infer(self):
        raise NotImplementedError

    def set_training_phase(self, *args):
        raise NotImplementedError

    def set_validation_phase(self, *args):
        raise NotImplementedError

    def set_inference_phase(self, *args):
        raise NotImplementedError

    def config_for_training(self, *args):
        raise NotImplementedError

    def config_for_inference(self, *args):
        raise NotImplementedError

    def config_for_saving(self, model_name, model_dir_root='saved_models'):
        self.model_name = model_name
        self.model_dir_root = model_dir_root
        self.model_dir_path = create_dir_and_clear_if_exists(self.model_dir_root, model_name)

    def config_task(self, task):
        self.task = task(learner=self)

    def save(self):
        path = join(self.model_dir_path, 'learner.pkl')
        with open(path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

