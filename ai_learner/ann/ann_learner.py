import torch
from apex import amp

from . import CallbacksGroup
from .callbacks import SaveModel, SimpleProgressBar, MetricsCB
from ..base.learner import Learner
from .train_functions import train_ann
from .utils import save_torch_state_dict, save_torch_model


class AnnLearner(Learner):
    def __init__(self, model, loss=None, optimizer=None, metrics=None, apex=True, device='cuda'):
        super().__init__(model)
        self.loss = loss
        self.optimizer = optimizer
        if metrics:
            self.metrics = metrics
            self.main_metric = metrics[0]
        else:
            self.metrics = None

        self.apex = apex
        self.device = device
        if self.apex:
            self.model, self.optimizer = amp.initialize(self.model.to(self.device), self.optimizer)

        self.training_phase = None
        self.validation_phase = None
        self.inference_phase = None

        # paths
        self.best_model_path = None
        self.best_model_dict_path = None
        self.last_model_path = None
        self.last_model_dict_path = None
        self.best_model_score = None

    def train(self, training_phase, validation_phase, callbacks=None, epochs=100):
        if callbacks is None:
            callbacks = []
        callbacks = [MetricsCB(), SimpleProgressBar(), SaveModel(verbose=True)] + callbacks
        callbacks = CallbacksGroup(learner=self, callbacks=callbacks)

        self.training_phase = training_phase
        self.validation_phase = validation_phase

        train_ann(learner=self, model=self.model, loss=self.loss, optimizer=self.optimizer,
                  training_phase=training_phase, validation_phase=validation_phase, callbacks=callbacks,
                  epochs=epochs, device=self.device)

    def infer(self, inference_phase, **kwargs):
        raise NotImplementedError

    def save_state_dict(self, path):
        save_torch_state_dict(self.model, path)

    def save_model(self, path):
        save_torch_model(self.model, path)

    def load_best_model(self):
        if self.best_model_path:
            self.model = torch.load(self.best_model_path)

    def load_last_model(self):
        if self.last_model_path:
            self.model = torch.load(self.last_model_path)







