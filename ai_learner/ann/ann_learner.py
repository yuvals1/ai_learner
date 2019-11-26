from . import CallbacksGroup
from .callbacks import SaveModel, SimpleProgressBar, MetricsCB
from ..base.learner import Learner
from .train_functions import train_ann
from .utils import save_torch_state_dict


class AnnLearner(Learner):
    def __init__(self, model, loss=None, optimizer=None, metrics=None):
        super().__init__(model)
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = [loss] + metrics
        self.main_metric = metrics[0]

        # phases
        self.training_phase = None
        self.validation_phase = None
        self.inference_phase = None

        # paths
        self.best_model_path = None
        self.last_model_path = None

    def train(self, training_phase, validation_phase, epochs=10, callbacks=None, device='cuda', apex=True):
        if callbacks is None:
            callbacks = []
        callbacks = [MetricsCB(), SimpleProgressBar(), SaveModel(verbose=True)] + callbacks
        callbacks = CallbacksGroup(learner=self, callbacks=callbacks)

        train_ann(epochs, model=self.model, loss=self.loss,optimizer=self.optimizer,
                  training_phase=training_phase, validation_phase=validation_phase,
                  callbacks=callbacks, device=device, apex=apex)

    def infer(self, inference_phase, device='cuda', **kwargs):
        raise NotImplementedError

    def save_model(self, path):
        save_torch_state_dict(self.model, path)







