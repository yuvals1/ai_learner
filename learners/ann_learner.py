from base.learner import Learner
import torch
from apex import amp


def to_device(z, device):
    if type(z) == list and len(z) > 1:
        z = [item.to(device) for item in z]
    elif type(z) == list and len(z) == 1:
        z = z[0].to(device)
    else:
        raise ValueError('...')
    return z


def unwrap_batch(batch, device, only_input=False):
    if not only_input:
        x, y = batch
        x = to_device(x, device)
        y = to_device(y, device)
        return x, y
    else:
        x, name, dataset = batch
        x = to_device(x, device)
        return x, name, dataset


class AnnLearner(Learner):
    def __init__(self, model, loss=None, optimizer=None, metrics=None, apex=True):
        self.apex = apex
        if self.apex:
            model, optimizer = amp.initialize(model, optimizer)
        super().__init__(model, loss, optimizer, metrics)

        self.finish_training = False
        self.train_stage = 1
        self.best_model_now = True

    def train(self, epochs=10, device='cuda'):

        self.model.to(device)
        self.callbacks.training_started()

        for epoch in range(1, epochs + 1):
            self.callbacks.epoch_started()

            for phase in (self.training_phase, self.validation_phase):
                self.callbacks.phase_started()
                is_training = phase.name == 'training'
                self.model.train(is_training)

                for batch in phase.loader:
                    self.callbacks.batch_started()

                    x, y = unwrap_batch(batch, device)

                    with torch.set_grad_enabled(is_training):
                        out = self.model(x)
                        loss_score = self.loss(out, y)

                    if is_training:
                        self.optimizer.zero_grad()
                        if self.apex:
                            with amp.scale_loss(loss_score, self.optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss_score.backward()
                        self.optimizer.step()

                    self.callbacks.batch_ended(phase=phase, pr=out, gt=y, x=x)
                self.callbacks.phase_ended()
            self.callbacks.epoch_ended(epoch=epoch)
            if self.finish_training:
                break
        self.callbacks.training_ended()

    def validate(self, device='cuda'):
        raise NotImplementedError

    def infer(self, device='cuda', **kwargs):

        self.model.load(self.model_path)
        self.model.to(device)
        self.model.train(False)

        for batch in self.inference_phase.loader:
            x, img_name, dataset_name = unwrap_batch(batch, device, only_input=True)
            img_name = img_name[0]
            dataset_name = dataset_name[0]
            self.task.infer(model=self.model, x=x, img_name=img_name, dataset=dataset_name, **kwargs)

    def train_and_infer(self, epochs=10, device='cuda'):

        self.train(epochs=epochs, device=device)
        self.infer(device=device)

    def config_for_training(self, callbacks, data_dir):

        callbacks = [MetricsCB(), SimpleProgressBar(), SaveModel(verbose=True)] + callbacks
        self.callbacks = CallbacksGroup(learner=self, callbacks=callbacks)
        self.data_dir = data_dir

    def set_training_phase(self, **kwargs):
        self.task.set_training_phase(**kwargs)

    def set_validation_phase(self, **kwargs):
        self.task.set_validation_phase(**kwargs)

    def set_inference_phase(self, **kwargs):
        self.task.set_inference_phase(**kwargs)

