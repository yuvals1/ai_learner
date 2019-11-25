from .utils import to_snake_case, get_class_name
import torch
import os
from os.path import join
import shutil
from collections import OrderedDict
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Callback:
    """
    The base class inherited by callbacks.

    Provides a lot of hooks invoked on various stages of the training loop
    execution. The signature of functions is as broad as possible to allow
    flexibility and customization in descendant classes.
    """

    def training_started(self, **kwargs): pass

    def training_ended(self, **kwargs): pass

    def epoch_started(self, **kwargs): pass

    def phase_started(self, **kwargs): pass

    def phase_ended(self, **kwargs): pass

    def epoch_ended(self, **kwargs): pass

    def batch_started(self, **kwargs): pass

    def batch_ended(self, **kwargs): pass

    def before_forward_pass(self, **kwargs): pass

    def after_forward_pass(self, **kwargs): pass

    def before_backward_pass(self, **kwargs): pass

    def after_backward_pass(self, **kwargs): pass


class CallbacksGroup(Callback):
    """
    Groups together several callbacks and delegates training loop
    notifications to the encapsulated objects.
    """

    def __init__(self, learner, callbacks):
        self.learner = learner
        self.callbacks = callbacks
        self.named_callbacks = {to_snake_case(get_class_name(cb)): cb for cb in self.callbacks}

    def __getitem__(self, item):
        item = to_snake_case(item)
        if item in self.named_callbacks:
            return self.named_callbacks[item]
        raise KeyError(f'callback name is not found: {item}')

    def training_started(self, **kwargs):
        self.invoke('training_started', **kwargs)

    def training_ended(self, **kwargs):
        self.invoke('training_ended', **kwargs)

    def epoch_started(self, **kwargs):
        self.invoke('epoch_started', **kwargs)

    def phase_started(self, **kwargs):
        self.invoke('phase_started', **kwargs)

    def phase_ended(self, **kwargs):
        self.invoke('phase_ended', **kwargs)

    def epoch_ended(self, **kwargs):
        self.invoke('epoch_ended', **kwargs)

    def batch_started(self, **kwargs):
        self.invoke('batch_started', **kwargs)

    def batch_ended(self, **kwargs):
        self.invoke('batch_ended', **kwargs)

    def before_forward_pass(self, **kwargs):
        self.invoke('before_forward_pass', **kwargs)

    def after_forward_pass(self, **kwargs):
        self.invoke('after_forward_pass', **kwargs)

    def before_backward_pass(self, **kwargs):
        self.invoke('before_backward_pass', **kwargs)

    def after_backward_pass(self, **kwargs):
        self.invoke('after_backward_pass', **kwargs)

    def invoke(self, method, **kwargs):
        for cb in self.callbacks:
            getattr(cb, method)(learner=self.learner, **kwargs)


class MetricsCB(Callback):

    def batch_ended(self, learner, phase, pr, gt, **kwargs):
        phase.metrics_holder.batch_ended_update(pr, gt)

    def epoch_ended(self, learner, **kwargs):
        learner.training_phase.metrics_holder.epoch_ended_update()
        learner.validation_phase.metrics_holder.epoch_ended_update()


class SimpleProgressBar(Callback):
    def __init__(self):
        self.start_time = None

    def epoch_started(self, learner, **kwargs):
        self.start_time = time.time()

    def epoch_ended(self, learner, epoch, **kwargs):
        epoch_time = (time.time() - self.start_time)
        string = f'epoch {epoch}, time:{epoch_time:.2f} seconds'
        for phase in (learner.training_phase, learner.validation_phase):
            string += '\n%10s - ' % phase.name
            for metric, history in phase.metrics_holder.epochs_history.items():
                string += f'{metric}:{history[-1]:.3f}, '

        print(string)


class SaveModel(Callback):
    def __init__(self, verbose=True):
        self.verbose = verbose

    def epoch_ended(self, learner, **kwargs):

        best_score_str = str(learner.validation_phase.metrics_holder.main_metric_best_score)[:6]
        curr_score_str = str(learner.validation_phase.metrics_holder.main_metric_curr_score)[:4]

        model_dir_new_name = '_'.join([learner.model_name, best_score_str])
        model_dir_new_path = join(learner.model_dir_root, model_dir_new_name)
        model_dir_old_path = learner.model_dir_path

        if not os.path.exists(model_dir_new_path):
            shutil.move(model_dir_old_path, model_dir_new_path)
        learner.model_dir_path = model_dir_new_path
        learner.model_path = join(learner.model_dir_path, 'state_dict' + '.pt')

        torch.save(learner.model.state_dict(), join(learner.model_dir_path, 'state_dict_last' + '.pt'))
        if learner.validation_phase.metrics_holder.main_metric_improved:
            torch.save(learner.model.state_dict(), learner.model_path)
            if self.verbose:
                print(f'main metric improved  -->  Saving model ...')

        learner.training_phase.metrics_holder.save_history(filename='training_history.csv',
                                                           dir_path=learner.model_dir_path)
        learner.validation_phase.metrics_holder.save_history(filename='validation_history.csv',
                                                             dir_path=learner.model_dir_path)


class ReduceLRCB(Callback):
    def __init__(self, optimizer, factor=0.75, patience=1, verbose=True):
        self.scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience=patience, verbose=verbose)

    def epoch_ended(self, learner, **kwargs):
        learner_loss_name = learner.loss.__name__
        train_loss = learner.training_phase.metrics_holder.epochs_history[learner_loss_name][-1]
        self.scheduler.step(train_loss)


class EarlyStopping(Callback):
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.stop = False

    def epoch_ended(self, learner, **kwargs):
        if not learner.validation_phase.metrics_holder.main_metric_improved:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                learner.finish_training = True
                print(f'early stopping..')
        else:
            self.counter = 0


# out of use

class ProgressBar(Callback):

    def epoch_started(self, phases, epoch, **kwargs):
        bars = OrderedDict()
        for phase in phases:
            if phase.name == 'train':
                bars[phase.name] = tqdm(total=len(phase.loader), desc=str(epoch),
                                        position=None, leave=True)
        self.bars = bars

    def batch_ended(self, phase, **kwargs):
        if phase.name == 'train':
            string = '|'
            for metric, val in phase.metrics_epoch_history.items():
                string += metric + ':' + f'{val[-1]:.2f}' + ','

            bar = self.bars[phase.name]
            bar.set_postfix_str(string)
            bar.update(1)
            bar.refresh()

    def epoch_ended(self, phases, **kwargs):
        string = ''
        for phase in phases:
            string += '|' + phase.name + '- '
            for metric, val in phase.metrics_train_history.items():
                string += metric + ':' + f'{val[-1]:.2f}' + ','

        for phase in phases:
            if phase.name == 'train':
                bar = self.bars[phase.name]
                bar.set_postfix_str(string)
                bar.n = len(phase.loader)
                bar.refresh()
