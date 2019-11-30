import multiprocessing
from torch.utils.data import DataLoader
from ai_learner.ann.metrics_holder import MetricsHolder


class Phase:
    """
    Model training loop phase.

    Each model's training loop iteration could be separated into (at least) two
    phases: training and validation. The instances of this class track
    metrics and counters, related to the specific phase, and keep the reference
    to subset of data, used during phase.
    """

    def __init__(self, learner, name):
        self.learner = learner
        self.name = name
        self.metrics_holder = NotImplemented
        self.loader = NotImplemented


class BasicPhase(Phase):
    def __init__(self, learner, name, loader):
        super().__init__(learner, name)
        self.loader = loader
        self.metrics_holder = MetricsHolder(learner=learner)





