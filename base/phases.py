import multiprocessing
from torch.utils.data import DataLoader
from base.metrics_holder import MetricsHolder

num_workers = multiprocessing.cpu_count() - 1


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

        if self.name is not 'inference':
            self.metrics_holder = MetricsHolder(learner=learner)


class BasicPhase(Phase):
    def __init__(self, learner, name, dataset, bs, shuffle):
        super().__init__(learner, name)
        self.bs = bs
        self.shuffle = shuffle
        self.dataset = dataset
        self.loader = DataLoader(self.dataset, batch_size=self.bs, shuffle=self.shuffle, num_workers=num_workers)

