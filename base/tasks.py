class Task:
    def __init__(self, learner):
        self.learner = learner

    def set_training_phase(self, *args):
        raise NotImplementedError

    def set_validation_phase(self, *args):
        raise NotImplementedError

    def set_inference_phase(self, *args):
        raise NotImplementedError


class AnnTask(Task):
    def __init__(self, learner):
        super().__init__(learner)


