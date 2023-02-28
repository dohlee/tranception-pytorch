from torch.optim.lr_scheduler import _LRScheduler

class LinearAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, num_annealing_steps, num_total_steps):
        self.num_annealing_steps = num_annealing_steps
        self.num_total_steps = num_total_steps

        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count <= self.num_annealing_steps:
            multiplier = self._step_count / self.num_annealing_steps
            return [base_lr * multiplier for base_lr in self.base_lrs]
        else:
            multiplier = (self.num_total_steps - self._step_count) / (self.num_total_steps - self.num_annealing_steps)
            return [base_lr * multiplier for base_lr in self.base_lrs]
