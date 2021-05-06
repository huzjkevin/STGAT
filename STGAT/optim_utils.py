from torch import optim


class Optim:
    def __init__(self, model, cfg):
        self.optimizer = optim.Adam(model.parameters(), amsgrad=True)
        self.lr_scheduler = ExpDecayScheduler(**cfg)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def set_lr(self, epoch):
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr_scheduler(epoch)

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


class ExpDecayScheduler:
    """
    Return `v0` until `e` reaches `e0`, then exponentially decay
    to `v1` when `e` reaches `e1` and return `v1` thereafter, until
    reaching `eNone`, after which it returns `None`.
    """

    def __init__(self, epoch0, lr0, epoch1, lr1):
        self.epoch0 = epoch0
        self.epoch1 = epoch1
        self.lr0 = lr0
        self.lr1 = lr1

    def __call__(self, epoch):
        if epoch < self.epoch0:
            return self.lr0
        elif epoch > self.epoch1:
            return self.lr1
        else:
            return self.lr0 * (self.lr1 / self.lr0) ** (
                (epoch - self.epoch0) / (self.epoch1 - self.epoch0)
            )
