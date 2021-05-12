from torch import optim


class Optim:
    def __init__(self, model, cfg=None, default_lr=1e-3):
        self.optimizer = optim.Adam(model.parameters(), amsgrad=True)
        if cfg is not None:
            self.lr_scheduler = ExpDecayScheduler(**cfg)
            self.scheduler_name = "exp"
        else:
            self.lr_scheduler = ConstantScheduler(default_lr)
            self.scheduler_name = "const"

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def set_lr(self, epoch):
        if self.scheduler_name == "exp":
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr_scheduler(epoch)
        else:
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr_scheduler()

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def set_scheduler(self, scheduler_name, params=None):
        if scheduler_name == "exp":
            assert params is not None, "Need configuration for exponential scheduler!"
            self.lr_scheduler = ExpDecayScheduler(**params)
            self.scheduler_name = "exp"
        elif scheduler_name == "const":
            assert params is not None, "invalid lr for constant scheduler"
            self.lr_scheduler = ConstantScheduler(params)
            self.scheduler_name = "const"
        else:
            raise NotImplementedError


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

class ConstantScheduler:
    def __init__(self, lr):
        self.lr = lr

    def __call__(self):
        return self.lr
