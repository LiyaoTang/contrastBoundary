import numpy as np

class StepScheduler(object):
    def __init__(self, name, base_value, decay_rate, decay_step, max_steps, clip_min=0):
        self.name = name
        self.clip_min = clip_min
        self.cur_step = 0
        self.values = [base_value * decay_rate ** (i // decay_step) for i in range(max_steps)]

    def reset(self):
        self.cur_step = 0

    def step(self):
        # cur_value = self.base_value * self.decay_rate ** (cur_step // decay_step)
        cur_value = max(self.values[self.cur_step], self.clip_min)
        self.cur_step += 1
        return cur_value

class LrScheduler(object):
    def __init__(self, config):
        self.config = config
        self.start_lr = float(config.learning_rate)
        self.clip_min = config.clip_min if config.clip_min else 0
        self.reset()

    def reset(self):
        self.cur_ep = 0
        self.cur_step = 0
        self.learning_rate = self.start_lr

    def _get_lr(self):
        cfg = self.config
        cur_ep = self.cur_ep
        cur_lr = self.learning_rate

        # normal decay
        if cfg.decay_step:
            times = self.cur_step // cfg.decay_step if isinstance(cfg.decay_step, int) else (np.array(cfg.decay_step) <= self.cur_step).sum()
        else:
            decay_epoch = cfg.decay_epoch if cfg.decay_epoch else 1  # decay per epoch by default
            times = self.cur_ep // decay_epoch if isinstance(decay_epoch, int) else (np.array(decay_epoch) <= self.cur_ep).sum()

        cum_decay = (cfg.decay_rate ** times) if type(cfg.decay_rate) in [int, float] else np.prod(cfg.decay_rate[:times])  # np.prod([]) = 1.0
        cur_lr = self.start_lr * cum_decay
        return cur_lr

    def to_list(self, max_epoch):
        return [self._get_lr(i, self.learning_rate) for i in range(max_epoch)]

    def step(self, epoch, step):
        self.cur_ep += epoch
        self.cur_step += step
        cur_lr = max(self._get_lr(), self.clip_min)
        self.learning_rate = cur_lr
        return cur_lr
