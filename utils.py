from torch.utils.tensorboard import SummaryWriter


class MySummaryWriter(SummaryWriter):
    def __init__(self, step=0, steps_to_log=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_step = step
        self.steps_to_log = steps_to_log

    def update_global_step(self, global_step):
        self.global_step = global_step

    def check_steps(self):
        return self.global_step % self.steps_to_log == 0


if __name__ == '__main__':
    MySummaryWriter(0, 100)
