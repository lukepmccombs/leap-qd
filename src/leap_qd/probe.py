from leap_ec.ops import Operator
from leap_ec.util import get_step
from leap_ec.global_vars import context
import os


class CheckpointMetric(Operator):

    def __init__(self, metric, base_dir, format_str, context=context):
        super().__init__()
        self.metric = metric
        self.base_dir = base_dir
        self.format_str = format_str
        self.context = context

        os.makedirs(base_dir, exist_ok=True)

    def __call__(self, arg):
        step = get_step(self.context)
        step_dir = os.path.join(self.base_dir, self.format_str.format(step=step))
        self.metric.save(step_dir)
        return arg
