### https://github.com/HazyResearch/transformers/blob/master/src/callbacks/speed_monitor.py
import os
import sys
# Adapted from https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/gpu_stats_monitor.html#GPUStatsMonitor
# We only need the speed monitoring, not the GPU monitoring
import time
from typing import Any

from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.utilities.types import STEP_OUTPUT

def filter(metrics):
    filtered = {}
    for k,v in metrics.items():
        if "val" in k or "test" in k:
            filtered[k]=float(v.item())
    return filtered

def is_better(a,b,metric):
    if "acc" in metric:
        return a>b
    else:
        return a<b



def get_important_metric(metrics):
    important_list = ["test/acc","test/accuracy","test/ppl","test/mse","test/loss","val/acc","val/accuracy","val/ppl","val/mse","val/loss"]
    for m in important_list:
        if m in metrics.keys():
            v = metrics[m]
            if "acc" in m:
                v = 100*v

            return f"{m}: {v:0.3f}"
    return "N/A"


class Score(Callback):
    """Monitor the speed of each step and each epoch.
    """
    def __init__(self,  enable
    ):
        super().__init__()
        self._enable = enable
        self._best_metrics =None


    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule,) -> None:
        if not self._enable:
            return

        if trainer.sanity_checking:
            return

        metrics = filter(trainer.callback_metrics)
        selected_metric = None
        look_at_list = ["val/acc","val/accuracy","val/ppl", "val/mse", "val/loss"]
        for m in look_at_list:
            if m in metrics.keys():
                selected_metric = m
                break
        if selected_metric is None:
            print(f"Could not find any monitorable metric! ({list(metrics.keys())})\n")
            return
        if self._best_metrics is None:
            self._best_metrics = metrics
            print(f"Init metric: {self._best_metrics}\n")
        else:
            if is_better(metrics[selected_metric],self._best_metrics[selected_metric],selected_metric):
                self._best_metrics = metrics
                print(f"New best metric: {self._best_metrics}\n")

    # @rank_zero_only
    # def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule,) -> None:
    #     if not self._enable:
    #         return
    #     metrics = filter(trainer.callback_metrics)
    #     print("test metric: ",metrics)

    @rank_zero_only
    def on_fit_end(self,trainer, pl_module):
        if not self._enable:
            return
        if self._best_metrics is None:
            print("No best metrics")
            return
        # dirname = os.path.dirname(os.path.abspath(__file__))
        dirname = os.path.abspath(__file__).replace("src/callbacks/score.py","")
        # abspath = os.path.abspath("global_summary.txt")
        with open(f"{dirname}global_summary.txt", "a") as f:
            m = get_important_metric(self._best_metrics)
            cmd_line = " ".join(sys.argv[1:])
            f.write(f"{m} python3 train.py {cmd_line} # {self._best_metrics}")
            f.write("\n")

        print(f"{dirname}global_summary.txt updated")