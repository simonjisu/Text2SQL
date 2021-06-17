import torch
import torchmetrics

class Perplexity(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.add_state("pp", default=torch.FloatTensor([0]), dist_reduce_fx="sum")
        self.add_state("cnt", default=torch.FloatTensor([0]), dist_reduce_fx="sum")
        
    def update(self, pp: torch.Tensor):
        self.pp += pp
        self.cnt += 1

    def compute(self):
        return self.pp / self.cnt