from typing import Any, Mapping

import torch


class EMA:
    def __init__(
        self,
        src_model: torch.nn.Module,
        ema_model: torch.nn.Module,
        beta: float = 0.999,
        update_after_step: int = 100,
        update_every: int = 10,
    ):
        super(EMA, self).__init__()
        self.beta = beta
        self.update_after_step = update_after_step
        self.update_every = update_every

        self.src_model = src_model
        self.ema_model = ema_model

        self.param_key = [k for k, _ in src_model.named_parameters()]
        self.buffer_key = [k for k, _ in src_model.named_buffers()]

        self.step = 0
        self.init = False

    def copy_model(self):
        model_state_dict = self.src_model.state_dict()
        ema_state_dict = self.ema_model.state_dict()
        for key in self.param_key:
            ema_state_dict[key].copy_(model_state_dict[key].detach())
        for key in self.buffer_key:
            ema_state_dict[key].copy_(model_state_dict[key].detach())

    def lerp_model(self, weight: float):
        model_state_dict = self.src_model.state_dict()
        ema_state_dict = self.ema_model.state_dict()
        for key in self.param_key:
            ema_state_dict[key].copy_(
                weight * ema_state_dict[key]
                + (1.0 - weight) * model_state_dict[key].detach()
            )
        for key in self.buffer_key:
            ema_state_dict[key].copy_(model_state_dict[key].detach())

    @torch.no_grad()
    def update(self):
        current_step = self.step
        self.step += 1
        if (current_step % self.update_every) != 0:
            return
        if current_step < self.update_after_step:
            self.copy_model()
            return
        if not self.init:
            self.copy_model()
            self.init = True
        self.lerp_model(self.beta)

    def state_dict(self):
        return {
            "step": self.step,
            "init": self.init,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        self.step = state_dict["step"]
        self.init = state_dict["init"]
