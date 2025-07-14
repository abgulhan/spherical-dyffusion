from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from src.experiment_types._base_experiment import BaseExperiment
from src.utilities.utils import (
    rrearrange, to_tensordict
)


class InterpolationExperiment(BaseExperiment):
    r"""Base class for all interpolation experiments."""

    def __init__(self, stack_window_to_channel_dim: bool = True, inference_val_every_n_epochs=None, **kwargs):
        super().__init__(**kwargs)
        if inference_val_every_n_epochs is not None:
            self.log_text.warning("``inference_val_every_n_epochs`` will be ignored for interpolation experiments.")
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.hidden_dims
        #self.save_hyperparameters(ignore=["model", "seed"])
        self.save_hyperparameters(ignore=["model", "seed"])
        assert self.horizon >= 2, "horizon must be >=2 for interpolation experiments"
        if hasattr(self.model, "set_min_max_time"):
            self.model.set_min_max_time(min_time=self.horizon_range[0], max_time=self.horizon_range[-1])

    @property
    def horizon_range(self) -> List[int]:
        # h = horizon
        # We use timesteps  w-l+1, ..., w-1, w+h to predict timesteps w, ..., w+h-1
        # interpolate between step t=0 and t=horizon
        return list(np.arange(1, self.horizon))

    @property
    def true_horizon(self) -> int:
        return self.horizon

    @property
    def horizon_name(self) -> str:
        s = f"{self.true_horizon}h"
        return s

    @property
    def short_description(self) -> str:
        name = super().short_description
        name += f" (h={self.horizon_name})"
        return name

    @property
    def WANDB_LAST_SEP(self) -> str:
        return "/"  # /ipol/"

    @property
    def num_conditional_channels(self) -> int:
        """The number of channels that are used for conditioning as auxiliary inputs."""
        nc = super().num_conditional_channels
        factor = self.window + 0 + 0  # num inputs before target + num targets + num inputs after target
        return nc * factor

    def actual_num_input_channels(self, num_input_channels: int) -> int:
        if self.hparams.stack_window_to_channel_dim:
            return num_input_channels * self.window + num_input_channels
        return 2 * num_input_channels  # inputs and targets are concatenated

    def postprocess_inputs(self, inputs):
        inputs = self.pack_data(inputs, input_or_output="input")
        if self.hparams.stack_window_to_channel_dim:  # and inputs.shape[1] == self.window:
            # Handle both 5D (2D spatial) and 6D (3D spatial) inputs
            if inputs.ndim == 5:
                # 2D spatial data: (batch, window, channels, lat, lon)
                inputs = rrearrange(inputs, "b window c lat lon -> b (window c) lat lon")
            elif inputs.ndim == 6:
                # 3D spatial data: (batch, window, channels, depth, lat, lon)
                inputs = rrearrange(inputs, "b window c depth lat lon -> b (window c) depth lat lon")
            else:
                raise ValueError(f"Expected 5D or 6D input tensor, got {inputs.ndim}D with shape {inputs.shape}")
        return inputs

    @torch.inference_mode()
    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        dataloader_idx: int = None,
        aggregators: Dict[str, Callable] = None,
        return_only_preds_and_targets: bool = False,
    ):
        no_aggregators = aggregators is None or len(aggregators.keys()) == 0
        main_data_raw = batch.pop("raw_dynamics")
        dynamics = batch["dynamics"]  # dynamics is a (b, t, c, h, w) tensor

        return_dict = dict()
        extra_kwargs = {}
        dynamical_cond = batch.pop("dynamical_condition", None)
        if dynamical_cond is not None:
            assert "condition" not in batch, "condition should not be in batch if dynamical_condition is present"
        inputs = self.get_evaluation_inputs(dynamics, split=split)
        for k, v in batch.items():
            if k != "dynamics":
                extra_kwargs[k] = self.get_ensemble_inputs(v, split=split, add_noise=False)

        for t_step in self.horizon_range:
            # dynamics[, self.window] is already the first target frame (t_step=1)
            target_time = self.window + t_step - 1
            time = torch.full((inputs.shape[0],), t_step, device=self.device, dtype=torch.long)
            if dynamical_cond is not None:
                extra_kwargs["condition"] = self.get_ensemble_inputs(
                    self.get_dynamical_condition(dynamical_cond, target_time), split=split, add_noise=False
                )
            results = self.predict(inputs, time=time, **extra_kwargs)
            preds = results["preds"]

            targets_tensor_t = main_data_raw[:, target_time, ...]
            targets = self.get_target_variants(targets_tensor_t, is_normalized=False)
            results["targets"] = targets
            results = {f"t{t_step}_{k}": v for k, v in results.items()}

            if return_only_preds_and_targets:
                return_dict[f"t{t_step}_preds"] = preds
                return_dict[f"t{t_step}_targets"] = targets
            else:
                return_dict = {**return_dict, **results}

            if no_aggregators:
                continue

            PREDS_NORMED_K = f"t{t_step}_preds_normed"
            PREDS_RAW_K = f"t{t_step}_preds"
            targets_normed = targets["targets_normed"] if targets is not None else None
            targets_raw = targets["targets"] if targets is not None else None
            aggregators[f"t{t_step}"].record_batch(
                target_data=targets_raw,
                gen_data=results[PREDS_RAW_K],
                target_data_norm=targets_normed,
                gen_data_norm=results[PREDS_NORMED_K],
            )

        return return_dict

    def get_dynamical_condition(
        self, dynamical_condition: Optional[Tensor], target_time: Union[int, Tensor]
    ) -> Tensor:
        if dynamical_condition is not None:
            if isinstance(target_time, int):
                return dynamical_condition[:, target_time, ...]
            else:
                if not torch.is_tensor(target_time):
                    target_time = torch.tensor(target_time, device=dynamical_condition.device)
                return dynamical_condition[torch.arange(dynamical_condition.shape[0]), target_time.long(), ...]        
        return None

    def get_inputs_from_dynamics(self, dynamics: Tensor, **kwargs) -> Tensor:
        """Get the inputs from the dynamics tensor.
        Since we are doing interpolation, this consists of the first window frames plus the last frame.
        
        dynamics must be a TensorDict
        """
        past_steps = dynamics[:, : self.window, ...]  # (b, window, c, lat, lon) at time 0
        last_step = dynamics[:, -1:, ...]  # (b, c, lat, lon) at time t=window+horizon
        past_steps = self.postprocess_inputs(past_steps)
        last_step = self.postprocess_inputs(last_step)
        inputs = torch.cat([past_steps, last_step], dim=1)  # (b, window*c + c, lat, lon)
        return inputs

    def interpolate(self, dynamics1: Dict[str, Tensor], dynamics2: Dict[str, Tensor], target_times: List[int], batch: bool = False) -> List[Tensor]:
        """
        Interpolate between two single timesteps using the SOMA interpolator model.
        
        Args:
            dynamics1: Tensor of shape (1, channels, z, y, x) for the first timestep.
            dynamics2: Tensor of shape (1, channels, z, y, x) for the second timestep.
            target_times: List of target times to interpolate to.
            batch: If True, treat dynamics1 and dynamics2 as batches (shape: (b, channels, z, y, x)).
        
        Returns:
            A list of interpolated tensors for each target time.
        """
        
        if isinstance(dynamics1, dict) and ("condition" in dynamics1.keys() or "dynamical_condition" in dynamics1.keys()):
            raise ValueError("Conditioning is currently not supported for interpolation. Remove 'condition' or 'dynamical_condition' from dynamics1 and dynamics2.")
        
        if "dynamics" in dynamics1:
            dynamics1 = dynamics1["dynamics"]
        if "dynamics" in dynamics2:
            dynamics2 = dynamics2["dynamics"]
        
        
        start_tensor = to_tensordict(dynamics1)
        end_tensor = to_tensordict(dynamics2)

        if not batch:
            # If not batch, we need to add a batch dimension
            start_tensor = start_tensor.unsqueeze(0)
            end_tensor = end_tensor.unsqueeze(0)

        start_tensor = self.postprocess_inputs(start_tensor)
        end_tensor = self.postprocess_inputs(end_tensor)
        inputs = torch.cat([start_tensor, end_tensor], dim=1)  # (b, window*c + c, lat, lon)
        # Move inputs to the same device as the model
        inputs = inputs.to(self.device)


        interpolated_tensors = []
        for target_time in target_times:
            # Construct time tensor
            time_tensor = torch.full((inputs.shape[0],), target_time,
                                   device=self.device, dtype=torch.long)
            interpolated_tensor = self.predict(inputs, time=time_tensor)
            #keys are: preds and preds_normed
            
            # save gpu memory
            for k in interpolated_tensor.keys():
                interpolated_tensor[k] = interpolated_tensor[k].detach().cpu()
            interpolated_tensors.append(interpolated_tensor)

        return interpolated_tensors

    def get_evaluation_inputs(self, dynamics: Tensor, split: str, **kwargs) -> Tensor:
        inputs = self.get_inputs_from_dynamics(dynamics)
        inputs = self.get_ensemble_inputs(inputs, split)
        return inputs

    # --------------------------------- Training
    def get_loss(self, batch: Any, optimizer_idx: int = 0) -> Tensor:
        r"""Compute the loss for the given batch."""
        dynamics = batch["dynamics"]  # dynamics is a (b, t, c, h, w) tensor
        inputs = self.get_inputs_from_dynamics(dynamics)  # (b, c, h, w) at time 0
        b = dynamics.shape[0]

        possible_times = torch.tensor(self.horizon_range, device=self.device, dtype=torch.long)  # (h,)
        # take random choice of time
        t = possible_times[torch.randint(len(possible_times), (b,), device=self.device, dtype=torch.long)]  # (b,)
        target_time = self.window + t - 1
        # t = torch.randint(start_t, max_t, (b,), device=self.device, dtype=torch.long)  # (b,)
        targets = dynamics[torch.arange(b), target_time, ...]  # (b, c, h, w)
        targets = self.pack_data(targets, input_or_output="output")
        # We use timesteps  w-l+1, ..., w-1, w+h to predict timesteps w, ..., w+h-1
        # so t=0 corresponds to interpolating w, t=1 to w+1, ..., t=h-1 to w+h-1

        loss = self.model.get_loss(
            inputs=inputs,
            targets=targets,
            condition=self.get_dynamical_condition(batch.pop("dynamical_condition", None), target_time=target_time),
            time=t,
            **{k: v for k, v in batch.items() if k != "dynamics"},
        )  # function of BaseModel or BaseDiffusion classes
        return loss
