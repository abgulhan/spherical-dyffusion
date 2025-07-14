from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
import hydra # Add hydra import
from omegaconf import DictConfig # Add DictConfig import

from src.experiment_types._base_experiment import BaseExperiment
from src.utilities.utils import (
    rrearrange,
)
# Import your new loss config classes
import src.ace_inference.core.loss as loss_module  # Import the module to access the classes


class InterpolationExperiment(BaseExperiment):
    r"""Base class for all interpolation experiments."""

    def original__init__(self, stack_window_to_channel_dim: bool = True, inference_val_every_n_epochs=None, **kwargs):
        super().__init__(**kwargs)
        if inference_val_every_n_epochs is not None:
            self.log_text.warning("``inference_val_every_n_epochs`` will be ignored for interpolation experiments.")
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.hidden_dims
        self.save_hyperparameters(ignore=["model"])
        assert self.horizon >= 2, "horizon must be >=2 for interpolation experiments"
        if hasattr(self.model, "set_min_max_time"):
            self.model.set_min_max_time(min_time=self.horizon_range[0], max_time=self.horizon_range[-1])


    def __init__(self, stack_window_to_channel_dim: bool = True, inference_val_every_n_epochs=None, 
                 # Assuming the full config 'cfg' or a specific 'loss_config' is available
                 # Let's assume 'cfg' (the whole DictConfig) is passed or accessible via self.hparams
                 **kwargs):
        super().__init__(**kwargs)
        if inference_val_every_n_epochs is not None:
            self.log_text.warning("``inference_val_every_n_epochs`` will be ignored for interpolation experiments.")
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.hidden_dims
        self.save_hyperparameters(ignore=["model"])
        assert self.horizon >= 2, "horizon must be >=2 for interpolation experiments"
        if hasattr(self.model, "set_min_max_time"):
            self.model.set_min_max_time(min_time=self.horizon_range[0], max_time=self.horizon_range[-1])

        # --- Instantiate new loss calculators ---
        # Access the main configuration (e.g., self.hparams if populated by Pytorch Lightning, or if cfg is passed)
        # For this example, let's assume self.hparams contains the full config or relevant loss part
        cfg_for_loss = self.hparams # Or directly use 'cfg' if it's an argument to __init__

        self.total_volume_loss_calculator = None
        self.total_volume_loss_weight = 0.0
        if cfg_for_loss and "loss" in cfg_for_loss and "total_volume_loss" in cfg_for_loss.loss:
            # Ensure cfg_for_loss.loss.total_volume_loss is the DictConfig node for TotalVolumeLossConfig
            vol_config_hydra = cfg_for_loss.loss.total_volume_loss
            if isinstance(vol_config_hydra, DictConfig): # Check if it's a DictConfig node
                 vol_config_obj = hydra.utils.instantiate(vol_config_hydra)
                 if isinstance(vol_config_obj, loss_module.TotalVolumeLossConfig):
                    self.total_volume_loss_calculator = vol_config_obj.build()
                    self.total_volume_loss_weight = vol_config_obj.weight
                 else:
                    self.log_text.warning("Failed to instantiate TotalVolumeLossConfig correctly.")
            else:
                self.log_text.warning("total_volume_loss config is not a DictConfig node.")


        self.total_salt_loss_calculator = None
        self.total_salt_loss_weight = 0.0
        if cfg_for_loss and "loss" in cfg_for_loss and "total_salt_loss" in cfg_for_loss.loss:
            salt_config_hydra = cfg_for_loss.loss.total_salt_loss
            if isinstance(salt_config_hydra, DictConfig):
                salt_config_obj = hydra.utils.instantiate(salt_config_hydra)
                if isinstance(salt_config_obj, loss_module.TotalSaltLossConfig):
                    self.total_salt_loss_calculator = salt_config_obj.build()
                    self.total_salt_loss_weight = salt_config_obj.weight
                else:
                    self.log_text.warning("Failed to instantiate TotalSaltLossConfig correctly.")
            else:
                self.log_text.warning("total_salt_loss config is not a DictConfig node.")

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
            inputs = rrearrange(inputs, "b window c lat lon -> b (window c) lat lon")
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
        """
        past_steps = dynamics[:, : self.window, ...]  # (b, window, c, lat, lon) at time 0
        last_step = dynamics[:, -1:, ...]  # (b, c, lat, lon) at time t=window+horizon
        past_steps = self.postprocess_inputs(past_steps)
        last_step = self.postprocess_inputs(last_step)
        inputs = torch.cat([past_steps, last_step], dim=1)  # (b, window*c + c, lat, lon)
        return inputs

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
        targets_tensor = dynamics[torch.arange(b), target_time, ...]  # (b, c, h, w)
        targets_for_main_loss = self.pack_data(targets_tensor, input_or_output="output")
        # We use timesteps  w-l+1, ..., w-1, w+h to predict timesteps w, ..., w+h-1
        # so t=0 corresponds to interpolating w, t=1 to w+1, ..., t=h-1 to w+h-1

        # --- Main model loss ---
        # The model's forward pass for interpolation might be called inside model.get_loss
        # or you might need to call self.model(inputs, time=t, ...) to get predictions first.
        # For this example, assume model.get_loss handles the prediction.
        main_loss = self.model.get_loss(
            inputs=inputs,
            targets=targets_for_main_loss, # Targets for the main pixel/field-wise loss
            condition=self.get_dynamical_condition(batch.get("dynamical_condition"), target_time=target_time),
            time=t,
            **{k: v for k, v in batch.items() if k != "dynamics"},
        )  # function of BaseModel or BaseDiffusion classes
        
        current_split = "train" if self.training else "val" # Or determine split differently
        self.log(f"{current_split}/loss_main", main_loss.detach(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        # --- Prepare data for custom losses ---
        # The custom losses expect a dictionary of predictions and targets.
        # You need to get the model's prediction output.
        # This might involve a separate forward call if model.get_loss doesn't return it,
        # or if model.get_loss is stateful (less common for the loss calculation itself).
        
        # Conceptual: Get predictions for the custom losses.
        # This depends on how your model's forward pass is structured for interpolation.
        # Let's assume you need to call the model to get a prediction dictionary.
        # This is a simplified placeholder. You'll need to adapt this to how your
        # `self.model.predict_step` or a similar method works to get the full output dictionary.
        
        # Placeholder: In a real scenario, you'd get `predicted_data_map` from your model's output
        # for the given `inputs` and `time`.
        # For example, if your model's forward pass or a predict method returns a dict:
        # with torch.no_grad(): # If only for loss calculation and not for main_loss's graph
        #    predicted_data_map = self.model(inputs, time=t, condition=..., **kwargs_for_model) 
        # This is highly dependent on your `self.model`'s API.
        # For now, we'll assume `batch` contains target data with the right keys,
        # and we need a placeholder for `predicted_data_map`.
        
        # Create a placeholder `predicted_data_map`. You MUST replace this with actual model predictions
        # structured as a dictionary with keys like 'timeDaily_avg_layerThickness'.
        predicted_data_map = {}
        if self.total_volume_loss_calculator or self.total_salt_loss_calculator:
            # This is a HACK. You need to get actual predictions from your model.
            # For example, if your model's forward pass returns a dict:
            #   model_output_dict = self.model(inputs, time=t, condition=self.get_dynamical_condition(batch.get(\"dynamical_condition\"), target_time=target_time), **other_kwargs)
            #   predicted_data_map = model_output_dict # if it has the right keys
            # Or you might need to reconstruct it.
            # For demonstration, let's assume the 'targets_tensor' shape can be used for a dummy prediction.
            dummy_pred_tensor = targets_tensor.detach().clone() # Replace with actual model output for these variables
            predicted_data_map = {
                'timeDaily_avg_layerThickness': dummy_pred_tensor, # Replace with actual prediction
                'timeDaily_avg_activeTracers_salinity': dummy_pred_tensor # Replace with actual prediction
                # Add other variables your model predicts if they are used by these losses or others
            }

        # The `batch` itself should serve as the `target_data_map` if it contains the raw variable names.
        # Ensure `batch` has keys like 'timeDaily_avg_layerThickness', 'timeDaily_avg_activeTracers_salinity'.
        # If `batch` is structured differently (e.g., `batch['dynamics']` holds all time steps),
        # you need to select the target time step and potentially rename/restructure.
        # For this example, we assume `batch` can be directly used or easily adapted.
        # Let's refine target_data_map to be more explicit for the specific time step
        target_data_map = {}
        for key in batch.keys():
            if key not in ["dynamics", "dynamical_condition"] and isinstance(batch[key], torch.Tensor) and batch[key].ndim > 1:
                 # This is a heuristic. You need to ensure these are the correct target variables
                 # for the current time step `target_time`.
                 # If batch[key] is like (B, T, C, H, W), you need to select target_time.
                 # If batch[key] is already (B, C, H, W) for the target time, it's fine.
                 # This part is CRUCIAL and depends on your dataloader's output structure.
                 # For now, let's assume batch contains keys that directly map to target variables
                 # for the current timestep, or that `targets_tensor` can be used if it's a single variable.
                 # A more robust way:
                 # target_data_map[key] = batch[key][torch.arange(b), target_time, ...] if batch[key].shape[1] == total_time_steps
                 # For simplicity, if your batch has 'timeDaily_avg_layerThickness' directly for the target step:
                 if key in predicted_data_map: # Only consider keys that model predicts for these losses
                    target_data_map[key] = batch[key] # This assumes batch[key] is the target for the current step

        # A more robust way to get target_data_map for the specific time step:
        # This assumes your datamodule provides data in a way that `batch` contains
        # all variables for all timesteps, and you select the `target_time`.
        # Or, if `dynamics` is the primary multi-channel, multi-timestep tensor:
        # target_data_map_from_dynamics = self.datamodule.tensor_to_dict(dynamics[torch.arange(b), target_time, ...])
        # This requires a `tensor_to_dict` method in your datamodule.
        # For now, we'll use a simplified approach assuming `batch` contains the necessary keys
        # or that `targets_tensor` (if it's the only variable for main loss) can be adapted.
        
        # Let's assume `batch` itself can be used as `target_data_map` if it has the right keys,
        # or you construct it appropriately.
        # For the custom losses, they expect a dictionary.
        # `targets_tensor` is (b, c, h, w). If your variables are channels in this, you need to map them.
        # If your `batch` is already a dict like `{'timeDaily_avg_layerThickness': tensor, ...}` for the target time, use it.
        
        # Refined target_data_map (assuming batch contains the necessary keys for the target time step)
        # This is still a simplification. You need to ensure this `target_data_map`
        # correctly represents the ground truth for the variables your custom losses need,
        # at the specific `target_time`.
        target_data_map_for_custom_loss = {}
        if 'timeDaily_avg_layerThickness' in batch: # Check if keys exist directly in batch for the target time
            target_data_map_for_custom_loss['timeDaily_avg_layerThickness'] = batch['timeDaily_avg_layerThickness']
        if 'timeDaily_avg_activeTracers_salinity' in batch:
            target_data_map_for_custom_loss['timeDaily_avg_activeTracers_salinity'] = batch['timeDaily_avg_activeTracers_salinity']
        # If not, you need to extract them from `dynamics` at `target_time` and map channel indices to names.


        # --- Calculate and add custom losses ---
        if self.total_volume_loss_calculator and self.total_volume_loss_weight > 0:
            if predicted_data_map and target_data_map_for_custom_loss.get('timeDaily_avg_layerThickness') is not None:
                volume_loss = self.total_volume_loss_calculator(predicted_data_map, target_data_map_for_custom_loss)
                self.log(f"{current_split}/loss_total_volume", volume_loss.detach(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
                main_loss += self.total_volume_loss_weight * volume_loss
            else:
                self.log_text.warning(f"Skipping total_volume_loss due to missing data in predicted_data_map or target_data_map for {current_split}.")
        
        if self.total_salt_loss_calculator and self.total_salt_loss_weight > 0:
            if predicted_data_map and target_data_map_for_custom_loss.get('timeDaily_avg_layerThickness') is not None and target_data_map_for_custom_loss.get('timeDaily_avg_activeTracers_salinity') is not None:
                salt_loss = self.total_salt_loss_calculator(predicted_data_map, target_data_map_for_custom_loss)
                self.log(f"{current_split}/loss_total_salt", salt_loss.detach(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
                main_loss += self.total_salt_loss_weight * salt_loss
            else:
                self.log_text.warning(f"Skipping total_salt_loss due to missing data in predicted_data_map or target_data_map for {current_split}.")

        self.log(f"{current_split}/loss_total", main_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return main_loss


    # --------------------------------- Training
    def get_loss_original(self, batch: Any, optimizer_idx: int = 0) -> Tensor:
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