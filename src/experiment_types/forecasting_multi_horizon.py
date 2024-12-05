from __future__ import annotations

import inspect
import math
from abc import ABC
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
import torch
from tensordict import TensorDict
from torch import Tensor
from tqdm.auto import tqdm

from src.diffusion.dyffusion import BaseDYffusion
from src.experiment_types._base_experiment import BaseExperiment
from src.utilities.utils import (
    multiply_by_scalar,
    rrearrange,
    split3d_and_merge_variables,
    torch_select,
    torch_to_numpy,
)


class AbstractMultiHorizonForecastingExperiment(BaseExperiment, ABC):
    PASS_METADATA_TO_MODEL = True

    def __init__(
        self,
        autoregressive_steps: int = 0,
        prediction_timesteps: Optional[Sequence[float]] = None,
        empty_cache_at_autoregressive_step: bool = False,
        inference_val_every_n_epochs: int = 1,
        return_outputs_at_evaluation: str | bool = "auto",
        stack_window_to_channel_dim=True,
        **kwargs,
    ):
        assert autoregressive_steps >= 0, f"Autoregressive steps must be >= 0, but is {autoregressive_steps}"
        assert autoregressive_steps == 0, "Autoregressive steps are not yet supported for this experiment type."
        self.stack_window_to_channel_dim = stack_window_to_channel_dim
        super().__init__(**kwargs)
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.autoregressive_steps
        self.save_hyperparameters(ignore=["model"])
        self.USE_TIME_AS_EXTRA_INPUT = False
        self._prediction_timesteps = prediction_timesteps
        self.hparams.pop("prediction_timesteps", None)
        if prediction_timesteps is not None:
            self.log_text.info(f"Using prediction timesteps {prediction_timesteps}")

        val_time_range = self.valid_time_range_for_backbone_model
        if hasattr(self.model, "set_min_max_time"):
            self.model.set_min_max_time(min_time=val_time_range[0], max_time=val_time_range[-1])
        elif hasattr(self.model, "model") and hasattr(self.model.model, "set_min_max_time"):
            # For diffusion models
            self.model.model.set_min_max_time(min_time=val_time_range[0], max_time=val_time_range[-1])

    @property
    def horizon_range(self) -> List[int]:
        return list(np.arange(1, self.horizon + 1))

    @property
    def valid_time_range_for_backbone_model(self) -> List[int]:
        return self.horizon_range

    @property
    def true_horizon(self) -> int:
        return self.horizon

    @property
    def horizon_name(self) -> str:
        s = f"{self.true_horizon}h"
        return s

    @property
    def prediction_timesteps(self) -> List[float]:
        """By default, we predict the timesteps in the horizon range (i.e. at data resolution)"""
        return self._prediction_timesteps or self.horizon_range

    @prediction_timesteps.setter
    def prediction_timesteps(self, value: List[float]):
        assert max(value) <= self.horizon_range[-1], f"Prediction range {value} exceeds {self.horizon_range=}"
        self._prediction_timesteps = value

    def num_autoregressive_steps_for_horizon(self, horizon: int) -> int:
        return max(1, math.ceil(horizon / self.true_horizon)) - 1

    @property
    def short_description(self) -> str:
        name = super().short_description
        name += f" (h={self.horizon_name})"
        return name

    def actual_num_input_channels(self, num_input_channels: int) -> int:
        # if we use the inputs as conditioning, and use an output-shaped input (e.g. for DDPM),
        # we need to use the output channels here!
        is_standard_diffusion = self.is_diffusion_model and "dyffusion" not in self.diffusion_config._target_.lower()
        is_dyffusion = self.is_diffusion_model and "dyffusion" in self.diffusion_config._target_.lower()
        if is_standard_diffusion:
            return self.actual_num_output_channels(self.dims["output"])
        elif is_dyffusion:
            return num_input_channels  # window is used as conditioning
        if self.stack_window_to_channel_dim:
            return multiply_by_scalar(num_input_channels, self.window)
        return num_input_channels

    def get_horizon(self, split: str, dataloader_idx: int = 0) -> int:
        if self.datamodule is not None and hasattr(self.datamodule, "get_horizon"):
            return self.datamodule.get_horizon(split, dataloader_idx=dataloader_idx)
        self.log_text.warning(f"Using default horizon {self.horizon} for split ``{split}``.")
        return self.horizon

    @property
    def prediction_horizon(self) -> int:
        if hasattr(self.datamodule_config, "prediction_horizon") and self.datamodule_config.prediction_horizon:
            return self.datamodule_config.prediction_horizon
        return self.horizon * (self.hparams.autoregressive_steps + 1)

    # def on_train_start(self) -> None:
    # def on_fit_start(self) -> None:
    def on_any_start(self, stage: str = None) -> None:
        super().on_any_start(stage)
        horizon = self.get_horizon(stage)
        ar_steps = self.num_autoregressive_steps_for_horizon(horizon)
        # max_horizon = horizon * (ar_steps + 1)
        self.log_text.info(f"Using {ar_steps} autoregressive steps for stage ``{stage}`` with horizon={horizon}.")

    # --------------------------------- Metrics
    def get_epoch_aggregators(self, split: str, dataloader_idx: int = None) -> dict:
        assert split in ["val", "test", "predict"], f"Invalid split {split}"
        is_inference_val = split == "val" and dataloader_idx == 1
        if is_inference_val and self.current_epoch % self.hparams.inference_val_every_n_epochs != 0:
            # Skip inference on validation set for this epoch (for efficiency)
            return {}

        return super().get_epoch_aggregators(split, dataloader_idx)

    @torch.inference_mode()  # torch.no_grad()
    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        dataloader_idx: int = None,
        return_outputs: bool | str = None,
        # "auto",  #  True = -> "preds" + "targets". False: None "all": all outputs
        boundary_conditions: Callable = None,
        t0: float = 0.0,
        dt: float = 1.0,
        aggregators: Dict[str, Callable] = None,
        verbose: bool = True,
        prediction_horizon: int = None,
    ):
        # todo: for huge horizons: load full dynamics + dynamics condition on CPU and send to GPU piece by piece
        return_dict = dict()
        if prediction_horizon is not None:
            assert split == "predict", "Prediction horizon only to be used for split='predict'"
        else:
            prediction_horizon = self.get_horizon(split, dataloader_idx=dataloader_idx)

        return_outputs = return_outputs or self.hparams.return_outputs_at_evaluation
        if return_outputs == "auto":
            return_outputs = "all" if split == "predict" and prediction_horizon < 1500 else False
        no_aggregators = aggregators is None or len(aggregators.keys()) == 0
        if not no_aggregators:
            split3d_and_merge_variables_p = (
                partial(split3d_and_merge_variables, level_names=self.datamodule.hparams.pressure_levels)
                if hasattr(self.datamodule.hparams, "pressure_levels")
                else lambda x: x
            )

        # Get predictions mask if available (applied to preds and targets, e.g. for spatially masked predictions)
        predictions_mask = batch.pop("predictions_mask", None)  # pop to ensure that it's not used in model
        if predictions_mask is not None:
            predictions_mask = predictions_mask[0, ...]  # e.g. (2, 40, 80) -> (40, 80)

        main_data_raw = batch.pop("raw_dynamics", None)  # Unnormalized (raw scale) data, used to compute targets
        dynamic_conds = batch.pop("dynamical_condition", None)  # will be added back to batch later, piece by piece
        # main_batch = batch.copy()
        # Compute how many autoregressive steps to complete
        if dataloader_idx is not None and dataloader_idx > 0 and no_aggregators:
            self.log_text.info(f"No aggregators for {split=} {dataloader_idx=} {self.current_epoch=}")
            return {}
        else:
            assert split in ["val", "test", "predict"] + self.test_set_names, f"Invalid split {split}"
            n_outer_loops = self.num_autoregressive_steps_for_horizon(prediction_horizon) + 1
            dyn_any = main_data_raw if main_data_raw is not None else batch["dynamics"]
            if dyn_any.shape[1] < prediction_horizon:
                raise ValueError(f"Prediction horizon {prediction_horizon} is larger than {dyn_any.shape}[1]")

        # Remove the last part of the dynamics that is not needed for prediction inside the module/model
        # dynamics = batch["dynamics"].clone()
        batch["dynamics"] = batch["dynamics"][:, : self.window + self.true_horizon, ...]

        if self.is_diffusion_model and split == "val" and dataloader_idx in [0, None]:
            # log validation loss
            if dynamic_conds is not None:
                # first window of dyn. condition
                batch["dynamical_condition"] = dynamic_conds[:, : self.window + self.true_horizon]
            loss = self.get_loss(batch)
            if isinstance(loss, dict):
                # add split/ prefix if not already there
                log_dict = {f"{split}/{k}" if not k.startswith(split) else k: float(v) for k, v in loss.items()}
            elif torch.is_tensor(loss):
                log_dict = {f"{split}/loss": float(loss)}
            self.log_dict(log_dict, on_step=False, on_epoch=True)

        # Initialize autoregressive loop
        autoregressive_inputs = None
        total_t = t0
        predicted_range_last = [0.0] + self.prediction_timesteps[:-1]
        ar_window_steps_t = self.horizon_range[-self.window :]  # autoregressive window steps (all after input window)
        pbar = tqdm(
            range(n_outer_loops),
            desc="Autoregressive Step",
            position=0,
            leave=True,
            disable=not self.verbose or n_outer_loops <= 1,
        )
        # Loop over autoregressive steps (to cover timesteps beyond training horizon)
        for ar_step in pbar:
            self.print_gpu_memory_usage(tqdm_bar=pbar, empty_cache=self.hparams.empty_cache_at_autoregressive_step)
            ar_window_steps = []
            # Loop over training horizon
            for t_step_last, t_step in zip(predicted_range_last, self.prediction_timesteps):
                total_horizon = ar_step * self.true_horizon + t_step
                if total_horizon > prediction_horizon:
                    # May happen if we have a prediction horizon that is not a multiple of the true horizon
                    break
                PREDS_NORMED_K = f"t{t_step}_preds_normed"
                PREDS_RAW_K = f"t{t_step}_preds"
                pr_kwargs = {} if autoregressive_inputs is None else {"num_predictions": 1}
                if dynamic_conds is not None:  # self.true_horizon=1
                    # ar_step = 0 --> slice(0, H+1), ar_step = 1 --> slice(H, 2H+1), etc.
                    current_slice = slice(ar_step * self.true_horizon, (ar_step + 1) * self.true_horizon + 1)
                    batch["dynamical_condition"] = dynamic_conds[:, current_slice]

                results = self.get_preds_at_t_for_batch(
                    batch, t_step, split, is_autoregressive=ar_step > 0, ensemble=True, **pr_kwargs
                )
                total_t += dt * (t_step - t_step_last)  # update time, by default this is == dt

                if float(total_horizon).is_integer() and main_data_raw is not None:
                    target_time = self.window + int(total_horizon) - 1
                    targets_tensor_t = main_data_raw[:, target_time, ...]
                    targets = self.get_target_variants(targets_tensor_t, is_normalized=False)
                else:
                    targets = None

                targets_normed = targets["targets_normed"] if targets is not None else None
                targets_raw = targets["targets"] if targets is not None else None
                # Apply boundary conditions to predictions, if any
                if boundary_conditions is not None:
                    data_t = main_data_raw[:, target_time, ...]
                    for k in [PREDS_NORMED_K, "preds_autoregressive_init_normed"]:
                        if k in results:
                            results[k] = boundary_conditions(
                                preds=results[k],
                                targets=targets_normed,
                                metadata=batch.get("metadata", None),
                                data=data_t,
                                time=total_t,
                            )
                preds_normed = results.pop(PREDS_NORMED_K)
                if return_outputs in [True, "all"]:
                    return_dict[f"t{total_horizon}_targets_normed"] = torch_to_numpy(targets_normed)
                    return_dict[f"t{total_horizon}_preds_normed"] = torch_to_numpy(preds_normed)
                elif return_outputs == "preds_only":
                    return_dict[f"t{total_horizon}_preds_normed"] = torch_to_numpy(preds_normed)

                if return_outputs == "all":
                    return_dict[f"t{total_horizon}_targets"] = torch_to_numpy(targets_raw)
                    return_dict.update(
                        {k.replace(f"t{t_step}", f"t{total_horizon}"): torch_to_numpy(v) for k, v in results.items()}
                    )  # update keys to total horizon (instead of relative horizon of autoregressive step)

                if t_step in ar_window_steps_t:
                    # if predicted_range == self.horizon_range and window == 1, then this is just the last step :)
                    # Need to keep the last window steps that are INTEGER steps!
                    ar_init = results.pop("preds_autoregressive_init_normed", preds_normed)
                    if self.use_ensemble_predictions(split):
                        ar_init = rrearrange(ar_init, "N B ... -> (N B) ...")  # flatten ensemble dimension
                    ar_window_steps += [ar_init]  # keep t,c,z,h,w

                if not float(total_horizon).is_integer():
                    self.log_text.info(f"Skipping non-integer total horizon {total_horizon}")
                    continue

                if no_aggregators:
                    continue

                with self.timing_scope(context=f"aggregators_{split}", no_op=True):
                    assert predictions_mask is None, "Predictions mask not yet supported for aggregators"
                    pred_data = split3d_and_merge_variables_p(results[PREDS_RAW_K])
                    target_data = split3d_and_merge_variables_p(targets_raw)
                    aggregators[f"t{total_horizon}"].record_batch(
                        target_data=target_data,
                        gen_data=pred_data,
                        target_data_norm=split3d_and_merge_variables_p(targets_normed),
                        gen_data_norm=split3d_and_merge_variables_p(preds_normed),
                        predictions_mask=predictions_mask,
                    )
                    if "time_mean" in aggregators:
                        aggregators["time_mean"].record_batch(
                            target_data=target_data, gen_data=pred_data, predictions_mask=predictions_mask
                        )
                del results, targets

            if ar_step < n_outer_loops - 1:  # if not last step, then update dynamics
                autoregressive_inputs = torch.stack(ar_window_steps, dim=1)  # shape (b, window, c, h, w)
                if not torch.is_tensor(autoregressive_inputs):
                    # Rename keys to make clear that these are treated as inputs now
                    for k in list(autoregressive_inputs.keys()):
                        autoregressive_inputs[k.replace("preds", "inputs")] = autoregressive_inputs.pop(k)
                batch["dynamics"] = autoregressive_inputs
            del ar_window_steps

        self.on_autoregressive_loop_end(split, dataloader_idx=dataloader_idx)
        return return_dict

    def on_autoregressive_loop_end(self, split: str, dataloader_idx: int = None, **kwargs):
        pass

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None, **kwargs):
        return super().test_step(batch, batch_idx, dataloader_idx, **kwargs)

    def on_test_epoch_end(self, **kwargs) -> None:
        return super().on_test_epoch_end(**kwargs)

    def get_preds_at_t_for_batch(
        self,
        batch: Dict[str, Tensor],
        horizon: int | float,
        split: str,
        ensemble: bool = False,
        is_autoregressive: bool = False,
        prepare_inputs: bool = True,
        **kwargs,
    ) -> Dict[str, Tensor]:
        b, t = batch["dynamics"].shape[0:2]  # batch size, time steps
        assert 0 < horizon <= self.true_horizon, f"horizon={horizon} must be in [1, {self.true_horizon}]"

        isi1 = isinstance(self, MHDYffusionAbstract)
        isi2 = isinstance(self, SimultaneousMultiHorizonForecasting)
        cache_preds = isi1 or isi2
        if not cache_preds or horizon == self.prediction_timesteps[0]:
            if self.prediction_timesteps != self.horizon_range:
                if isi1:
                    self.model.hparams.prediction_timesteps = [p_h for p_h in self.prediction_timesteps]
            # create time tensor full of t_step, with batch size shape
            if prepare_inputs:
                inputs, extra_kwargs = self.get_inputs_and_extra_kwargs(
                    batch, time=None, split=split, is_autoregressive=is_autoregressive, ensemble=ensemble
                )
            else:
                inputs = batch.pop(self.inputs_data_key)
                extra_kwargs = batch

            # inputs may be a repeated version of batch["dynamics"] for ensemble predictions
            with torch.inference_mode():
                self._current_preds = self.predict(inputs, **extra_kwargs, **kwargs)
                # for k, v, in {**self._current_preds, "dynamics": batch["dynamics"]}.items():
                # log.info(f"key={k}, shape={v.shape}, min={v.min()}, max={v.max()}, mean={v.mean()}, std={v.std()}")

        if cache_preds:
            # for this model, we can cache the multi-horizon predictions
            preds_key = f"t{horizon}_preds"  # key for this horizon's predictions
            results = {k: self._current_preds.pop(k) for k in list(self._current_preds.keys()) if preds_key in k}
            if horizon == self.horizon_range[-1]:
                assert all(
                    ["preds" not in k or "preds_autoregressive_init" in k for k in self._current_preds.keys()]
                ), (
                    f'{preds_key=} must be the only key containing "preds" in last prediction. '
                    f"Got: {list(self._current_preds.keys())}"
                )
                results = {**results, **self._current_preds}  # add the rest of the results, if any
                del self._current_preds
        else:
            results = {f"t{horizon}_{k}": v for k, v in self._current_preds.items()}
        return results

    def get_inputs_from_dynamics(self, dynamics: Tensor | Dict[str, Tensor]) -> Tensor | Dict[str, Tensor]:
        return dynamics[:, : self.window, ...]  # (b, window, c, lat, lon) at time 0

    def get_condition_from_dynamica_cond(
        self, dynamics: Tensor | Dict[str, Tensor], **kwargs
    ) -> Tensor | Dict[str, Tensor]:
        dynamics_cond = self.get_inputs_from_dynamics(dynamics)
        dynamics_cond = self.transform_inputs(dynamics_cond, **kwargs)
        return dynamics_cond

    def transform_inputs(
        self,
        inputs: Tensor,
        time: Tensor = None,
        ensemble: bool = True,
        stack_window_to_channel_dim: bool = None,
        **kwargs,
    ) -> Tensor:
        if stack_window_to_channel_dim is None:
            stack_window_to_channel_dim = self.stack_window_to_channel_dim
        if stack_window_to_channel_dim:
            inputs = rrearrange(inputs, "b window c ... -> b (window c) ...")
        if ensemble:
            inputs = self.get_ensemble_inputs(inputs, **kwargs)
        return inputs

    def get_extra_model_kwargs(
        self,
        batch: Dict[str, Tensor],
        split: str,
        time: Tensor = None,
        ensemble: bool = False,
        is_autoregressive: bool = False,
    ) -> Dict[str, Any]:
        extra_kwargs = dict()
        ensemble_k = ensemble and not is_autoregressive
        if self.USE_TIME_AS_EXTRA_INPUT:
            batch["time"] = time
        for k, v in batch.items():
            if k == "dynamics":
                continue
            elif k == "metadata":
                if self.PASS_METADATA_TO_MODEL:
                    extra_kwargs[k] = self.get_ensemble_inputs(v, split=split, add_noise=False) if ensemble_k else v
            elif k == "predictions_mask":
                extra_kwargs[k] = v[0, ...]  # e.g. (2, 40, 80) -> (40, 80)
            elif k in ["static_condition", "time", "lookback"]:
                # Static features or time: simply add ensemble dimension and done
                extra_kwargs[k] = self.get_ensemble_inputs(v, split=split, add_noise=False) if ensemble else v
            elif "dynamical_condition" == k:  # k in ["condition", "time_varying_condition"]:
                # Time-varying features
                extra_kwargs[k] = self.get_condition_from_dynamica_cond(
                    v, split=split, time=time, ensemble=ensemble, add_noise=False
                )
            else:
                raise ValueError(f"Unsupported key {k} in batch")
        return extra_kwargs

    def get_inputs_and_extra_kwargs(
        self,
        batch: Dict[str, Tensor],
        time: Tensor = None,
        split: str = None,
        ensemble: bool = False,
        is_autoregressive: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        inputs = self.get_inputs_from_dynamics(batch["dynamics"])
        ensemble_inputs = ensemble and not is_autoregressive
        inputs = self.pack_data(inputs, input_or_output="input")
        inputs = self.transform_inputs(inputs, split=split, ensemble=ensemble_inputs)
        extra_kwargs = self.get_extra_model_kwargs(
            batch, split=split, time=time, ensemble=ensemble, is_autoregressive=is_autoregressive
        )
        return inputs, extra_kwargs


class MHDYffusionAbstract(AbstractMultiHorizonForecastingExperiment):
    PASS_METADATA_TO_MODEL = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.diffusion_config is not None, "diffusion config must be set. Use ``diffusion=<dyffusion>``!"
        assert self.diffusion_config.timesteps == self.horizon, "diffusion timesteps must be equal to horizon"


# This class is a subclass of MHDYffusionAbstract for multi-horizon forecasting using diffusion
# models.
class MultiHorizonForecastingDYffusion(MHDYffusionAbstract):
    model: BaseDYffusion

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Problematic when module.torch_compile="model":
        # assert isinstance(
        #     self.model, BaseDYffusion
        # ), f"Model must be an instance of BaseDYffusion, but got {type(self.model)}"
        if hasattr(self.model, "interpolator"):
            # self.log_text.info(f"------------------- Setting num_predictions={self.hparams.num_predictions}")
            self.model.interpolator.hparams.num_predictions = self.hparams.num_predictions
            self.model.interpolator.num_predictions_in_mem = self.num_predictions_in_mem

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if hasattr(self.model, "interpolator"):
            self.model.interpolator._datamodule = self.datamodule

    @property
    def valid_time_range_for_backbone_model(self) -> List[int]:
        return self.model.valid_time_range_for_backbone_model

    def get_condition_from_dynamica_cond(
        self, dynamics: Tensor | Dict[str, Tensor], **kwargs
    ) -> Tensor | Dict[str, Tensor]:
        # selection of times will be handled inside src.diffusion.dyffusion
        return self.transform_inputs(dynamics, stack_window_to_channel_dim=False, **kwargs)

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        split = "train" if self.training else "val"
        dynamics = batch["dynamics"]
        x_last = dynamics[:, -1, ...]
        x_last = self.pack_data(x_last, input_or_output="output")
        inputs, extra_kwargs = self.get_inputs_and_extra_kwargs(batch, split=split, ensemble=False)

        loss = self.model.p_losses(input_dynamics=inputs, xt_last=x_last, **extra_kwargs)
        return loss

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        #  Skip loading the interpolator state_dict, as its weights are loaded in src.diffusion.dyffusion.__init__
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("model.interpolator")}
        return super().load_state_dict(state_dict, strict=False)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        # Pop the interpolator state_dict from the checkpoint, as it is not needed
        checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if "model.interpolator" not in k}


class AbstractSimultaneousMultiHorizonForecastingModule(AbstractMultiHorizonForecastingExperiment):
    _horizon_at_once: int = None

    def __init__(self, horizon_at_once: int = None, autoregressive_loss_weights: Sequence[float] = None, **kwargs):
        """Simultaneous multi-horizon forecasting module.

        Args:
            horizon_at_once (int, optional): Number of time steps to forecast at once. Defaults to None.
                If None, then the full horizon is forecasted at once.
                Otherwise, only ``horizon_at_once`` time steps are forecasted at once and trained autoregressively until the full horizon is reached.
        """
        super().__init__(**kwargs)
        self.autoregressive_train_steps = self.horizon // self.horizon_at_once
        if self.autoregressive_train_steps > 1:
            self.log_text.info(
                f"Training autoregressively for {self.autoregressive_train_steps} steps with horizon_at_once={self.horizon_at_once}"
            )
        if autoregressive_loss_weights is None:
            autoregressive_loss_weights = [
                1.0 / self.autoregressive_train_steps for _ in range(self.autoregressive_train_steps)
            ]
        assert (
            len(autoregressive_loss_weights) == self.autoregressive_train_steps
        ), f"Expected {self.autoregressive_train_steps} autoregressive loss weights, but got {len(autoregressive_loss_weights)}"
        self.autoregressive_loss_weights = autoregressive_loss_weights

        if self.stack_window_to_channel_dim:
            # Need to reshape the predictions to (b, t, c, h, w), where t = num_time_steps predicted
            # if self.horizon_at_once > 1:
            self.targets_pre_process = partial(rrearrange, pattern="b t c ... -> b (t c) ...", t=self.horizon_at_once)
            # else:
            #     self.targets_pre_process = lambda x: x
            self.predictions_post_process = partial(
                rrearrange, pattern="b (t c) ... -> b t c ...", t=self.horizon_at_once
            )
        else:
            self.predictions_post_process = self.targets_pre_process = None

    @property
    def horizon_at_once(self) -> int:
        if self._horizon_at_once is None:
            self._horizon_at_once = self.hparams.horizon_at_once or self.horizon
            assert self.horizon % self.horizon_at_once == 0, "horizon must be divisible by horizon_at_once"
        return self._horizon_at_once

    @property
    def true_horizon(self) -> int:
        return self.horizon_at_once

    @property
    def horizon_range(self) -> List[int]:
        return list(range(1, self.horizon_at_once + 1))

    def actual_num_output_channels(self, num_output_channels: int) -> int:
        num_output_channels = super().actual_num_output_channels(num_output_channels)
        if self.stack_window_to_channel_dim:
            return multiply_by_scalar(num_output_channels, self.horizon_at_once)
        return num_output_channels

    def reshape_predictions(self, results: TensorDict) -> TensorDict:
        """Reshape and unpack the predictions from the model. This modifies the input dictionary in-place.
        Args:
           results (Dict[str, Tensor]): The model outputs. Access the predictions via results['preds'].
        """
        # reshape predictions to (b, t, c, h, w), where t = num_time_steps predicted
        # ``b`` corresponds to the batch dimension and potentially the ensemble dimension
        results["preds"] = self.predictions_post_process(results["preds"])
        # for k in list(results.keys()):
        # results[k] = rrearrange(results[k], "b (t c) ... -> b t c ...", t=self.horizon)
        # if isinstance(results, TensorDictBase):
        #     results.batch_size = [*results.batch_size, self.horizon]
        return super().reshape_predictions(results)

    def unpack_predictions(self, results: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Unpack the predictions from the model. This modifies the input dictionary in-place.
        Args:
           results (Dict[str, Tensor]): The model outputs. Access the predictions via results['preds'].
        """
        horizon_dim = 1 if self.num_predictions == 1 else 2  # self.CHANNEL_DIM - 1  # == -4
        preds = results.pop("preds")
        assert (
            preds.shape[horizon_dim] == self.horizon_at_once
        ), f"Expected {preds.shape=} with dim {horizon_dim}={self.horizon_at_once}"
        for h in self.horizon_range:
            results[f"t{h}_preds"] = torch_select(preds, dim=horizon_dim, index=h - 1)
            # th_pred.shape = (E, B, C, H, W); E = ensemble, B = batch, C = channels, H = height, W = width
        return super().unpack_predictions(results)


class SimultaneousMultiHorizonForecasting(AbstractSimultaneousMultiHorizonForecastingModule):
    def __init__(self, timestep_loss_weights: Sequence[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model", "timestep_loss_weights"])

        if timestep_loss_weights is None:
            timestep_loss_weights = [1.0 / self.horizon_at_once for _ in range(self.horizon_at_once)]
        self.timestep_loss_weights = timestep_loss_weights

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        dynamics = batch["dynamics"]
        split = "train" if self.training else "val"
        inputs, extra_kwargs = self.get_inputs_and_extra_kwargs(batch, split=split, ensemble=False)

        losses = dict(loss=0.0)
        for ar_step in range(self.autoregressive_train_steps):
            offset_left = self.window + self.horizon_at_once * ar_step
            offset_right = self.window + self.horizon_at_once * (ar_step + 1)
            targets = dynamics[:, offset_left:offset_right, ...]
            targets = self.pack_data(targets, input_or_output="output")
            # if self.stack_window_to_channel_dim:
            # =========== THE BELOW GIVES TERRIBLE LOSS CURVES ==========================
            # DO NOT DO THIS: targets = rrearrange(targets, "b t c ... -> b (t c) ...") |
            # ===========================================================================
            # targets = self.targets_pre_process(targets)  # This will still do it, but only if t > 1
            loss_ar_i, preds = self.model.get_loss(
                inputs=inputs,
                targets=targets,
                return_predictions=True,
                predictions_post_process=self.predictions_post_process,
                targets_pre_process=self.targets_pre_process,
                **extra_kwargs,
            )
            if isinstance(loss_ar_i, dict):
                losses["loss"] += loss_ar_i.pop("loss") * self.autoregressive_loss_weights[ar_step]
                for k, v in loss_ar_i.items():
                    k_ar = f"{k}_ar{ar_step}" if ar_step > 0 else k
                    losses[k_ar] = float(v)
            else:
                losses["loss"] += loss_ar_i * self.autoregressive_loss_weights[ar_step]

            if ar_step < self.autoregressive_train_steps - 1:
                if isinstance(preds, dict):
                    # log.info(f"inputs.shape={inputs.shape}, preds.shape={preds['preds'].shape}")
                    inputs = preds.pop("preds")  # use the predictions as inputs for the next autoregressive step
                    for k, v in preds.items():
                        # log.info(f"Adding {k} to loss_ar_i, shape={v.shape}, before: {extra_kwargs.get(k).shape}")
                        extra_kwargs[k] = v  # overwrite other kwargs for the next step
                else:
                    inputs = preds
                inputs = inputs[:, -self.window :, ...].squeeze(1)  # keep only the last window steps

        return losses


def infer_class_from_ckpt(ckpt_path: str, state=None) -> Type[AbstractMultiHorizonForecastingExperiment]:
    """Infer the experiment class from the checkpoint path."""
    ckpt = torch.load(ckpt_path, map_location="cpu") if state is None else state
    module_config = ckpt["hyper_parameters"]
    abstract_kwargs = inspect.signature(AbstractMultiHorizonForecastingExperiment).parameters
    base_kwargs = {k: v for k, v in module_config.items() if k not in abstract_kwargs}
    diffusion_cfg = module_config["diffusion_config"]
    if diffusion_cfg is not None:
        if "dyffusion" in diffusion_cfg.get("_target_", ""):
            return MultiHorizonForecastingDYffusion
        return SimultaneousMultiHorizonForecasting
    elif "timestep_loss_weights" in base_kwargs.keys():
        return SimultaneousMultiHorizonForecasting
    else:
        raise ValueError(f"Could not infer class from {ckpt_path=}")
