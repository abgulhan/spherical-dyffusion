import dataclasses
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

import torch

from src.ace_inference.core.aggregator.climate_data import ClimateData, compute_dry_air_absolute_differences
from src.ace_inference.core.data_loading.data_typing import SigmaCoordinates
from src.ace_inference.core.device import get_device


def get_dry_air_nonconservation(
    data: Mapping[str, torch.Tensor],
    area_weights: torch.Tensor,
    sigma_coordinates: SigmaCoordinates,
):
    """
    Computes the time-average one-step absolute difference in surface pressure due to
    changes in globally integrated dry air.

    Args:
        data: A mapping from variable name to tensor of shape
            [sample, time, lat, lon], in physical units. specific_total_water in kg/kg
            and surface_pressure in Pa must be present.
        area_weights: The area of each grid cell as a [lat, lon] tensor, in m^2.
        sigma_coordinates: The sigma coordinates of the model.
    """
    return compute_dry_air_absolute_differences(
        ClimateData(data), area=area_weights, sigma_coordinates=sigma_coordinates
    ).mean()


class ConservationLoss:
    def __init__(
        self,
        config: "ConservationLossConfig",
        area_weights: torch.Tensor,
        sigma_coordinates: SigmaCoordinates,
    ):
        """
        Args:
            config: configuration options.
            area_weights: The area of each grid cell as a [lat, lon] tensor, in m^2.
            sigma_coordinates: The sigma coordinates of the model.
        """
        self._config = config
        self._area_weights = area_weights.to(get_device())
        self._sigma_coordinates = sigma_coordinates.to(get_device())

    def __call__(self, gen_data: Mapping[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute loss and metrics related to conservation.

        Args:
            gen_data: A mapping from variable name to tensor of shape
                [sample, time, lat, lon], in physical units.
        """
        conservation_metrics = {}
        loss = torch.tensor(0.0, device=get_device())
        if self._config.dry_air_penalty is not None:
            dry_air_loss = self._config.dry_air_penalty * get_dry_air_nonconservation(
                gen_data,
                area_weights=self._area_weights,
                sigma_coordinates=self._sigma_coordinates,
            )
            conservation_metrics["dry_air_loss"] = dry_air_loss.detach()
            loss += dry_air_loss
        return conservation_metrics, loss

    def get_state(self):
        return {
            "config": dataclasses.asdict(self._config),
            "sigma_coordinates": self._sigma_coordinates,
            "area_weights": self._area_weights,
        }

    @classmethod
    def from_state(cls, state) -> "ConservationLoss":
        return cls(
            config=ConservationLossConfig(**state["config"]),
            sigma_coordinates=state["sigma_coordinates"],
            area_weights=state["area_weights"],
        )


@dataclasses.dataclass
class ConservationLossConfig:
    """
    Attributes:
        dry_air_penalty: A constant by which to multiply one-step non-conservation
            of surface pressure due to dry air in Pa as an L1 loss penalty. By
            default, no such loss will be included.
    """

    dry_air_penalty: Optional[float] = None

    def build(self, area_weights: torch.Tensor, sigma_coordinates: SigmaCoordinates) -> ConservationLoss:
        return ConservationLoss(
            config=self,
            area_weights=area_weights,
            sigma_coordinates=sigma_coordinates,
        )


class LpLoss(torch.nn.Module):
    def __init__(self, p=2):
        """
        Args:
            p: Lp-norm type. For example, p=1 for L1-norm, p=2 for L2-norm.
        """
        super(LpLoss, self).__init__()

        if p <= 0:
            raise ValueError("Lp-norm type should be positive")

        self.p = p

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        return torch.mean(diff_norms / y_norms)

    def __call__(self, x, y):
        return self.rel(x, y)


class AreaWeightedMSELoss(torch.nn.Module):
    def __init__(self, area: torch.Tensor):
        super(AreaWeightedMSELoss, self).__init__()
        self._area_weights = area / area.mean()

    def __call__(self, x, y):
        return torch.mean((x - y) ** 2 * self._area_weights)


class WeightedSum(torch.nn.Module):
    """
    A module which applies multiple loss-function modules (taking two inputs)
    to the same input and returns a tensor equal to the weighted sum of the
    outputs of the modules.
    """

    def __init__(self, modules: List[torch.nn.Module], weights: List[float]):
        """
        Args:
            modules: A list of modules, each of which takes two tensors and
                returns a scalar tensor.
            weights: A list of weights to apply to the outputs of the modules.
        """
        super().__init__()
        if len(modules) != len(weights):
            raise ValueError("modules and weights must have the same length")
        self._wrapped = modules
        self._weights = weights

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return sum(w * module(x, y) for w, module in zip(self._weights, self._wrapped))


class GlobalMeanLoss(torch.nn.Module):
    """
    A module which computes a loss on the global mean of each sample.
    """

    def __init__(self, area: torch.Tensor, loss: torch.nn.Module):
        """
        Args:
            area: A tensor of shape (n_lat, n_lon) containing the area of
                each grid cell.
            loss: A loss function which takes two tensors of shape
                (n_samples, n_timesteps, n_channels) and returns a scalar
                tensor.
        """
        super().__init__()
        self.global_mean = GlobalMean(area)
        self.loss = loss

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.global_mean(x)
        y = self.global_mean(y)
        return self.loss(x, y)


class GlobalMean(torch.nn.Module):
    def __init__(self, area: torch.Tensor):
        """
        Args:
            area: A tensor of shape (n_lat, n_lon) containing the area of
                each grid cell.
        """
        super().__init__()
        self.area_weights = area / area.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: A tensor of shape (n_samples, n_timesteps, n_channels, n_lat, n_lon)
        """
        return (x * self.area_weights[None, None, None, :, :]).sum(dim=(3, 4))


@dataclasses.dataclass
class LossConfig:
    """
    A dataclass containing all the information needed to build a loss function,
    including the type of the loss function and the data needed to build it.

    Args:
        type: the type of the loss function
        kwargs: data for a loss function instance of the indicated type
        global_mean_type: the type of the loss function to apply to the global
            mean of each sample, by default no loss is applied
        global_mean_kwargs: data for a loss function instance of the indicated
            type to apply to the global mean of each sample
        global_mean_weight: the weight to apply to the global mean loss
            relative to the main loss
    """

    type: Literal["LpLoss", "MSE", "AreaWeightedMSE"] = "LpLoss"
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {})
    global_mean_type: Optional[Literal["LpLoss"]] = None
    global_mean_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {})
    global_mean_weight: float = 1.0

    def __post_init__(self):
        if self.type not in ("LpLoss", "MSE", "AreaWeightedMSE"):
            raise NotImplementedError(self.type)
        if self.global_mean_type is not None and self.global_mean_type != "LpLoss":
            raise NotImplementedError(self.global_mean_type)

    def build(self, area: torch.Tensor) -> Any:
        """
        Args:
            area: A tensor of shape (n_lat, n_lon) containing the area of
                each grid cell.
        """
        area = area.to(get_device())
        if self.type == "LpLoss":
            main_loss = LpLoss(**self.kwargs)
        elif self.type == "MSE":
            main_loss = torch.nn.MSELoss(reduction="mean")
        elif self.type == "AreaWeightedMSE":
            main_loss = AreaWeightedMSELoss(area)

        if self.global_mean_type is not None:
            global_mean_loss = GlobalMeanLoss(area=area, loss=LpLoss(**self.global_mean_kwargs))
            final_loss = WeightedSum(
                modules=[main_loss, global_mean_loss],
                weights=[1.0, self.global_mean_weight],
            )
        else:
            final_loss = main_loss
        return final_loss.to(device=get_device())


# Helper function to compute total quantity (sum over all non-batch dimensions)
def _compute_total_quantity(tensor: torch.Tensor) -> torch.Tensor:
    """Computes the sum of a tensor over all dimensions except the first (batch) dimension.
    Args:
        tensor: A tensor of shape [batch, ...].
    Returns:
        A tensor of shape [batch] representing the sum for each sample.
    """
    if tensor.ndim == 0: # Scalar tensor
        return tensor
    if tensor.ndim == 1: # Already [batch]
        return tensor
    return tensor.reshape(tensor.size(0), -1).sum(dim=1)


class TotalVolumeLoss(torch.nn.Module):
    """
    Computes a loss on the difference between predicted and target total volume.
    Total volume is calculated as the sum of all layer thickness values for each sample.
    """
    def __init__(self, loss_fn: torch.nn.Module, layer_thickness_key: str = 'timeDaily_avg_layerThickness'):
        """
        Args:
            loss_fn: The base loss function (e.g., nn.MSELoss(), nn.L1Loss()) to apply
                     to the difference in total volumes.
            layer_thickness_key: The key for layer thickness data in the input mappings.
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.layer_thickness_key = layer_thickness_key

    def forward(self, pred_data: Mapping[str, torch.Tensor], target_data: Mapping[str, torch.Tensor]) -> torch.Tensor:
        pred_layer_thickness = pred_data.get(self.layer_thickness_key)
        target_layer_thickness = target_data.get(self.layer_thickness_key)

        if pred_layer_thickness is None:
            raise KeyError(f"Layer thickness key '{self.layer_thickness_key}' not found in predicted data.")
        if target_layer_thickness is None:
            raise KeyError(f"Layer thickness key '{self.layer_thickness_key}' not found in target data.")

        pred_total_volume = _compute_total_quantity(pred_layer_thickness)
        target_total_volume = _compute_total_quantity(target_layer_thickness)

        return self.loss_fn(pred_total_volume, target_total_volume)


class TotalSaltLoss(torch.nn.Module):
    """
    Computes a loss on the difference between predicted and target total salt.
    Total salt is calculated as the sum of (layer_thickness * salinity) for each cell,
    summed up for each sample.
    """
    def __init__(self,
                 loss_fn: torch.nn.Module,
                 layer_thickness_key: str = 'timeDaily_avg_layerThickness',
                 salinity_key: str = 'timeDaily_avg_activeTracers_salinity'):
        """
        Args:
            loss_fn: The base loss function (e.g., nn.MSELoss(), nn.L1Loss()) to apply
                     to the difference in total salt.
            layer_thickness_key: The key for layer thickness data in the input mappings.
            salinity_key: The key for salinity data in the input mappings.
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.layer_thickness_key = layer_thickness_key
        self.salinity_key = salinity_key

    def forward(self, pred_data: Mapping[str, torch.Tensor], target_data: Mapping[str, torch.Tensor]) -> torch.Tensor:
        pred_layer_thickness = pred_data.get(self.layer_thickness_key)
        pred_salinity = pred_data.get(self.salinity_key)
        target_layer_thickness = target_data.get(self.layer_thickness_key)
        target_salinity = target_data.get(self.salinity_key)

        if pred_layer_thickness is None:
            raise KeyError(f"Layer thickness key '{self.layer_thickness_key}' not found in predicted data.")
        if pred_salinity is None:
            raise KeyError(f"Salinity key '{self.salinity_key}' not found in predicted data.")
        if target_layer_thickness is None:
            raise KeyError(f"Layer thickness key '{self.layer_thickness_key}' not found in target data.")
        if target_salinity is None:
            raise KeyError(f"Salinity key '{self.salinity_key}' not found in target data.")

        pred_salt_integrand = pred_layer_thickness * pred_salinity
        target_salt_integrand = target_layer_thickness * target_salinity

        pred_total_salt = _compute_total_quantity(pred_salt_integrand)
        target_total_salt = _compute_total_quantity(target_salt_integrand)

        return self.loss_fn(pred_total_salt, target_total_salt)


# Example of how you might define configuration dataclasses for these (optional):
@dataclasses.dataclass
class TotalVolumeLossConfig:
    weight: float = 1.0
    loss_fn_type: Literal["MSE", "L1"] = "MSE"
    layer_thickness_key: str = 'timeDaily_avg_layerThickness'

    def build(self) -> TotalVolumeLoss:
        if self.loss_fn_type == "MSE":
            loss_fn = torch.nn.MSELoss()
        elif self.loss_fn_type == "L1":
            loss_fn = torch.nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss_fn_type: {self.loss_fn_type}")
        return TotalVolumeLoss(loss_fn=loss_fn.to(get_device()), layer_thickness_key=self.layer_thickness_key)

@dataclasses.dataclass
class TotalSaltLossConfig:
    weight: float = 1.0
    loss_fn_type: Literal["MSE", "L1"] = "MSE"
    layer_thickness_key: str = 'timeDaily_avg_layerThickness'
    salinity_key: str = 'timeDaily_avg_activeTracers_salinity'

    def build(self) -> TotalSaltLoss:
        if self.loss_fn_type == "MSE":
            loss_fn = torch.nn.MSELoss()
        elif self.loss_fn_type == "L1":
            loss_fn = torch.nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss_fn_type: {self.loss_fn_type}")
        return TotalSaltLoss(
            loss_fn=loss_fn.to(get_device()),
            layer_thickness_key=self.layer_thickness_key,
            salinity_key=self.salinity_key
        )
