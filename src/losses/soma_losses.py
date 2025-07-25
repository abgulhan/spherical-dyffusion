"""
SOMA-specific loss functions that handle denormalization for physically meaningful training.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class SOMADenormalizingLoss(nn.Module):
    """
    Loss wrapper that denormalizes predictions and targets before computing loss.
    This is critical for SOMA training to ensure physically meaningful gradients.
    
    Can be instantiated via Hydra config:
        loss_function:
          _target_: src.losses.soma_losses.SOMADenormalizingLoss
          base_loss_fn:
            _target_: torch.nn.MSELoss
            reduction: mean
    """
    
    def __init__(self, base_loss_fn: nn.Module = None, datamodule: Any = None):
        """
        Args:
            base_loss_fn: The underlying loss function (e.g., MSELoss). 
                         If None, defaults to MSELoss.
            datamodule: SOMA datamodule with denormalization capabilities.
                       Will be set automatically by the experiment.
        """
        super().__init__()
        
        # Handle case where base_loss_fn is not provided (for Hydra instantiation)
        if base_loss_fn is None:
            base_loss_fn = nn.MSELoss()
        
        self.base_loss_fn = base_loss_fn
        self.datamodule = datamodule
        
        # Log that SOMA loss is being used
        logger.info("="*60)
        logger.info("  SOMA DENORMALIZING LOSS INITIALIZED!")
        logger.info(f"   Base loss function: {type(base_loss_fn).__name__}")
        logger.info(f"   Datamodule provided: {datamodule is not None}")
        logger.info("="*60)
        
    def set_datamodule(self, datamodule):
        """Set or update the datamodule reference."""
        self.datamodule = datamodule
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                variable_name: Optional[str] = None) -> torch.Tensor:
        """
        Compute loss after denormalizing both predictions and targets.
        
        Args:
            predictions: Model predictions (normalized)
            targets: Ground truth targets (normalized)
            variable_name: Name of the variable for proper denormalization
            
        Returns:
            Loss computed on denormalized data
        """
        # Log that SOMA loss forward is being called
        logger.debug("  SOMA LOSS FORWARD CALLED - Computing loss on denormalized data")
        
        if self.datamodule is None:
            logger.warning("No datamodule set for SOMADenormalizingLoss, computing loss on normalized data")
            return self.base_loss_fn(predictions, targets)
            
        # Check if datamodule has denormalization capability
        if not hasattr(self.datamodule, 'denormalize_for_loss'):
            logger.warning("Datamodule does not support denormalization, computing loss on normalized data")
            return self.base_loss_fn(predictions, targets)
            
        try:
            # Denormalize both predictions and targets
            predictions_denorm = self.datamodule.denormalize_for_loss(predictions, variable_name=variable_name)
            targets_denorm = self.datamodule.denormalize_for_loss(targets, variable_name=variable_name)
            
            # Compute loss on denormalized data
            return self.base_loss_fn(predictions_denorm, targets_denorm)
            
        except Exception as e:
            logger.error(f"Error during denormalization in loss computation: {e}")
            logger.warning("Falling back to normalized data for loss computation")
            return self.base_loss_fn(predictions, targets)


class SOMAMultiVariableLoss(nn.Module):
    """
    Loss function that handles multiple variables with proper denormalization for each.
    """
    
    def __init__(self, base_loss_fn: nn.Module, datamodule: Any = None, 
                 variable_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            base_loss_fn: The underlying loss function (e.g., MSELoss)
            datamodule: SOMA datamodule with denormalization capabilities
            variable_weights: Optional weights for different variables
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.datamodule = datamodule
        self.variable_weights = variable_weights or {}
        
    def set_datamodule(self, datamodule):
        """Set or update the datamodule reference."""
        self.datamodule = datamodule
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted loss across multiple variables with denormalization.
        
        Args:
            predictions: Dict of model predictions {variable_name: tensor}
            targets: Dict of ground truth targets {variable_name: tensor}
            
        Returns:
            Weighted sum of losses across variables
        """
        total_loss = 0.0
        total_weight = 0.0
        
        for var_name in predictions.keys():
            if var_name not in targets:
                logger.warning(f"Variable {var_name} in predictions but not in targets, skipping")
                continue
                
            pred = predictions[var_name]
            target = targets[var_name]
            
            # Get variable-specific loss
            if self.datamodule is not None and hasattr(self.datamodule, 'denormalize_for_loss'): # check if soma datamodule
                try:
                    pred_denorm = self.datamodule.denormalize_for_loss(pred, variable_name=var_name)
                    target_denorm = self.datamodule.denormalize_for_loss(target, variable_name=var_name)
                    var_loss = self.base_loss_fn(pred_denorm, target_denorm)
                except Exception as e:
                    logger.error(f"Error denormalizing {var_name}: {e}")
                    var_loss = self.base_loss_fn(pred, target)
            else:
                var_loss = self.base_loss_fn(pred, target)
            
            # Apply variable weight
            weight = self.variable_weights.get(var_name, 1.0)
            total_loss += weight * var_loss
            total_weight += weight
            
        # Return weighted average
        if total_weight > 0:
            return total_loss / total_weight
        else:
            return total_loss


class DirectionLoss(nn.Module):
    """
    Computes direction loss for velocity components (zonal and meridional).
    This loss ensures that the predicted velocity direction is consistent with the target.
    """
    
    def __init__(self, zonal_var: str = "u", meridional_var: str = "v", 
                 loss_type: str = "angular", eps: float = 1e-8):
        """
        Args:
            zonal_var: Name of the zonal velocity variable
            meridional_var: Name of the meridional velocity variable  
            loss_type: Type of direction loss ('angular', 'cosine', 'magnitude_weighted')
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.zonal_var = zonal_var
        self.meridional_var = meridional_var
        self.loss_type = loss_type.lower()
        self.eps = eps
        
        logger.info(f"DirectionLoss initialized: {zonal_var}/{meridional_var}, type={loss_type}")
        
    def forward(self, predictions: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Compute direction loss between predicted and target velocity components.
        
        Args:
            predictions: Either dict with velocity components or stacked tensor [u, v, ...]
            targets: Either dict with velocity components or stacked tensor [u, v, ...]
            
        Returns:
            Direction loss value
        """
        # Extract velocity components
        if isinstance(predictions, dict):
            pred_u = predictions[self.zonal_var]
            pred_v = predictions[self.meridional_var]
            target_u = targets[self.zonal_var] 
            target_v = targets[self.meridional_var]
        else:
            # Assume first two channels are u, v
            pred_u, pred_v = predictions[:, 0], predictions[:, 1]
            target_u, target_v = targets[:, 0], targets[:, 1]
            
        return self._compute_direction_loss(pred_u, pred_v, target_u, target_v)
    
    def _compute_direction_loss(self, pred_u: torch.Tensor, pred_v: torch.Tensor,
                               target_u: torch.Tensor, target_v: torch.Tensor) -> torch.Tensor:
        """Compute the actual direction loss based on loss_type."""
        
        if self.loss_type == "angular":
            return self._angular_loss(pred_u, pred_v, target_u, target_v)
        elif self.loss_type == "cosine":
            return self._cosine_loss(pred_u, pred_v, target_u, target_v)
        elif self.loss_type == "magnitude_weighted":
            return self._magnitude_weighted_loss(pred_u, pred_v, target_u, target_v)
        else:
            raise ValueError(f"Unknown direction loss type: {self.loss_type}")
    
    def _angular_loss(self, pred_u, pred_v, target_u, target_v):
        """Angular difference between velocity vectors."""
        # Compute angles
        pred_angle = torch.atan2(pred_v, pred_u + self.eps)
        target_angle = torch.atan2(target_v, target_u + self.eps)
        
        # Angular difference (accounting for periodicity)
        angle_diff = pred_angle - target_angle
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        
        return torch.mean(angle_diff ** 2)
    
    def _cosine_loss(self, pred_u, pred_v, target_u, target_v):
        """Cosine similarity loss between velocity vectors."""
        # Normalize vectors
        pred_mag = torch.sqrt(pred_u**2 + pred_v**2 + self.eps)
        target_mag = torch.sqrt(target_u**2 + target_v**2 + self.eps)
        
        pred_u_norm = pred_u / pred_mag
        pred_v_norm = pred_v / pred_mag
        target_u_norm = target_u / target_mag
        target_v_norm = target_v / target_mag
        
        # Cosine similarity
        cosine_sim = pred_u_norm * target_u_norm + pred_v_norm * target_v_norm
        
        # Convert to loss (1 - cosine_similarity)
        return torch.mean(1.0 - cosine_sim)
    
    def _magnitude_weighted_loss(self, pred_u, pred_v, target_u, target_v):
        """Direction loss weighted by velocity magnitude."""
        # Compute magnitudes
        target_mag = torch.sqrt(target_u**2 + target_v**2 + self.eps)
        
        # Normalize and compute angular loss
        angular_loss = self._angular_loss(pred_u, pred_v, target_u, target_v)
        
        # Weight by magnitude (higher weight for stronger velocities)
        weights = target_mag / (torch.mean(target_mag) + self.eps)
        
        return torch.mean(weights * angular_loss)


class SOMACompositeLoss(nn.Module):
    """
    Composite loss that combines multiple loss components with configurable weights.
    Supports both denormalizing and non-denormalizing losses.
    """
    
    def __init__(self, 
                 primary_loss: nn.Module,
                 datamodule: Any = None,
                 direction_loss: Optional[DirectionLoss] = None,
                 direction_weight: float = 0.1,
                 additional_losses: Optional[Dict[str, Dict]] = None,
                 denormalize_primary: bool = False,
                 denormalize_direction: bool = False):
        """
        Args:
            primary_loss: Main loss function (e.g., MSELoss)
            datamodule: SOMA datamodule with denormalization capabilities
            direction_loss: Optional DirectionLoss instance
            direction_weight: Weight for direction loss component
            additional_losses: Dict of additional losses {"name": {"loss": loss_fn, "weight": float, "denormalize": bool}}
            denormalize_primary: Whether to denormalize data for primary loss
            denormalize_direction: Whether to denormalize data for direction loss
        """
        super().__init__()
        
        self.primary_loss = primary_loss
        self.datamodule = datamodule
        self.direction_loss = direction_loss
        self.direction_weight = direction_weight
        self.additional_losses = additional_losses or {}
        self.denormalize_primary = denormalize_primary
        self.denormalize_direction = denormalize_direction
        
        # Store primary loss name for better logging
        self.primary_loss_name = getattr(primary_loss, '__class__', type(primary_loss)).__name__.lower().replace('loss', '')
        
        # Log configuration
        logger.info("="*60)
        logger.info("  SOMA COMPOSITE LOSS INITIALIZED!")
        logger.info(f"   Primary loss: {type(primary_loss).__name__}")
        logger.info(f"   Direction loss enabled: {direction_loss is not None}")
        if direction_loss is not None:
            logger.info(f"   Direction weight: {direction_weight}")
            logger.info(f"   Direction denormalize: {denormalize_direction}")
        logger.info(f"   Primary denormalize: {denormalize_primary}")
        logger.info(f"   Additional losses: {list(self.additional_losses.keys())}")
        logger.info("="*60)
        
    def set_datamodule(self, datamodule):
        """Set or update the datamodule reference."""
        self.datamodule = datamodule
        
    def forward(self, predictions: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                variable_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss with all components.
        
        Returns:
            Dict with individual loss components and total loss
        """
        loss_components = {}
        
        # Primary loss (with optional denormalization)
        if self.denormalize_primary and self._can_denormalize():
            try:
                pred_denorm = self._denormalize(predictions, variable_name)
                target_denorm = self._denormalize(targets, variable_name)
                primary_loss_val = self.primary_loss(pred_denorm, target_denorm)
            except Exception as e:
                logger.warning(f"Primary loss denormalization failed: {e}")
                primary_loss_val = self.primary_loss(predictions, targets)
        else:
            primary_loss_val = self.primary_loss(predictions, targets)
            
        # Add descriptive loss component names for WandB
        primary_loss_name = type(self.primary_loss).__name__.lower().replace('loss', '')
        loss_components['primary'] = primary_loss_val
        loss_components[f'train/loss_{primary_loss_name}'] = primary_loss_val
        
        # Direction loss (with optional denormalization)
        if self.direction_loss is not None:
            if self.denormalize_direction and self._can_denormalize():
                try:
                    pred_denorm = self._denormalize(predictions, variable_name)
                    target_denorm = self._denormalize(targets, variable_name)
                    direction_loss_val = self.direction_loss(pred_denorm, target_denorm)
                except Exception as e:
                    logger.warning(f"Direction loss denormalization failed: {e}")
                    direction_loss_val = self.direction_loss(predictions, targets)
            else:
                direction_loss_val = self.direction_loss(predictions, targets)
                
            loss_components['direction'] = direction_loss_val

            loss_components['train/loss_direction'] = direction_loss_val
        
        # Additional losses
        for loss_name, loss_config in self.additional_losses.items():
            loss_fn = loss_config['loss']
            weight = loss_config.get('weight', 1.0)
            denormalize = loss_config.get('denormalize', False)
            
            if denormalize and self._can_denormalize():
                try:
                    pred_denorm = self._denormalize(predictions, variable_name)
                    target_denorm = self._denormalize(targets, variable_name)
                    additional_loss_val = loss_fn(pred_denorm, target_denorm)
                except Exception as e:
                    logger.warning(f"Additional loss {loss_name} denormalization failed: {e}")
                    additional_loss_val = loss_fn(predictions, targets)
            else:
                additional_loss_val = loss_fn(predictions, targets)
                
            loss_components[loss_name] = additional_loss_val
            loss_components[f'train/loss_{loss_name}'] = additional_loss_val
        
        # Compute total weighted loss
        total_loss = loss_components['primary']
        
        if 'direction' in loss_components:
            total_loss = total_loss + self.direction_weight * loss_components['direction']
            
        for loss_name, loss_config in self.additional_losses.items():
            weight = loss_config.get('weight', 1.0)
            total_loss = total_loss + weight * loss_components[loss_name]
        
        loss_components['total'] = total_loss
        loss_components['loss'] = total_loss  # For compatibility with training_step
        
        return loss_components
    
    def _can_denormalize(self) -> bool:
        """Check if denormalization is possible."""
        return (self.datamodule is not None and 
                hasattr(self.datamodule, 'denormalize_for_loss'))
    
    def _denormalize(self, data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                    variable_name: Optional[str] = None):
        """Denormalize data using datamodule."""
        if isinstance(data, dict):
            # Handle dict of tensors
            denorm_data = {}
            for var_name, tensor in data.items():
                denorm_data[var_name] = self.datamodule.denormalize_for_loss(
                    tensor, variable_name=var_name)
            return denorm_data
        else:
            # Handle single tensor
            return self.datamodule.denormalize_for_loss(data, variable_name=variable_name)


def make_soma_loss(base_loss_name: str = "mse", datamodule: Any = None, 
                     **loss_kwargs) -> SOMADenormalizingLoss:
    """
    Factory function to make SOMA denormalizing loss.
    
    Args:
        base_loss_name: Name of base loss function ("mse", "l1", etc.)
        datamodule: SOMA datamodule with denormalization capabilities
        **loss_kwargs: Additional arguments for base loss function
        
    Returns:
        SOMADenormalizingLoss instance
    """
    # Make base loss function
    if base_loss_name.lower() in ["mse", "l2", "mean_squared_error"]:
        base_loss = nn.MSELoss(**loss_kwargs)
    elif base_loss_name.lower() in ["l1", "mae", "mean_absolute_error"]:
        base_loss = nn.L1Loss(**loss_kwargs)
    elif base_loss_name.lower() in ["smooth_l1", "huber"]:
        base_loss = nn.SmoothL1Loss(**loss_kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {base_loss_name}")
    
    return SOMADenormalizingLoss(base_loss, datamodule)


def make_soma_composite_loss(
    base_loss_name: str = "mse", 
    datamodule: Any = None,
    enable_direction_loss: bool = False,
    direction_loss_config: Optional[Dict] = None,
    direction_weight: float = 0.1,
    denormalize_primary: bool = False,
    denormalize_direction: bool = False,
    additional_losses: Optional[Dict] = None,
    **loss_kwargs) -> Union[SOMADenormalizingLoss, SOMACompositeLoss]:
    """
    Factory function to make SOMA composite loss with optional direction loss.
    
    Args:
        base_loss_name: Name of base loss function ("mse", "l1", etc.)
        datamodule: SOMA datamodule with denormalization capabilities
        enable_direction_loss: Whether to include direction loss
        direction_loss_config: Config for DirectionLoss {"zonal_var": "u", "meridional_var": "v", "loss_type": "angular"}
        direction_weight: Weight for direction loss component
        denormalize_primary: Whether to denormalize for primary loss
        denormalize_direction: Whether to denormalize for direction loss  
        additional_losses: Additional loss components
        **loss_kwargs: Additional arguments for base loss function
        
    Returns:
        SOMACompositeLoss or SOMADenormalizingLoss instance
    """
    # Make base loss function
    if base_loss_name.lower() in ["mse", "l2", "mean_squared_error"]:
        base_loss = nn.MSELoss(**loss_kwargs)
    elif base_loss_name.lower() in ["l1", "mae", "mean_absolute_error"]:
        base_loss = nn.L1Loss(**loss_kwargs)
    elif base_loss_name.lower() in ["smooth_l1", "huber"]:
        base_loss = nn.SmoothL1Loss(**loss_kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {base_loss_name}")
    
    # If no additional components, return simple denormalizing loss
    if not enable_direction_loss and not additional_losses:
        return SOMADenormalizingLoss(base_loss, datamodule)
    
    # Make direction loss if requested
    direction_loss = None
    if enable_direction_loss:
        dir_config = direction_loss_config or {}
        direction_loss = DirectionLoss(
            zonal_var=dir_config.get("zonal_var", "u"),
            meridional_var=dir_config.get("meridional_var", "v"),
            loss_type=dir_config.get("loss_type", "angular"),
            eps=dir_config.get("eps", 1e-8)
        )
    
    return SOMACompositeLoss(
        primary_loss=base_loss,
        datamodule=datamodule,
        direction_loss=direction_loss,
        direction_weight=direction_weight,
        additional_losses=additional_losses,
        denormalize_primary=denormalize_primary,
        denormalize_direction=denormalize_direction
    )
