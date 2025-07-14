"""
SOMA-specific loss functions that handle denormalization for physically meaningful training.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
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


def create_soma_loss(base_loss_name: str = "mse", datamodule: Any = None, 
                     **loss_kwargs) -> SOMADenormalizingLoss:
    """
    Factory function to create SOMA denormalizing loss.
    
    Args:
        base_loss_name: Name of base loss function ("mse", "l1", etc.)
        datamodule: SOMA datamodule with denormalization capabilities
        **loss_kwargs: Additional arguments for base loss function
        
    Returns:
        SOMADenormalizingLoss instance
    """
    # Create base loss function
    if base_loss_name.lower() in ["mse", "l2", "mean_squared_error"]:
        base_loss = nn.MSELoss(**loss_kwargs)
    elif base_loss_name.lower() in ["l1", "mae", "mean_absolute_error"]:
        base_loss = nn.L1Loss(**loss_kwargs)
    elif base_loss_name.lower() in ["smooth_l1", "huber"]:
        base_loss = nn.SmoothL1Loss(**loss_kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {base_loss_name}")
    
    return SOMADenormalizingLoss(base_loss, datamodule)
