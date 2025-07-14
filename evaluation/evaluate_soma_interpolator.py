#!/usr/bin/env python3
"""
Comprehensive evaluation script for SOMA interpolator model with proper data loading and normalization.

Usage:
    cd evaluation/
    python evaluate_soma_interpolator.py --ckpt_path /path/to/checkpoint.ckpt \
        --data_path /path/to/soma_data.h5 \
        --timestep1 0 --timestep2 24 --target_times 6,12,18

Functions:
    - interpolate_timesteps(): Main function for interpolation that can be imported
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import h5py
from typing import List, Union, Tuple, Dict, Any

# Add the parent directory (project root) to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.experiment_types.interpolation import InterpolationExperiment
from src.datamodules.SOMA_datamodule import MyCustomDataModule

# Import metrics for optional evaluation
try:
    from src.evaluation import metrics
except ImportError:
    metrics = None

from src.utilities.utils import to_tensordict

def interpolate(    
    timestep_pair: Dict[str, torch.Tensor],
    target_times: Union[int, List[int]] = 12,
    variable_names: List[str] = None,
    device: str = "auto",
    verbose: bool = False,
    model=None,
    datamodule=None,
) -> Dict[str, Any]:
    """
    Interpolate between two consecutive timesteps using the SOMA interpolator model.
    
    Args:
        timestep_pair: Dict containing dynamics with 2 consecutive timesteps from SOMA datamodule
                      Format: {'dynamics': {var_name: tensor_of_shape(2, depth, height, width), ...}}
        target_times: Time parameter(s) for interpolation - can be int or list of ints
        variable_names: List of variable names to interpolate (if None, uses all available)
        device: Device to use ("auto", "cpu", "cuda")
        verbose: Whether to print detailed progress information
        model: Pre-loaded model instance (required)
        datamodule: Pre-loaded datamodule instance (required for normalization)
    
    Returns:
        Dictionary containing:
            - 'predictions': Dict of predictions for each target time
            - 'target_times': List of target times predicted
            - 'metadata': Additional info about shapes, variables, etc.
    """
    
    if model is None:
        raise ValueError("Model instance must be provided")
    if datamodule is None:
        raise ValueError("Datamodule instance must be provided for normalization")
    
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    model.eval()
    
    # Override num_predictions for single-sample evaluation
    original_num_predictions = model.num_predictions
    model.num_predictions = 1
    
    if verbose:
        print(f"Model's original num_predictions: {original_num_predictions}")
        print(f"Overriding to num_predictions=1 for interpolation")
        print(f"Model horizon: {model.horizon}, window: {model.window}")
    
    # Ensure the model's datamodule reference is set for pack_data/unpack_data
    model._datamodule = datamodule
    
    # Get dataset for normalization/denormalization
    dataset = datamodule._data_train if datamodule._data_train is not None else datamodule._data_val
    if dataset is None:
        raise ValueError("No dataset available from datamodule for normalization")
    
    # Ensure target_times is a list
    if isinstance(target_times, int):
        target_times = [target_times]
    
    # Extract dynamics from timestep_pair
    if 'dynamics' not in timestep_pair:
        raise ValueError("timestep_pair must contain 'dynamics' key")
    
    dynamics_dict = timestep_pair['dynamics']
    
    if verbose:
        print(f"Target interpolation times: {target_times}")
        print(f"Dynamics keys: {list(dynamics_dict.keys())}")
        
        # Verify we have exactly 2 timesteps
        first_var = list(dynamics_dict.values())[0]
        if first_var.shape[0] != 2:
            raise ValueError(f"Expected 2 timesteps in dynamics, got {first_var.shape[0]}")
        print(f"Confirmed 2 timesteps in dynamics data")
    
    # Filter variable names if specified
    available_vars = list(dynamics_dict.keys())
    if variable_names is not None:
        available_vars = [var for var in variable_names if var in available_vars]
        if verbose:
            print(f"Filtering to variables: {available_vars}")
    
    # Move tensors to device
    for var_name in available_vars:
        dynamics_dict[var_name] = dynamics_dict[var_name].to(device)
    
    if verbose:
        print(f"Moved dynamics to device: {device}")
        for var_name in available_vars:
            print(f"  {var_name}: {dynamics_dict[var_name].shape}")
    
    # Extract start and end timesteps from the 2-timestep window
    start_timestep = {}
    end_timestep = {}
    
    for var_name in available_vars:
        var_tensor = dynamics_dict[var_name]  # Shape: (2, depth, height, width) or (2, height, width)
        start_timestep[var_name] = var_tensor[0]  # First timestep
        end_timestep[var_name] = var_tensor[1]    # Second timestep
        
        if verbose:
            print(f"Extracted {var_name}: start {start_timestep[var_name].shape}, end {end_timestep[var_name].shape}")
    
    # Create interpolation window by combining start and end timesteps
    # The model expects a window of timesteps, so we need to create a proper sequence
    # For interpolation, we use the first timestep(s) as context and the last as boundary condition
    
    if verbose:
        print("Creating interpolation window from start and end timesteps...")
    
    # Create a dynamics dict that contains the proper window structure
    # We'll create a sequence where:
    # - First model.window timesteps are from start_timestep (repeated to fill window)
    # - Last timestep is end_timestep (at position model.horizon)
    
    batch_dynamics_dict = {}
    
    for var_name in available_vars:
        start_var = start_timestep[var_name]  # Shape: (depth, height, width) or (height, width)
        end_var = end_timestep[var_name]      # Shape: (depth, height, width) or (height, width)
        
        # Create a sequence of length (model.window + model.horizon) to match training expectations
        total_timesteps = model.window + model.horizon
        
        if start_var.dim() == 3:  # (depth, height, width)
            sequence_shape = (total_timesteps,) + start_var.shape
        elif start_var.dim() == 2:  # (height, width)
            sequence_shape = (total_timesteps,) + start_var.shape
        else:
            raise ValueError(f"Unexpected variable {var_name} shape: {start_var.shape}")
        
        # Create the sequence
        var_sequence = torch.zeros(sequence_shape, device=device, dtype=start_var.dtype)
        
        # Fill the window portion with start_timestep (repeated)
        for i in range(model.window):
            var_sequence[i] = start_var
        
        # Fill the horizon portion with linear interpolation between start and end
        # This provides a reasonable initialization for the sequence
        for i in range(model.horizon):
            t_idx = model.window + i
            alpha = (i + 1) / model.horizon  # Interpolation weight
            var_sequence[t_idx] = (1 - alpha) * start_var + alpha * end_var
        
        # Set the final timestep explicitly to end_timestep
        var_sequence[-1] = end_var
        
        # Add batch dimension
        batch_dynamics_dict[var_name] = var_sequence.unsqueeze(0)  # (1, time, ...)
        
        if verbose:
            print(f"Created sequence for {var_name}: {batch_dynamics_dict[var_name].shape}")
    
    # Convert to TensorDict for model processing
    if verbose:
        print("Converting dynamics dictionary to TensorDict...")
    
    dynamics_tensordict = to_tensordict(batch_dynamics_dict, find_batch_size_max=True)
    
    if verbose:
        print(f"TensorDict created successfully")
        print(f"TensorDict batch size: {dynamics_tensordict.batch_size}")
        print(f"TensorDict keys: {list(dynamics_tensordict.keys())}")
    
    # Get model inputs
    if verbose:
        print("Processing inputs through model...")
    
    ## converts input data to compatible format to interpolator
    inputs = model.get_inputs_from_dynamics(dynamics_tensordict)
    
    if verbose:
        print(f"Processed inputs shape: {inputs.shape}")
    
    # Run interpolation predictions
    if verbose:
        print(f"\nRunning interpolation for {len(target_times)} target time(s)...")
    
    predictions = {}
    with torch.no_grad():
        for target_time in target_times:
            if verbose:
                print(f"Predicting timestep {target_time}...")
            
            # Create time tensor
            time_tensor = torch.full((inputs.shape[0],), target_time,
                                   device=device, dtype=torch.long)
            
            # Use predict_packed method
            results = model.predict_packed(inputs, time=time_tensor)
            pred = results["preds"]
            
            if verbose:
                print(f"Raw prediction shape: {pred.shape}")
            predictions[target_time] = pred
    
    # Process and denormalize predictions
    if verbose:
        print("\nProcessing and denormalizing predictions...")
    
    denormalized_predictions = {}
    for target_time, pred in predictions.items():
        if verbose:
            print(f"Processing prediction for t={target_time}...")
            print(f"Raw prediction shape: {pred.shape}")
        
        # Create results dict for unpacking
        results_dict = {"preds": pred}
        
        # Unpack the predictions using the model's unpack_predictions method
        unpacked_results = model.unpack_predictions(results_dict.copy())
        preds_tensordict = unpacked_results['preds']
        
        if verbose:
            print(f"Unpacked variables: {list(preds_tensordict.keys())}")
        
        # Extract and denormalize each variable
        denorm_vars = {}
        
        for var_name in available_vars:
            if var_name in preds_tensordict:
                var_tensor_normalized = preds_tensordict[var_name]  # Shape: (batch, depth, height, width) or (batch, height, width)
                var_tensor_normalized = var_tensor_normalized.squeeze(0)  # Remove batch dimension
                
                # Denormalize using the dataset's inverse_transform
                try:
                    var_tensor_denormalized = dataset.inverse_transform(var_tensor_normalized, variable_name=var_name)
                    denorm_vars[var_name] = var_tensor_denormalized.cpu().numpy()
                    
                    if verbose:
                        print(f"  Denormalized {var_name}: {denorm_vars[var_name].shape}")
                        
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Failed to denormalize {var_name}: {e}")
                    # Fallback: just convert to numpy without denormalization
                    denorm_vars[var_name] = var_tensor_normalized.cpu().numpy()
            else:
                if verbose:
                    print(f"⚠️ Variable {var_name} not found in unpacked predictions")
        
        denormalized_predictions[target_time] = denorm_vars
    
    # Prepare metadata
    spatial_shape = inputs.shape[1:]  # Get spatial shape from processed inputs
    metadata = {
        'model_horizon': model.horizon,
        'model_window': model.window,
        'spatial_shape': spatial_shape,
        'variable_names': available_vars,
        'device': device,
        'input_dynamics_shape': {k: v.shape for k, v in batch_dynamics_dict.items()},
        'normalized_input_shape': inputs.shape,
        'timestep_pair_shape': {k: v.shape for k, v in dynamics_dict.items()},
        'start_timestep_shape': {k: v.shape for k, v in start_timestep.items()},
        'end_timestep_shape': {k: v.shape for k, v in end_timestep.items()},
    }
    
    result = {
        'predictions': denormalized_predictions,
        'target_times': target_times,
        'metadata': metadata,
    }
    
    if verbose:
        print(f"\n✅ Interpolation completed successfully!")
        print(f"Generated {len(denormalized_predictions)} interpolated timesteps")
    
    return result


def interpolate_timesteps(
    checkpoint_path: str = None,
    data_path: str = None,
    timestep1: int = 0,
    timestep2: int = 24,
    target_times: Union[int, List[int]] = 12,
    sample_idx: int = 0,
    variable_names: List[str] = None,
    device: str = "auto",
    num_predictions: int = None,
    compute_metrics: bool = False,
    verbose: bool = False,
    model=None,
    datamodule=None,
) -> Dict[str, Any]:
    """
    Main interpolation function that can be imported and used in other scripts.

    Args:
        checkpoint_path: Path to trained model checkpoint (optional if model is provided)
        data_path: Path to SOMA HDF5 data file (optional if datamodule is provided)
        timestep1: First input timestep (boundary condition)
        timestep2: Second input timestep (boundary condition)
        target_times: Time parameter(s) for interpolation - can be int or list of ints
        sample_idx: Which sample within the batch to use
        variable_names: List of variable names to load (if None, loads all)
        device: Device to use ("auto", "cpu", "cuda")
        num_predictions: Override model's num_predictions for evaluation (if None, defaults to 1)
        compute_metrics: Whether to compute metrics comparing predictions to ground truth (default: False)
        verbose: Whether to print detailed progress information
        model: Pre-loaded model instance (optional)
        datamodule: Pre-loaded datamodule instance (optional)

    Returns:
        Dictionary containing:
            - 'model': The loaded model
            - 'predictions': Dict of predictions for each target time
            - 'inputs': The input data used
            - 'target_times': List of target times predicted
            - 'metadata': Additional info about shapes, variables, etc.
    """

    # 1) Load the model to GPU if not provided
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Either 'model' or 'checkpoint_path' must be provided.")
        if verbose:
            print(f"Loading model from: {checkpoint_path}")
        model = InterpolationExperiment.load_from_checkpoint(checkpoint_path)
        model.eval()
    else:
        if verbose:
            print("Using provided model instance.")

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    if verbose:
        print(f"✅ Model moved to {device}")

    # Override num_predictions for evaluation if needed
    # The model was trained with num_predictions=2 but we can use any value for inference
    original_num_predictions = model.num_predictions
    if verbose:
        print(f"Model's original num_predictions: {original_num_predictions}")

    if num_predictions is not None:
        model.num_predictions = num_predictions
        if verbose:
            print(f"Overriding num_predictions to: {model.num_predictions} for evaluation")
    else:
        # For single-sample evaluation, default to 1 to avoid batch size issues
        model.num_predictions = 1
        if verbose:
            print(f"Setting num_predictions to: {model.num_predictions} for evaluation (default override)")

    if verbose:
        print(f"Model horizon: {model.horizon}, window: {model.window}")

    # 2) Load the dataset (raw data) and get normalized data from datamodule if not provided
    if datamodule is None:
        if data_path is None:
            raise ValueError("Either 'datamodule' or 'data_path' must be provided.")
        if verbose:
            print(f"\nLoading data from: {data_path}")

        # Set up datamodule using model's configuration
        datamodule_config = getattr(model, "hparams", {}).get('datamodule_config', {})
        if isinstance(datamodule_config, dict):
            # Update with our data path
            datamodule_config['data_path'] = data_path
            datamodule = MyCustomDataModule(**datamodule_config)
        else:
            # Fallback: create with minimal config
            datamodule = MyCustomDataModule(
                data_path=data_path,
                batch_size=1,
                window=model.window,
                horizon=model.horizon,
                num_workers=0
            )

        # Setup the datamodule
        if verbose:
            print("Setting up datamodule...")
        datamodule.setup("fit")
        if verbose:
            print(f"✅ Datamodule setup complete. Has in_packer: {hasattr(datamodule, 'in_packer')}")
    else:
        if verbose:
            print("Using provided datamodule instance.")

    # Ensure the model's datamodule reference is set for pack_data/unpack_data
    # Set the private _datamodule attribute directly to avoid Lightning restrictions
    model._datamodule = datamodule
    if verbose:
        print(f"✅ Model datamodule reference set")

    # Get the dataset which has built-in normalization
    dataset = datamodule._data_train if datamodule._data_train is not None else datamodule._data_val
    if dataset is None:
        raise ValueError("No dataset available from datamodule. Check data path and setup.")

    if verbose:
        print(f"Dataset has {len(dataset)} samples")
        print(f"Dataset param_names: {dataset.param_names}")

    # 3) Get properly formatted data from the dataset (already normalized)
    if verbose:
        print(f"\nGetting data sample from dataset...")

    # Get a sample to understand the data structure
    sample = dataset[sample_idx]
    if verbose:
        print(f"Sample keys: {sample.keys()}")

    # The sample contains 'dynamics' which is a dict of tensors (one per variable)
    dynamics_dict = sample['dynamics']  # This is already normalized by the dataset
    if verbose:
        print(f"Dynamics dict keys: {list(dynamics_dict.keys())}")
        print(f"Dynamics dict type: {type(dynamics_dict)}")

    # Print shapes of each variable
    if verbose:
        for var_name, var_tensor in dynamics_dict.items():
            print(f"Variable {var_name}: {var_tensor.shape}")

    # Move dynamics_dict to device first (but keep as dict for proper processing)
    for key in dynamics_dict:
        dynamics_dict[key] = dynamics_dict[key].to(device)

    if verbose:
        print(f"Input dynamics dict moved to device: {device}")

    # Add batch dimension to each variable in the dict
    batch_dynamics_dict = {}
    for var_name, var_tensor in dynamics_dict.items():
        if var_tensor.dim() == 4:  # (time, depth, height, width) - 3D case
            batch_dynamics_dict[var_name] = var_tensor.unsqueeze(0)  # (1, time, depth, height, width)
        elif var_tensor.dim() == 3:  # (time, height, width) - 2D case
            batch_dynamics_dict[var_name] = var_tensor.unsqueeze(0)  # (1, time, height, width)
        else:
            raise ValueError(f"Unexpected variable {var_name} shape: {var_tensor.shape}. Expected 3D or 4D tensor.")

    if verbose:
        print(f"Batch dynamics dict shapes:")
        for var_name, var_tensor in batch_dynamics_dict.items():
            print(f"  {var_name}: {var_tensor.shape}")

    # Extract timesteps we need for interpolation
    sample_timesteps = list(batch_dynamics_dict.values())[0].shape[1]  # Get time dimension from any variable
    if verbose:
        print(f"Total timesteps in sample: {sample_timesteps}")

    if sample_timesteps < 2:
        raise ValueError(f"Need at least 2 timesteps for interpolation, got {sample_timesteps}")

    if verbose:
        print(f"Sample will use first {model.window} timesteps and last timestep as boundary conditions")

    # Convert the dynamics dictionary to a TensorDict before passing to get_inputs_from_dynamics
    # This matches what happens in the training pipeline where to_tensordict() is called
    if verbose:
        print("Converting dynamics dictionary to TensorDict...")
    from src.utilities.utils import to_tensordict
    dynamics_tensordict = to_tensordict(batch_dynamics_dict, find_batch_size_max=True)
    if verbose:
        print(f"TensorDict created successfully")
        print(f"TensorDict batch size: {dynamics_tensordict.batch_size}")
        print(f"TensorDict keys: {list(dynamics_tensordict.keys())}")

    if verbose:
        print("Passing TensorDict to model for processing...")
    inputs = model.get_inputs_from_dynamics(dynamics_tensordict)
    if verbose:
        print(f"Processed inputs shape: {inputs.shape}")

    # Get metadata about the data
    variable_names = list(dynamics_dict.keys())  # Use the original variable names
    spatial_shape = inputs.shape[1:]  # Get spatial shape from processed inputs

    # Ensure target_times is a list
    if isinstance(target_times, int):
        target_times = [target_times]

    if verbose:
        print(f"Target interpolation times: {target_times}")

    # 4) Run interpolation and get output frames
    if verbose:
        print(f"\nRunning interpolation for {len(target_times)} target time(s)...")

    predictions = {}
    raw_predictions_for_comparison = []
    
    with torch.no_grad():
        for i, target_time in enumerate(target_times):
            if verbose:
                print(f"Predicting timestep {target_time}...")

            # Construct time tensor
            time_tensor = torch.full((inputs.shape[0],), target_time,
                                   device=device, dtype=torch.long)
            
            if verbose:
                print(f"   Time tensor: {time_tensor} (shape: {time_tensor.shape}, device: {time_tensor.device})")
                print(f"   Input tensor shape: {inputs.shape}, device: {inputs.device}")

            # Use predict_packed method directly to bypass ensemble logic issues
            # The model is configured for ensemble predictions (num_predictions=2) but we have batch_size=1
            results = model.predict_packed(inputs, time=time_tensor)
            pred = results["preds"]

            if verbose:
                print(f"Raw prediction shape: {pred.shape}")
                
            # Store for detailed comparison
            pred_cpu = pred.detach().cpu()
            raw_predictions_for_comparison.append(pred_cpu)
            
            # Print detailed statistics for each prediction
            print(f"🔍 Target time {target_time} detailed stats:")
            print(f"   Raw tensor mean: {pred.mean().item():.10f}")
            print(f"   Raw tensor std:  {pred.std().item():.10f}")
            print(f"   Raw tensor min:  {pred.min().item():.10f}")
            print(f"   Raw tensor max:  {pred.max().item():.10f}")
            print(f"   Raw tensor sum:  {pred.sum().item():.10f}")
            
            # Compare with previous predictions if we have any
            if i > 0:
                prev_pred = raw_predictions_for_comparison[i-1].to(pred.device)
                
                # Detailed comparison
                abs_diff = torch.abs(pred - prev_pred)
                rel_diff = abs_diff / (torch.abs(prev_pred) + 1e-8)
                
                print(f"   📊 Comparison with t={target_times[i-1]}:")
                print(f"      Absolute diff mean: {abs_diff.mean().item():.12f}")
                print(f"      Absolute diff max:  {abs_diff.max().item():.12f}")
                print(f"      Relative diff mean: {rel_diff.mean().item():.12f}")
                print(f"      Relative diff max:  {rel_diff.max().item():.12f}")
                print(f"      Are tensors equal?  {torch.equal(pred, prev_pred)}")
                print(f"      Are tensors close? (1e-6): {torch.allclose(pred, prev_pred, atol=1e-6, rtol=1e-6)}")
                print(f"      Are tensors close? (1e-8): {torch.allclose(pred, prev_pred, atol=1e-8, rtol=1e-8)}")
                print(f"      Are tensors close? (1e-10): {torch.allclose(pred, prev_pred, atol=1e-10, rtol=1e-10)}")
                
                # Check specific locations for differences
                if not torch.equal(pred, prev_pred):
                    diff_locations = torch.where(abs_diff > 1e-10)
                    if len(diff_locations[0]) > 0:
                        print(f"      Number of differing elements: {len(diff_locations[0])}")
                        print(f"      Fraction differing: {len(diff_locations[0]) / pred.numel():.6f}")
                        # Show a few examples
                        for j in range(min(5, len(diff_locations[0]))):
                            idx = tuple(loc[j].item() for loc in diff_locations)
                            val1 = prev_pred[idx].item()
                            val2 = pred[idx].item()
                            print(f"         Location {idx}: {val1:.12f} → {val2:.12f} (diff: {val2-val1:.12e})")
                else:
                    print(f"      ⚠️ WARNING: Tensors are EXACTLY identical!")
            
            predictions[target_time] = pred
    
    # Final comprehensive comparison of all predictions
    print(f"\n🔬 COMPREHENSIVE PREDICTION COMPARISON:")
    print(f"   Total target times: {len(target_times)}")
    
    # Create a matrix of pairwise comparisons
    comparison_matrix = []
    for i, time_i in enumerate(target_times):
        row = []
        for j, time_j in enumerate(target_times):
            if i == j:
                row.append("SELF")
            else:
                pred_i = raw_predictions_for_comparison[i]
                pred_j = raw_predictions_for_comparison[j]
                
                # Check equality
                are_equal = torch.equal(pred_i, pred_j)
                are_close_1e6 = torch.allclose(pred_i, pred_j, atol=1e-6, rtol=1e-6)
                are_close_1e8 = torch.allclose(pred_i, pred_j, atol=1e-8, rtol=1e-8)
                are_close_1e10 = torch.allclose(pred_i, pred_j, atol=1e-10, rtol=1e-10)
                
                if are_equal:
                    status = "IDENTICAL"
                elif are_close_1e10:
                    status = "CLOSE-1e10"
                elif are_close_1e8:
                    status = "CLOSE-1e8"
                elif are_close_1e6:
                    status = "CLOSE-1e6"
                else:
                    status = "DIFFERENT"
                
                row.append(status)
        comparison_matrix.append(row)
    
    # Print comparison matrix
    print(f"   Pairwise comparison matrix:")
    print(f"      ", end="")
    for time_j in target_times:
        print(f"{time_j:>12}", end="")
    print()
    
    for i, time_i in enumerate(target_times):
        print(f"   t={time_i:2d}", end="")
        for j, status in enumerate(comparison_matrix[i]):
            print(f"{status:>12}", end="")
        print()
    
    # Check if ALL predictions are identical
    all_identical = True
    for i in range(1, len(raw_predictions_for_comparison)):
        if not torch.equal(raw_predictions_for_comparison[0], raw_predictions_for_comparison[i]):
            all_identical = False
            break
    
    if all_identical:
        print(f"   🚨 CRITICAL: ALL PREDICTIONS ARE IDENTICAL!")
        print(f"   🚨 This suggests the model is NOT using time conditioning properly!")
    else:
        print(f"   ✅ SUCCESS: Predictions are different across time steps!")
    
    # Check model's time conditioning capability
    print(f"\n🔍 MODEL TIME CONDITIONING CHECK:")
    print(f"   Model class: {type(model.model).__name__}")
    print(f"   Model has 'time' in forward signature: {hasattr(model.model, 'forward') and 'time' in str(model.model.forward.__code__.co_varnames)}")
    
    # Try to inspect the model's forward method
    try:
        import inspect
        forward_signature = inspect.signature(model.model.forward)
        print(f"   Model forward signature: {forward_signature}")
    except Exception as e:
        print(f"   Could not inspect forward signature: {e}")
    
    print(f"   Model horizon range: {model.horizon_range}")
    print(f"   Target times used: {target_times}")
    print(f"   Valid target times (in horizon range): {[t for t in target_times if t in model.horizon_range]}")
    
    invalid_times = [t for t in target_times if t not in model.horizon_range]
    if invalid_times:
        print(f"   ⚠️ WARNING: Some target times are outside model's trained horizon range: {invalid_times}")
        print(f"   ⚠️ This could explain identical predictions!")
    
    print(f"\n" + "="*80)

    # 5) Use the model's postprocess_predictions to handle unpacking and denormalization
    if verbose:
        print("\nProcessing predictions using model's postprocess_predictions...")

    denormalized_predictions = {}
    for target_time, pred in predictions.items():
        if verbose:
            print(f"Processing prediction for t={target_time}...")

        # pred shape: (batch, channels, depth, height, width) or (batch, channels, height, width)
        if verbose:
            print(f"Raw prediction shape: {pred.shape}")

        # Use the model's postprocess_predictions method, which handles:
        # 1. Reshaping predictions (if needed for ensemble)
        # 2. Unpacking predictions from packed tensor to individual variables
        # 3. Denormalization using the datamodule's inverse_transform

        # Create the same format that the training pipeline expects
        results_dict = {"preds": pred}

        # Apply the same postprocessing as in training
        if verbose:
            print(f"Results dict keys before postprocessing: {results_dict.keys()}")
            print(f"'preds' shape: {results_dict['preds'].shape}")

        # Use manual unpacking and denormalization (more reliable than postprocess_predictions)
        if verbose:
            print("🔍 Unpacking and denormalizing predictions...")

        # Step 1: Unpack the predictions using the model's unpack_predictions method
        unpacked_results = model.unpack_predictions(results_dict.copy())
        preds_tensordict = unpacked_results['preds']

        if verbose:
            print(f"Unpacked variables: {list(preds_tensordict.keys())}")

        # Step 2: Extract individual variables from TensorDict and denormalize them
        denorm_vars = {}

        for var_name in preds_tensordict.keys():
            var_tensor_normalized = preds_tensordict[var_name]  # Shape: (batch, depth, height, width)
            var_tensor_normalized = var_tensor_normalized.squeeze(0)  # Remove batch dimension

            # Step 3: Denormalize using the dataset's inverse_transform
            try:
                var_tensor_denormalized = dataset.inverse_transform(var_tensor_normalized, variable_name=var_name)
                denorm_vars[var_name] = var_tensor_denormalized.cpu().numpy()
            except Exception as e:
                print(f"⚠️ Failed to denormalize {var_name}: {e}")
                # Fallback: just convert to numpy without denormalization
                denorm_vars[var_name] = var_tensor_normalized.cpu().numpy()

        if verbose:
            print(f"Denormalized variables: {list(denorm_vars.keys())}")
            for var_name, var_tensor in denorm_vars.items():
                print(f"  {var_name}: {var_tensor.shape}")

        denormalized_predictions[target_time] = denorm_vars

    # 6) Compute metrics comparing predictions to ground truth (optional)
    computed_metrics = {}
    if compute_metrics:
        if verbose:
            print("\n📊 Computing metrics comparing predictions to ground truth...")

        if metrics is None:
            print("⚠️ Metrics module not available, skipping metrics computation")
        else:
            # Get area weights for proper spatial averaging (SOMA uses uniform grid)
            try:
                area_weights = datamodule.calculate_area_weights(use_mesh_file=False)
                area_weights = area_weights.to(device)
                if verbose:
                    print(f"Using area weights with shape: {area_weights.shape}")
            except Exception as e:
                if verbose:
                    print(f"⚠️ Could not load area weights: {e}, using uniform weights")
                y, x = batch_dynamics_dict[list(batch_dynamics_dict.keys())[0]].shape[-2:]
                area_weights = torch.ones((y, x), device=device)

            # Compute metrics for each target time
            for target_time in target_times:
                if verbose:
                    print(f"Computing metrics for t={target_time}...")

                # Calculate the ground truth time index in the dynamics tensor
                # Following the same logic as in training: target_time_index = window + t - 1
                # where t is the time parameter passed to the model (target_time here)
                ground_truth_time_index = model.window + target_time - 1

                if ground_truth_time_index >= batch_dynamics_dict[list(batch_dynamics_dict.keys())[0]].shape[1]:
                    print(f"⚠️ Ground truth time index {ground_truth_time_index} out of bounds for data with {batch_dynamics_dict[list(batch_dynamics_dict.keys())[0]].shape[1]} timesteps")
                    continue

                # Extract ground truth for this target time
                ground_truth_vars = {}
                for var_name in denormalized_predictions[target_time].keys():
                    if var_name in batch_dynamics_dict:
                        # Get ground truth tensor: shape (1, time, depth, height, width)
                        gt_tensor_normalized = batch_dynamics_dict[var_name][0, ground_truth_time_index, ...]  # (depth, height, width)

                        # Denormalize ground truth using the same method as predictions
                        try:
                            gt_tensor_denormalized = dataset.inverse_transform(gt_tensor_normalized, variable_name=var_name)
                            ground_truth_vars[var_name] = gt_tensor_denormalized
                        except Exception as e:
                            print(f"⚠️ Failed to denormalize ground truth for {var_name}: {e}")
                            continue

                # Compute metrics for each variable
                target_time_metrics = {}
                for var_name in denormalized_predictions[target_time].keys():
                    if var_name not in ground_truth_vars:
                        continue

                    # Convert predictions back to tensor for metrics computation
                    pred_tensor = torch.from_numpy(denormalized_predictions[target_time][var_name]).to(device)
                    gt_tensor = ground_truth_vars[var_name].to(device)

                    # Ensure same shape
                    if pred_tensor.shape != gt_tensor.shape:
                        if verbose:
                            print(f"⚠️ Shape mismatch for {var_name}: pred {pred_tensor.shape} vs gt {gt_tensor.shape}")
                        continue

                    if verbose:
                        print(f"  Computing metrics for {var_name}: shape {pred_tensor.shape}")

                    # Compute various metrics (spatial averaging over last 2 dimensions)
                    spatial_dims = (-2, -1)  # Height and width dimensions

                    try:
                        # Root Mean Squared Error
                        rmse = metrics.root_mean_squared_error(
                            truth=gt_tensor, predicted=pred_tensor,
                            weights=area_weights, dim=spatial_dims
                        ).mean().item()  # Average over remaining dimensions (depth)

                        # Mean Bias
                        bias = metrics.weighted_mean_bias(
                            truth=gt_tensor, predicted=pred_tensor,
                            weights=area_weights, dim=spatial_dims
                        ).mean().item()

                        # Mean Squared Error
                        mse = metrics.mean_squared_error(
                            truth=gt_tensor, predicted=pred_tensor,
                            weights=area_weights, dim=spatial_dims
                        ).mean().item()

                        target_time_metrics[var_name] = {
                            'rmse': rmse,
                            'bias': bias,
                            'mse': mse,
                            'pred_mean': pred_tensor.mean().item(),
                            'gt_mean': gt_tensor.mean().item(),
                            'pred_std': pred_tensor.std().item(),
                            'gt_std': gt_tensor.std().item(),
                        }

                        if verbose:
                            print(f"    {var_name}: RMSE={rmse:.6f}, Bias={bias:.6f}, MSE={mse:.6f}")

                    except Exception as e:
                        if verbose:
                            print(f"⚠️ Error computing metrics for {var_name}: {e}")
                        continue

                computed_metrics[f"t{target_time}"] = target_time_metrics

            # Compute overall metrics across all target times and variables
            if computed_metrics:
                if verbose:
                    print("\n📈 Summary metrics:")
                all_rmse = []
                all_bias = []
                for t_key, t_metrics in computed_metrics.items():
                    for var_name, var_metrics in t_metrics.items():
                        all_rmse.append(var_metrics['rmse'])
                        all_bias.append(var_metrics['bias'])
                        if verbose:
                            print(f"  {t_key}_{var_name}: RMSE={var_metrics['rmse']:.6f}")

                computed_metrics['summary'] = {
                    'mean_rmse': np.mean(all_rmse) if all_rmse else 0.0,
                    'mean_bias': np.mean(all_bias) if all_bias else 0.0,
                    'std_rmse': np.std(all_rmse) if all_rmse else 0.0,
                    'std_bias': np.std(all_bias) if all_bias else 0.0,
                }
                if verbose:
                    print(f"  Overall: Mean RMSE={computed_metrics['summary']['mean_rmse']:.6f}, Mean Bias={computed_metrics['summary']['mean_bias']:.6f}")

    # Prepare metadata
    metadata = {
        'model_horizon': model.horizon,
        'model_window': model.window,
        'spatial_shape': spatial_shape,
        'variable_names': variable_names,
        'input_timesteps': [timestep1, timestep2],
        'device': device,
        'input_dynamics_shape': {k: v.shape for k, v in batch_dynamics_dict.items()},
        'normalized_input_shape': inputs.shape,
        'dataset_length': len(dataset),
        'sample_idx_used': sample_idx,
    }

    # Check if interpolated time steps are identical
    identical_timesteps = []
    pred_arrays = []
    for target_time in target_times:
        # Concatenate all variables for this target_time
        concat_vars = []
        for var_name in denormalized_predictions[target_time]:
            arr = denormalized_predictions[target_time][var_name]
            concat_vars.append(arr.flatten())
        if concat_vars:
            pred_arrays.append(np.concatenate(concat_vars))
    # Compare each prediction array to the next
    for i in range(len(pred_arrays) - 1):
        if np.array_equal(pred_arrays[i], pred_arrays[i+1]):
            identical_timesteps.append((target_times[i], target_times[i+1]))
    if identical_timesteps:
        print(f"⚠️ Warning: The following interpolated time steps are identical: {identical_timesteps}")

    result = {
        'model': model,
        'predictions': denormalized_predictions,
        'inputs': {'dynamics_dict': {k: v.cpu().numpy() for k, v in batch_dynamics_dict.items()}},
        'target_times': target_times,
        'metadata': metadata,
        'metrics': computed_metrics,  # Add computed metrics to result
    }

    if verbose:
        print(f"\n✅ Interpolation completed successfully!")
    return result



def evaluate_interpolator(checkpoint_path, data_path=None):
    """
    Simple evaluation function that tests the model with synthetic data.
    For real data interpolation, use interpolate_timesteps() function.
    """
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Load the trained model
    model = InterpolationExperiment.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    if torch.cuda.is_available():
        model.cuda()
        print("✅ Model moved to GPU")
    
    # Override num_predictions for evaluation if needed
    # The model was trained with num_predictions=2 but we can use any value for inference
    original_num_predictions = model.num_predictions
    print(f"Model's original num_predictions: {original_num_predictions}")
    
    # For single-sample evaluation, we can set it to 1 to avoid batch size issues
    # Or keep it at 2 for ensemble predictions (but need to duplicate input)
    model.num_predictions = 1  # Change this to desired value
    print(f"Setting num_predictions to: {model.num_predictions} for evaluation")
    
    print(f"Model horizon: {model.horizon}, window: {model.window}")
    
    # Check model's expected input channels
    if hasattr(model.model, 'encoder') and hasattr(model.model.encoder, 'weight'):
        expected_channels = model.model.encoder.weight.shape[1]
        print(f"Model expects {expected_channels} input channels")
    else:
        # Try to infer from model attributes
        expected_channels = 12  # Default based on error message
        print(f"Model expects {expected_channels} input channels (inferred)")
    
    # Create a simple synthetic test batch to verify the model works
    print("\n🧪 Testing model with synthetic data...")
    
    # Create synthetic input data matching the expected format
    batch_size = 2
    channels = expected_channels  # Use the expected number of channels
    height, width, depth = 100, 100, 60  # SOMA spatial dimensions
    
    # Create synthetic data - FNO3D expects [batch, channels, depth, height, width]
    synthetic_dynamics = torch.randn(batch_size, channels, depth, height, width)
    
    if torch.cuda.is_available():
        synthetic_dynamics = synthetic_dynamics.cuda()
    
    print(f"Synthetic input shape: {synthetic_dynamics.shape}")
    print(f"Using {channels} channels to match model expectations")
    
    # Test model forward pass with time conditioning
    try:
        with torch.no_grad():
            # Test different time steps
            for test_time in [1, 6, 12, 18, 23]:
                time_tensor = torch.full((batch_size,), test_time, 
                                       device=synthetic_dynamics.device, dtype=torch.long)
                
                # Use the proper predict method with time conditioning
                results = model.predict_packed(synthetic_dynamics, time=time_tensor)
                predictions = results["preds"]
                
                print(f"✅ Time step {test_time}: {synthetic_dynamics.shape} → {predictions.shape}")
        
        print(f"\n✅ Model forward pass with time conditioning successful!")
        
        # Calculate some basic statistics for the last prediction
        pred_mean = predictions.mean().item()
        pred_std = predictions.std().item()
        pred_min = predictions.min().item()
        pred_max = predictions.max().item()
        
        print(f"\nPrediction statistics (t={test_time}):")
        print(f"  Mean: {pred_mean:.6f}")
        print(f"  Std:  {pred_std:.6f}")
        print(f"  Min:  {pred_min:.6f}")
        print(f"  Max:  {pred_max:.6f}")
        
        print(f"\n🎯 Model evaluation completed successfully!")
        print(f"💡 To use with real data, call interpolate_timesteps() function")
        
        return True
        
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Evaluate SOMA interpolator model')
    parser.add_argument('--checkpoint_path', '--ckpt_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, 
                       help='Path to SOMA HDF5 data file (required for real data interpolation)')
    
    # Parameters for real data interpolation
    parser.add_argument('--timestep1', type=int, default=0,
                       help='First input timestep (boundary condition)')
    parser.add_argument('--timestep2', type=int, default=24,
                       help='Second input timestep (boundary condition)')
    parser.add_argument('--target_times', type=str, default="12",
                       help='Comma-separated list of target times to interpolate (e.g., "6,12,18")')
    parser.add_argument('--variables', type=str, default=None,
                       help='Comma-separated list of variables to use (if None, uses all)')
    parser.add_argument('--batch_idx', type=int, default=0,
                       help='Batch index to use from dataset')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index within batch')
    parser.add_argument('--device', type=str, default="auto",
                       choices=["auto", "cpu", "cuda"], help='Device to use')
    parser.add_argument('--num_predictions', type=int, default=None,
                       help='Override model num_predictions for evaluation (if None, uses model default)')
    parser.add_argument('--compute_metrics', action='store_true', default=False,
                       help='Compute metrics comparing predictions to ground truth (default: False)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint file not found: {args.checkpoint_path}")
        return
    
    # Parse target times
    target_times = [int(t.strip()) for t in args.target_times.split(',')]
    
    # Parse variables if provided
    variables = None
    if args.variables:
        variables = [v.strip() for v in args.variables.split(',')]
    
    if args.data_path and os.path.exists(args.data_path):
        print("🔄 Running interpolation with real data...")
        try:
            result = interpolate_timesteps(
                checkpoint_path=args.checkpoint_path,
                data_path=args.data_path,
                timestep1=args.timestep1,
                timestep2=args.timestep2,
                target_times=target_times,
                #batch_idx=args.batch_idx,
                sample_idx=args.sample_idx,
                variable_names=variables,
                device=args.device,
                num_predictions=args.num_predictions,
                compute_metrics=args.compute_metrics
            )
            
            print("\n📊 Interpolation Results Summary:")
            print(f"Input timesteps: {result['metadata']['input_timesteps']}")
            print(f"Target times: {result['target_times']}")
            print(f"Variables: {result['metadata']['variable_names']}")
            print(f"Spatial shape: {result['metadata']['spatial_shape']}")
            
            # Display metrics if computed
            if result['metrics']:
                print(f"\n📈 Metrics Summary:")
                if 'summary' in result['metrics']:
                    summary = result['metrics']['summary']
                    print(f"  Overall Mean RMSE: {summary['mean_rmse']:.6f}")
                    print(f"  Overall Mean Bias: {summary['mean_bias']:.6f}")
                print(f"  Metrics computed for {len([k for k in result['metrics'].keys() if k != 'summary'])} target times")
            
            # Save visualizations for all variables and time steps
            try:
                # Ensure the 'figures' directory exists
                figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
                os.makedirs(figures_dir, exist_ok=True)
                
                total_saved = 0
                for target_time, predictions in result['predictions'].items():
                    for var_name, pred_data in predictions.items():
                        # Take a middle slice for visualization
                        if len(pred_data.shape) == 3:  # 3D data
                            mid_depth = pred_data.shape[0] // 2
                            slice_data = pred_data[mid_depth, :, :]
                        else:  # 2D data
                            slice_data = pred_data
                        
                        plt.figure(figsize=(10, 8))
                        plt.imshow(slice_data, cmap='viridis')
                        plt.colorbar()
                        plt.title(f'Interpolated {var_name} at t={target_time}\n'
                                f'Input boundary: t={args.timestep1} and t={args.timestep2}')
                        plt.xlabel('Longitude index')
                        plt.ylabel('Latitude index')
                        
                        filename = os.path.join(figures_dir, f'interpolation_{var_name}_t{target_time}.png')
                        plt.savefig(filename, dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"✅ Saved visualization: {filename}")
                        total_saved += 1
                
                print(f"📈 Total visualizations saved: {total_saved}")
            except Exception as e:
                print(f"⚠️ Could not create visualization: {e}")
            
            print("\n✅ SUCCESS: Real data interpolation completed!")
            
        except Exception as e:
            print(f"❌ Real data interpolation failed: {e}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
    else:
        if args.data_path:
            print(f"⚠️ Data file not found: {args.data_path}")
        print("🔄 Running basic model evaluation with synthetic data...")
        
        success = evaluate_interpolator(args.checkpoint_path, args.data_path)
        
        if success:
            print("\n✅ SUCCESS: Model evaluation completed!")
            print("The model checkpoint is working correctly and can make predictions.")
            if not args.data_path:
                print("💡 Provide --data_path to run interpolation with real data")
        else:
            print("\n❌ FAILED: Model evaluation encountered issues")

if __name__ == "__main__":
    main()
