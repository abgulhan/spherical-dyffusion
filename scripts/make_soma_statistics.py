#!/usr/bin/env python3
"""
Make training statistics for SOMA dataset to enable normalization/masking.
Based on the pattern from SOMA_adjoint_deep_ensemble/utils/helper.py
"""

import numpy as np
import h5py
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_training_statistics(data_path):
    """
    Use ALL sequences from training data to compute statistics.
    Compute per z-level, per-channel statistics: (z, channels)
    """
    data_path = Path(data_path)

    logger.info(f"Opening HDF5 file: {data_path}")
    
    with h5py.File(data_path, "r") as f:
        # Get all run keys
        all_keys = list(f.keys())
        logger.info(f"Found {len(all_keys)} runs in HDF5 file")
        
        # Sample from available runs
        if len(all_keys) == 0:
            raise ValueError("No data found in HDF5 file")
            
        # Use ALL data from ALL runs for statistics calculation
        logger.info(f"Computing statistics using ALL {len(all_keys)} runs (optimized for speed)...")
        
        # Get shape from first run
        first_key = all_keys[0]
        first_data = f[first_key]
        time_dim, z_dim, y_dim, x_dim, c_dim = first_data.shape
        logger.info(f"Data dimensions: time={time_dim}, z={z_dim}, y={y_dim}, x={x_dim}, channels={c_dim}")
        
        # Only compute statistics for channels 0-4 (physical variables)
        # Channel 5 (var_x) is constant and should not be normalized
        channels_to_normalize = 5  # Channels 0-4
        
        # Initialize statistics arrays
        sum_values = np.zeros((z_dim, channels_to_normalize), dtype=np.float64)
        sum_squared = np.zeros((z_dim, channels_to_normalize), dtype=np.float64)
        count = np.zeros((z_dim, channels_to_normalize), dtype=np.int64)
        
        total_timesteps = 0
        
        # Process each run separately to avoid memory issues
        for run_idx, key in enumerate(all_keys):
            logger.info(f"Processing run {run_idx+1}/{len(all_keys)}: {key}")
            
            data = f[key]
            logger.info(f"Run {key}: shape {data.shape}")
            
            # Use ALL timesteps (no sampling) - process in efficient chunks
            chunk_size = min(data.shape[0], 20)  # Process 20 timesteps at a time for memory efficiency
            
            for start_t in range(0, data.shape[0], chunk_size):
                end_t = min(start_t + chunk_size, data.shape[0])
                
                # Load chunk: (chunk_time, z, y, x, c) - NO spatial subsampling
                chunk = data[start_t:end_t, ...]
                
                # Create mask for valid values
                valid_mask = np.abs(chunk) < 1e30
                
                # Vectorized statistics update per z-level and channel (only channels 0-4)
                for z in range(z_dim):
                    for c in range(channels_to_normalize):  # Only process channels 0-4
                        z_c_data = chunk[:, z, :, :, c]  # (chunk_time, y, x)
                        z_c_mask = valid_mask[:, z, :, :, c]  # (chunk_time, y, x)
                        
                        if np.any(z_c_mask):
                            valid_values = z_c_data[z_c_mask]
                            n_valid = len(valid_values)
                            
                            if n_valid > 0:
                                # Accumulate statistics
                                sum_values[z, c] += np.sum(valid_values)
                                sum_squared[z, c] += np.sum(valid_values**2)
                                count[z, c] += n_valid
                
                total_timesteps += (end_t - start_t)
                
                # Progress reporting every 10 chunks
                if ((start_t // chunk_size) % 10 == 0):
                    logger.info(f"  Processed {end_t}/{data.shape[0]} timesteps in run {key}")
                
            logger.info(f"Completed run {key}: processed ALL {data.shape[0]} timesteps")
        
        # Compute final statistics from accumulators
        mean = np.zeros((z_dim, c_dim), dtype=np.float64)  # Full 6 channels
        std = np.ones((z_dim, c_dim), dtype=np.float64)    # Full 6 channels
        
        # Compute statistics for channels 0-4
        valid_count_mask = count > 0
        mean[:, :channels_to_normalize][valid_count_mask] = sum_values[valid_count_mask] / count[valid_count_mask]
        
        # Compute variance and std for channels 0-4
        for z in range(z_dim):
            for c in range(channels_to_normalize):
                if count[z, c] > 1:
                    variance = (sum_squared[z, c] / count[z, c]) - (mean[z, c] ** 2)
                    std[z, c] = np.sqrt(max(variance, 1e-12))  # Avoid zero std
        
        # Set Channel 5 (var_x) to neutral normalization (mean=0, std=1)
        mean[:, 5] = 0.0
        std[:, 5] = 1.0
        
        logger.info(f"Total timesteps processed: {total_timesteps}")
        logger.info(f"Statistics computed for {np.sum(count > 0)} out of {z_dim * channels_to_normalize} (z, channel) combinations")
        logger.info(f"Channel 5 (var_x) set to neutral normalization: mean=0, std=1")
        
        # Handle cases with no valid data in channels 0-4
        no_data_mask = count == 0
        if np.any(no_data_mask):
            logger.warning(f"Found {np.sum(no_data_mask)} combinations with no valid data in channels 0-4, setting to mean=0, std=1")
            for z in range(z_dim):
                for c in range(channels_to_normalize):
                    if count[z, c] == 0:
                        mean[z, c] = 0.0
                        std[z, c] = 1.0
        
        logger.info(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
        logger.info(f"Channels 0-4 mean range: [{np.nanmin(mean[:, :5]):.6f}, {np.nanmax(mean[:, :5]):.6f}]")
        logger.info(f"Channels 0-4 std range: [{np.nanmin(std[:, :5]):.6f}, {np.nanmax(std[:, :5]):.6f}]")
        logger.info(f"Channel 5 (var_x): mean={mean[0, 5]:.1f}, std={std[0, 5]:.1f} (neutral)")
        
        # Check for zero std (could cause division by zero) - only in channels 0-4
        zero_std_mask = (std[:, :channels_to_normalize] == 0)
        if np.any(zero_std_mask):
            logger.warning(f"Found {np.sum(zero_std_mask)} elements with zero std in channels 0-4, setting to 1.0")
            std[:, :channels_to_normalize][zero_std_mask] = 1.0

    # Save statistics
    stats_path = data_path.parent / "corrected_training_statistics_per_level_state.npz"
    logger.info(f"Saving corrected statistics to: {stats_path}")
    
    np.savez(
        stats_path,
        mean=mean,
        std=std,
    )
    
    logger.info("Statistics saved successfully!")
    return stats_path

if __name__ == "__main__":
    data_path = "/global/homes/a/abgulhan/WORKING/ml_converted/data-gm-year7-22-biweekly.hdf5"
    
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        exit(1)
    
    stats_path = extract_training_statistics(data_path)
    logger.info(f"Training statistics made at: {stats_path}")
