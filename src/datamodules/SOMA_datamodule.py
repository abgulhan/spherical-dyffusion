import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Dict, Sequence, Union, Any
import pytorch_lightning as pl
import logging
import os
from omegaconf import DictConfig
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree
#from src.ace_inference.core.prescriber import Prescriber
from src.utilities.packer import Packer
from src.evaluation.aggregators.main import OneStepAggregator
from src.evaluation.aggregators.time_mean import TimeMeanAggregator
from src.evaluation import metrics
from src.utilities.utils import get_logger, to_torch_and_device

# Ensure the project root is in sys.path for module import
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    from src.datamodules.abstract_datamodule import BaseDataModule
except ImportError:
    try:
        from abstract_datamodule import BaseDataModule
    except ImportError:
        raise ImportError("Could not import BaseDataModule. Ensure the path is correct or the module exists.")


logger = logging.getLogger(__name__)

class SOMAh5Dataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 run_keys: List[str], 
                 time_steps_per_run: int = -1, 
                 transform=None,
                 param_names: Optional[List[str]] = None,
                 #norm_param_names: Optional[List[str]] = None,
                 time_interval: int = 14,
                 horizon: int = 1,
                 window: int = 1,
                 standardize: bool = True,
                 stats_path: Optional[str] = None,
                 window_step: int = 1,
                 stack_z: bool = False,
                 return_mask: bool = True):
        """
        Custom Dataset for HDF5 files with multiple runs.
        Args:
            data_path (str): Path to the HDF5 file.
            run_keys (List[str]): List of HDF5 group keys, each corresponding to a data run for this split.
            time_steps_per_run (int): Number of time steps in each run. Set as -1 to use all time steps.
            transform (callable, optional): Optional transform to be applied on a sample.
            param_names (List[str], optional): List of variable names for channels.
            norm_param_names (List[str], optional): List of variable names to normalize and denormalize.
            time_interval (int): Interval between samples (in days). Default is 14 (2 weeks).
            horizon (int): Number of consecutive time steps to return per sample.
            window (int): Number of timesteps used as input (context). Total timesteps returned = window + horizon.
            standardize (bool): Whether to normalize data using precomputed statistics.
            stats_path (str, optional): Path to statistics file. If None, will look for default location.
            window_step (int): Step size distance between window indices. Increase to decrease number of data points.
            stack_z (bool): Whether to stack z-dimension for 2D interpolation.
            return_mask (bool): Whether to return a mask for valid data points when __getitem__ is called. 
                Masked regions will be set as `predictions_mask` in return dictionary. Note: in model.get_loss() this mask will be used in loss computation.
        """
        self.data_path = data_path
        self.run_keys = run_keys
        self.time_steps_per_run = time_steps_per_run
        self.transform = transform
        self.param_names = param_names
        #self.norm_param_names = norm_param_names
        #if self.norm_param_names is None:
        #    self.norm_param_names = self.param_names  # Use param_names if norm_param_names not provided
        #    print(f"[SOMAh5Dataset] norm_param_names not provided, using param_names: {self.param_names}")
        self.time_interval = time_interval
        self.horizon = horizon
        self.window = window
        self.standardize = standardize
        self.total_timesteps = window + horizon  # Total timesteps to return per sample
        self.window_step = window_step  # Step size distance between window indices
        
        # Don't keep HDF5 file open - open as needed to avoid lock issues
        # self.data = h5py.File(self.data_path, 'r')  # Open the HDF5 file
        self.data = None  # Will open file as needed in __getitem__

        # Load normalization statistics if standardization is enabled
        self.mean = None
        self.std = None
        self.mask = None
        self.stack_z = stack_z  # Whether to stack z-dimension for 2D interpolation
        self.return_mask = return_mask  # Whether to return a mask for valid data points
        
        if self.standardize:
            self._load_stats(stats_path)

        # Determine time_steps_per_run if needed
        if not self.run_keys:
            logger.warning(f"No run keys provided for dataset using HDF5 file: {data_path}. Dataset will be empty, unless setup() is called -- in which all keys will be used.")
            self._window_indices = []
        else:
            # Temporarily open HDF5 file to get metadata
            try:
                with h5py.File(self.data_path, 'r', locking=False) as hf:
                    if self.time_steps_per_run <= 0:
                        self.time_steps_per_run = hf[self.run_keys[0]].shape[0]
                        logger.info(f"Determined time_steps_per_run from HDF5 file: {self.time_steps_per_run}")
                    
                    # Build index: list of (run_idx, start_time_idx) for each valid window
                    # Each sample needs total_timesteps consecutive timesteps
                    self._window_indices = []
                    for run_idx, run_key in enumerate(self.run_keys):
                        if run_key not in hf:
                            logger.error(f"Run key '{run_key}' not found in HDF5 file {data_path}")
                            continue
                        n_steps = hf[run_key].shape[0]
                        for start in range(0, n_steps - self.total_timesteps + 1, self.window_step):
                            self._window_indices.append((run_idx, start))
                            
            except Exception as e:
                logger.error(f"Error accessing HDF5 file {data_path} during initialization: {e}")
                self._window_indices = []
                
        logger.info(f"Initialized SOMAh5Dataset with {len(self._window_indices)} samples (windows of window+horizon={self.total_timesteps}) from {len(self.run_keys)} runs.")

    def _load_stats(self, stats_path: Optional[str] = None):
        """Load normalization statistics from file."""
        if stats_path is None:
            # Default location relative to data file - try corrected stats first
            stats_path = os.path.join(os.path.dirname(self.data_path), "corrected_training_statistics_per_level_state.npz")
            # full_stats_path = os.path.join(os.path.dirname(self.data_path), "full_training_statistics_per_level_state.npz")
        
        if not os.path.exists(stats_path):
            logger.warning(f"Statistics file not found at {stats_path}. Standardization will be disabled.")
            self.standardize = False
            return
            
        try:
            stats = np.load(stats_path)
            self.mean = stats["mean"]  # Shape: (z, channels)
            self.std = stats["std"]    # Shape: (z, channels)
            
            # Expand dimensions to match data: (1, z, 1, 1, channels) for broadcasting
            self.mean = self.mean[None, :, None, None, :]
            self.std = self.std[None, :, None, None, :]
            
            logger.info(f"Loaded normalization statistics from {stats_path}")
            logger.info(f"Mean shape: {self.mean.shape}, Std shape: {self.std.shape}")
            logger.info(f"Mean value range: [{np.min(stats['mean']):.6f}, {np.max(stats['mean']):.6f}]")
            logger.info(f"Std value range: [{np.min(stats['std']):.6f}, {np.max(stats['std']):.6f}]")
            
            # Check for invalid statistics
            if np.any(np.isnan(self.mean)) or np.any(np.isnan(self.std)):
                logger.warning("Found NaN values in statistics, disabling standardization")
                self.standardize = False
                return
                
            if np.any(self.std <= 0):
                logger.warning("Found zero or negative std values, disabling standardization")
                self.standardize = False
                return
                
        except Exception as e:
            logger.error(f"Error loading statistics from {stats_path}: {e}")
            self.standardize = False

    def _transform_data(self, data: np.ndarray) -> np.ndarray:
        """
        Apply normalization and masking to data.
        
        Args:
            data: Input data with shape (time, z, y, x, channels)
            
        Returns:
            Transformed data with same shape
        """
        # Create mask for valid values (abs(value) < 1e30)
        valid_mask = np.abs(data) < 1e30
        self.mask = valid_mask  # Store mask for later use
        
        # Apply standardization if enabled
        if self.standardize and self.mean is not None and self.std is not None:
            # Only standardize output channels (not auxiliary channels like var_x)
            # Create a copy to avoid modifying the original
            normalized_data = data.copy()
            
            if hasattr(self, 'out_indices') and self.out_indices:
                # Only normalize the output channels
                for channel_idx in self.out_indices:
                    if channel_idx < data.shape[-1]:  # Safety check
                        channel_data = data[..., channel_idx:channel_idx+1]
                        channel_valid_mask = valid_mask[..., channel_idx:channel_idx+1]
                        channel_mean = self.mean[..., channel_idx:channel_idx+1]
                        channel_std = self.std[..., channel_idx:channel_idx+1]
                        
                        # Apply normalization only to this channel
                        normalized_channel = np.where(
                            channel_valid_mask, 
                            (channel_data - channel_mean) / channel_std, 
                            channel_data
                        )
                        normalized_data[..., channel_idx:channel_idx+1] = normalized_channel
                        
                logger.debug(f"Applied normalization to output channels: {self.out_indices}")
            else:
                # Fallback: normalize all channels (original behavior)
                logger.warning("out_indices not found, normalizing all channels as fallback")
                normalized_data = np.where(valid_mask, (data - self.mean) / self.std, data)
                
            data = normalized_data
            
        # Set invalid regions to 0 (following SOMA adjoint pattern)
        data = np.where(valid_mask, data, 0.0)
        
        return data

    def inverse_transform(self, data, variable_name: str = None, strict: bool = False):
        """
        Denormalize data for loss computation.

        Args:
            data: Normalized data tensor - single variable tensor, multi-channel tensor, or TensorDict
            variable_name: Name of the variable (to look up channel index)
            strict: If True, raises error for non-tensor data. If False, returns data unchanged.

        Returns:
            Denormalized data tensor or TensorDict
            
        Raises:
            TypeError: If strict=True and data is not a tensor or TensorDict
        """
        if not self.standardize or self.mean is None or self.std is None:
            return data

        # Handle TensorDict case (from model predictions)
        if hasattr(data, 'keys') and callable(getattr(data, 'keys')):
            # This is a TensorDict - denormalize each variable separately
            result = {}
            for var_name in data.keys():
                if var_name in self.param_names:
                    result[var_name] = self.inverse_transform(data[var_name], variable_name=var_name)
                else:
                    # Unknown variable, return as-is
                    result[var_name] = data[var_name]
            return type(data)(result)  # Preserve the original type (TensorDict)
        
        # Handle regular tensor case
        if not torch.is_tensor(data):
            if strict:
                raise TypeError(f"Expected tensor or TensorDict, got {type(data)}. "
                              f"Use strict=False to allow non-tensor data to pass through unchanged.")
            return data
            
        ndim = data.dim()
        # Detect z-dimension
        if ndim == 5:
            z_dim = -3
            c_dim = 1
        elif ndim == 4:
            z_dim = -3
            c_dim = 0
        elif ndim == 3:
            z_dim = -3
            c_dim = None
        else:
            logger.warning(f"Unexpected tensor rank {ndim} for data shape {data.shape}, assuming z at index -3")
            z_dim = -3
            c_dim = -4 if ndim >= 4 else None

        if variable_name is not None and self.param_names is not None:
            # Single variable denormalization
            try:
                channel_idx = self.param_names.index(variable_name)
            except ValueError:
                raise ValueError(f"Variable {variable_name} not found in param_names.")
            mean_np = self.mean[0, :, 0, 0, channel_idx]  # Shape: (z,)
            std_np = self.std[0, :, 0, 0, channel_idx]    # Shape: (z,)
            mean_torch = torch.from_numpy(mean_np).to(data.device, dtype=data.dtype)
            std_torch = torch.from_numpy(std_np).to(data.device, dtype=data.dtype)
            shape = [1] * ndim
            shape[z_dim] = len(mean_torch)
            mean_torch = mean_torch.view(shape)
            std_torch = std_torch.view(shape)
            return data * std_torch + mean_torch
        else:
            # Denormalize all channels using out_indices if available, otherwise all channels
            if c_dim is None:
                logger.error("Cannot denormalize all channels: channel dimension not found.")
                return data
            # Determine which channels to denormalize
            if hasattr(self, 'out_indices') and self.out_indices is not None and len(self.out_indices) > 0:
                channel_indices = self.out_indices
            else:
                # Denormalize all channels
                channel_indices = list(range(self.mean.shape[-1]))
            data_out = data.clone()
            for i, idx in enumerate(channel_indices):
                mean_np = self.mean[0, :, 0, 0, idx]  # Shape: (z,)
                std_np = self.std[0, :, 0, 0, idx]    # Shape: (z,)
                mean_torch = torch.from_numpy(mean_np).to(data.device, dtype=data.dtype)
                std_torch = torch.from_numpy(std_np).to(data.device, dtype=data.dtype)
                shape = [1] * ndim
                shape[z_dim] = len(mean_torch)
                mean_torch = mean_torch.view(shape)
                std_torch = std_torch.view(shape)
                # Index into channel dimension
                if ndim >= 4:
                    data_out[..., i, :, :, :] = data[..., i, :, :, :] * std_torch + mean_torch
                elif ndim == 3:
                    data_out[i, :, :] = data[i, :, :] * std_torch.squeeze() + mean_torch.squeeze()
            return data_out

    def __len__(self):
        return len(self._window_indices)

    def __getitem__(self, idx: int):
        """
        Retrieves a sample from the dataset and returns a dictionary with 'dynamics' and optionally 'dynamical_condition' keys.
        """
        
        if isinstance(idx, int):    
            if not (0 <= idx < len(self._window_indices)):
                raise IndexError(f"Index {idx} out of bounds for number of windows {len(self._window_indices)}")
            run_idx, start_time_idx = self._window_indices[idx]
            run_key = self.run_keys[run_idx]
        
        elif isinstance(idx, tuple) and len(idx) == 2:
            assert isinstance(idx[1], int), "Second element of index tuple must be an integer (run index)"
            if isinstance(idx[0], int):
                run_idx, start_time_idx = idx
                run_key = self.run_keys[run_idx]
            elif isinstance(idx[0], str):
                run_key = idx[0]
                if run_key not in self.run_keys:
                    raise KeyError(f"Run key '{run_key}' not found in run_keys")
                run_idx = self.run_keys.index(run_key)
                start_time_idx = idx[1]
                
            if not (0 <= run_idx < len(self.run_keys)):
                raise IndexError(f"Run index {run_idx} out of bounds for run keys {self.run_keys}")
            if not (0 <= start_time_idx < self.time_steps_per_run):
                raise IndexError(f"Start time index {start_time_idx} out of bounds for time steps per run {self.time_steps_per_run}")

        else:
            raise TypeError(f"Unsupported index type: {type(idx)}. Expected int or tuple of (run_idx, start_time_idx).")
        
        # Open HDF5 file for this specific read operation
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with h5py.File(self.data_path, 'r', locking=False) as hf:
                    data_for_window = hf[run_key][start_time_idx:start_time_idx+self.total_timesteps, ...]
                    # Convert to numpy array immediately while file is open
                    data_cleaned_np = np.array(data_for_window, dtype=np.float32)
                break  # Success, exit retry loop
            except OSError as e:
                if "524" in str(e) or "unable to lock file" in str(e).lower():
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(0.1 * (attempt + 1))  # Short retry delay
                        continue
                    else:
                        logger.error(f"Failed to read from HDF5 file after {max_retries} attempts: {e}")
                        raise
                else:
                    logger.error(f"Error reading HDF5 file {self.data_path} for run {run_key}: {e}")
                    raise
            except KeyError as e:
                logger.error(f"KeyError accessing HDF5 file: {e}. Run key: {run_key}")
                raise
            except Exception as e:
                logger.error(f"Error reading HDF5 file {self.data_path} for run {run_key}: {e}")
                raise
        
        # Apply normalization and masking
        data_cleaned_np = self._transform_data(data_cleaned_np)
        
        features_tensor = torch.from_numpy(data_cleaned_np).float()  # (horizon, Z, Y, X, P)
        # print(f"[SOMAh5Dataset] features_tensor (before permute) shape: {features_tensor.shape}")
        features_tensor = features_tensor.permute(0, 4, 1, 2, 3)  # (horizon, P, Z, Y, X)
        predictions_mask = None
        if self.return_mask:
            # Select only the first item in the horizon dimension for the mask
            predictions_mask = torch.from_numpy(self.mask).permute(0, 4, 1, 2, 3)  # (horizon, P, Z, Y, X)
            predictions_mask = predictions_mask[0:1] # Select first time step only and squeeze singleton dims
            #predictions_mask = predictions_mask  
            # print(f"[SOMAh5Dataset] features_tensor (after permute) shape: {features_tensor.shape}")
        assert all(s > 0 for s in features_tensor.shape), f"Zero-length dimension in features_tensor: {features_tensor.shape} (idx={idx}, run_key={run_key})"
        param_dict = {}
        predictions_mask_dict = {}
        if self.param_names is not None:
            for i, name in enumerate(self.param_names):
                param_dict[name] = features_tensor[:, i, ...]  # (horizon, Z, Y, X)
                if self.return_mask:
                    predictions_mask_dict[name] = predictions_mask[:, i, ...].squeeze(0) # Now shape: (P, Z, Y, X)
        else:
            for i in range(features_tensor.shape[1]):
                param_dict[f"channel_{i}"] = features_tensor[:, i, ...]
                if self.return_mask:
                    predictions_mask_dict[f"channel_{i}"] = predictions_mask[:, i, ...].squeeze(0)
        
        # If stack_z is True, average across z-axis (dim=2) and reduce to 2D grid
        if getattr(self, "stack_z", False):
            for k in param_dict:
                # param_dict[k] shape: (horizon, Z, Y, X)
                param_dict[k] = param_dict[k].mean(dim=1)  # Now shape: (horizon, Y, X)
            if self.return_mask:
                raise NotImplementedError("Masking not implemented for stacked z-dimension. Please implement if needed.")

        if self.transform:
            raise NotImplementedError("Transform function is not implemented in SOMAh5Dataset. Please implement it if needed.")
            # for k in param_dict:
            #     param_dict[k] = self.transform(param_dict[k])

        # Compose output dictionary similar to the provided function
        tensors = param_dict.copy()
        data = {}
        if hasattr(self, 'forcing_names') and self.forcing_names is not None:
            forcings = {k: tensors.pop(k) for k in list(tensors.keys()) if k in self.forcing_names}
            if hasattr(self, 'forcing_packer') and hasattr(self, 'forcing_normalizer'):
                forcings = self.forcing_packer.pack(self.forcing_normalizer.normalize(forcings))
            data = {"dynamics": tensors, "dynamical_condition": forcings}
        else:
            data = {"dynamics": tensors}
            
        if self.return_mask:
            # Add mask to the output dictionary
            data['predictions_mask'] = predictions_mask_dict
        return data


class MyCustomDataModule(BaseDataModule): # Ensure it inherits from BaseDataModule
    
    """
    ----------------------------------------------------------------------------------------------------------
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """
    def __init__(self,
                 data_path: str,
                 model_config: Optional[DictConfig] = None, 
                 param_names: Optional[List[Union[str, int]]] = ['timeDaily_avg_layerThickness',
                                                    'timeDaily_avg_velocityZonal',
                                                    'timeDaily_avg_velocityMeridional',
                                                    'timeDaily_avg_activeTracers_temperature',
                                                    'timeDaily_avg_activeTracers_salinity',
                                                    'var_x'] ,
                 forcing_names: Optional[List[Union[str, int]]] = None, # either names in param_names or indices
                 in_names: Optional[List[Union[str, int]]] = None,
                 out_names: Optional[List[Union[str, int]]] = None,
                 
                 horizon: int = 10, # number of consecutive time steps to return per sample
                 window: int = 1, # number of timesteps used as input (context)
                 
                 batch_size: int = 32,
                 eval_batch_size: Optional[int] = None, 
                 num_workers: int = 4,
                 
                 train_val_test_split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 seed_data: int = 42,
                 time_steps_per_run: Optional[int] = None, 
                 #var_param_channel_index: Optional[int] = None, 
                 pin_memory: bool = True,
                 data_dir: Optional[str] = None, 
                 time_interval: int = 14, # time between each step,
                 prescriber = None, #Optional[Prescriber] = None,
                 mesh_file_path: Optional[str] = None,  # Path to mesh file for area weights
                 use_mesh_area_weights: bool = False,  # Whether to load area weights from mesh file
                 standardize: bool = True,  # Whether to normalize data using precomputed statistics
                 stats_path: Optional[str] = None,  # Path to statistics file
                 window_step: int = 1,  # Step size distance between window indices. Increase to decrease number of data points
                 stack_z: bool = False,  # Whether to stack z-dimension for 2D interpolation
                 **kwargs): 
        """
        Args:
            data_path: Path to HDF5 data file
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            horizon: Prediction horizon
            window: Input window size
            time_interval: Time interval between samples
            prediction_horizon: Validation horizon 1
            prediction_horizon_long: Validation horizon 2
            train_val_test_split: Split ratios for train/val/test
            seed: Random seed for data splitting
            standardize: Whether to standardize the data
            stats_path: Path to precomputed statistics
            pin_memory: Whether to pin memory in data loaders
        """
        print(f"[MyCustomDataModule] Initializing MyCustomDataModule!")
        print(f"[MyCustomDataModule] Class name: {self.__class__.__name__}")
        print(f"[MyCustomDataModule] Module: {self.__class__.__module__}")
        
        # Extract parameters that are not accepted by parent class
        self.mesh_file_path = mesh_file_path
        self.use_mesh_area_weights = use_mesh_area_weights
        self.standardize = standardize
        self.stats_path = stats_path
        self.stack_z = stack_z  # Whether to stack z-dimension for 2D interpolation
        
        effective_data_dir = data_dir if data_dir is not None else os.path.dirname(data_path)
        super().__init__(data_dir=effective_data_dir, 
                         model_config=model_config, 
                         batch_size=batch_size, 
                         eval_batch_size=eval_batch_size or batch_size, 
                         num_workers=num_workers, 
                         pin_memory=pin_memory, 
                         **kwargs)
        
        # Save all relevant hyperparameters for reproducibility
        self.save_hyperparameters(
            "data_path",
            "train_val_test_split_ratios",
            # "seed",  # Exclude seed to avoid Lightning hparams conflict # TODO check if this effect reproducability for data divisions
            "time_steps_per_run",
            "param_names",
            "forcing_names",
            "in_names",
            "out_names",
            "horizon",
            "window",
            "time_interval",
            "mesh_file_path",
            "use_mesh_area_weights",
            "standardize",
            "stats_path",
            "window_step",)
        
        # Store param_names and resolve indices for in/out/forcing
        self.param_names = param_names
        self._param_name_to_index = {name: idx for idx, name in enumerate(param_names)} if param_names else {}
        self.use_mesh_area_weights = use_mesh_area_weights

        if prescriber is not None:
            logger.warning("Prescriber is not used in MyCustomDataModule. It is only for compatibility with BaseDataModule.")

        def resolve_indices(names):
            if names is None:
                return None
            indices = []
            for n in names:
                if isinstance(n, int):
                    indices.append(n)
                elif isinstance(n, str):
                    if n in self._param_name_to_index:
                        indices.append(self._param_name_to_index[n])
                    else:
                        raise ValueError(f"Parameter name '{n}' not found in param_names.")
                else:
                    raise ValueError(f"Parameter specifier '{n}' must be int or str.")
            return indices

        self.in_indices = resolve_indices(in_names) if in_names is not None else list(range(len(param_names)))
        self.out_indices = resolve_indices(out_names) if out_names is not None else []
        self.forcing_indices = resolve_indices(forcing_names) if forcing_names is not None else []
        self.out_names = out_names

        # These will be determined in setup() or from hparams
        self._time_steps_per_run: Optional[int] = time_steps_per_run
        self._n_input_channels: Optional[int] = None
        self._data_spatial_shape_zyx: Optional[Tuple[int, int, int]] = None
        
        # Attributes for datasets, aligned with BaseDataModule's expected attributes
        self._data_train: Optional[SOMAh5Dataset] = None 
        self._data_val: Optional[SOMAh5Dataset] = None
        self._data_test: Optional[SOMAh5Dataset] = None
        self._data_predict: Optional[SOMAh5Dataset] = None

        self.all_run_keys: List[str] = []
        self.time_interval = time_interval
        self.window_step = window_step  # Step size distance between window indices

    def _get_all_run_keys(self) -> List[str]:
        """Helper to get all top-level group keys from the HDF5 file, assumed to be runs."""
        import time
        max_retries = 5
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Use file locking mode that's more compatible with distributed systems
                with h5py.File(self.hparams.data_path, 'r', locking=False) as hf:
                    return list(hf.keys())
            except OSError as e:
                if "524" in str(e) or "unable to lock file" in str(e).lower():
                    logger.warning(f"File lock error on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        logger.error(f"Failed to access HDF5 file after {max_retries} attempts")
                        return []
                else:
                    logger.error(f"Could not read run keys from HDF5 file {self.hparams.data_path}: {e}")
                    return []
            except Exception as e:
                logger.error(f"Could not read run keys from HDF5 file {self.hparams.data_path}: {e}")
                return []

    def prepare_data(self):
        """Checks for data file existence. Called once per node. Not for state assignment."""
        if not os.path.exists(self.hparams.data_path):
            raise FileNotFoundError(f"HDF5 file not found: {self.hparams.data_path}")
        logger.info(f"HDF5 file found at: {self.hparams.data_path}")
        # BaseDataModule might have its own prepare_data logic, consider calling super().prepare_data()
        # super().prepare_data() # If BaseDataModule implements it and it's relevant

    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set internal variables: self._data_train, self._data_val, self._data_test.
        Determines data shape and channel info from the HDF5 file.
        """
        # super().setup(stage) # If BaseDataModule implements setup and it's relevant before this logic

        if not self.all_run_keys: 
            self.all_run_keys = self._get_all_run_keys()

        if not self.all_run_keys:
            logger.error("No run keys found in HDF5 file. Cannot setup datasets.")
            return

        if self._time_steps_per_run is None or self._n_input_channels is None or self._data_spatial_shape_zyx is None:
            try:
                with h5py.File(self.hparams.data_path, 'r', locking=False) as hf:
                    first_run_key = self.all_run_keys[0]
                    data_shape_from_file = hf[first_run_key].shape 
                    
                    determined_time_steps = data_shape_from_file[0]
                    if self.hparams.time_steps_per_run is not None and self.hparams.time_steps_per_run != determined_time_steps:
                        logger.warning(f"User-provided time_steps_per_run ({self.hparams.time_steps_per_run}) "
                                       f"differs from HDF5 file ({determined_time_steps}) for run {first_run_key}. "
                                       f"Using value from HDF5: {determined_time_steps}.")
                        self._time_steps_per_run = determined_time_steps
                    elif self.hparams.time_steps_per_run is None:
                        self._time_steps_per_run = determined_time_steps
                    # If user provided and matches, or if user didn't provide, self._time_steps_per_run is now set.

                    self._data_spatial_shape_zyx = data_shape_from_file[1:4] 
                    self._n_input_channels = data_shape_from_file[4] 

                    logger.info(f"Determined data characteristics from HDF5: "
                                f"time_steps_per_run={self._time_steps_per_run}, "
                                f"spatial_shape_zyx={self._data_spatial_shape_zyx}, "
                                f"n_input_channels={self._n_input_channels}")
            except Exception as e:
                logger.error(f"Could not determine data characteristics from HDF5 file {self.hparams.data_path}: {e}")
                if self._time_steps_per_run is None: self._time_steps_per_run = 417 
                if self._n_input_channels is None: self._n_input_channels = 6 
                if self._data_spatial_shape_zyx is None: self._data_spatial_shape_zyx = (60,100,100)
                logger.warning("Falling back to default/user-provided data characteristics for setup.")

        # Create packers for input and output data
        if not hasattr(self, 'in_packer'):
            in_names = [self.param_names[i] for i in self.in_indices] if self.param_names and self.in_indices else []
            if in_names:
                channel_axis = -4  # For tensors of shape (batch, time, channels, z, y, x)
                self.in_packer = Packer(in_names, axis=channel_axis)
            
        if not hasattr(self, 'out_packer'):
            out_names = [self.param_names[i] for i in self.out_indices] if self.param_names and self.out_indices else []
            if not out_names and self.param_names:
                # Default to first 5 parameters if out_names not specified (excluding var_x)
                out_names = [name for name in self.param_names if name != 'var_x'][:5]
                self.out_indices = [self._param_name_to_index[name] for name in out_names]
            if out_names:
                channel_axis = -4  # For tensors of shape (batch, time, channels, z, y, x)
                self.out_packer = Packer(out_names, axis=channel_axis)

        if self._time_steps_per_run is None or self._time_steps_per_run <= 0:
            # This should ideally use the hparams value if available and valid, or the inferred one.
            # The logic above tries to set self._time_steps_per_run from hparams or inference.
            # If it's still None or invalid here, it's a problem.
            raise ValueError(f"time_steps_per_run is invalid or could not be determined: {self._time_steps_per_run}")

        #np.random.seed(self.hparams.seed) 
        shuffled_run_keys = np.random.permutation(self.all_run_keys)

        num_runs = len(shuffled_run_keys)
        assert self.hparams.train_val_test_split_ratios is not None, \
            "train_val_test_split_ratios must be provided in hparams."
        train_ratio, val_ratio, _ = self.hparams.train_val_test_split_ratios
        logger.info(f"Using train/val/test split ratios: {train_ratio}, {val_ratio}, {1 - train_ratio - val_ratio}")

        train_idx_end = int(num_runs * train_ratio)
        val_idx_end = train_idx_end + int(num_runs * val_ratio)

        train_run_keys = list(shuffled_run_keys[:train_idx_end])
        val_run_keys = list(shuffled_run_keys[train_idx_end:val_idx_end])
        test_run_keys = list(shuffled_run_keys[val_idx_end:])
        
        logger.info(f"Total runs: {num_runs}. Splitting into: "
                    f"Train runs: {len(train_run_keys)}, "
                    f"Val runs: {len(val_run_keys)}, "
                    f"Test runs: {len(test_run_keys)}")
        logger.info("="*20)
        logger.info(f"Train run keys: {train_run_keys}")
        logger.info(f"Validation run keys: {val_run_keys}")
        logger.info(f"Test run keys: {test_run_keys}")
        logger.info("="*20)

        dataset_common_args = {
            'data_path': self.hparams.data_path, 
            'time_steps_per_run': self._time_steps_per_run,
            'param_names': self.param_names,
            'time_interval': self.hparams.time_interval,
            'horizon': self.hparams.horizon,
            'window': getattr(self.hparams, 'window', 1),  # Default window size of 1
            'standardize': self.standardize,
            'stats_path': self.stats_path,
            'stack_z': self.stack_z,
            'window_step': self.window_step,  # Default step size of 1
        }

        if stage == 'fit' or stage is None:
            self._data_train = SOMAh5Dataset(run_keys=train_run_keys, **dataset_common_args)
            self._data_val = SOMAh5Dataset(run_keys=val_run_keys, **dataset_common_args)
            # Pass out_indices to datasets for channel-selective normalization
            if hasattr(self, 'out_indices') and self.out_indices is not None:
                self._data_train.out_indices = self.out_indices
                self._data_val.out_indices = self.out_indices
        if stage == 'test' or stage is None:
            self._data_test = SOMAh5Dataset(run_keys=test_run_keys, **dataset_common_args)
            if hasattr(self, 'out_indices') and self.out_indices is not None:
                self._data_test.out_indices = self.out_indices
        if stage == 'predict' or stage is None: 
            self._data_predict = SOMAh5Dataset(run_keys=self.all_run_keys, **dataset_common_args)
            if hasattr(self, 'out_indices') and self.out_indices is not None:
                self._data_predict.out_indices = self.out_indices

    def train_dataloader(self) -> DataLoader:
        if not self._data_train:
            logger.error("Train dataset not initialized. Call setup(stage='fit') first.")
            self.setup(stage='fit') # Attempt to setup if not already
            if not self._data_train: # Check again
                raise RuntimeError("Train dataset still not initialized after attempting setup.")
        return DataLoader(
            self._data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True, 
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0
        )

    def val_dataloader(self) -> Union[DataLoader, Sequence[DataLoader]]:
        '''
        Construct and return a DataLoader for the validation dataset.
        '''
        if not self._data_val:
            logger.error("Validation dataset not initialized. Call setup(stage='fit') first.")
            self.setup(stage='fit') # Attempt to setup if not already
            if not self._data_val: # Check again
                raise RuntimeError("Validation dataset still not initialized after attempting setup.")
        return DataLoader(
            self._data_val,
            batch_size=self.hparams.eval_batch_size or self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0
        )

    def test_dataloader(self) -> Union[DataLoader, Sequence[DataLoader]]: 
        '''
        Construct and return a DataLoader for the test dataset.
        '''
        if not self._data_test:
            logger.error("Test dataset not initialized. Call setup(stage='test') first.")
            self.setup(stage='test') # Attempt to setup if not already
            if not self._data_test: # Check again
                raise RuntimeError("Test dataset still not initialized after attempting setup.")
        return DataLoader(
            self._data_test,
            batch_size=self.hparams.eval_batch_size or self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0
        )

    @property
    def n_input_channels(self) -> int:
        '''
        Number of input channels based on in_indices or the total number of channels in the dataset.
        If in_indices is specified, it returns the length of in_indices.
        Otherwise, it returns the total number of channels in the dataset.
        '''
        if self.in_indices is not None:
            return len(self.in_indices)
        if self._n_input_channels is None:
            logger.debug("n_input_channels accessed before being set. Attempting to run setup().")
            self.setup() 
            if self._n_input_channels is None:
                raise RuntimeError("Could not determine n_input_channels even after calling setup(). Ensure HDF5 file is valid and accessible.")
        return self._n_input_channels

    @property
    def data_shape(self) -> Tuple[int, int, int, int]: 
        '''
        Returns the shape of the data as a tuple (C, Z, Y, X) where:
        - C is the number of input channels (length of in_indices if specified, otherwise total channels)
        - Z, Y, X are the spatial dimensions of the data.
        If in_indices is specified, it returns the shape based on the number of in_indices.
        Otherwise, it returns the shape based on the total number of channels in the dataset.
        '''
        # Data shape is (input_channels, Z, Y, X)
        if (self.in_indices is not None and self._data_spatial_shape_zyx is not None):
            return (len(self.in_indices), *self._data_spatial_shape_zyx)
        if self._n_input_channels is None or self._data_spatial_shape_zyx is None:
            logger.debug("data_shape accessed before components are set. Attempting to run setup().")
            self.setup() 
            if self._n_input_channels is None or self._data_spatial_shape_zyx is None:
                raise RuntimeError("Could not determine data_shape components even after calling setup(). Ensure HDF5 file is valid and accessible.")
        return (self._n_input_channels, *self._data_spatial_shape_zyx)

    def get_static_features(self) -> Optional[torch.Tensor]:
        logger.info("MyCustomDataModule.get_static_features() called. No separate static features are provided.")
        return None

    def get_variable_names(self) -> List[str]:
        '''
        Returns a list of variable names based on in_indices or param_names.
        If in_indices is specified, it returns the names corresponding to those indices.
        Otherwise, it returns the names from param_names.
        If neither is specified, it returns generic channel names like "channel_0", "channel_1", etc.
        '''
        # Prefer param_names if available, otherwise fallback to generic channel names
        if self.param_names is not None:
            names = [self.param_names[i] if isinstance(i, int) and i < len(self.param_names) else str(i) for i in self.in_indices] if self.in_indices is not None else list(self.param_names)
            return names
        if self._n_input_channels is None:
            logger.debug("get_variable_names accessed before n_input_channels is set. Attempting to run setup().")
            self.setup()
            if self._n_input_channels is None:
                logger.error("Could not determine n_input_channels for get_variable_names.")
                return []
        return [f"channel_{i}" for i in range(self._n_input_channels)]

    def get_lat_lon_grid(self) -> Optional[Dict[str, np.ndarray]]:
        logger.warning("MyCustomDataModule.get_lat_lon_grid() not implemented, returning None.")
        return None

    def get_static_features_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        logger.warning("MyCustomDataModule.get_static_features_dict() not implemented, returning None.")
        return None

    def split_channels_to_variables(self, tensor_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Split a multi-channel tensor into a dictionary of named variables.
        
        Args:
            tensor_data: Tensor of shape (..., C, Z, Y, X) where C is the number of channels
            
        Returns:
            Dictionary mapping variable names to tensors of shape (..., Z, Y, X)
        """
        if not hasattr(self, 'param_names') or self.param_names is None:
            # Fallback to generic names if param_names not available
            n_channels = tensor_data.shape[-4]  # Channel dimension is -4 for (..., C, Z, Y, X)
            param_names = [f"var_{i}" for i in range(n_channels)]
        else:
            param_names = self.param_names
            
        result = {}
        for i, var_name in enumerate(param_names):
            # Extract channel i: tensor[..., i, :, :, :]
            result[var_name] = tensor_data[..., i, :, :, :]
            
        return result

    def get_inverse_transform(self):
        """
        Get the inverse transform function for denormalizing data in loss calculation.
        
        Returns:
            Function that can denormalize data, or None if no normalization is applied.
        """
        if hasattr(self, '_data_train') and self._data_train is not None:
            return self._data_train.inverse_transform
        elif hasattr(self, '_data_val') and self._data_val is not None:
            return self._data_val.inverse_transform
        elif hasattr(self, '_data_test') and self._data_test is not None:
            return self._data_test.inverse_transform
        else:
            logger.warning("No dataset available to get inverse transform from")
            return None

    def denormalize_for_loss(self, data: torch.Tensor, variable_name: str = None) -> torch.Tensor:
        """
        Denormalize data for loss computation.
        
        Args:
            data: Normalized data tensor
            variable_name: Name of the variable (for proper denormalization)
            
        Returns:
            Denormalized data tensor
        """
        inverse_transform = self.get_inverse_transform()
        if inverse_transform is not None:
            return inverse_transform(data, variable_name=variable_name)
        else:
            return data

    def get_epoch_aggregators(
        self,
        split: str,
        is_ensemble: bool,
        dataloader_idx: int = 0,
        experiment_type: str = None,
        device: torch.device = None,
        verbose: bool = True,
    ) -> Dict[str, OneStepAggregator]:
        """Return the epoch aggregators for the given split."""
        print(f"[get_epoch_aggregators] Called with split='{split}', experiment_type='{experiment_type}'")
        
        # Calculate proper area weights for horizontal dimensions (Y, X)
        area_weights = self.calculate_area_weights(use_mesh_file=self.use_mesh_area_weights)
        if device is not None:
            area_weights = to_torch_and_device(area_weights, device)
        
        aggr_kwargs = dict(area_weights=area_weights, is_ensemble=is_ensemble)
        aggregators = {}
        
        # Determine horizon range based on experiment type
        if "interpolation" in experiment_type.lower():
            horizon_range = range(1, self.hparams.horizon)
        else:
            # For forecasting experiments
            horizon_range = range(1, self.hparams.horizon + 1)
        
        # Make aggregators for each time step
        for h in horizon_range:
            aggregators[f"t{h}"] = OneStepAggregator(
                use_snapshot_aggregator=False,  # Disable snapshots for now to save memory
                record_normed=False, # disable area normalization since grid is uniform
                record_abs_values=True,
                verbose=verbose and (h == 1),
                name=f"t{h}",
                **aggr_kwargs,
            )
        
        # Always add time mean aggregator - provides valuable MSE metrics for all experiments and splits
        aggregators["time_mean"] = SOMATimeMeanAggregator(**aggr_kwargs, name="time_mean")
        
        return aggregators

    @property
    def spatial_shape_zyx(self) -> Tuple[int, int, int]:
        '''
        Returns the spatial shape of the data as a tuple (Z, Y, X).
        '''
        if self._data_spatial_shape_zyx is None:
            logger.debug("spatial_shape_zyx accessed before being set. Attempting to run setup().")
            self.setup() 
            if self._data_spatial_shape_zyx is None:
                raise RuntimeError("Could not determine spatial_shape_zyx even after calling setup(). Ensure HDF5 file is valid and accessible.")
        return self._data_spatial_shape_zyx

    def calculate_area_weights(self, use_mesh_file: bool = False) -> torch.Tensor:
        """
        Calculate proper area weights for the grid.
        
        Args:
            use_mesh_file: If True, loads area weights from the original mesh file.
                          If False (default), uses uniform weights since SOMA grid is uniform.
        
        For SOMA dataset, all grid cells are uniform and have the same area,
        so uniform weights are appropriate. The mesh file option is kept for
        other datasets that may have non-uniform grids.
        """
        z, y, x = self.spatial_shape_zyx
        
        if use_mesh_file:
            try:
                # Load actual area weights from original mesh
                area_weights = self._load_area_weights_from_file()
                logger.info(f"Loaded actual area weights from mesh file: shape {area_weights.shape}")
                return area_weights
            except Exception as e:
                logger.warning(f"Failed to load area weights from file: {e}")
                logger.warning("Falling back to uniform area weights")
        
        # Use uniform weights (default for SOMA since grid is uniform)
        area_weights = torch.ones((y, x), dtype=torch.float32)
        logger.info(f"Using uniform area weights for uniform grid: shape {area_weights.shape}")
        
        return area_weights
    
    def _load_area_weights_from_file(self) -> torch.Tensor:
        """
        Load area weights from the original mesh file and interpolate to current grid.
        """
        import xarray as xr
        from scipy.spatial import cKDTree
        import numpy as np
        
        # Path to the original mesh file
        mesh_file = "/global/homes/a/abgulhan/WORKING/git_repos/spherical-dyffusion/initial_state.nc"
        
        # Load the mesh data
        ds = xr.open_dataset(mesh_file)
        
        # Get original mesh coordinates and areas
        lat_orig = ds['latCell'].values  # radians
        lon_orig = ds['lonCell'].values  # radians  
        area_orig = ds['areaCell'].values  # m²
        
        # Convert to degrees for easier handling
        lat_orig_deg = np.rad2deg(lat_orig)
        lon_orig_deg = np.rad2deg(lon_orig)
        
        # Create regular grid coordinates for current 100x100 grid
        z, y, x = self.spatial_shape_zyx
        
        # Estimate the domain bounds from original mesh
        lat_min, lat_max = lat_orig_deg.min(), lat_orig_deg.max()
        lon_min, lon_max = lon_orig_deg.min(), lon_orig_deg.max()
        
        # Create regular grid
        lat_grid = np.linspace(lat_min, lat_max, y)
        lon_grid = np.linspace(lon_min, lon_max, x)
        lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
        
        # Flatten grid points for interpolation
        grid_points = np.column_stack([lat_2d.ravel(), lon_2d.ravel()])
        orig_points = np.column_stack([lat_orig_deg, lon_orig_deg])
        
        # Use KDTree for nearest neighbor interpolation
        tree = cKDTree(orig_points)
        distances, indices = tree.query(grid_points)
        
        # Interpolate area values
        area_interp = area_orig[indices].reshape(y, x)
        
        # Normalize to relative weights (optional - makes values more manageable)
        area_weights = area_interp / area_interp.mean()
        
        # Convert to torch tensor
        area_weights = torch.tensor(area_weights, dtype=torch.float32)
        
        logger.info(f"Interpolated area weights: min={area_weights.min():.3f}, max={area_weights.max():.3f}, mean={area_weights.mean():.3f}")
        
        return area_weights

class SOMATimeMeanAggregator(TimeMeanAggregator):
    """
    Custom TimeMeanAggregator for SOMA data that handles proper dimension reduction.
    
    The standard TimeMeanAggregator assumes that metrics functions return scalars when
    called on individual variables. However, our SOMA data structure causes metrics 
    to return tensors that need dimension reduction.
    """
    
    def __init__(self, **kwargs):
        print("[SOMATimeMeanAggregator] Initializing SOMATimeMeanAggregator!")
        super().__init__(**kwargs)
    
    @torch.inference_mode()
    def _record_batch(self, target_data, gen_data):
        print(f"[SOMATimeMeanAggregator] Recording batch, current n_batches: {self._n_batches}")
        return super()._record_batch(target_data, gen_data)
    
    def get_logs(self, **kwargs):
        print(f"[SOMATimeMeanAggregator] get_logs() called with {self._n_batches} batches recorded")
        return super().get_logs(**kwargs)

    @torch.inference_mode()
    def _get_logs(self, **kwargs) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Returns logs as can be reported to WandB.
        """
        if self._n_batches == 0:
            print(f"[SOMATimeMeanAggregator] No data recorded yet, returning empty logs")
            return {}, {}
        
        area_weights = self._area_weights
        logs = {}
        
        for name in self._gen_data.keys():
            gen = self._gen_data[name] / self._n_batches
            target = self._target_data[name] / self._n_batches
            
            if self._is_ensemble:
                gen_ens_mean = gen.mean(dim=0)
                logs[f"rmse_member_avg/{name}"] = np.mean(
                    [
                        float(
                            metrics.root_mean_squared_error(
                                predicted=gen[i], truth=target, weights=area_weights,
                                dim=(-2, -1)  # Reduce over spatial dimensions
                            ).mean().cpu().numpy()  # Take mean over remaining dimensions
                        )
                        for i in range(gen.shape[0])
                    ]
                )
                logs[f"mse_member_avg/{name}"] = np.mean(
                    [
                        float(
                            metrics.mean_squared_error(
                                predicted=gen[i], truth=target, weights=area_weights,
                                dim=(-2, -1)
                            ).mean().cpu().numpy()
                        )
                        for i in range(gen.shape[0])
                    ]
                )
            else:
                gen_ens_mean = gen

            # Call metrics with proper dimension reduction
            logs[f"rmse/{name}"] = float(
                metrics.root_mean_squared_error(
                    predicted=gen_ens_mean, truth=target, weights=area_weights,
                    dim=(-2, -1)  # Reduce over spatial dimensions
                ).mean().cpu().numpy()  # Take mean over any remaining dimensions
            )
            logs[f"mse/{name}"] = float(
                metrics.mean_squared_error(
                    predicted=gen_ens_mean, truth=target, weights=area_weights,
                    dim=(-2, -1)
                ).mean().cpu().numpy()
            )

            logs[f"bias/{name}"] = float(
                metrics.time_and_global_mean_bias(
                    predicted=gen_ens_mean, truth=target, weights=area_weights,
                    time_dim=0, spatial_dims=(-2, -1)  # Specify time and spatial dims
                ).mean().cpu().numpy()  # Take mean over any remaining dimensions
            )
            
            logs[f"crps/{name}"] = float(
                metrics.crps_ensemble(
                    predicted=gen, truth=target, weights=area_weights,
                    dim=(-2, -1)  # Reduce over spatial dimensions
                ).mean().cpu().numpy()  # Take mean over any remaining dimensions
            )
        
        # # Debug print to verify MSE is in logs
        # print(f"[SOMATimeMeanAggregator] Generated logs keys: {list(logs.keys())}")
        # mse_keys = [k for k in logs.keys() if 'mse' in k]
        # print(f"[SOMATimeMeanAggregator] MSE keys found: {mse_keys}")
        
        return logs, {}

# Example usage (for testing the datamodule directly, not part of the library)
if __name__ == '__main__':
    import argparse

    # Dummy AbstractDataModule for local testing if needed
    class AbstractDataModule(pl.LightningDataModule):
        def get_static_features(self): return None
        def get_variable_names(self): return []
        @property
        def n_input_channels(self):
            return 0

        def data_shape(self):
            return (0, 0, 0, 0)

    parser = argparse.ArgumentParser(description="Test MyCustomDataModule with dummy or real HDF5 dataset.")
    parser.add_argument('data_path', nargs='?', default=None, help='Path to the HDF5 dataset file. If not provided, a dummy file will be created and used.')
    args = parser.parse_args()

    print("Running MyCustomDataModule example...")
    dummy_h5_path = 'dummy_test_data.h5'
    num_dummy_runs = 5
    time_steps = 10 
    Z, Y, X, P = 60, 100, 100, 6

    if args.data_path is not None:
        data_path = args.data_path
        if not os.path.exists(data_path):
            print(f"Provided HDF5 file does not exist: {data_path}")
            exit(1)
    else:
        data_path = dummy_h5_path
        if not os.path.exists(dummy_h5_path):
            print(f"Creating dummy HDF5 file: {dummy_h5_path}")
            with h5py.File(dummy_h5_path, 'w') as hf:
                for i in range(num_dummy_runs):
                    run_key = f'run_{i}'
                    run_data = np.random.rand(time_steps, Z, Y, X, P).astype(np.float32)
                    run_data[..., P-1] = float(i) 
                    if i == 0 and time_steps > 0:
                        run_data[0, 0, 0, 0, 0] = 2e30
                        run_data[0, 1, 0, 0, 0] = -3e30
                    hf.create_dataset(run_key, data=run_data)
            print("Dummy HDF5 file created.")
        else:
            print(f"Dummy HDF5 file {dummy_h5_path} already exists.")

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        dm = MyCustomDataModule(
            data_path=data_path,
            batch_size=2,
            num_workers=0, 
            var_param_channel_index= (P-1) if P > 0 else None # Make it robust for P=0 case
        )
        dm.prepare_data()
        dm.setup(stage='fit')

        print(f"Number of input channels: {dm.n_input_channels}")
        print(f"Data shape (C, Z, Y, X): {dm.data_shape}")
        print(f"Variable names: {dm.get_variable_names()}")

        if dm._data_train and len(dm._data_train) > 0:
            print(f"Number of training samples: {len(dm._data_train)}")
            train_loader = dm.train_dataloader()
            for i, features in enumerate(train_loader):
                print(f"Train Batch {i+1}:")
                print(f"  Features shape: {features.shape}")
                print(f"  Feature mean: {features.mean()}, Feature NaN count: {torch.isnan(features).sum()}")
                if i == 1:
                    break
        else:
            print("No training data loaded.")

        if dm._data_val and len(dm._data_val) > 0:
            print(f"Number of validation samples: {len(dm._data_val)}")
            val_loader = dm.val_dataloader()
            for i, features in enumerate(val_loader):
                print(f"Val Batch {i+1}:")
                print(f"  Features shape: {features.shape}")
                if i == 1:
                    break
        else:
            print("No validation data loaded.")
            
    except Exception as e:
        print(f"An error occurred during the example run: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Example finished. If you re-run, delete dummy_test_data.h5 manually if needed or uncomment cleanup.")

