import h5py
import torch
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys

# Import modules with fallback handling
try:
    from src.experiment_types.interpolation import InterpolationExperiment
    from src.datamodules.SOMA_datamodule import MyCustomDataModule
    from src.utilities.utils import to_tensordict
except ModuleNotFoundError:
    # Try relative import if running as a module
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()  # fallback if __file__ not defined
    project_root = os.path.abspath(os.path.join(current_dir, '..', 'git_repos', 'spherical-dyffusion'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.experiment_types.interpolation import InterpolationExperiment
    from src.datamodules.SOMA_datamodule import MyCustomDataModule
    from src.utilities.utils import to_tensordict


# # Import SOMA interpolation with availability check
# try:
#     from evaluation.evaluate_soma_interpolator import interpolate
#     INTERPOLATION_AVAILABLE = True
# except ImportError as e:
#     INTERPOLATION_AVAILABLE = False

# Load global dataset for backward compatibility
# (main_interpolated_animation loads its own data as needed)
# file_path = '/global/homes/a/abgulhan/WORKING/ml_converted/data-gm-year7-22-biweekly.hdf5'
# try:
#     f = h5py.File(file_path, 'r', locking=False)
#     fwd = "forward_2"
#     data_shape = f[fwd].shape
#     time_steps, depth_layers, y_size, x_size, n_vars = data_shape
    
#     # Variable list for reference (time_step, z=60, y=100, x=100, parameters=6)
#     var_list = [
#         'timeDaily_avg_layerThickness',
#         'timeDaily_avg_velocityZonal', 
#         'timeDaily_avg_velocityMeridional',
#         'timeDaily_avg_activeTracers_temperature',
#         'timeDaily_avg_activeTracers_salinity',
#         'var_x'  # all values are same for f[fwd][..., 5]
#     ]
# except Exception as e:
#     print(f"⚠️ Warning: Could not load global dataset: {e}")
#     f = None
#     time_steps = y_size = x_size = 0


# def clean_outliers(data, threshold=1e30):
#     """
#     Clean outliers in the data by replacing values greater than the threshold with NaN.
#     """
#     data = np.where(np.abs(data) > threshold, np.nan, data)
#     return data




class MidpointFocusNorm(mcolors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0.5, gamma=0.3, emphasis_below=True, clip=False):
        """
        Custom normalization with proper below-midpoint emphasis.
        
        Args:
            vmin/vmax: Data range
            midpoint: Target value (0.0-1.0 normalized)
            gamma: Emphasis strength (lower = more emphasis)
            emphasis_below: Emphasize values below midpoint
        """
        self.midpoint = midpoint
        self.gamma = gamma
        self.emphasis_below = emphasis_below
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Normalize data to [0,1] range
        x = (value - self.vmin) / (self.vmax - self.vmin)
        
        # Handle special cases
        if self.midpoint <= 0 or self.midpoint >= 1:
            return x ** self.gamma
        
        # Apply proper transformation for below-midpoint emphasis
        if self.emphasis_below:
            # CORRECTED TRANSFORMATION
            with np.errstate(invalid='ignore'):
                return np.where(
                    x < self.midpoint,
                    (x / self.midpoint) ** self.gamma,  # FIXED: Use gamma directly
                    (x - self.midpoint) / (1 - self.midpoint) * (1 - self.midpoint**self.gamma) + self.midpoint**self.gamma
                )
        else:
            # Original around-midpoint transformation
            with np.errstate(invalid='ignore'):
                return np.where(
                    x < self.midpoint,
                    0.5 * (x / self.midpoint) ** self.gamma,
                    1 - 0.5 * ((1 - x) / (1 - self.midpoint)) ** self.gamma
                )




def main_interpolated_animation(
    checkpoint_path: str,
    data_path: str,
    data_key: str = "forward_2",
    total_original_frames: int = 10,
    original_fps: float = 1.0,
    num_interpolated_frames: int = 6,
    output_dir: str = "figures",
    midpoint: float = 0.15,
    colorbar_gamma: float = 1.0,
    cmap: str = 'rainbow',
    emphasize_below: bool = False,
    layer_start: int = 0,
    layer_end: int = 10,
    #subsample: int = 3,
    device: str = "auto",
    #batch_idx: int = 0,
    #sample_idx: int = 0,
    #forward_idx = int = 0,
):
    """
    Main function for creating interpolated ocean current animations with proper speed matching.
    
    This function:
    1. Loads ocean velocity data (zonal and meridional only)
    2. Uses SOMA interpolator to generate intermediate frames
    3. Creates animation with speed matching (interpolated frames play at same temporal rate)
    4. Focuses only on velocity components for interpolation
    
    Args:
        checkpoint_path (str): Path to trained SOMA interpolator model checkpoint
        data_path (str): Path to SOMA HDF5 data file
        data_key (str): Key in HDF5 file containing the ocean data (default: "forward_2")
        total_original_frames (int): Number of original frames to use from dataset
        original_fps (float): Original frames per second rate for the data
        num_interpolated_frames (int): Number of frames to interpolate between each real frame pair
                                     Must be a divisor of the model's horizon (typically 24)
        output_dir (str): Directory to save the animation
        midpoint (float): Midpoint for color normalization (0.0-1.0)
        colorbar_gamma (float): Gamma correction for colorbar sensitivity
        cmap (str): Colormap to use ('rainbow', 'viridis', 'plasma', etc.)
        emphasize_below (bool): Whether to emphasize values below midpoint
        layer_start (int): Starting depth layer index for averaging
        layer_end (int): Ending depth layer index for averaging
        subsample (int): Spatial subsampling factor for streamplot visualization
        device (str): Device to use for model inference ("auto", "cpu", "cuda")
        batch_idx (int): Which batch to use from the dataset
        sample_idx (int): Which sample within the batch to use
        
    Returns:
        str: Path to the saved animation file
        
    Raises:
        ImportError: If SOMA interpolation module is not available
        ValueError: If num_interpolated_frames is not a valid divisor of model horizon
        FileNotFoundError: If checkpoint_path or data_path don't exist
    """
    

    
    # Validate inputs
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    
    # Load and validate the model
    print(f"Loading SOMA interpolator model from: {checkpoint_path}")
    
    model = InterpolationExperiment.load_from_checkpoint(checkpoint_path)
    print(f"Loaded experiment class: {type(model).__name__}")
    print(f"Loaded model architecture: {type(model.model).__name__}")
    print(f"Model class location: {type(model.model).__module__}.{type(model.model).__name__}")
    model.eval()
    model_horizon = model.horizon
    print(f"Model horizon: {model_horizon}")
    

    # Validate tat num_interpolated_frames is a divisor of model horizon
    if num_interpolated_frames > 0 and model_horizon % num_interpolated_frames != 0:
        raise ValueError(f"num_interpolated_frames ({num_interpolated_frames}) must be a divisor of model horizon ({model_horizon})")

    
    # Set up prediction datamodule to fetch timestep pairs
    print(f"Setting up SOMA datamodule for timestep pairs from: {data_path}")
    
    # Create a dedicated datamodule for prediction/data access
    # Use window=1, horizon=1 to get consecutive timestep pairs
    datamodule = MyCustomDataModule(
        data_path=data_path,
        batch_size=1,
        window=1,  # Start with 1 timestep as context
        window_step=1,
        horizon=0,  # Predict  timestep ahead -> gives us 1 single timesteps
        num_workers=0,
        time_interval=1,  # Access consecutive timesteps
        standardize=True,  # Use normalized data like the model expects
        time_steps_per_run=None  # Use all available timesteps
    )
    datamodule.setup("predict")
    
    # Get data information from the datamodule
    dataset = datamodule._data_predict
    if dataset is None:
        raise ValueError("Failed to initialize prediction dataset")
    
    model._datamodule = datamodule  # Set the datamodule in the model for access
    model.num_predictions = 1  # Set number of predictions to 1 for single timestep pairs
    model.eval()  # Ensure model is in evaluation mode
    
    # The dataset now provides timestep pairs (window=1 + horizon=1 = 2 consecutive timesteps)
    time_steps_available = len(dataset)
    print(f"Datamodule initialized with {time_steps_available} available timesteps")
    
    # Get spatial dimensions by fetching a sample
    sample_data = dataset[data_key, 0]
    sample_tensor =  to_tensordict(sample_data['dynamics'], find_batch_size_max=True)  # Shape: (window+horizon, channels, z, y, x) = (1, channels, z, y, x)
    #dynamics_tensordict = to_tensordict(batch_dynamics_dict, find_batch_size_max=True)
    
    # # Add batch dimension to each variable in the dict
    # batch_dynamics_dict = {}
    # for var_name, var_tensor in dynamics_dict.items():
    #     if var_tensor.dim() == 4:  # (time, depth, height, width) - 3D case
    #         batch_dynamics_dict[var_name] = var_tensor.unsqueeze(0)  # (1, time, depth, height, width)
    #     elif var_tensor.dim() == 3:  # (time, height, width) - 2D case
    #         batch_dynamics_dict[var_name] = var_tensor.unsqueeze(0)  # (1, time, height, width)
    #     else:
    #         raise ValueError(f"Unexpected variable {var_name} shape: {var_tensor.shape}. Expected 3D or 4D tensor.")
    
    
    print(type(sample_tensor))
    print(len(sample_tensor))
    print(sample_tensor.keys())
    print(sample_tensor.shape)
    n_timesteps, z_dim, y_size, x_size = sample_tensor.shape
    
    if n_timesteps != 1:
        raise ValueError(f"Expected 1 timestep from datamodule sample, got {n_timesteps}")
    
    print(f"Data dimensions: {len(sample_tensor.keys())} channels, {z_dim}z x {y_size}y x {x_size}x spatial grid")
    print(f"Each sample contains {n_timesteps} consecutive timesteps")

    if total_original_frames < 2:
        raise ValueError("Need at least 2 original frames for interpolation")
    
    if total_original_frames > time_steps_available:
        raise ValueError(f"Warning: Need {original_time_intervals} timestep pairs, but only {time_steps_available} available.")
    
    # Calculate animation timing correctly
    # The goal is uniform temporal spacing between ALL frames (real + interpolated)
    # 
    # Example: If we have 3 real frames (covering 2 time intervals) with 2 interpolated frames between each:
    # Real[0] at t=0, Interp at t=0.33, Interp at t=0.67, Real[1] at t=1.0, Interp at t=1.33, Interp at t=1.67, Real[2] at t=2.0
    # 
    # So we have 7 total frames covering 2 time intervals
    # If original_fps = 5, then each time interval = 1/5 = 0.2 seconds
    # Total animation time = 2 * 0.2 = 0.4 seconds
    # Animation FPS = 7 frames / 0.4 seconds = 17.5 fps
    
    total_animation_frames = total_original_frames + (total_original_frames - 1) * num_interpolated_frames
    
    # Each "time interval" in the original data corresponds to 1/original_fps seconds
    # Calculate animation FPS to maintain the same total duration as original
    # Original covers (total_original_frames - 1) intervals at original_fps
    # Animation should cover the same time with (total_animation_frames - 1) intervals
    original_time_intervals = total_original_frames - 1
    animation_time_intervals = total_animation_frames - 1
    total_animation_time = original_time_intervals / original_fps
    animation_fps = animation_time_intervals / total_animation_time
    
    print(f"Animation FPS calculation:")
    print(f"  Original: {total_original_frames} frames at {original_fps} fps = {total_animation_time:.3f}s duration")
    print(f"  Animation: {total_animation_frames} frames at {animation_fps:.2f} fps = {animation_time_intervals / animation_fps:.3f}s duration")
    
    print(f"Animation parameters:")
    print(f"  - Original frames: {total_original_frames}")
    print(f"  - Interpolated frames per pair: {num_interpolated_frames}")
    print(f"  - Total animation frames: {total_animation_frames}")
    print(f"  - Original FPS: {original_fps}")
    print(f"  - Animation FPS: {animation_fps:.2f} (to maintain temporal rate)")
    print(f"  - Animation duration: {animation_time_intervals / animation_fps:.3f} seconds")
    
    
    # Prepare containers for all frames (real + interpolated)
    frames_to_animate = []
    
    animation_frame_idx = 0
    
    print(f"standardize: {dataset.standardize}")
    print(f"param_names: {dataset.param_names}")
    print(f"out_indices: {dataset.out_indices}")
    
    print(f"Generating interpolated animation...")
    
    
    ############################################
    # Generate interpolated frames between each pair of consecutive real frames
    for real_idx in range(total_original_frames):
        print(f"Processing timestep pair {real_idx + 1}/{total_original_frames}")
        
        # Fetch the timestep pair from datamodule

        dataset.standardize = False
        start_real = dataset[data_key, real_idx]['dynamics']  # Shape: (1, channels, z, y, x)
        end_real = dataset[data_key, real_idx+1]['dynamics']
        dataset.standardize = True
        start_real_norm = dataset[data_key, real_idx]['dynamics']  # Shape: (1, channels, z, y, x)
        end_real_norm = dataset[data_key, real_idx+1]['dynamics']
        
        # Set all values of 'var_x' in end_real_norm to 0.5

        end_real_norm['var_x'] = torch.full_like(end_real_norm['var_x'], 0.0)
        start_real_norm['var_x'] = torch.full_like(start_real_norm['var_x'], 0.0)

        if len(frames_to_animate) == 0:
            frames_to_animate.append(start_real)
            frames_to_animate.append(dataset.inverse_transform(start_real_norm)) # debugging
            
            print(f"  Stored real frame {real_idx} at animation index {len(frames_to_animate)}")
        
        
        if num_interpolated_frames > 0:
            # Calculate the time points for interpolation
            # Distribute interpolation times evenly across the model horizon, but exclude the final time
            # which would be equivalent to the next real frame
            time_step_size = model_horizon // (num_interpolated_frames + 1)
            target_times = [i * time_step_size for i in range(1, num_interpolated_frames + 1)]
            
            # Ensure we don't interpolate at the horizon time (which equals the next real frame)
            target_times = [t for t in target_times if t < model_horizon]
            
            print(f"  Interpolating between consecutive timesteps at times: {target_times}")
            
            predictions=None
            try:
                # Use the new interpolate function that takes a timestep pair directly
                with torch.no_grad():
                    predictions = model.interpolate(
                        dynamics1=start_real_norm,
                        dynamics2=end_real_norm,
                        target_times=target_times,
                        batch=False,  # No batch dimension needed for single pair
                    )
                # each element has keys preds and preds_normed
                
                # # Debug: Check if predictions are different
                # print(f"  Generated {len(predictions)} interpolated frames")
                # if len(predictions) > 1:
                #     # Compare first and last interpolated frames
                #     frame1 = predictions[0]['preds']['timeDaily_avg_velocityZonal']
                #     frame_last = predictions[-1]['preds']['timeDaily_avg_velocityZonal']
                #     diff = torch.abs(frame1 - frame_last).mean().item()
                #     print(f"  Difference between first and last interpolated frames: {diff:.8f}")
                #     if diff < 1e-10:
                #         print("  ⚠️ WARNING: Interpolated frames appear to be identical!")
                
                # Extract the velocity predictions
                #predictions = results['predictions']
                
            except Exception as e:
                raise ValueError(f"Interpolation failed for timestep frame {real_idx}. "
                                f"Error: {e}")
            
            for pred in predictions:
                interp = pred['preds']  #pred['preds_normed']  # Use normalized predictions
                denorm_interp = dataset.inverse_transform(interp)  # Apply inverse transform to get back to original scale
                #frames_to_animate.append(interp) # debugging
                frames_to_animate.append(denorm_interp)  
                
            del predictions
            torch.cuda.empty_cache()
        
        # add the real frame
        frames_to_animate.append(end_real)
        frames_to_animate.append(dataset.inverse_transform(end_real_norm)) # debugging

        print(f"  Stored real frame {real_idx+1} at animation index {len(frames_to_animate)}")

    
    print(f"Generated {len(frames_to_animate)} total frames ({total_original_frames} real + {len(frames_to_animate) - total_original_frames} interpolated)")
    
    # Debug: Check frame structures
    print("Debugging frame structures...")
    for i, frame in enumerate(frames_to_animate[:5]):  # Check first 5 frames
        print(f"Frame {i}: type={type(frame)}")
        if isinstance(frame, dict):
            print(f"  Keys: {list(frame.keys())}")
            if 'timeDaily_avg_velocityZonal' in frame:
                print(f"  Velocity shape: {frame['timeDaily_avg_velocityZonal'].shape}")
        else:
            print(f"  TensorDict keys: {list(frame.keys()) if hasattr(frame, 'keys') else 'No keys method'}")
            if hasattr(frame, '__getitem__') and 'timeDaily_avg_velocityZonal' in frame:
                print(f"  Velocity shape: {frame['timeDaily_avg_velocityZonal'].shape}")
    
    # Extract zonal and meridional velocity components from the frames
    u_all = []
    v_all = []
    ocean_mask = None  # Will be extracted from first frame
    
    for i, frame in enumerate(frames_to_animate):
        try:
            if isinstance(frame, dict):
                # Interpolated frame - access velocities from dict
                u_frame = frame['timeDaily_avg_velocityZonal']
                v_frame = frame['timeDaily_avg_velocityMeridional']
            else:
                # Real frame - access velocities from TensorDict
                u_frame = frame['timeDaily_avg_velocityZonal']
                v_frame = frame['timeDaily_avg_velocityMeridional']
            
            # Ensure on CPU and convert to numpy
            if hasattr(u_frame, 'cpu'):
                u_frame = u_frame.cpu()
            if hasattr(v_frame, 'cpu'):
                v_frame = v_frame.cpu()
            
            u_np = np.squeeze(u_frame.numpy() if hasattr(u_frame, 'numpy') else u_frame)
            v_np = np.squeeze(v_frame.numpy() if hasattr(v_frame, 'numpy') else v_frame)
            
            # Extract ocean mask from the first frame (frame 0)
            if i == 0:
                # Ocean mask is where data is NOT NaN and NOT 0.0 (land areas are often 0.0)
                ocean_mask = ~(np.isnan(u_np) | np.isnan(v_np) | (u_np == 0.0) | (v_np == 0.0))
                print(f"Extracted ocean mask from frame 0: {np.sum(ocean_mask)} ocean points out of {ocean_mask.size} total")
            else:
                # Apply the ocean mask to all subsequent frames
                # Set land areas (where mask is False) to NaN
                u_np = np.where(ocean_mask, u_np, np.nan)
                v_np = np.where(ocean_mask, v_np, np.nan)
            
            print(f"Frame {i}: u_shape={u_np.shape}, v_shape={v_np.shape}")
            
            u_all.append(u_np)
            v_all.append(v_np)
            
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            print(f"Frame type: {type(frame)}")
            if hasattr(frame, 'keys'):
                print(f"Frame keys: {list(frame.keys())}")
            raise
    
    # Check if all frames have the same shape
    shapes_u = [u.shape for u in u_all]
    shapes_v = [v.shape for v in v_all]
    print(f"All u shapes: {shapes_u}")
    print(f"All v shapes: {shapes_v}")
    
    u_all = np.array(u_all)
    v_all = np.array(v_all)
    
    # Debug frame uniqueness BEFORE cleaning and averaging
    print(f"Debug: Checking frame uniqueness before processing...")
    is_unique = debug_frame_uniqueness(u_all, v_all, num_interpolated_frames)
    if not is_unique:
        print("⚠️ WARNING: Duplicate frames detected! Animation may appear jerky.")
    
    # Clean outliers in velocity data
    u_all = np.where(np.abs(u_all) > 1e30, np.nan, u_all)
    v_all = np.where(np.abs(v_all) > 1e30, np.nan, v_all)
    # Check if we have enough frames to animate
    if len(u_all) < 2 or len(v_all) < 2:
        raise ValueError("Not enough valid frames to create animation. Check data integrity.")
    # Average over depth layers if specified
    if layer_start < 0 or layer_end > z_dim or layer_start >= layer_end:
        raise ValueError(f"Invalid layer range: {layer_start}-{layer_end} for depth dimension {z_dim}")
    if layer_start == 0 and layer_end == z_dim:
        print("Using all depth layers for averaging.")
    else:
        print(f"Averaging over depth layers {layer_start} to {layer_end} (exclusive)")
    print(f"before shapes: u_all.shape = {u_all.shape}, v_all.shape = {v_all.shape}")
    u_all = np.nanmean(u_all[:, layer_start:layer_end, :, :], axis=1)  # Average over depth layers
    v_all = np.nanmean(v_all[:, layer_start:layer_end, :, :], axis=1)  # Average over depth layers
    # Check if we have valid data after averaging
    if np.isnan(u_all).all() or np.isnan(v_all).all():
        raise ValueError("All averaged velocity data is NaN. Check input data integrity.")
    
    # Make the animation
    print("Making animation...")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get the actual 2D dimensions from the velocity arrays (after depth averaging)
    # u_all and v_all have shape (frames, y, x) after depth averaging
    actual_y_size, actual_x_size = u_all.shape[1], u_all.shape[2]
    
    # Prepare meshgrid for streamplot
    # Create coordinate arrays that match the actual 2D data shape
    x_coords = np.arange(actual_x_size)
    y_coords = np.arange(actual_y_size)
    x_sub, y_sub = np.meshgrid(x_coords, y_coords)
    
    # Set colorbar limits based on all frames for consistency
    all_speeds = np.sqrt(u_all**2 + v_all**2)
    vmin = np.nanmin(all_speeds)
    vmax = np.nanmax(all_speeds)
    
    print(f"Velocity range: {vmin:.6f} to {vmax:.6f} m/s")
    print(f"Data shape: u_all.shape = {u_all.shape}, v_all.shape = {v_all.shape}")
    print(f"Meshgrid shape: x_sub.shape = {x_sub.shape}, y_sub.shape = {y_sub.shape}")
    
    # Apply custom normalization
    norm = MidpointFocusNorm(
        vmin=vmin, 
        vmax=vmax, 
        midpoint=midpoint, 
        gamma=colorbar_gamma,
        emphasis_below=emphasize_below
    )

    # Create colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    cbar.set_label('Current Speed (m/s)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    def update(frame):
        # Clear the axis for the new frame
        ax.cla()
        ax.set_xlim(0, actual_x_size)
        ax.set_ylim(0, actual_y_size)
        ax.set_aspect('equal')
        ax.set_xlabel("X Position", fontsize=12)
        ax.set_ylabel("Y Position", fontsize=12)

        # Calculate current speed for color mapping
        current_speed = np.sqrt(u_all[frame]**2 + v_all[frame]**2)
        
        # Handle NaN values in velocity data for streamplot
        u_frame = u_all[frame].copy()
        v_frame = v_all[frame].copy()
        
        # Replace NaN values with zero for streamplot (streamplot doesn't handle NaN well)
        u_frame_clean = np.where(np.isnan(u_frame), 0, u_frame)
        v_frame_clean = np.where(np.isnan(v_frame), 0, v_frame)
        
        # Ensure all arrays have the same shape
        # The velocity arrays should have shape (actual_y_size, actual_x_size) to match the meshgrid
        if u_frame_clean.shape != (actual_y_size, actual_x_size):
            print(f"Warning: u_frame shape {u_frame_clean.shape} doesn't match expected ({actual_y_size}, {actual_x_size})")
            # If the shape is (actual_x_size, actual_y_size), transpose it
            if u_frame_clean.shape == (actual_x_size, actual_y_size):
                u_frame_clean = u_frame_clean.T
                v_frame_clean = v_frame_clean.T
                current_speed = current_speed.T
        
        # Create a mask for valid data
        # TODO don't make ocean mask a global variable, pass it as an argument
        if ocean_mask is None:
            valid_mask = ~(np.isnan(u_frame) | np.isnan(v_frame))
        else:
            valid_mask = ocean_mask

        # Only draw streamlines where we have valid data
        if np.any(valid_mask):
            # Verify shapes match before calling streamplot
            if (u_frame_clean.shape == x_sub.shape and 
                v_frame_clean.shape == y_sub.shape and 
                current_speed.shape == x_sub.shape):
                
                # Draw streamlines
                strm = ax.streamplot(
                    x_sub, y_sub, u_frame_clean, v_frame_clean,
                    color=current_speed,
                    cmap=cmap,
                    linewidth=2,
                    density=3,
                    arrowsize=1.5,
                    norm=norm
                )
            else:
                print(f"Shape mismatch:")
                print(f"  x_sub.shape = {x_sub.shape}")
                print(f"  y_sub.shape = {y_sub.shape}")
                print(f"  u_frame_clean.shape = {u_frame_clean.shape}")
                print(f"  v_frame_clean.shape = {v_frame_clean.shape}")
                print(f"  current_speed.shape = {current_speed.shape}")
                strm = None
        else:
            # If no valid data, create empty plot with proper axes
            strm = None
        
        # Determine if this is a real frame or interpolated frame
        # Real frames occur at indices: 0, 1+num_interpolated_frames, 1+2*(num_interpolated_frames+1), etc.
        real_frame_indices = [i * (num_interpolated_frames + 1) for i in range(total_original_frames)]
        is_real_frame = frame in real_frame_indices
        frame_type = "Real" if is_real_frame else "Interpolated"
        
        ax.set_title(f"Ocean Current Velocity - {frame_type} Frame ({frame+1}/{total_animation_frames})", fontsize=16)
        if strm is not None:
            return strm.lines,
        else:
            return [],

    # Create animation with calculated FPS to maintain temporal rate
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_animation_frames,
        interval=int(1000 / animation_fps),  # Convert FPS to milliseconds
        blit=False,  # Disable blitting to avoid rendering issues
        repeat=True,
        cache_frame_data=True  # Cache frames to reduce stuttering
    )
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save animation with descriptive filename
    output_filename = (f"{data_key}_{total_original_frames}real_{num_interpolated_frames}interp_"
                      f"{cmap}_gamma{colorbar_gamma}_mid{midpoint}_fps{original_fps:.1f}.gif")
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Saving animation to: {output_path}")
    ani.save(output_path,
             writer='ffmpeg',
             fps=animation_fps,
             dpi=150,
             bitrate=600) #1800
    plt.close(fig)
    
    print(f"✅ Animation saved successfully!")
    print(f"📊 Animation details:")
    print(f"   - File: {output_path}")
    print(f"   - Real frames: {total_original_frames}")
    print(f"   - Interpolated frames: {total_animation_frames - total_original_frames}")
    print(f"   - Total frames: {total_animation_frames}")
    print(f"   - FPS: {animation_fps:.1f} (maintaining {original_fps:.1f} temporal rate)")
    print(f"   - Duration: {(total_animation_frames - 1) / animation_fps:.3f} seconds")
    print(f"   - Variables: Zonal and Meridional velocity only")
    print(f"   - Depth layers: {layer_start}-{layer_end} (averaged)")
    
    return output_path


def debug_frame_uniqueness(u_all, v_all, num_interpolated_frames):
    """
    Debug function to check if frames are unique and detect duplications.
    """
    print(f"\n🔍 FRAME UNIQUENESS ANALYSIS:")
    print(f"Total frames: {u_all.shape[0]}")
    
    # Calculate frame statistics
    frame_stats = []
    for i in range(u_all.shape[0]):
        u_mean = np.nanmean(u_all[i])
        v_mean = np.nanmean(v_all[i])
        u_std = np.nanstd(u_all[i])
        v_std = np.nanstd(v_all[i])
        frame_stats.append((u_mean, v_mean, u_std, v_std))
    
    # Check for duplicates
    duplicates_found = []
    tolerance = 1e-10
    for i in range(len(frame_stats)):
        for j in range(i + 1, len(frame_stats)):
            u_diff = abs(frame_stats[i][0] - frame_stats[j][0])
            v_diff = abs(frame_stats[i][1] - frame_stats[j][1])
            if u_diff < tolerance and v_diff < tolerance:
                duplicates_found.append((i, j))
    
    if duplicates_found:
        print(f"❌ Found {len(duplicates_found)} duplicate frame pairs:")
        for i, j in duplicates_found[:5]:  # Show first 5
            print(f"  Frames {i} and {j} are identical")
    else:
        print(f"✅ All frames are unique!")
    
    # Check pattern of real vs interpolated frames
    print(f"\n📊 Frame pattern analysis (first 20 frames):")
    step_size = 1 + num_interpolated_frames
    for i in range(min(20, len(frame_stats))):
        if i % step_size == 0:
            frame_type = "REAL"
        else:
            frame_type = f"INTERP-{i % step_size}"
        
        u_mean, v_mean = frame_stats[i][:2]
        print(f"  Frame {i:2d}: {frame_type:8s} u_mean={u_mean:8.6f} v_mean={v_mean:8.6f}")
    
    return len(duplicates_found) == 0




def interpolate(model, timestep_start, timestep_end, target_times, device="auto"):
    """
    Interpolate between two single timesteps using the SOMA interpolator model.
    
    Args:
        model: Trained SOMA interpolation model
        data1_dict: First timestep data as tensordict (single timestep, not windowed)
        data2_dict: Second timestep data as tensordict (single timestep, not windowed)
        target_times: List of target interpolation times within model horizon
        device: Device to run inference on
        
    Returns:
        dict: Dictionary with target_time -> interpolated_data mappings
    """
    import torch
    
    # Set up device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    model.eval()
    
    # Prepare the input tensors
    # The model expects input format similar to training: (batch, time, channels, z, y, x)
    dynamics1 = timestep_start['dynamics'].to(device)  # Shape: (1, channels, z, y, x)
    dynamics2 = timestep_end['dynamics'].to(device)  # Shape: (1, channels, z, y, x)
    
    # Add batch dimension and combine timesteps
    # Create input tensor with shape (batch=1, time=2, channels, z, y, x)
    input_tensor = torch.stack([dynamics1[0], dynamics2[0]], dim=0).unsqueeze(0)  # (1, 2, channels, z, y, x)
    
    # Create the input dictionary for the model
    input_dict = {
        'dynamics': input_tensor
    }
    
    # Run interpolation
    with torch.no_grad():
        try:
            # The model's interpolate method expects input similar to training format
            # and returns interpolated frames at the specified target times
            interpolation_results = model.interpolate(
                input_dict, 
                target_times=target_times
            )
            
            # Process results - convert back to numpy and organize by target_time
            predictions = {}
            
            for i, target_time in enumerate(target_times):
                if i < len(interpolation_results):
                    # Extract the interpolated frame
                    interp_frame = interpolation_results[i]  # Shape: (channels, z, y, x)
                    
                    # Convert to numpy
                    if isinstance(interp_frame, torch.Tensor):
                        interp_frame_np = interp_frame.cpu().numpy()
                    else:
                        interp_frame_np = interp_frame
                    
                    # Create result dictionary with variable names
                    result_dict = {
                        'timeDaily_avg_layerThickness': interp_frame_np[0],        # Channel 0
                        'timeDaily_avg_velocityZonal': interp_frame_np[1],         # Channel 1
                        'timeDaily_avg_velocityMeridional': interp_frame_np[2],    # Channel 2
                        'timeDaily_avg_activeTracers_temperature': interp_frame_np[3],  # Channel 3
                        'timeDaily_avg_activeTracers_salinity': interp_frame_np[4],     # Channel 4
                        'var_x': interp_frame_np[5] if interp_frame_np.shape[0] > 5 else None  # Channel 5
                    }
                    
                    predictions[target_time] = result_dict
            
            return {'predictions': predictions}
            
        except Exception as e:
            raise Exception(f"Error during interpolation: {e}")



def extract_velocity_from_timestep_pair(timestep_sample, timestep_idx=0, layer_start=0, layer_end=10):
    """
    Extract velocity components from a timestep pair sample.
    
    Args:
        timestep_pair_sample: Dictionary containing 'dynamics' tensor with 2 consecutive timesteps
        timestep_idx: Which timestep to extract (0 for first, 1 for second)
        layer_start: Starting depth layer for averaging
        layer_end: Ending depth layer for averaging
        
    Returns:
        tuple: (u_velocity, v_velocity) as numpy arrays with shape (y, x)
    """
    # Extract the dynamics tensor - shape: (2, channels, z, y, x) for timestep pairs
    dynamics = timestep_sample['dynamics']
    
    if dynamics.shape[0] != 2:
        raise ValueError(f"Expected 2 timesteps in dynamics, got {dynamics.shape[0]}")
    
    # Take the specified timestep - shape: (channels, z, y, x)
    timestep_data = dynamics[timestep_idx]
    
    # Extract velocity channels (indices 1 and 2 for zonal and meridional velocity)
    u_velocity = timestep_data[1]  # timeDaily_avg_velocityZonal - shape: (z, y, x)
    v_velocity = timestep_data[2]  # timeDaily_avg_velocityMeridional - shape: (z, y, x)
    
    # Convert to numpy and average over depth layers
    u_np = u_velocity.numpy()
    v_np = v_velocity.numpy()
    
    # Average over specified depth layers
    u_avg = np.nanmean(u_np[layer_start:layer_end], axis=0)  # Shape: (y, x)
    v_avg = np.nanmean(v_np[layer_start:layer_end], axis=0)  # Shape: (y, x)
    
    return u_avg, v_avg



if __name__ == '__main__':
    
    checkpoint_path = '/global/homes/a/abgulhan/WORKING/git_repos/spherical-dyffusion/results/checkpoints/40587104/SOMA-Ipol24h_None_epoch037_seed3.ckpt'#'/global/homes/a/abgulhan/WORKING/checkpoints/SOMA-Ipol24h_None_epoch005_seed4.ckpt'#'/global/homes/a/abgulhan/WORKING/checkpoints/SOMA-Ipol24h_None_epoch044_seed4.ckpt'  
    #'/global/homes/a/abgulhan/WORKING/git_repos/spherical-dyffusion/results/checkpoints/40587104/SOMA-Ipol24h_None_epoch037_seed3.ckpt'#
    data_path = '/global/homes/a/abgulhan/WORKING/ml_converted/data-gm-year7-22-biweekly.hdf5'
    
    output = main_interpolated_animation(
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        data_key='forward_1',
        total_original_frames=3,  # Reduced for debugging
        original_fps=0.9,
        num_interpolated_frames=3,
        output_dir='figures/test_horizon',
        midpoint=0.0,
        colorbar_gamma=1.0,
        cmap='rainbow',
        device='auto',
        layer_start=0,
        layer_end=10
    )

