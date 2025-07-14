from omegaconf import DictConfig
import importlib

def get_dims_of_dataset(datamodule_config: DictConfig):
    """Returns the number of features for the given dataset."""
    target = datamodule_config.get("_target_", datamodule_config.get("name"))
    conditional_dim = 0
    spatial_dims_out = None
    if "fv3gfs" in target:
        input_dim = len(datamodule_config.in_names)
        output_dim = len(datamodule_config.out_names)
        spatial_dims = (180, 360)
        conditional_dim = len(datamodule_config.forcing_names) if datamodule_config.forcing_names is not None else 0
    elif target == "src.datamodules.SOMA_datamodule.MyCustomDataModule":
        # Dynamically import and instantiate the datamodule
        module_path, class_name = target.rsplit(".", 1)
        module = importlib.import_module(module_path)
        DataModuleClass = getattr(module, class_name)
        # Instantiate the datamodule
        datamodule = DataModuleClass(
            data_path=datamodule_config.data_path,
            model_config=None,
            param_names=datamodule_config.get("param_names", []),
            forcing_names=datamodule_config.get("forcing_names", None),
            in_names=datamodule_config.get("in_names", []),
            out_names=datamodule_config.get("out_names", []),
            horizon=datamodule_config.get("horizon", 1),
            batch_size=1,
            eval_batch_size=1,
            num_workers=0,
            train_val_test_split_ratios=(0.11, 0.0, 0.0),
            time_steps_per_run=datamodule_config.get("time_steps_per_run", None),
            pin_memory=False,
            data_dir=datamodule_config.get("data_dir", None),
            time_interval=datamodule_config.get("time_interval", 14),
            window_step=datamodule_config.get("window_step", 1),  
            stack_z=datamodule_config.get("stack_z", False),
        )
        datamodule.setup()
        dataset = datamodule._data_train  # or _data_val/_data_test as appropriate
        if len(dataset) == 0:
            raise RuntimeError(
                f"SOMA dataset is empty during dimension inference! "
                f"Check that run keys are set and the HDF5 file is accessible. "
                f"datamodule_config: {datamodule_config}"
            )
        sample = dataset[0]['dynamics']  # Get a sample from the dataset
        # Defensive: sample should be (dict, xr.DataArray) or similar
        if isinstance(sample, dict):
            # If sample is a dict of tensors
            first_tensor = next(iter(sample.values()))
        elif isinstance(sample, (tuple, list)) and isinstance(sample[0], dict):
            # If sample is (dict, ...)
            first_tensor = next(iter(sample[0].values()))
        else:
            raise RuntimeError(f"Unexpected sample format from SOMA dataset: {type(sample)}")
        # Defensive: first_tensor should be a torch.Tensor or np.ndarray
        if hasattr(first_tensor, 'shape'):
            spatial_shape = tuple(first_tensor.shape[-3:])  # (D, H, W) or (Z, Y, X)
        else:
            raise RuntimeError(f"First tensor in SOMA sample is not a tensor/array: {type(first_tensor)}")
        input_dim = len(datamodule_config.get("in_names", []))
        output_dim = len(datamodule_config.get("out_names", []))
        conditional_dim = len(datamodule_config.get("forcing_names", [])) if datamodule_config.get("forcing_names", None) is not None else 0
        result =  {
            "input": input_dim,
            "output": output_dim,
            "spatial_in": spatial_shape,
            "spatial_out": spatial_shape,
            "conditional": conditional_dim,
        }
        #print(f"====> stack_z is set to {datamodule.stack_z}")
        print(f"SOMA dataset dimensions: {result}")
        return result
    elif "debug_datamodule" in target:
        input_dim = output_dim = datamodule_config.channels
        spatial_dims = (datamodule_config.height, datamodule_config.width)

    else:
        raise ValueError(f"Unknown dataset: {target}")
    result = {
        "input": input_dim,
        "output": output_dim,
        "spatial_in": spatial_dims,
        "spatial_out": spatial_dims_out if spatial_dims_out is not None else spatial_dims,
        "conditional": conditional_dim,
    }
    
    print(f"Dataset dimensions: {result}")
    return result
