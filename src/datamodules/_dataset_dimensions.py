from omegaconf import DictConfig


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

    elif "debug_datamodule" in target:
        input_dim = output_dim = datamodule_config.channels
        spatial_dims = (datamodule_config.height, datamodule_config.width)

    else:
        raise ValueError(f"Unknown dataset: {target}")
    return {
        "input": input_dim,
        "output": output_dim,
        "spatial_in": spatial_dims,
        "spatial_out": spatial_dims_out if spatial_dims_out is not None else spatial_dims,
        "conditional": conditional_dim,
    }
