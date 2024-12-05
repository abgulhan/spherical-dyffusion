import time
from typing import Dict, Optional

from omegaconf import DictConfig


def _shared_prefix(config: DictConfig, init_prefix: str = "") -> str:
    """This is a prefix for naming the runs for a more agreeable logging."""
    s = init_prefix if isinstance(init_prefix, str) else ""
    if not config.get("model"):
        return s
    # Find mixer type if it is a transformer model (e.g. self-attention or FNO mixing)
    kwargs = dict(mixer=config.model.mixer._target_) if config.model.get("mixer") else dict()
    s += clean_name(config.model._target_, **kwargs)
    return s.lstrip("_")


def get_name_for_hydra_config_class(config: DictConfig) -> Optional[str]:
    """Will return a string that can describe the class of the (sub-)config."""
    if "name" in config and config.get("name") is not None:
        return config.get("name")
    elif "_target_" in config:
        return config._target_.split(".")[-1]
    return None


def get_clean_float_name(lr: float) -> str:
    """Stringify floats <1 into very short format (use for learning rates, weight-decay etc.)"""
    # basically, map Ae-B to AB (if lr<1e-5, else map 0.0001 to 1e-4)
    # convert first to scientific notation:
    if lr >= 0.1:
        return str(lr)
    lr_e = f"{lr:.1e}"  # 1e-2 -> 1.0e-02, 0.03 -> 3.0e-02
    # now, split at the e into the mantissa and the exponent
    lr_a, lr_b = lr_e.split("e-")
    # if the decimal point is 0 (e.g 1.0, 3.0, ...), we return a simple string
    if lr_a[-1] == "0":
        return f"{lr_a[0]}{int(lr_b)}"
    else:
        return str(lr).replace("e-", "")


def remove_float_prefix(string, prefix_name: str = "lr", separator="_") -> str:
    # Remove the lr and/or wd substrings like:
    # 0.0003lr_0.01wd -> ''
    # 0.0003lr -> ''
    # 0.0003lr_0.5lrecs_0.01wd -> '0.5lrecs'
    # 0.0003lr_0.5lrecs -> '0.5lrecs'
    # 0.0003lr_0.5lrecs_0.01wd_0.5lrecs -> '0.5lrecs_0.5lrecs'
    if prefix_name not in string:
        return string
    part1, part2 = string.split(prefix_name)
    # split at '_' and keep all but the last part
    part1keep = "_".join(part1.split(separator)[:-1])
    return part1keep + part2


def get_loss_name(loss):
    if isinstance(loss, str):
        loss_name = loss.lower()
    elif loss.get("_target_", "").endswith("LpLoss"):
        p, is_relative = loss.get("p", 2), loss.get("relative")
        loss_name = f"l{p}r" if is_relative else f"l{p}a"
    else:
        assert loss.get("_target_") is not None, f"Unknown loss ``{loss}``"
        loss_name = loss.get("_target_").split(".")[-1].lower().replace("loss_function", "").replace("loss", "")
    return loss_name


def get_detailed_name(config, add_unique_suffix: bool = True) -> str:
    """This is a detailed name for naming the runs for logging."""
    s = config.get("name") + "_" if config.get("name") is not None else ""
    hor = config.datamodule.get("horizon", 1)
    if (
        hor > 1
        and f"H{hor}" not in s
        and f"horizon{hor}" not in s.lower()
        and f"h{hor}" not in s.lower()
        and f"{hor}h" not in s.lower()
        and f"{hor}l" not in s.lower()
    ):
        print(
            f"WARNING: horizon {hor} not in name, but should be!",
            s,
            config.get("name_suffix"),
        )
        s = s[:-1] + f"-MH{hor}_"

    s += str(config.get("name_suffix")) + "_" if config.get("name_suffix") is not None else ""
    s += _shared_prefix(config) + "_"

    w = config.datamodule.get("window", 1)
    if w > 1:
        s += f"{w}w_"

    if config.datamodule.get("train_start_date") is not None:
        s += f"{config.datamodule.train_start_date}tst_"

    if config.get("model") is None:
        return s.rstrip("_-").lstrip("_-")  # for "naive" baselines, e.g. climatology

    use_ema, ema_decay = config.module.get("use_ema", False), config.module.get("ema_decay", 0.9999)
    if use_ema:
        s += "EMA_"
        if ema_decay != 0.9999:
            s = s.replace("EMA", f"EMA{config.module.ema_decay}")

    is_diffusion = config.get("diffusion") is not None
    if is_diffusion:
        if config.diffusion.get("interpolator_run_id"):
            int_run_id = config.diffusion.interpolator_run_id
            replace = {
                "SOME_RUN_ID": "SOME_SIMPLER_ALIAS",
            }
            int_run_id = replace.get(int_run_id, int_run_id)
            s += f"{int_run_id}-ipolID_"

        fcond = config.diffusion.get("forward_conditioning")
        if fcond != "none":
            s += f"{fcond}-fcond_" if "noise" not in fcond else f"{fcond}_"

        if config.diffusion.get("time_encoding", "dynamics") != "dynamics":
            tenc = config.diffusion.get("time_encoding")
            if tenc == "continuous":
                s += "ContTime_"
            elif tenc == "dynamics":
                pass  # s += "DynT_"
            else:
                s += f"{config.diffusion.time_encoding}-timeEnc_"

    hdims = config.model.get("hidden_dims")
    if hdims is None:
        num_L = config.model.get("num_layers") or config.model.get("depth")
        if num_L is None:
            dim_mults = config.model.get("dim_mults") or config.model.get("channel_mult")
            if dim_mults is None:
                pass
            elif tuple(dim_mults) == (1, 2, 4):
                num_L = "3"
            else:
                num_L = "-".join([str(d) for d in dim_mults])

        possible_dim_names = ["dim", "hidden_dim", "embed_dim", "hidden_size", "model_channels"]
        hdim = None
        for name in possible_dim_names:
            hdim = config.model.get(name)
            if hdim is not None:
                break

        if hdim is not None:
            hdims = f"{hdim}x{num_L}" if num_L is not None else f"{hdim}"
    elif all([h == hdims[0] for h in hdims]):
        hdims = f"{hdims[0]}x{len(hdims)}"
    else:
        hdims = str(hdims)

    s += f"{hdims}d_" if hdims is not None else ""
    if config.model.get("mlp_ratio", 4.0) != 4.0:
        s += f"{config.model.mlp_ratio}dxMLP_"

    if is_diffusion and config.diffusion.get("loss_function") is not None:
        loss = config.diffusion.get("loss_function")
        loss = get_loss_name(loss)
        if loss not in ["mse", "l2"]:
            s += f"{loss.upper()}_"
    else:
        loss = config.model.get("loss_function")
        loss = get_loss_name(loss)
        if loss in ["mse", "l2"]:
            pass
        elif loss in ["l2_rel", "l1_rel"]:
            s += f"{loss.upper().replace('_REL', 'rel')}_"
        else:
            s += f"{loss.upper()}_"

    time_emb = config.model.get("with_time_emb", False)
    if time_emb not in [False, True, "scale_shift"]:
        s += f"{time_emb}_"
    if (isinstance(time_emb, str) and "scale_shift" in time_emb) and not config.model.get(
        "time_scale_shift_before_filter"
    ):
        s += "tSSA_"  # time scale shift after filter

    optim = config.module.get("optimizer")
    if optim is not None:
        if "adamw" not in optim.name.lower():
            s += f"{optim.name.replace('Fused', '').replace('fused', '')}_"
        if "fused" in optim.name.lower() or optim.get("fused", False):
            s = s[:-1] + "F_"
    scheduler_cfg = config.module.get("scheduler")
    lr = config.get("base_lr") or optim.get("lr")
    s += f"{get_clean_float_name(lr)}lr_"
    if scheduler_cfg is not None and "warmup_epochs" in scheduler_cfg:
        s += f"LC{scheduler_cfg.warmup_epochs}:{scheduler_cfg.max_epochs}_"

    if is_diffusion:
        lam1 = config.diffusion.get("lambda_reconstruction")
        lam2 = config.diffusion.get("lambda_reconstruction2")
        all_lams = [lam1, lam2]
        nonzero_lams = len([1 for lam in all_lams if lam is not None and lam > 0])
        uniform_lams = [
            1 / nonzero_lams if nonzero_lams > 0 else 0,
            0.33 if nonzero_lams == 3 else 0,
        ]
        if config.diffusion.get("lambda_reconstruction2", 0) > 0:
            if lam1 == lam2:
                s += f"{lam1}lRecs_"
            else:
                s += f"{lam1}-{lam2}lRecs_"

            if config.diffusion.get("reconstruction2_detach_x_last", False):
                s += "detX0_"
        elif lam1 is not None and lam1 not in uniform_lams:
            s += f"{lam1}lRec_"

    dropout = {
        "": config.model.get("dropout", 0),
        "in": config.model.get("input_dropout", 0),
        "pos": config.model.get("pos_emb_dropout", 0),
        "at": config.model.get("attn_dropout", 0),
        "b": config.model.get("block_dropout", 0),
        "b1": config.model.get("block_dropout1", 0),
        "ft": config.model.get("dropout_filter", 0),
        "mlp": config.model.get("dropout_mlp", 0),
    }
    any_nonzero = any([d > 0 for d in dropout.values() if d is not None])
    for k, d in dropout.items():
        if d is not None and d > 0:
            s += f"{int(d * 100)}{k}"
    if any_nonzero:  # remove redundant 'Dr_'
        s += "Dr_"

    if any_nonzero and is_diffusion and config.diffusion.get("enable_interpolator_dropout", False):
        s += "iDr_"  # interpolator dropout

    if config.model.get("drop_path_rate", 0) > 0:
        s += f"{int(config.model.drop_path_rate * 100)}dpr_"

    if config.module.optimizer.get("weight_decay") and config.module.optimizer.get("weight_decay") > 0:
        s += f"{get_clean_float_name(config.module.optimizer.get('weight_decay'))}wd_"

    if config.get("suffix", "") != "":
        s += f"{config.get('suffix')}_"

    wandb_cfg = config.get("logger", {}).get("wandb", {})
    if wandb_cfg.get("resume_run_id") and wandb_cfg.get("id", "$") != wandb_cfg.get("resume_run_id", "$"):
        s += f"{wandb_cfg.get('resume_run_id')}rID_"

    if add_unique_suffix:
        s += f"{config.get('seed')}seed"
        s += "_" + time.strftime("%Hh%Mm%b%d")
        wandb_id = wandb_cfg.get("id")
        if wandb_id is not None:
            s += f"_{wandb_id}"

    return s.replace("None", "").rstrip("_-").lstrip("_-")


def clean_name(class_name, mixer=None, dm_type=None) -> str:
    """This names the model class paths with a more concise name."""
    if "SphericalFourierNeuralOperatorNet" in class_name:
        return "SFNO"
    elif "unet_simple" in class_name:
        s = "SimpleUnet"
    elif "Unet" in class_name:
        s = "UNetR"
    elif "SimpleConvNet" in class_name:
        s = "SimpleCNN"
    else:
        raise ValueError(f"Unknown class name: {class_name}, did you forget to add it to the clean_name function?")

    return s


def get_group_name(config) -> str:
    """
    This is a group name for wandb logging.
    On Wandb, the runs of the same group are averaged out when selecting grouping by `group`
    """
    # s = get_name_for_hydra_config_class(config.model)
    # s = s or _shared_prefix(config, init_prefix=s)
    return get_detailed_name(config, add_unique_suffix=False)


def var_names_to_clean_name() -> Dict[str, str]:
    """This is a clean name for the variables (e.g. for plotting)"""
    var_dict = {
        "tas": "Air Temperature",
        "psl": "Sea-level Pressure",
        "ps": "Surface Pressure",
        "pr": "Precipitation",
        "sst": "Sea Surface Temperature",
    }
    return var_dict


variable_name_to_metadata = {
    "DLWRFsfc": {"units": "W/m**2", "long_name": "surface downward longwave flux"},
    "DSWRFsfc": {
        "units": "W/m**2",
        "long_name": "averaged surface downward shortwave flux",
    },
    "DSWRFtoa": {
        "units": "W/m**2",
        "long_name": "top of atmos downward shortwave flux",
    },
    "GRAUPELsfc": {
        "units": "kg/m**2/s",
        "long_name": "bucket surface graupel precipitation rate",
    },
    "HGTsfc": {"units": "m", "long_name": "surface height"},
    "ICEsfc": {
        "units": "kg/m**2/s",
        "long_name": "bucket surface ice precipitation rate",
    },
    "LHTFLsfc": {"units": "w/m**2", "long_name": "surface latent heat flux"},
    "PRATEsfc": {
        "units": "kg/m**2/s",
        "long_name": "bucket surface precipitation rate",
    },
    "PRESsfc": {"units": "Pa", "long_name": "surface pressure"},
    "SHTFLsfc": {"units": "w/m**2", "long_name": "surface sensible heat flux"},
    "SNOWsfc": {
        "units": "kg/m**2/s",
        "long_name": "bucket surface snow precipitation rate",
    },
    "ULWRFsfc": {"units": "W/m**2", "long_name": "surface upward longwave flux"},
    "ULWRFtoa": {"units": "W/m**2", "long_name": "top of atmos upward longwave flux"},
    "USWRFsfc": {
        "units": "W/m**2",
        "long_name": "averaged surface upward shortwave flux",
    },
    "USWRFtoa": {"units": "W/m**2", "long_name": "top of atmos upward shortwave flux"},
    "air_temperature_0": {"units": "K", "long_name": "temperature level-0"},
    "air_temperature_1": {"units": "K", "long_name": "temperature level-1"},
    "air_temperature_2": {"units": "K", "long_name": "temperature level-2"},
    "air_temperature_3": {"units": "K", "long_name": "temperature level-3"},
    "air_temperature_4": {"units": "K", "long_name": "temperature level-4"},
    "air_temperature_5": {"units": "K", "long_name": "temperature level-5"},
    "air_temperature_6": {"units": "K", "long_name": "temperature level-6"},
    "air_temperature_7": {"units": "K", "long_name": "temperature level-7"},
    "ak_0": {"units": "Pa", "long_name": "ak"},
    "ak_1": {"units": "Pa", "long_name": "ak"},
    "ak_2": {"units": "Pa", "long_name": "ak"},
    "ak_3": {"units": "Pa", "long_name": "ak"},
    "ak_4": {"units": "Pa", "long_name": "ak"},
    "ak_5": {"units": "Pa", "long_name": "ak"},
    "ak_6": {"units": "Pa", "long_name": "ak"},
    "ak_7": {"units": "Pa", "long_name": "ak"},
    "ak_8": {"units": "Pa", "long_name": "ak"},
    "bk_0": {"units": "", "long_name": "bk"},
    "bk_1": {"units": "", "long_name": "bk"},
    "bk_2": {"units": "", "long_name": "bk"},
    "bk_3": {"units": "", "long_name": "bk"},
    "bk_4": {"units": "", "long_name": "bk"},
    "bk_5": {"units": "", "long_name": "bk"},
    "bk_6": {"units": "", "long_name": "bk"},
    "bk_7": {"units": "", "long_name": "bk"},
    "bk_8": {"units": "", "long_name": "bk"},
    "eastward_wind_0": {"units": "m/sec", "long_name": "zonal wind level-0"},
    "eastward_wind_1": {"units": "m/sec", "long_name": "zonal wind level-1"},
    "eastward_wind_2": {"units": "m/sec", "long_name": "zonal wind level-2"},
    "eastward_wind_3": {"units": "m/sec", "long_name": "zonal wind level-3"},
    "eastward_wind_4": {"units": "m/sec", "long_name": "zonal wind level-4"},
    "eastward_wind_5": {"units": "m/sec", "long_name": "zonal wind level-5"},
    "eastward_wind_6": {"units": "m/sec", "long_name": "zonal wind level-6"},
    "eastward_wind_7": {"units": "m/sec", "long_name": "zonal wind level-7"},
    "land_fraction": {
        "units": "dimensionless",
        "long_name": "fraction of grid cell area occupied by land",
    },
    "northward_wind_0": {"units": "m/sec", "long_name": "meridional wind level-0"},
    "northward_wind_1": {"units": "m/sec", "long_name": "meridional wind level-1"},
    "northward_wind_2": {"units": "m/sec", "long_name": "meridional wind level-2"},
    "northward_wind_3": {"units": "m/sec", "long_name": "meridional wind level-3"},
    "northward_wind_4": {"units": "m/sec", "long_name": "meridional wind level-4"},
    "northward_wind_5": {"units": "m/sec", "long_name": "meridional wind level-5"},
    "northward_wind_6": {"units": "m/sec", "long_name": "meridional wind level-6"},
    "northward_wind_7": {"units": "m/sec", "long_name": "meridional wind level-7"},
    "ocean_fraction": {
        "units": "dimensionless",
        "long_name": "fraction of grid cell area occupied by ocean",
    },
    "sea_ice_fraction": {
        "units": "dimensionless",
        "long_name": "fraction of grid cell area occupied by sea ice",
    },
    "soil_moisture": {
        "units": "kg/m**2",
        "long_name": "total column soil moisture content",
    },
    "specific_total_water_0": {
        "units": "kg/kg",
        "long_name": "specific total water level-0",
    },
    "specific_total_water_1": {
        "units": "kg/kg",
        "long_name": "specific total water level-1",
    },
    "specific_total_water_2": {
        "units": "kg/kg",
        "long_name": "specific total water level-2",
    },
    "specific_total_water_3": {
        "units": "kg/kg",
        "long_name": "specific total water level-3",
    },
    "specific_total_water_4": {
        "units": "kg/kg",
        "long_name": "specific total water level-4",
    },
    "specific_total_water_5": {
        "units": "kg/kg",
        "long_name": "specific total water level-5",
    },
    "specific_total_water_6": {
        "units": "kg/kg",
        "long_name": "specific total water level-6",
    },
    "specific_total_water_7": {
        "units": "kg/kg",
        "long_name": "specific total water level-7",
    },
    "surface_temperature": {"units": "K", "long_name": "surface temperature"},
    "tendency_of_total_water_path": {
        "units": "kg/m^2/s",
        "long_name": "time derivative of total water path",
    },
    "tendency_of_total_water_path_due_to_advection": {
        "units": "kg/m^2/s",
        "long_name": "tendency of total water path due to advection",
    },
    "total_water_path": {"units": "kg/m^2", "long_name": "total water path"},
}


def full_variable_name_with_units(variable: str, formatted: bool = True, capitalize: bool = True) -> str:
    """This is a full name for the variable (e.g. for plotting)"""
    if variable not in variable_name_to_metadata:
        return variable
    data = variable_name_to_metadata[variable]
    long_name = data.get("long_name", variable)
    if capitalize:
        long_name = long_name.capitalize()
    # Make long name bold in latex, and units italic
    if formatted is True:
        name = long_name.replace("_", " ").replace(" ", "\\ ")
        if data["units"] == "":
            return f"$\\bf{{{name}}}$"
        else:
            return f'$\\bf{{{name}}}$ [$\\it{{{data["units"]}}}$]'
    elif formatted == "units":
        if data["units"] == "":
            return f"{long_name}"
        else:
            return f'{long_name} [$\\it{{{data["units"]}}}$]'
    else:
        if data["units"] == "":
            return f"{long_name}"
        else:
            return f'{long_name} [{data["units"]}]'


def formatted_units(variable: str) -> str:
    """This is a full name for the variable (e.g. for plotting)"""
    if variable not in variable_name_to_metadata:
        return ""
    data = variable_name_to_metadata[variable]
    return f"[$\\it{{{data['units']}}}$]"


def formatted_long_name(variable: str, capitalize: bool = True) -> str:
    """This is a full name for the variable (e.g. for plotting)"""
    if variable not in variable_name_to_metadata:
        return variable
    data = variable_name_to_metadata[variable]
    long_name = data.get("long_name", variable)
    if capitalize:
        long_name = long_name.capitalize()
    long_name = long_name.replace("_", " ").replace(" ", "\\ ")
    return f"$\\bf{{{long_name}}}$"


def clean_metric_name(metric: str) -> str:
    """This is a clean name for the metrics (e.g. for plotting)"""
    metric_dict = {
        "mae": "MAE",
        "mse": "MSE",
        "crps": "CRPS",
        "rmse": "RMSE",
        "bias": "Bias",
        "mape": "MAPE",
        "ssr": "Spread / RMSE",
        "ssr_abs_dist": "abs(1 - Spread / RMSE)",
        "ssr_squared_dist": "(1 - Spread / RMSE)^2",
        "nll": "NLL",
        "r2": "R2",
        "corr": "Correlation",
        "corrcoef": "Correlation",
        "corr_mem_avg": "Corr. Mem. Avg.",
        "corr_spearman": "Spearman Correlation",
        "corr_kendall": "Kendall Correlation",
        "corr_pearson": "Pearson Correlation",
        "grad_mag_percent_diff": "Gradient Mag. % Diff",
    }
    for k in ["crps", "ssr", "rmse", "grad_mag_percent_diff", "bias"]:
        metric_dict[f"weighted_{k}"] = metric_dict[k]

    return metric_dict.get(metric.lower(), metric)
