import os
import re
from typing import Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from src.utilities.utils import get_logger


log = get_logger(__name__)


# try:
#     torch.serialization.add_safe_globals([ListConfig])
# except AttributeError:
#     log.warning("torch.serialization.add_safe_globals([ListConfig]) not supported in this version of PyTorch")


def get_local_ckpt_path(
    config: DictConfig,
    wandb_run,  #: wandb.apis.public.Run,
    ckpt_filename: str = "last.ckpt",
    throw_error_if_local_not_found: bool = False,
) -> Optional[str]:
    potential_dirs = [
        config.ckpt_dir,
        os.path.join(config.work_dir.replace("-test", ""), "checkpoints"),
        os.path.join(os.getcwd(), "results", "checkpoints"),
    ]
    for callback_k in config.get("callbacks", {}).keys():
        if "checkpoint" in callback_k and config.callbacks[callback_k] is not None:
            potential_dirs.append(config.callbacks[callback_k].dirpath)

    for local_dir in potential_dirs:
        log.info(f"Checking {local_dir}. {os.path.exists(local_dir)=}")
        if not os.path.exists(local_dir):
            continue
        if wandb_run.id not in local_dir:
            local_dir = os.path.join(local_dir, wandb_run.id)
            if not os.path.exists(local_dir):
                continue
        ckpt_files = [f for f in os.listdir(local_dir) if f.endswith(".ckpt")]
        if ckpt_filename == "last.ckpt":
            ckpt_files = [f for f in ckpt_files if "last" in f]
            if len(ckpt_files) == 0:
                continue
            elif len(ckpt_files) == 1:
                latest_ckpt_file = ckpt_files[0]
            else:
                # Get their epoch numbers from inside the file
                # epochs = [torch.load(os.path.join(local_dir, f), weights_only=True)["epoch"] for f in ckpt_files]
                epochs = [torch.load(os.path.join(local_dir, f))["epoch"] for f in ckpt_files]
                # Find the ckpt file with the latest epoch
                latest_ckpt_file = ckpt_files[np.argmax(epochs)]
                log.info(
                    f"Found multiple last-v<V>.ckpt files. Using the one with the highest epoch: {latest_ckpt_file}. ckpt_to_epoch: {dict(zip(ckpt_files, epochs))}"
                )
            return os.path.join(local_dir, latest_ckpt_file)

        elif ckpt_filename in ["earliest_epoch", "latest_epoch", "earliest_epoch_any", "latest_epoch_any"]:
            # Find the earliest epoch ckpt file
            if ckpt_filename in ["earliest_epoch_any", "latest_epoch_any"]:
                ckpt_files = [f for f in ckpt_files if "epoch" in f]
            else:
                ckpt_files = [f for f in ckpt_files if "epoch" in f and "epochepoch=" not in f]
            if len(ckpt_files) == 0:
                continue

            # Function to extract the epoch number from the filename
            def get_epoch_number(filename):
                if "_any" in ckpt_filename:
                    filename = filename.replace("epochepoch=", "epoch")  # Fix for a bug in the filename
                match = re.search(r"_epoch(\d+)_", filename)
                return int(match.group(1))

            # Find the ckpt file with the earliest epoch
            min_or_max = min if ckpt_filename == "earliest_epoch" else max
            earliest_ckpt_file = min_or_max(ckpt_files, key=lambda f: get_epoch_number(f))
            log.info(f"For ckpt_filename={ckpt_filename}, found ckpt file: {earliest_ckpt_file} in {local_dir}")
            return os.path.join(local_dir, earliest_ckpt_file)

        ckpt_path = os.path.join(local_dir, ckpt_filename)
        if os.path.exists(ckpt_path):
            return ckpt_path
        else:
            log.warning(f"{local_dir} exists but could not find {ckpt_filename=}. Files in dir: {ckpt_files}.")
    if ckpt_filename in ["earliest_epoch", "latest_epoch", "earliest_epoch_any", "latest_epoch_any"]:
        raise NotImplementedError("Not implemented")
    if throw_error_if_local_not_found:
        raise FileNotFoundError(
            f"Could not find ckpt file {ckpt_filename} in any of the potential dirs: {potential_dirs}"
        )
    return None


def download_model_from_hf(
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    hf_path: Optional[str] = None,
    cache_dir: str = "auto",
):
    """
    Downloads a model file from Hugging Face Hub

    Args:
        repo_id (str): Hugging Face repository ID, e.g. "username/repo_name"
        filename (str): Name of the file to download, e.g. "model.pt"
        hf_path (str): Path to the model on Hugging Face Hub. "<repo_id>/<filename>"
        cache_dir (str): Local directory to save the model

    Returns:
        str: Path to the downloaded file
    """
    if hf_path is not None:
        assert repo_id is None and filename is None, "hf_path should be used alone"
        # After last / is the filename
        repo_id, filename = hf_path.rsplit("/", 1)
    else:
        assert repo_id is not None and filename is not None, "repo_id and filename should be used together"

    if filename.endswith(".ckpt") or filename.endswith(".pt"):
        dtype = "model"
        cache_dir = ".cache/models/" if cache_dir == "auto" else cache_dir
    elif filename.endswith(".yaml"):
        dtype = "config"
        cache_dir = ".cache/configs/" if cache_dir == "auto" else cache_dir
    else:
        dtype = "data"
        cache_dir = ".cache/data/" if cache_dir == "auto" else cache_dir

    os.makedirs(cache_dir, exist_ok=True)

    log.info(f"Downloading {dtype} from Hugging Face Hub: {repo_id}/{filename}. Saving to {cache_dir=}")
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)

    return model_path


def local_path_to_absolute_and_download_if_needed(path: str) -> Optional[str]:
    """
    Convert a local path to an absolute path and download the file if it is a Hugging Face Hub path (starts with "hf:")
    """
    if path is None:
        return None
    if path.startswith("hf:"):
        # Download ckpt from huggingface hub
        #   e.g. "hf:salv47/spherical-dyffusion/interpolator-sfno-best-val_avg_crps.ckpt"
        hf_path = path.replace("hf:", "")
        path = download_model_from_hf(hf_path=hf_path)
    path = os.path.abspath(path)
    return path
