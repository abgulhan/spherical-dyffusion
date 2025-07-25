import os

import hydra
import wandb
from omegaconf import DictConfig

from src.train import run_model
from src.utilities.utils import get_logger

log = get_logger(__name__)

log.setLevel("DEBUG") ###

if "CONFIG_PATH" in os.environ:
    # Split config path and config name from config path (split by last '/')
    config_path, config_name = os.environ["CONFIG_PATH"].rsplit("/", 1)
    log.info(f"Using config path from environment variable: {os.environ['CONFIG_PATH']}")
else:
    config_path = "src/configs/"
    config_name = "main_config.yaml"


@hydra.main(config_path=config_path, config_name=config_name, version_base=None)
def main(config: DictConfig) -> float:
    """Run/train model based on the config file configs/main_config.yaml (and any command-line overrides)."""
    return run_model(config)


if __name__ == "__main__":
    if "WANDB_API_KEY" in os.environ:
        if str(os.environ.get("WANDB_MODE")).lower() in ["disabled", "offline"] or str(os.environ.get("WANDB_DISABLED")).lower() == "true":
            print("Weights & Biases logging is disabled.")
        else:
            print("Logging to Weights & Biases")
            wandb.login(key=os.environ["WANDB_API_KEY"])

    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
