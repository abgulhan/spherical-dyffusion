#   Usage:
#       python run_inference.py <path_to_yaml_config>
#
#   Debug with:
#       python run_inference.py "src/configs/inference/ckpt_from_local.yaml"
import argparse
from src.ace_inference.inference.inference import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str, help="Path to the yaml config file for inference", default="src/configs/inference/ckpts_from_huggingface_debug.yaml")

    args = parser.parse_args()
    main(yaml_config=args.yaml_config)
