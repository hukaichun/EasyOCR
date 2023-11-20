import os

import torch
import torch.optim as optim
import yaml

import dataset as DS
import finetune as FT

import logging
logging.basicConfig(level=logging.WARNING)




def main():
    root_path = os.path.dirname(__file__)
    config_path = "./config_files/fintuning_config.yaml"
    
    with open(config_path, 'r', encoding="utf8") as f:
        general_config = yaml.safe_load(f)

    with open("/".join([root_path, general_config["data_config"]]), "r", encoding="utf8") as f:
        data_config = yaml.safe_load(f)
        print(data_config)
    
    with open("/".join([root_path, general_config["model_config"]]), "r", encoding="utf8") as f:
        model_config = yaml.safe_load(f)
        print(model_config)

    with open("/".join([root_path, general_config["training_config"]]), "r", encoding="utf8") as f:
        training_config = yaml.safe_load(f)
        print(training_config)


if __name__ == "__main__":
    main()