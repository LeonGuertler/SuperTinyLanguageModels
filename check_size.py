"""
The main training code
"""
import os

import hydra

from models.build_models import build_model
from trainers.build_trainers import build_trainer, ddp_setup
from trainers.utils import create_folder_structure, init_print_override, restore_print_override
from models.utils import print_model_stats

import torch
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp

from trainers.prepare import prepare_data



@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    if "full_configs" in cfg:
        cfg = cfg["full_configs"]
    model = build_model(model_cfg=cfg["model"])

    # print full parameter count
    print_model_stats(model)



if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter