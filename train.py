import hydra.utils
from omegaconf import DictConfig, OmegaConf
from trainers import base_trainer


@hydra.main(config_path="configs/train/", config_name="baseline.yaml")
def main(model_cfg: DictConfig) -> None:
    # Load the general config file
    general_cfg_path = hydra.utils.to_absolute_path("configs/general_config.yaml")
    general_cfg = OmegaConf.load(general_cfg_path)

    # Merge the general configuration with the nanoGPT configuration
    cfg = OmegaConf.merge(general_cfg, model_cfg)

    # Trainer
    trainer = base_trainer.build_trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
