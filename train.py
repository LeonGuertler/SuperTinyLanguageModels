"""
The main training code
"""
import hydra

from trainers.build_trainers import build_trainer

@hydra.main(config_path="configs/train", config_name="baseline")
def main(cfg):
    input(cfg)
    # load the relevant trainer
    trainer = build_trainer(
        cfg=cfg,
    )
    # preprocess the training data
    trainer.preprocess_data()

    # train the model
    trainer.train()



if __name__ == "__main__":
    main()



