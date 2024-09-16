"""
Check model parameter count
"""
import hydra
from models.build_models import build_model
from models.utils import print_model_stats

@hydra.main(config_path="configs/train", config_name="baseline-10m")
def main(cfg):
    if "full_configs" in cfg:
        cfg = cfg["full_configs"]
    model, _ = build_model(model_cfg=cfg["model"])

    # print full parameter count
    print_model_stats(model)



if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter