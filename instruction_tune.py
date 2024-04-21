"""
Simple instruction tuning abstraction.
"""
import hydra 
from tuners.build_tuners import build_tuner


@ hydra.main(config_path="configs/tune", config_name="baseline")