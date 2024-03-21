import torch 
from dataclasses import dataclass 
from models.baseline import baseGPT, the10mmodel

MODEL_CLASSES = {
    "baseline": baseGPT,
    "baseline_ffn_sharing": baseGPT,
    "the_10m_model": the10mmodel,
}


def build_model(config=None, ckpt_path=None):
    # check if model is loaded
    if ckpt_path is not None:
        checkpoint = torch.load(
            config["ckpt_path"],
            map_location="cpu"
        )

        # load model with correct architecture
        model = MODEL_CLASSES[checkpoint['config']['arch']['model']](
            config=checkpoint['config']['arch']
        )
        
        # load model weights
        model.load_state_dict(checkpoint['model_state_dict'])

        return model 
    

    else:
        # build model from scratch
        model = MODEL_CLASSES[config['arch']['model']](
            config=config
        )
        
        return model
    