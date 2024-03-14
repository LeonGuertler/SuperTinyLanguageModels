import torch 
from dataclasses import dataclass 
from models.baseline import baseGPT 

MODEL_CLASSES = {
    "baseline": baseGPT,
    "baseline_ffn_sharing": baseGPT,
}


def build_model(config=None, ckpt_path=None):
    # check if model is loaded
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)

        # load model with correct architecture
        model = MODEL_CLASSES[checkpoint['config']['model_type']](
            config=checkpoint['config']['arch']
        )
        
        # load model weights
        model.load_state_dict(checkpoint['model_state_dict'])

        return model 
    

    else:
        # build model from scratch
        model = MODEL_CLASSES[config.model_type](
            config=config['arch']
        )
        
        return model
    