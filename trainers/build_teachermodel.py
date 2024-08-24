'''
Class for the projection layer of the unnormalized attention mechanism.
'''

import hydra
import torch
import torch.nn as nn
from models.build_models import build_model

class ProjectionLayers(nn.Module):

    def __init__(self, model_cfg, teacher_model_cfg):
        super(ProjectionLayers, self).__init__()

        ## hidden state
        self.student_hs_dim = model_cfg.hidden_dim
        self.teacher_hs_dim = teacher_model_cfg.hidden_size
        self.projection_hs = nn.Linear(self.student_hs_dim, self.teacher_hs_dim)

        ## embedding state
        self.projection_emb = nn.Linear(self.student_hs_dim, self.teacher_hs_dim)


def build_projection(model_cfg, teacher_model_cfg):

    return ProjectionLayers(model_cfg, teacher_model_cfg)

def init_teachermodel(cfg):

    # load the teacher model
    if "model_ckpt" in cfg.teachermodel:
        # set the checkpoint path to absolute path
        cfg.teachermodel["model_ckpt"] = hydra.utils.to_absolute_path(cfg.teachermodel["model_ckpt"]) # get the absolute path of the teacher model checkpoint
        teacher_model = build_model(checkpoint=torch.load(cfg.teachermodel["model_ckpt"])) # load the teacher model
    # otherwise build the model from scratch (e.g. for external pretrained models)
    else:
        teacher_model = build_model(model_cfg=cfg.teachermodel["model"])
        
    teacher_model.to(cfg["general"]["device"]) # move the teacher model to the device
    teacher_model.eval() # set the teacher model to evaluation mode
    teacher_model_cfg = teacher_model.core_model.model.config # get the teacher model configuration

    ## determine if should build the attention projection layer
    if cfg.teachermodel.get("build_projection", None):
        attention_projection = build_projection(model_cfg=cfg["model"], teacher_model_cfg=teacher_model_cfg)
        attention_projection.to(cfg["general"]["device"])
        print("Attention projection layer built")
    else:
        attention_projection = None
        print("No attention projection layer built")

    return teacher_model, attention_projection