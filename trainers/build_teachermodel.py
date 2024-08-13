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
        ## attention heads
        self.student_dim = model_cfg.core_model.attn.num_heads
        self.teacher_dim = teacher_model_cfg.num_attention_heads
        self.projection_attn = nn.Linear(self.student_dim, self.teacher_dim, bias=False)

        ## hidden state
        self.student_hs_dim = model_cfg.hidden_dim
        self.teacher_hs_dim = teacher_model_cfg.hidden_size
        self.projection_hs = nn.Linear(self.student_hs_dim, self.teacher_hs_dim)

    def forward(self, attn, hidden_state):
        return self.projection_attn(attn), self.projection_hs(hidden_state)


def build_attention_projection(model_cfg, teacher_model_cfg):

    return ProjectionLayers(model_cfg, teacher_model_cfg)

def init_teachermodel(cfg):

    # load the teacher model
    cfg.teachermodel["model_ckpt"] = hydra.utils.to_absolute_path(cfg.teachermodel["model_ckpt"]) # get the absolute path of the teacher model checkpoint
    teacher_model = build_model(checkpoint=torch.load(cfg.teachermodel["model_ckpt"])) # load the teacher model
    teacher_model.to(cfg["general"]["device"]) # move the teacher model to the device
    teacher_model.eval() # set the teacher model to evaluation mode
    teacher_model_cfg = teacher_model.core_model.model.config # get the teacher model configuration

    # build the attention projection layer
    attention_projection = build_attention_projection(model_cfg=cfg["model"], teacher_model_cfg=teacher_model_cfg)
    attention_projection.to(cfg["general"]["device"])

    return teacher_model, attention_projection