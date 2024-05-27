"""
Pytest for debugging the llm eval code.
"""
from dataclasses import dataclass

import pytest
import torch

from models.experimental import hugging_face
from models import model_shell
from evals.llm_harness import LMEvalWrappedModel
from lm_eval.models import huggingface

@dataclass
class RequestModel:
    """Class for testing the LLM model."""
    args: tuple[str, str]


def test_lm_eval_wrapper():
    """
    Test the generic core model.
    """
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    modelpath = "microsoft/Phi-3-mini-4k-instruct"
    hf_model = huggingface.HFLM(
        pretrained=modelpath,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        dtype=torch.float16
    )
    model_config = {
            "model_string": modelpath
        }
    shell = model_shell.ModelShell(
        hugging_face.HFEmbedder(model_cfg=model_config),
        hugging_face.HFTransformerCore(model_cfg=model_config),
        hugging_face.HFLMHead(model_cfg=model_config)
    )
    shell.to(torch.device("cuda"))
    shell.eval()
    model = LMEvalWrappedModel(shell)
    context_str = "The capital of France is"
    target_str = "Paris"
    requests = [
        RequestModel(args=(context_str, target_str))
    ]
    results = model.loglikelihood(requests)
    results_target = hf_model.loglikelihood(requests)
    assert results[0] == results_target[0]
    assert results == results_target

