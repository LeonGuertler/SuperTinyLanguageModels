import os, time, math, pickle, hydra
from omegaconf import DictConfig, OmegaConf
from contextlib import nullcontext

import numpy as np
import torch

from models.build_models import build_model

import argparse
import os
import numpy as np
import pandas as pd
import time


"""
very simplistic bare-bones MMLU test

"""



@hydra.main(config_path="configs/train/", config_name="baseline.yaml")
def main(model_cfg: DictConfig) -> None:
    # Load the general config file
    general_cfg_path = hydra.utils.to_absolute_path("configs/general_config.yaml")
    general_cfg = OmegaConf.load(general_cfg_path)
    
    # Merge the general configuration with the nanoGPT configuration
    cfg = OmegaConf.merge(general_cfg, model_cfg)

    # set the random seed
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)


    model_path = "../../../pulled/2/ckpt.pt"

    model = build_model(
        config = cfg,
        ckpt_path = model_path
    )

    # generate some text
    text = "The quick brown fox jumps over the lazy dog."
    input(model.generate(text, max_new_tokens=100, top_k=10))


    choices = ["A", "B", "C", "D"]
    idx_list = []

    # determine model output idx for given choices
    # encode each token
    for token in choices:
        idx = model.tokenizer.encode_text(token, device="cpu")
        idx_list.append(idx.item())
        
    idx_list = torch.tensor(idx_list)

    # load questions


    df = pd.read_csv("../../../test.csv")

    idx_ans_dict = {
        0: "A",
        1: "B",
        2: "C",
        3: "D"
    }

    def create_full_text(prompt, A, B, C, D):
        return f"{prompt} A) {A} B) {B} C) {C} D) {D}"

    # iterate over rows and store answer
    y_pred = []
    y_true = []
    for idx in range(len(df)):
        prompt, A, B, C, D, answer = df.iloc[idx]

        # create the actual 2-shot prompt (following phi-1.5)
        # randomly sample 2 other question answer pairs
        # and append them to the prompt
        prompt_1, A_1, B_1, C_1, D_1, answer_1 = df.sample(1).iloc[0]
        prompt_2, A_2, B_2, C_2, D_2, answer_2 = df.sample(1).iloc[0]

        #full_prompt = create_full_text(prompt_1, A_1, B_1, C_1, D_1)
        #full_prompt += f"Answer: {answer_1}"

        #full_prompt += create_full_text(prompt_2, A_2, B_2, C_2, D_2)
        #full_prompt += f"Answer: {answer_2}"

        full_prompt = create_full_text(prompt, A, B, C, D)
        full_prompt += f"Answer: "


        # get answer from model
        token_ids = model.tokenizer.encode_text(full_prompt, device="cpu")
        # truncate to 511 tokens
        token_ids = token_ids[:,:511]
        logits, _ = model(
            token_ids, #model.tokenizer.encode_text(prompt, device="cpu")
        )
        #input(logits.size())
        # reduce to only the choices
        logits = torch.nn.functional.softmax(
            logits[:, :, idx_list], dim=-1
        )

        # get letter answer
        y_pred.append(idx_ans_dict[torch.argmax(logits).item()])
        y_true.append(answer)

        if not idx % 10:
            # get accuracy
            acc = sum([1 for i in range(len(y_pred)) if y_pred[i] == y_true[i]]) / len(y_pred)
            print(f"Accuracy: {acc}")

        #input(logits)

    # print model name and final accuracy
    print(f"Model: {model_path}")
    print(f"Final Accuracy: {sum([1 for i in range(len(y_pred)) if y_pred[i] == y_true[i]]) / len(y_pred)}")
    print(model.config)


main()