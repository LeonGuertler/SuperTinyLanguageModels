"""
A collection of data generating strategies that are used by the model to generate 
data responses for any given input, which the model can then use to train further.

The data generation strategies will can range from simple data generation strategies 
to monte carlo tree search strategies. 
"""

import hydra
import torch

from models.build_models import build_model
from models.components.utils.generation_utils import get_generator_type

@hydra.main(config_path="configs", config_name="process_generation")
def main(cfg):
    # set the checkpoint path to absolute path
    cfg["model_ckpt"] = hydra.utils.to_absolute_path(cfg["model_ckpt"])
    cfg["value_model_ckpt"] = hydra.utils.to_absolute_path(cfg["value_model_ckpt"])

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    
    ## load the base model
    model, _ = build_model(
        checkpoint_path=cfg["model_ckpt"],
        device=device,
        attention_type=cfg["attention_type"]
    )
    model.eval() # ensure that is in eval mode

    ## load the value model
    value_model, _ = build_model(
        checkpoint_path=cfg["value_model_ckpt"],
        device=device,
        attention_type=cfg["attention_type"]
    )
    value_model.eval() # ensure that is in eval mode

    ## get the generator type based on the strategy provided in the config
    ## the strategy should have the 
    generator = get_generator_type(
        model=model,
        value_model=value_model,
        generate_cfg=cfg["generator"],
        strategy_cfg=cfg["process_strategy"],
        device=device
    )

    ## load the input text which can be as a list or a dataset from huggingface.
    input_text = cfg["input_text"] ## TODO - dataloader or just self inserted in .yaml
    if isinstance(input_text, str):
        input_text = [input_text]
    
    ## generate N responses for each input text
    ## can this be made more efficient?
    N = cfg["samples_per_input_text"]
    input_text_data = []
    generated_data = []
    generated_values = []

    for text in input_text:
        for _ in range(N):
            generated_text, value = generator.generate_data(text) ## value *= 
            if value > cfg["process_strategy"]["value_threshold"]: ## only store the responses that are above the value threshold
                input_text_data.append(text)
                generated_data.append("".join(generated_text))
                generated_values.append(value)
        
    ## save the generated data responses and the value of the responses ## TODO
    with open(cfg["output_path"], "w") as f:
        f.write("Input Text,Generated Text,Value\n")
        for input_text, generated_text, value in zip(input_text_data, generated_data, generated_values):
            f.write(f"{input_text},{generated_text},{value}\n")

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter