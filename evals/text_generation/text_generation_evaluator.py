"""
Evaluate the quality of some generated text by calculating the everage perplexity
of three larger models on the generated text.
"""
import torch 
import numpy as np


class TextGenerationEvaluator:
    def __init__(self, model):
        self.model = model 

        # Ensuret he model is in evaluation mode
        self.model.eval()

        self.ppl_models = {
            "llama-3-8b": None
        }

        self.prompts = [
            "generate some text"
        ]

    def evaluate(self):
        """
        For each prompt, first, generate some text using the model.
        Then calculate the average perplexity for each control model.
        Finally, return the average perplexity of the control models.
        """
        
        perplexities = []
        for prompt in self.prompts:
            generated_text = self.model.generate(prompt)
            for model_name, model in self.ppl_models.items():
                # TODO use the model to calculate per-token perplexity
                perplexities.append(perplexity)

        
        return np.mean(perplexities)
        