"""
Evaluator class for evaluating models.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np 
from datasets import load_dataset
from Levenshtein import distance as levenshtein_distance
from evals.evaluator_interface import EvaluationInterface


class TextModelingEvaluator(EvaluationInterface):
    """
    Evaluator class that evaluates models on their language modeling 
    capabilities in a way that is agnostic to the tokenizer used, using byte-level accuracy.
    """
    def __init__(self, model, topic_list):
        self.model = model 
        self.topic_list = topic_list

        # Ensure the model is in evaluation mode
        self.model.eval()

        # Load the text data
        self.data = load_dataset("SuperTinyLanguageModels/text-modeling-eval")["train"]


    def _split_into_chunks(self, text, chunk_size=100):
        """
        Split the text into chunks of 'chunk_size' words.
        """
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def _process_chunk(self, chunk):
        """
        Process a chunk of text by predicting the next word after the chunk.
        """
        input_ids = self.model.embedding_model.tokenize_input(chunk) #, return_tensors="pt")

        # convert to tensor
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.model.device)

        # Get logits from the model (normal forward pass)
        with torch.no_grad():
            logits, _ = self.model(token_ids=input_ids)


        # Shift the input tokens to align them with the predicted tokens
        shift_labels = input_ids[:, 1:].contiguous()
        shift_logits = logits[:, :-1, :].contiguous()

        # Get the predicted tokens (the ones with the highest logit)
        predicted_token_ids = torch.argmax(shift_logits, dim=-1)

        return shift_labels, predicted_token_ids, shift_logits


    def evaluate(self):
        """
        Evaluate the model on text modeling capabilities.
        """
        results = {} 
        for i in range(len(self.data)):
            reference_text = self.data[i]["text"]
            category = self.data[i]["topic"]
            difficulty = self.data[i]["difficulty"]

            # Split the text into chunks
            chunks = self._split_into_chunks(reference_text)

            # TODO the chunks should be stacked and run simulataneously
            total_edit_distance = 0
            count = 0
            byte_correct = 0
            byte_count = 0

            byte_perplexity_total = 0

            for chunk in chunks:
                input_ids, predicted_ids, predicted_token_logits = self._process_chunk(chunk)

                for input_id, predicted_id, pred_logits in zip(input_ids[0], predicted_ids[0], predicted_token_logits[0]):
                    input_text = self.model.embedding_model.decode([[input_id.item()]])# , skip_special_tokens=True)
                    predicted_text = self.model.embedding_model.decode([[predicted_id.item()]]) #, skip_special_tokens=True)
                    input_text_enc = input_text[0].encode("utf-8")
                    total_edit_distance += levenshtein_distance(
                        input_text_enc, 
                        predicted_text[0].encode("utf-8")
                    )
                    # increment count by num bytes
                    count += len(input_text_enc)

                    # calculate byte accuracy
                    for byte_idx in range(len(input_text_enc)):
                        if byte_idx < len(predicted_text[0].encode("utf-8")):
                            if input_text_enc[byte_idx] == predicted_text[0].encode("utf-8")[byte_idx]:
                                byte_correct += 1
                            byte_count += 1

                    # calculate byte perplexity
                    byte_perplexity_total += torch.exp(
                        F.cross_entropy(
                            pred_logits.unsqueeze(0).softmax(dim=-1),
                            input_id.unsqueeze(0)
                        )
                    ).item()*len(input_text_enc)

            if category not in results:
                results[category] = {}

            if difficulty not in results[category]:
                results[category][difficulty] = {
                    'Byte Acc.': [],
                    'Norm. Lev. Dist': [],
                    'Byte Perplexity': []
                }


            results[category][difficulty]['Norm. Lev. Dist'].append(total_edit_distance / count)
            results[category][difficulty]["Byte Acc."].append(byte_correct / byte_count)
            results[category][difficulty]["Byte Perplexity"].append(byte_perplexity_total / count)

        structured_output = {}
        for category in results.keys():
            for difficulty in results[category].keys():
                structured_output[f"Text Modeling (Byte Acc.)/{category}-{difficulty}"] = np.mean(results[category][difficulty]["Byte Acc."])
                structured_output[f"Text Modeling (Byte Lev. Dist.)/{category}-{difficulty}"] = np.mean(results[category][difficulty]["Norm. Lev. Dist"])
                structured_output[f"Text Modeling (Byte Perplexity)/{category}-{difficulty}"] = np.mean(results[category][difficulty]["Byte Perplexity"])


                #structured_output[f"{category}/Byte Acc./{difficulty}"] = np.mean(results[category][difficulty]["Byte Acc."])
                #structured_output[f"{category}/Norm. Lev. Dist/{difficulty}"] = np.mean(results[category][difficulty]["Norm. Lev. Dist"])
                #structured_output[f"{category}/Byte Perplexity/{difficulty}"] = np.mean(results[category][difficulty]["Byte Perplexity"])

        return structured_output