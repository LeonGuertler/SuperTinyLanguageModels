"""
Evaluator class for evaluating models.
"""
import os
import torch
import torch.nn.functional as F
from Levenshtein import distance as levenshtein_distance
from evals.evaluator_interface import EvaluationInterface


class TextModelingEvaluator(EvaluationInterface):
    """
    Evaluator class that evaluates models on their language modeling 
    capabilities in a way that is agnostic to the tokenizer used, using byte-level accuracy.
    """
    def __init__(self, model, eval_dir):
        self.model = model 

        # Ensure the model is in evaluation mode
        self.model.eval()

        self.modeling_topics = os.listdir(
            os.path.join(eval_dir, "text_modeling", "test_data")
        )
        self.modeling_difficulties = ["easy", "medium", "hard"]
        self.eval_dir = eval_dir

        # Load the text data
        self._load_data()

    def _load_data(self):
        """
        Load all modeling texts into a dictionary.
        """
        self.data = {}
        for topic in self.modeling_topics:
            self.data[topic] = {}
            for difficulty in self.modeling_difficulties:
                file_path = os.path.join(
                        self.eval_dir, "text_modeling", "test_data", topic, f"{difficulty}.txt"
                    )
                with open(file_path, "r") as f:
                    self.data[topic][difficulty] = f.read()

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

        return shift_labels, predicted_token_ids


    def evaluate(self):
        """
        Evaluate the model on text modeling capabilities.
        """
        results = {} 
        for topic in self.modeling_topics:
            for difficulty in self.modeling_difficulties:
                reference_text = self.data[topic][difficulty]

                # Split the text into chunks
                chunks = self._split_into_chunks(reference_text)
                # TODO the chunks should be stacked and run simulataneously
                total_edit_distance = 0
                count = 0

                for chunk in chunks:
                    input_ids, predicted_ids = self._process_chunk(chunk)

                    for input_id, predicted_id in zip(input_ids[0], predicted_ids[0]):
                        input_text = self.model.embedding_model.decode([[input_id.item()]])# , skip_special_tokens=True)
                        predicted_text = self.model.embedding_model.decode([[predicted_id.item()]]) #, skip_special_tokens=True)
                        input_text_enc = input_text[0].encode("utf-8")
                        total_edit_distance += levenshtein_distance(
                            input_text_enc, 
                            predicted_text[0].encode("utf-8")
                        )
                        # increment count by num bytes
                        count += len(input_text_enc)

                # Average edit distance over all chunks
                avg_edit_distance = total_edit_distance / count

                if topic not in results:
                    results[topic] = {}
                results[topic][difficulty] = avg_edit_distance

        return results