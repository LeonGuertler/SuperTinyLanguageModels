import torch
from Levenshtein import distance as levenshtein_distance
from evals.core import BaseModelWrapper
from itertools import zip_longest

from typing import List, Dict, Any


def batch(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


class LoglikelihoodMCQModelWrapper(BaseModelWrapper):
    """ TODO """
    def __init__(self, model):
        """ TODO """
        super().__init__()
        self.model = model
        self.device = model.device

    @torch.no_grad()
    def __call__(
        self,
        prefixes: List[str], 
        continuations: List[str]
    ) -> List[float]:
        """ Compute the loglikelihood of given inputs """
        results = []
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                for prefix_batch, cont_batch in zip(
                    batch(prefixes, 32), batch(continuations, 32) # TODO (legacy) should not be hardcoded
                ):
                    ll = self.model.loglikelihood(prefix_batch, cont_batch)
                    results.extend(ll.cpu().numpy())
        return results


class TextModelingModelWrapper(BaseModelWrapper):
    """ TODO """
    def __init__(self, model, chunk_size):
        """ TODO """
        super().__init__()
        self.model = model 
        self.device = model.device
        self.chunk_size = chunk_size

        self.encoding_function = self.model.embedding_model.tokenize_input
        self.decoding_function = self.model.embedding_model.decode

    def _split_into_chunks(self, text):
        """
        Split the text into chunks of 'chunk_size' words.
        """
        words = text.split()
        return [' '.join(words[i:i + self.chunk_size]) for i in range(0, len(words), self.chunk_size)]
    
    @torch.no_grad()
    def _process_chunk(self, chunk):
        """ TODO """
        input_ids = self.encoding_function(chunk)

        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.model.device)

        logits, _ = self.model(token_ids=input_ids)

        # Shift the input tokens to align them with the predicted tokens
        shift_labels = input_ids[:, 1:].contiguous()
        shift_logits = logits[:, :-1, :].contiguous()

        # Get the predicted tokens (the ones with the highest logit)
        predicted_token_ids = torch.argmax(shift_logits, dim=-1)

        return shift_labels, predicted_token_ids, shift_logits

    @torch.no_grad()
    def __call__(self, reference_text: str) -> Dict[str, Any]:
        """
        Process the reference text and compute metrics.

        Args:
            reference_text (str): The text to process.

        Returns:
            Dict[str, Any]: A dictionary containing metrics.
        """
        # Split the input text into chunks
        chunks = self._split_into_chunks(reference_text)

        # Initialize accumulators
        total_edit_distance = 0
        total_bytes = 0
        total_correct_bytes = 0
        total_loss = 0.0
        total_tokens = 0

        for chunk in chunks:
            shift_labels, predicted_token_ids, shift_logits = self._process_chunk(chunk)

            # Decode input and predicted tokens
            input_text = ''.join(self.decoding_function([shift_labels.squeeze(0).cpu().tolist()]))
            predicted_text = ''.join(self.decoding_function([predicted_token_ids.squeeze(0).cpu().tolist()]))

            # Encode texts to bytes
            input_bytes = input_text.encode("utf-8")
            predicted_bytes = predicted_text.encode("utf-8")

            # Calculate Levenshtein distance
            total_edit_distance += levenshtein_distance(input_bytes, predicted_bytes)

            # Calculate byte accuracy
            for input_byte, predicted_byte in zip_longest(input_bytes, predicted_bytes):
                if input_byte == predicted_byte:
                    total_correct_bytes += 1
                total_bytes += 1

            # Calculate loss (negative log-likelihood)
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))   # Shape: (seq_len - 1, vocab_size)
            shift_labels = shift_labels.view(-1)                          # Shape: (seq_len - 1)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fct(shift_logits, shift_labels)
            total_loss += loss.item()
            total_tokens += shift_labels.numel()

        # Return accumulated results
        return {
            'edit_distance': total_edit_distance,
            'correct_bytes': total_correct_bytes,
            'bytes': total_bytes,
            'loss': total_loss,
            'tokens': total_tokens
        }