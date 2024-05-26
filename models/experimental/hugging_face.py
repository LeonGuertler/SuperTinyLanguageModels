"""An interface for loading in models from the Hugging Face model hub
This can be used for finetuning or training from scratch."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.components.tokenizers.base_class import Tokenizer
from models.embedding_models import EmbedderInterface


class HFTokenizerWrapper(Tokenizer):
    def __init__(self, hf_tokenizer_name):
        self.hf_tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
        self.eot_token = self.hf_tokenizer.eos_token_id
        self.pad_token = self.hf_tokenizer.pad_token_id
        self.vocab_size = self.hf_tokenizer.vocab_size

    def encode(self, text):
        """Encode a text into tokens."""
        return self.hf_tokenizer.encode(text, add_special_tokens=False)

    def encode_batch(self, texts):
        """Encode a batch of texts into tokens.

        Default implementation is to loop over the texts"""
        self.hf_tokenizer.batch_encode_plus(
            texts,
            padding=True,
            return_tensors="pt",
        )

    def pad_batch(self, token_lists):
        """Pad a list of token lists to the same length,
        and return the padded tensor, and mask tensor."""
        padded_tokens, mask = self.hf_tokenizer.pad(
            encoded_inputs=token_lists,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return padded_tokens, mask

    def decode(self, tokens):
        """Decode a list of tokens into a string."""
        return self.hf_tokenizer.decode(tokens, skip_special_tokens=True)

    def decode_batch(self, token_lists):
        """Decode a list of token lists into a list of strings.

        Default implementation is to loop over the token lists."""
        return self.hf_tokenizer.batch_decode(token_lists, skip_special_tokens=True)


class HFEmbedder(EmbedderInterface):
    """
    A class for loading in models from the Hugging Face model hub
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        model_string = model_cfg["model_string"]
        self.tokenizer = ...
        self.model = ...
        # load the model from the model hub
        self.load_model(model_string)

    def load_model(self, model_string):
        """
        Load the model from the Hugging Face model hub
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_string)
        self.embeddings = AutoModelForCausalLM.from_pretrained(
            model_string
        ).get_input_embeddings()

    def forward(self, token_ids):
        """
        Forward pass for the model
        """
        return self.embeddings(token_ids)

    def tokenize_input(self, input_string):
        """
        Tokenize the input string
        """
        return self.tokenizer(input_string, return_tensors="pt")["input_ids"]


class HFTransformerCore(torch.nn.Module):
    """Runs the huggingface transformer model"""

    def __init__(self, model_cfg):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_cfg["model_string"])

    def forward(self, x):
        """Calls the huggingface model in question"""
        return self.model(inputs_embeds=x).logits


class HFLMHead(torch.nn.Module):
    """Poses as the language model head but is just an identity function"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = torch.nn.Identity()

    def forward(self, x):
        """Should return the logits and optionally a loss"""
        return self.model(x), None