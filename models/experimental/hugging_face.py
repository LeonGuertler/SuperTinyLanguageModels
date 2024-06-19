"""An interface for loading in models from the Hugging Face model hub
This can be used for finetuning or training from scratch."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.components.tokenizers.base_class import Tokenizer
from models.embedding_models import EmbedderInterface
from models.model_shell import ModelShell
from trainers.base_trainer import BaseTrainer
#from trainers.datasets import BaseDataloader


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
        self.tokenizer = HFTokenizerWrapper(model_string)
        self.embeddings = AutoModelForCausalLM.from_pretrained(
            model_string,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).get_input_embeddings()

    def decode(self, token_ids):
        """
        Decode the token ids
        """
        return self.tokenizer.decode_batch(token_ids)

    def forward(self, token_ids):
        """
        Forward pass for the model
        """
        return self.embeddings(token_ids)

    def tokenize_input(self, input_string, truncate=False, add_eot=True):
        """This function should take a single input string and returns

        the tokenized input.
        Args:
            input_string: str
            truncate: bool - whether to perform (left) truncation
            add_eot: bool
        Returns:
            typically token_ids of shape (S,)
        """
        token_ids = self.tokenizer.encode(input_string)
        if add_eot:
            token_ids.append(self.tokenizer.eot_token)
        if truncate:
            token_ids = self.truncate([token_ids])[0]
        return token_ids

    def pad_batch(self, token_lists, direction="right"):
        """Pad the token lists into a tensor, and returns a mask"""
        return self.tokenizer.pad_batch(token_lists, direction)

    def truncate(self, token_lists):
        """Truncate the token lists to the max length of the model"""
        max_length = self.model_cfg["context_window"]
        return [tokens[-max_length:] for tokens in token_lists]


class HFTransformerCore(torch.nn.Module):
    """Runs the huggingface transformer model"""

    def __init__(self, model_cfg):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_string"],
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        )

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


class MockTrainer(BaseTrainer):
    """A trainer that skips the training step, but runs e.g. logging"""

    def __init__(
        self,
        cfg,
        model: ModelShell,
        optimizer,
        dataloader,
        loss_fn,
        lr_scheduler=None,
        dropout_scheduler=None,
    ) -> None:
        """Just forward the arguments to the parent class"""
        super().__init__(
            cfg, model, optimizer, dataloader, loss_fn, lr_scheduler, dropout_scheduler
        )

    def _run_step(self, *args, **kwargs):
        return torch.tensor(0.0)

    def _save_model(self, iter_num=0):
        """We don't want to save the model in this case..."""
        pass
