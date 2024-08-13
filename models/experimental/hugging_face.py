"""An interface for loading in models from the Hugging Face model hub
This can be used for finetuning or training from scratch."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.components.tokenizers.base_class import Tokenizer
from models.embedding_models import EmbedderInterface
from models.model_shell import ModelShell
from trainers.base_trainer import BaseTrainer
from trainers.dataloader import BaseDataloader

def build_model(model_cfg):
    '''
    Helper function to build a model from the huggingface model hub.
    '''
    ## get the model string
    model_str = model_cfg["model_string"]

    ## check if we are using flash attention
    flash_attn = model_cfg.get("flash_attention", False)
    if flash_attn:
        attn_impl = "flash_attention_2"
    else:
        attn_impl = "eager"

    ## load the model from the model hub
    model = AutoModelForCausalLM.from_pretrained(
        model_str,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        torch_dtype=torch.float32,
    )

    return model


class HFTokenizerWrapper(Tokenizer):
    def __init__(self, hf_tokenizer_name):
        self.hf_tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
        self.eot_token = self.hf_tokenizer.eos_token_id
        self.pad_token = self.hf_tokenizer.pad_token_id
        self.vocab_size = self.hf_tokenizer.vocab_size
        if self.pad_token is None:
            self.pad_token = self.eot_token

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
        self.tokenizer = HFTokenizerWrapper(model_string)
        self.embeddings = build_model(model_cfg).get_input_embeddings()

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
    """
    Hugging Face transformer class.
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.model = build_model(model_cfg = model_cfg)

        ## freeze the parameters
        print("Note: Freezing the parameters of the hf_core model.")
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Calls the huggingface model in question, and returns the last hidden state.
        """
        ## get the hidden states
        self.hidden_states = self.model(inputs_embeds = x, output_hidden_states = True).hidden_states

        ## return the last hidden state
        if isinstance(self.hidden_states, tuple):
            return self.hidden_states[-1]

        

class HFLMHead(torch.nn.Module):
    """
    Takes the language model head of a Hugging Face transformer class.
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.lm_head = build_model(model_cfg = model_cfg).get_output_embeddings()
    
    def forward(self, x):
        """
        Passes the input through the language model head to get logits.
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            x: torch.tensor(B, S, V)
        """
        # Apply the language model head to get logits
        return self.lm_head(x), None


class MockTrainer(BaseTrainer):
    """A trainer that skips the training step, but runs e.g. logging"""

    def __init__(
        self,
        cfg,
        model: ModelShell,
        optimizer,
        dataloader: BaseDataloader,
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
