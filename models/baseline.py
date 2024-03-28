import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

# import the layers
from models.layers import LayerNorm, CausalSelfAttention, FFN

from models.tokenizer import tokenizer


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config["arch"]["hidden_dim"], bias=config["arch"]["bias"])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config["arch"]["hidden_dim"], bias=config["arch"]["bias"])
        self.mlp = FFN(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class baseGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config["arch"]["vocab_size"] is not None
        assert config["arch"]["context_window"] is not None
        self.config = config
        self.tokenizer = tokenizer(config=config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer.device = self.device

        # prepare the dataset if necessary
        # self.tokenizer.prepare_dataset()

        # construct the actual model
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config["arch"]["vocab_size"], config["arch"]["hidden_dim"]
                ),
                wpe=nn.Embedding(
                    config["arch"]["context_window"], config["arch"]["hidden_dim"]
                ),
                drop=nn.Dropout(config["arch"]["dropout"]),
                h=nn.ModuleList(
                    [Block(config) for _ in range(config["arch"]["depth"])]
                ),
                ln_f=LayerNorm(
                    config["arch"]["hidden_dim"], bias=config["arch"]["bias"]
                ),
            )
        )
        self.lm_head = nn.Linear(
            config["arch"]["hidden_dim"], config["arch"]["vocab_size"], bias=False
        )
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config["arch"]["depth"])
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_batch(self, split="train"):
        return self.tokenizer.get_batch(split=split)

    def embed(self, idx):
        with torch.no_grad():
            x = self.transformer.wte(idx)
            pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
            pos = self.transformer.wpe(pos)
            x = x + pos
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)
        return x

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config["arch"]["context_window"]
        ), f"Cannot forward sequence of length {t}, block size is only {self.config['arch']['context_window']}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, input_text, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        idx = self.tokenizer.encode_text(input_text, device=self.device)

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # input(idx)
            idx_cond = (
                idx
                if idx.size(1) <= self.config["arch"]["context_window"]
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return self.tokenizer.decode_tokens(idx[0].tolist())
