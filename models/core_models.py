"""
Simple, flexible core models.
"""

import torch

from models.components.layers.transformer_blocks import GenericTransformerBlock


class GenericTransformer(torch.nn.Module):
    """
    Generic Transformer Class intended to be used for as
    broad a range of transformer models as possible.
    """

    def __init__(self, model_cfg, teacher_model_cfg=None):
        super().__init__()

        # build the transformer
        self.transformer = torch.nn.ModuleDict(
            {
                "drop": torch.nn.Dropout(),
                "h": torch.nn.ModuleList(
                    [
                        GenericTransformerBlock(
                            hidden_dim=model_cfg["hidden_dim"],
                            context_window=model_cfg["context_window"],
                            use_rope=model_cfg["positional_encoding_type"] == "rope",
                            ffn_cfg=model_cfg["core_model"]["ffn"],
                            attn_cfg=model_cfg["core_model"]["attn"],
                        )
                        for _ in range(model_cfg["core_model"]["num_layers"])
                    ]
                ),
            }
        )

    def forward(self, x):
        """
        Pass an input through the model
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            x: torch.tensor(B, S, H)
        """

        # apply dropout
        x = self.transformer.drop(x)

        # create the lists for attention and hidden state
        hidden_states = []
        qk_lists = []

        # pass through the transformer blocks
        for block in self.transformer.h:
            x = block(x)
            hidden_states.append(block.hidden_state)
            qk_lists.append(block.qk_list)

        # assign to class object it's own attention matrices and hidden states
        # this will avoid the need to return them from the forward pass 
        # which may conflict elsewhere.
        self.hidden_states = hidden_states
        self.qk_lists = [item for sublist in qk_lists for item in sublist]

        return x


class GenericFFNSharedTransfomer(GenericTransformer):
    """
    Generic Transformer Class that shares the weights
    between all FFN blocks (similar to
    https://arxiv.org/abs/2402.16840).
    """

    def __init__(self, model_cfg):
        super().__init__(model_cfg=model_cfg)

        # share the weights between transformer blocks
        ffn_0 = self.transformer.h[0].ffn

        for i in range(1, len(self.transformer.h)):
            # find all linear layers in the ffn subnets and tie them to the first layer
            for name, module in ffn_0.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_module = dict(self.transformer.h[i].ffn.named_modules())[
                        name
                    ]
                    target_module.weight = module.weight
                    target_module.bias = module.bias
