import torch 
from typing import Optional
from models import core_models, embedding_models, model_heads


class DualModelShell(torch.nn.Module):
    def __init__(
        self,
        model_cfg,
        embedding_model: embedding_models.EmbedderInterface,
        core_model: core_models.GenericTransformer,
        model_head: model_heads.AutoregressiveLMHead,
    ):
        super().__init__()
        self.embedding_model = embedding_model
        self.core_model = core_model
        self.model_head = model_head

        self.core_model.set_embedder(
            embedder=self.embedding_model
        )


        # check if embedding model weights are to be shared with the model head
        if model_cfg.get("embedding_weight_tying", True):
            # share the weights between the token embeddings and the final
            # logit layer, following: https://paperswithcode.com/method/weight-tying
            assert model_head.linear.weight.shape == embedding_model.token_embedder.weight.shape, \
                "The embedding model and the model head should have the same output dimension."
            embedding_model.token_embedder.weight = model_head.linear.weight

        self.device = ...

    # override to device to set the attribute
    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)


    def forward(self, token_ids, delimitations, attn_mask: Optional[torch.Tensor]=None):
        # embed the token ids
        x = self.embedding_model(token_ids)

        # pass the embedded tokens and delimitations to the core model 
        x = self.core_model(x, delimitations)

        # pass the core model output throught he model head
        x = self.model_head(x)

        return x 


