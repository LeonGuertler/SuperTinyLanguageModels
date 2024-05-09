"""
The standard Model Shell. It combines the embedding model, 
core model and LM head.
"""
import torch 





class ModelShell(torch.nn.Module):
    """
    Unify the embedding model, core model and LM head
    into a single object; initializes the weights
    and prints basic model statistics.
    """
    def __init__(
        self, 
        embedding_model, 
        core_model, 
        model_head,
        weight_init_func=None
    ):
        super().__init__()
        self.embedding_model = embedding_model
        self.core_model = core_model
        self.model_head = model_head

        # initialize model weights
        if weight_init_func is not None:
            self.apply(weight_init_func)


    def forward(self, token_ids):
        """
        The default forward pass is used for trianing and 
        accepts the token_ids as input.
        """

        # pass the token_ids through the embedding model 
        # to get B, S, H (with pos encoding if necessary)
        x = self.embedding_model(token_ids)

        # pass the embeddings through the core model
        x = self.core_model(x)

        # pass the core model output through the model head
        x = self.model_head(x)

        return x
    
    @torch.no_grad()
    def inference(self, model_input):
        """
        Takes a string or list of token ids as input,
        and returns the decoded model output. The actual 
        decoding should happen in the decoding generator.
        Args:
            model_input: str or torch.tensor(B, S)
        Returns:
            logits: torch.tensor(B, S, V)
        """

        # check if input is string 
        if isinstance(model_input, str):
            # use inference function of the embedding model
            x = self.embedding_model.inference(model_input)
        else:
            # use standard forward function of the embedding model
            x = self.embedding_model(model_input)

        # pass the embeddings through the core model
        x = self.core_model(x)

        # pass the core model output through the model head
        logits = self.model_head.inference(x)

        return logits 

