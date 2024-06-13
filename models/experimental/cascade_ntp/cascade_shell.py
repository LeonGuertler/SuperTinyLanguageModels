import torch
from models import embedding_models, model_heads, model_shell
from models.core_models import GenericTransformer

class CascadeTransformer(GenericTransformer):
    """
    Generic Transformer altered to return all the inputs as a list
    """

    def forward(self, x):
        """
        Pass an input through the model
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            xs: list[torch.tensor(B, S, H)]
        """

        # apply dropout
        x = self.transformer.drop(x)
        xs = []
        # pass through the transformer blocks
        for block in self.transformer.h:
            x = block(x)
            xs.append(x)

        return xs

class CascadeShell(model_shell.ModelShell):
    """
    Unify the embedding model, core model and LM head
    into a single object; initializes the weights
    and prints basic model statistics.
    """

    def __init__(
        self,
        embedding_model: embedding_models.EmbedderInterface,
        core_model: CascadeTransformer,
        model_head: model_heads.AutoregressiveLMHead,
        weight_init_func=None,
    ):
        super().__init__(
            embedding_model=embedding_model,
            core_model=core_model,
            model_head=model_head,
            weight_init_func=weight_init_func,
        )
        # define a set of thresholds per layer that decrease to 0 as cos(\pi i/2n)
        self.thresholds = torch.tensor(
            [torch.cos(torch.tensor(i * 3.1415 / (2 * len(self.core_model.transformer.h))))
             for i in range(len(self.core_model.transformer.h))],
            dtype=torch.float32,
        )
        # map to log space
        self.thresholds = torch.log(self.thresholds)

    def forward(self, token_ids):
        """
        The default forward pass is used for trianing and
        accepts the token_ids as input.
        """

        # pass the token_ids through the embedding model
        # to get B, S, H (with pos encoding if necessary)
        x = self.embedding_model(token_ids)

        # pass the embeddings through the core model
        xs = self.core_model(x)

        # loop over the outputs of the transformer blocks
        # and pass them through the model head
        logits = []
        aux_losses = []
        for x in xs:
            logit, aux_loss = self.model_head(x)
            logits.append(logit)
            aux_losses.append(aux_loss)
        return logits, aux_losses[-1] # TODO handle better...

    @torch.no_grad()
    def inference(self, model_input):
        """
        Takes a string or list of token ids as input,
        and returns the decoded model output. The actual
        decoding should happen in the decoding generator.
        Args:
            model_input: str or torch.tensor(B, S)
        Returns:
            logits: torch.tensor(B, S, V),
        """

        # check if input is string
        if isinstance(model_input, str):
            # use inference function of the embedding model
            model_input = self.embedding_model.tokenize_input(model_input, truncate=True, add_eot=False)
        model_input = torch.tensor(model_input, device=self.device, dtype=torch.long).unsqueeze(0)
        x = self.embedding_model(model_input)

        # pass the embeddings through the core model
        xs = self.core_model(x)

        # loop over the outputs of the transformer blocks
        # and pass them through the model head maybe
        # logits = torch.zeros(1, x.size(1), self.model_head.vocab_size, device=self.device)
        base_target = model_input[:, 1:]
        # extend the target with the greedy logit selection
        for i,x in enumerate(xs):
            pred_logit = self.model_head.forward(x) # 1, S, V
            greedy_target = pred_logit[:,-1].argmax(-1).unsqueeze(1)
            target = torch.cat([base_target, greedy_target], dim=1)
            path_prob = super().loglikelihood(pred_logit, target) / target.size(1)
            if path_prob < self.thresholds[i]:
                break
            return pred_logit
    
    def loglikelihood(self, logits, target_tensor, mask):
        """
        We use the confidences to determine at which layer the
        model would decode the token and use this to calculate
        the loss."""
        # run through from the first to the last layer, and
        # write the confidence for a token when the greedy
        # i.e. not the actual next token exceeds the threshold for that
        # layer
        confidences = torch.zeros_like(target_tensor, dtype=torch.float32)
        predicted_mask = torch.ones_like(target_tensor, dtype=torch.bool)
        for i, logit in enumerate(logits):
            # first we get the greedy
            _, predicted = logit.argmax(-1)
            # now we get a mask for tokens whereby the greedy loss
            # exceed the threshold
            mask = torch.nn.functional.cross_entropy(
                logit.view(-1, logit.size(-1)),
                predicted.view(-1),
            ) < self.thresholds[i]
            mask = mask.view(target_tensor.size())
            # and this with the predicted mask
            mask = mask & predicted_mask
            # now write out the confidence for any that pass the mask
            confidences[mask] = i
            # update the predicted mask
            predicted_mask = predicted_mask & ~mask
        # now we can calculate the loss
        return -confidences.sum()


def compute_cascade_loss(logits, targets, mask=None):
    """
    Compute the loss for the cascade model

    During training we train early layers on a given token only if
    every later layer is correct on that token. I.e. if layer 9,10
    are correct for token j, then we train only layer 8 to predict token j
    Args:
        logits: list[torch.tensor(B, S, V)]
        targets: torch.tensor(B, S)
        mask: torch.tensor(B, S)
    Returns:
        loss: torch.tensor(1)
    """

    # Initialize the main loss and auxiliary loss
    main_loss = torch.tensor(0.0, device=targets.device)

    # Create a mask to keep track of correctly predicted tokens
    correct_mask = torch.ones_like(targets, dtype=torch.bool, device=targets.device)
    if mask is not None:
        correct_mask = correct_mask & mask

    # Loop over logits from the last layer to the first
    for i in range(len(logits) - 1, -1, -1):
        logit = logits[i]
        _, predicted = torch.max(logit, dim=-1)
        delta_mask = predicted != targets & correct_mask

        # Calculate the cross-entropy loss for the current layer
        loss_fct = torch.nn.functional.cross_entropy
        current_loss = loss_fct(logit.view(-1, logit.size(-1)), targets.view(-1), reduction="none")
        current_loss = current_loss.view(targets.size())

        # Only consider the loss for tokens where all subsequent layers are correct
        current_loss = current_loss * delta_mask
        main_loss += current_loss.sum()

        # Update the correct mask for the current layer
        correct_mask = correct_mask * ~delta_mask



    # Normalize the loss by the number of tokens
    main_loss = main_loss / targets.numel()

    return main_loss

