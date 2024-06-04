"""Integration code"""

import torch

from models import embedding_models, generator, model_shell


def batch(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


class EvalWrapper:
    def __init__(self, model_shell: model_shell.ModelShell):
        self.model_shell = model_shell
        super().__init__()

    def loglikelihood(self, prefixes, continuations) -> list[float]:
        """
        Compute the loglikelihood of the continuations given the prefixes
        """
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        self.model_shell = self.model_shell.to(device)
        results = []
        for batch_requests in batch(list(zip(prefixes, continuations)), batch_size=8):
            with torch.no_grad():
                with torch.autocast(device_type=device_str):
                    context_strs = [request[0] for request in batch_requests]
                    target_strs = [request[1] for request in batch_requests]
                    embedding_model: embedding_models.EmbedderInterface = (
                        self.model_shell.embedding_model
                    )
                    # tokenize the inputs
                    context_tokens = [
                        embedding_model.tokenize_input(context_str)
                        for context_str in context_strs
                    ]
                    context_tokens = [
                        tokens if len(tokens) > 0 else [0] for tokens in context_tokens
                    ]
                    target_tokens = [
                        embedding_model.tokenize_input(target_str)
                        for target_str in target_strs
                    ]
                    target_tokens = embedding_model.truncate(target_tokens)

                    # append the target tokens to the input tokens
                    # remove the final tokens due to causal language modeling
                    unpadded_input_tokens = [
                        (context_tokens[i] + target_tokens[i])[
                            :-1
                        ]  # shifts the tokens by 1 back
                        for i in range(len(context_tokens))
                    ]
                    unpadded_input_tokens = embedding_model.truncate(
                        unpadded_input_tokens
                    )
                    # pad the input tokens to the max length in the batch
                    input_tokens, _ = embedding_model.pad_batch(unpadded_input_tokens)
                    input_tokens = torch.tensor(input_tokens, device=device).long()

                    # get the logits
                    logits, _ = self.model_shell(input_tokens)

                    for i, _ in enumerate(batch_requests):
                        logits_i = logits[i]
                        # remove the padding
                        logits_i = logits_i[: len(unpadded_input_tokens[i])]
                        target_tokens_i = torch.tensor(
                            target_tokens[i], device=device
                        ).long()
                        logits_i = logits_i[-len(target_tokens_i) :]
                        logits_i = logits_i.reshape(-1, logits_i.shape[-1])
                        target_tokens_i = target_tokens_i.reshape(-1)
                        # get the loglikelihood of the target string
                        ll = torch.nn.functional.cross_entropy(
                            logits_i, target_tokens_i, reduction="sum"
                        )

                        results.append(-ll.item())

        return results

    def generate(self, prefixes) -> list[str]:
        """
        Generate a continuation for a given prefix
        """
        model_generator = generator.StandardGenerator(
            self.model_shell,
            generate_cfg={
                "max_new_tokens": 128,
                "temperature": 0.7,
                "top_k": 0.9,
            },
        )
        for prefix in prefixes:
            # tokenize the inputs
            yield model_generator.default_generate(prefix)
