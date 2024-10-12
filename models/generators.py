"""
Generator Base Wrapper
"""

import torch
import torch.nn.functional as F

## libraries for entropy calculation
import math
from typing import Dict, Optional

class AdaptiveGenerator(torch.nn.Module):
    """
    Adaptive Generator with adaptive sampling strategies based on entropy metrics.
    Reference: https://github.com/xjdr-alt/entropix/tree/main
    """

    LN_2 = math.log(2)  # ln(2) for entropy calculations

    def __init__(self, model, generate_cfg: Dict, device: str = "cuda"):
        """
        Initialize the generator with the model and sampling configuration.

        Args:
            model: The language model to use for generation.
            generate_cfg (Dict): Configuration dictionary for sampler hyperparameters.
            device (str): Device to run the generator on ('cuda' or 'cpu').
        """
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.generate_config = generate_cfg

    def default_generate(self, input_text: str) -> str:
        """
        Generate text using the default generation method.

        Args:
            input_text (str): The initial text prompt.

        Returns:
            str: Generated text.
        """
        return self.generate(
            input_text,
            max_new_tokens=self.generate_config.max_new_tokens,  # Adjust as needed
            temperature=self.generate_config.temperature,
            top_k=self.generate_config.top_k,
            top_p=self.generate_config.top_p,
            min_p=self.generate_config.min_p
        )

    @torch.no_grad()
    def generate(
        self,
        input_text: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        clarifying_question_token: int = 2564
    ) -> str:
        """
        Generate text with advanced sampling strategies.

        Args:
            input_text (str): The initial text prompt.
            max_new_tokens (int): Number of tokens to generate.
            temperature (float, optional): Temperature for scaling logits.
            top_k (int, optional): Top-K sampling parameter.
            top_p (float, optional): Top-P (nucleus) sampling parameter.
            min_p (float, optional): Minimum probability threshold.
            clarifying_question_token (int, optional): Token ID for clarifying questions.

        Returns:
            str: Generated text.
        """
        # Tokenize input
        idx = self.model.embedding_model.tokenize_input(
            input_string=input_text,
            add_eot=False,
            truncate=True
        )
        # Convert to tensor and move to device
        idx = torch.tensor(idx, dtype=torch.long).unsqueeze(0).to(self.device)  # Shape: (batch_size=1, sequence_length)

        for _ in range(max_new_tokens):
            # Forward pass to get logits and attention scores
            logits, attention_scores = self.model.inference(idx)
            # Logits shape: (batch_size, sequence_length, vocab_size)
            # attention_scores shape: list of tensors, each of shape (batch_size, num_heads, sequence_length, sequence_length)

            # Debugging: Print shapes (optional, remove in production)
            # print(f"logits shape: {logits.shape}")
            # print(f"attention_scores length: {len(attention_scores)}")
            # for i, m in enumerate(self.model.core_model.matrices):
            #     print(f"matrix {i} scaled_scores shape: {m['scaled_scores'].shape}")

            # Take logits of the last token
            if logits.dim() == 3:
                logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
            elif logits.dim() == 2:
                logits = logits  # Shape: (batch_size, vocab_size)
            elif logits.dim() == 1:
                logits = logits.unsqueeze(0)  # Shape: (1, vocab_size)
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")

            # Ensure logits is 2D
            if logits.dim() != 2:
                raise ValueError(f"Logits should be 2D after processing, got shape: {logits.shape}")

            # Scale logits by temperature
            scaled_logits = logits / temperature  # Shape: (batch_size, vocab_size)

            # Retrieve and concatenate attention scores
            # Ensure that self.model.core_model.matrices is correctly structured
            attention_scores = torch.cat([attn_component['scaled_scores'] for attn_component in self.model.core_model.attn_components], dim=1)
            # attention_scores shape: (batch_size, total_heads, seq_length, seq_length)

            # Calculate metrics
            metrics = self._calculate_metrics(scaled_logits, attention_scores)

            # Decide on sampling strategy based on metrics
            next_token = self._decide_next_token(logits, metrics, clarifying_question_token)

            # Check if the next token is the end-of-text token
            if next_token.item() == self.model.embedding_model.eot_token:
                print("EOT token found, stopping generation.")
                break

            # Debugging: Print shapes (optional, remove in production)
            # print("next token shape!!!", next_token.shape)  # Should be [batch_size, 1]
            # print("idx shape!!!", idx.shape)  # Should be [batch_size, current_length]

            # Append the next token to the sequence
            idx = torch.cat((idx, next_token), dim=1)  # Shape: (batch_size, current_length + 1)
        # print(idx.ty)
        # Decode the generated tokens to string
        return self.model.embedding_model.decode(idx.tolist())

    def _calculate_metrics(self, logits: torch.Tensor, attention_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate entropy, varentropy, and other metrics from logits and attention scores.

        Args:
            logits (torch.Tensor): Logits from the model. Shape: (batch_size, vocab_size)
            attention_scores (torch.Tensor): Attention scores from the model. Shape: (batch_size, total_heads, seq_length, seq_length)

        Returns:
            Dict[str, torch.Tensor]: Dictionary of calculated metrics.
        """
        # Calculate log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, vocab_size)
        probs = torch.exp(log_probs)  # Shape: (batch_size, vocab_size)

        # Entropy: -sum(p * log2(p))
        entropy = -torch.sum(probs * log_probs, dim=-1) / self.LN_2  # Shape: (batch_size,)

        # Varentropy: sum(p * (log2(p) + entropy)^2)
        entropy_expanded = entropy.unsqueeze(-1)  # Shape: (batch_size, 1)
        varentropy = torch.sum(probs * (log_probs / self.LN_2 + entropy_expanded) ** 2, dim=-1)  # Shape: (batch_size,)

        # Attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)  # Shape: (batch_size, total_heads, seq_length, seq_length)

        # Attention entropy: -sum(p * log2(p))
        attn_entropy = -torch.sum(
            attention_probs * torch.log2(torch.clamp(attention_probs, min=1e-10)), 
            dim=-1
        )  # Shape: (batch_size, total_heads, seq_length)

        # Aggregate attention entropy by averaging over sequence length
        attn_entropy = torch.mean(attn_entropy, dim=-1)  # Shape: (batch_size, total_heads)

        # Attention varentropy: variance of attention entropy across heads
        attn_varentropy = torch.var(attn_entropy, dim=1)  # Shape: (batch_size,)

        # Mean attention
        mean_attention = torch.mean(attention_probs, dim=1)  # Shape: (batch_size, seq_length, seq_length)

        # Agreement: mean absolute difference from mean attention
        agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2, 3))  # Shape: (batch_size,)

        # Interaction strength: mean absolute attention scores
        interaction_strength = torch.mean(torch.abs(attention_scores), dim=(1, 2, 3))  # Shape: (batch_size,)

        return {
            "logits_entropy": entropy.mean(),
            "logits_varentropy": varentropy.mean(),
            "attn_entropy": attn_entropy.mean(),
            "attn_varentropy": attn_varentropy.mean(),
            "agreement": agreement.mean(),
            "interaction_strength": interaction_strength.mean()
        }

    def _decide_next_token(
        self,
        logits: torch.Tensor,
        metrics: Dict[str, torch.Tensor],
        clarifying_question_token: int
    ) -> torch.Tensor:
        """
        Decide the next token to generate based on the calculated metrics.

        Args:
            logits (torch.Tensor): Logits from the model. Shape: (batch_size, vocab_size)
            metrics (Dict[str, torch.Tensor]): Calculated metrics.
            clarifying_question_token (int): Token ID for clarifying questions.

        Returns:
            torch.Tensor: Next token ID. Shape: (batch_size, 1)
        """
        cfg = self.generate_config
        ent = metrics["logits_entropy"].item()
        vent = metrics["logits_varentropy"].item()
        attn_ent = metrics["attn_entropy"].item()
        attn_vent = metrics["attn_varentropy"].item()
        agreement = metrics["agreement"].item()
        interaction_strength = metrics["interaction_strength"].item()

        # Low Entropy, Low Varentropy: Greedy decoding
        if ent < cfg.low_ent_thresh and vent < cfg.low_vent_thresh:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # Shape: (batch_size, 1)
            return next_token

        # High Entropy, Low Varentropy: Insert clarifying question
        elif ent > cfg.high_ent_thresh and vent < cfg.low_vent_thresh:
            # Insert a clarifying question token for each sequence in the batch
            batch_size = logits.shape[0]
            next_token = torch.full((batch_size, 1), clarifying_question_token, device=self.device)  # Shape: (batch_size, 1)
            return next_token

        # Low Entropy, High Varentropy: Adjust temperature and top_k
        elif ent < cfg.high_ent_thresh and vent > cfg.high_vent_thresh:
            temp_adj = cfg.lehv_interaction_strength_offset + cfg.lehv_interaction_strength_coef * interaction_strength
            adjusted_temp = min(1.5, cfg.temperature * temp_adj)
            adjusted_top_k = max(5, int(cfg.top_k * (1 + 0.5 * (1 - agreement))))
            return self._sample(logits, temperature=adjusted_temp, top_p=cfg.top_p, top_k=adjusted_top_k, min_p=cfg.min_p)

        # High Entropy, High Varentropy: High temperature and adjusted top_p
        elif ent > cfg.med_ent_thresh and vent > cfg.high_vent_thresh:
            temp_adj = cfg.hehv_attn_vent_offset + cfg.hehv_attn_vent_coef * attn_vent
            adjusted_temp = max(2.0, cfg.temperature * temp_adj)
            top_p_adj = max(0.5, cfg.top_p - cfg.hehv_attn_ent_coef * attn_ent)
            return self._sample(logits, temperature=adjusted_temp, top_p=top_p_adj, top_k=cfg.top_k, min_p=cfg.min_p)

        # Middle ground: Adaptive sampling
        else:
            logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
            attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

            temperature = cfg.temperature * (
                1 
                + cfg.ada_temp_logits * logits_uncertainty 
                + cfg.ada_temp_attn * attn_uncertainty 
                - cfg.ada_temp_agree * agreement
            )
            top_p = torch.clamp(cfg.top_p * (1 + cfg.ada_top_p * metrics["attn_varentropy"]), 0.1, 1.0)
            top_k = int(
                torch.clamp(
                    torch.round(
                        torch.tensor(cfg.top_k) * (
                            1 
                            + cfg.ada_top_k_int * interaction_strength 
                            - cfg.ada_top_k_agree * agreement
                        )
                    ),
                    min=1,
                    max=100
                ).item()
            )
            min_p = torch.clamp(cfg.min_p * (1 - cfg.ada_min_p * logits_uncertainty), 0.01, 0.5)

            # Perform multiple adaptive samples and select the best one
            samples = []
            sample_scores = []
            for _ in range(cfg.n_adaptive_samples):
                sample = self._sample(
                    logits, 
                    temperature=temperature.item(), 
                    top_p=top_p.item(), 
                    top_k=top_k, 
                    min_p=min_p.item()
                )
                score = self._score_sample(sample, logits, metrics)
                samples.append(sample)
                sample_scores.append(score)

            # Select the sample with the highest score
            sample_scores = torch.tensor(sample_scores)
            best_sample_idx = torch.argmax(sample_scores).item()
            best_sample = samples[best_sample_idx]  # Shape: (batch_size, 1)
            return best_sample  # Shape: (batch_size, 1)

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float, top_k: int, min_p: float) -> torch.Tensor:
        """
        Sample a token based on adjusted logits with temperature, top_p, top_k, and min_p.

        Args:
            logits (torch.Tensor): Logits from the model. Shape: (batch_size, vocab_size)
            temperature (float): Temperature scaling factor.
            top_p (float): Cumulative probability threshold for nucleus sampling.
            top_k (int): Number of top tokens to consider for top-k sampling.
            min_p (float): Minimum probability threshold.

        Returns:
            torch.Tensor: Sampled token IDs. Shape: (batch_size, 1)
        """
        # Apply temperature
        scaled_logits = logits / temperature  # Shape: (batch_size, vocab_size)

        # Apply min_p sampling
        if min_p > 0.0:
            probs = F.softmax(scaled_logits, dim=-1)  # Shape: (batch_size, vocab_size)
            p_max, _ = torch.max(probs, dim=-1, keepdim=True)  # Shape: (batch_size, 1)
            indices_to_remove = probs < (min_p * p_max)  # Shape: (batch_size, vocab_size)
            scaled_logits = torch.where(indices_to_remove, torch.full_like(scaled_logits, -float('Inf')), scaled_logits)

        # Apply top_k sampling
        if top_k is not None and top_k > 0:
            topk_logits, topk_indices = torch.topk(scaled_logits, top_k, dim=-1)
            scaled_logits = torch.full_like(scaled_logits, -float('Inf')).scatter_(-1, topk_indices, topk_logits)

        # Apply top_p (nucleus) sampling
        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(F.softmax(scaled_logits, dim=-1), descending=True, dim=-1)  # [batch_size, vocab_size]
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # [batch_size, vocab_size]
            sorted_indices_to_remove = cumulative_probs > top_p  # [batch_size, vocab_size]
            # Shift the mask right to keep the first token above the threshold
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            # Scatter back to original ordering
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)  # [batch_size, vocab_size]
            scaled_logits = torch.where(indices_to_remove, torch.full_like(scaled_logits, -float('Inf')), scaled_logits)

        # Re-normalize the logits
        probs = F.softmax(scaled_logits, dim=-1)  # Shape: (batch_size, vocab_size)

        # Multinomial sampling
        next_token = torch.multinomial(probs, num_samples=1)  # Shape: (batch_size, 1)
        return next_token

    def _score_sample(self, sample: torch.Tensor, logits: torch.Tensor, metrics: Dict[str, torch.Tensor]) -> float:
        """
        Score a sampled token based on various confidence metrics.

        Args:
            sample (torch.Tensor): Sampled token ID. Shape: (batch_size, 1)
            logits (torch.Tensor): Logits from the model. Shape: (batch_size, vocab_size)
            metrics (Dict[str, torch.Tensor]): Calculated metrics.

        Returns:
            float: Confidence score for the sampled token.
        """
        # Calculate log probability of the sampled token
        log_probs = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, vocab_size)
        sample_log_prob = log_probs.gather(-1, sample).squeeze(-1).item()

        # Confidence score based on metrics and configuration
        cfg = self.generate_config
        confidence_score = (
            (1 - metrics["logits_entropy"].item()) * cfg.ada_score_logits_ent +
            (1 - metrics["attn_entropy"].item()) * cfg.ada_score_attn_ent +
            (1 - metrics["logits_varentropy"].item()) * cfg.ada_score_logits_vent +
            (1 - metrics["attn_varentropy"].item()) * cfg.ada_score_attn_vent +
            metrics["agreement"].item() * cfg.ada_score_agree +
            metrics["interaction_strength"].item() * cfg.ada_score_int
        )

        return sample_log_prob + confidence_score

    def forward(self, x):
        """Call the underlying model's forward method."""
        return self.model(x)

    def embed(self, x):
        """Embed the input using the underlying model's embedding."""
        return self.model.embed(x)

class BaseGenerator(torch.nn.Module):
    """ TODO """
    def __init__(self, model):
        super().__init__()
        """ TODOD """
        self.model = model 
        self.device = self.model.device

    @torch.no_grad()
    def generate(self, input_text):
        """ TODO """
        raise NotImplementedError("Each Generator needs to implement the generate function")

class EntropyTemperatureGenerator(BaseGenerator):
    '''
    From: https://arxiv.org/pdf/2403.14541
    Entropy based temepraure adjusts the temperature based on the entropy of the logits.
    If logits highly uncertain, entropy is high, temperature is increased.
    If logits are certain, entropy is low, temperature is decreased.
    '''

    def __init__(self, model, generate_cfg, device="cuda"):
        super().__init__()
        self.model = model
        self.device = device
        self.model = self.model.to(torch.device(self.device))
        self.generate_config = generate_cfg

    def default_generate(self, input_text):
        '''
        Generate text using the default generation method
        '''
        return self.generate(
            input_text,
            self.generate_config["max_new_tokens"],
            self.generate_config["temperature"],
            self.generate_config["top_k"],
            self.generate_config.get("temperature_scaling_factor", 0.1)
        )
    
    @torch.no_grad()
    def generate(self, input_text, max_new_tokens, temperature_ceiling, top_k, temperature_scaling_factor):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        idx = self.model.embedding_model.tokenize_input(
            input_string=input_text,
            add_eot=False,
            truncate=True
        )
        # push to device
        idx = torch.tensor(idx).unsqueeze(0).to(torch.device(self.device))
        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits, _ = self.model.inference(idx)
            # calculate the entropy of the logits
            entropy = self.calculate_entropy(F.softmax(logits[-1], dim=-1))
            # calculate the temperatures
            temperature = temperature_ceiling * (0.8 ** (temperature_scaling_factor / entropy)) # 0.8 is fixed based on the paper
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature
            # logits might have shape (b,t,v) or (t,v)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                ## for batched logits
                if len(logits.shape) == 3:
                    logits[logits < v[:, :, [-1]]] = -float("Inf")
                ## for single logits
                else:
                    logits[logits < v[:, [-1]]] = -float("Inf")

            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)

            # check if done
            if idx_next == self.model.embedding_model.eot_token:
                break

            idx = torch.cat((idx, idx_next), dim=1)

        return self.model.embedding_model.decode(idx.tolist())
    
    def calculate_entropy(self, probabilities):
        """
        Calculate the entropy of a probability distribution from softmaxed logits.
        
        :param probabilities: A PyTorch tensor of softmaxed logits
        :return: The entropy value as a PyTorch tensor
        """
       
        # Ensure the input is a PyTorch tensor
        if not isinstance(probabilities, torch.Tensor):
            probabilities = torch.tensor(probabilities)
        
        # Ensure the tensor is float for numerical stability
        probabilities = probabilities.float()
        
        return -torch.sum(probabilities * torch.log(probabilities), dim=-1)


    def forward(self, x):
        """Call the underlying model"""
        return self.model(x)

    def embed(self, x):
        """Embed the input"""
        return self.model.embed(x)



class BeamSearchGenerator(BaseGenerator):
    def __init__(self, model, generate_cfg, device="cuda"):
        super().__init__()
        self.model = model
        self.device = device
        self.model = self.model.to(torch.device(self.device))
        self.generate_config = generate_cfg

    def default_generate(self, input_text):
        return self.generate(
            input_text,
            self.generate_config["max_new_tokens"],
            self.generate_config["temperature"],
            self.generate_config["top_k"],
            self.generate_config.get("beam_width", 5),
            self.generate_config.get("use_sampling", False),
            self.generate_config.get("repetition_penalty", 1.2),
            self.generate_config.get("repetition_window", 32)
        )

    @torch.no_grad()
    def generate(self, input_text, max_new_tokens, temperature=1.0, top_k=None, beam_width=5, 
                 use_sampling=False, repetition_penalty=1.2, repetition_window=32):
        idx = self.model.embedding_model.tokenize_input(input_string=input_text, add_eot=False, truncate=True)
        idx = torch.tensor(idx).unsqueeze(0).to(torch.device(self.device))

        # Initialize beam with the input sequence
        beam = [(idx, 0.0)]

        for _ in range(max_new_tokens):
            all_candidates = []
            for seq, score in beam:
                # Get logits for the entire sequence
                logits = self.model.inference(seq)[0]
                # Consider only the last token's logits
                last_token_logits = logits / temperature

                # Apply repetition penalty
                if repetition_penalty > 1.0:
                    self.apply_repetition_penalty(last_token_logits, seq, repetition_penalty, repetition_window)

                if top_k is not None:
                    v, _ = torch.topk(last_token_logits, min(top_k, last_token_logits.size(-1)))
                    last_token_logits[last_token_logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(last_token_logits, dim=-1)

                if use_sampling:
                    # Sampling
                    sampled_indices = torch.multinomial(probs, num_samples=beam_width)
                    top_indices = sampled_indices[0]
                    top_probs = probs[0, top_indices]
                else:
                    # Greedy selection
                    top_probs, top_indices = torch.topk(probs, k=beam_width)
                    top_probs = top_probs[0]
                    top_indices = top_indices[0]

                for prob, idx_next in zip(top_probs, top_indices):
                    new_seq = torch.cat([seq, idx_next.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score - torch.log(prob).item()
                    all_candidates.append((new_seq, new_score))

            # Select top beam_width candidates
            beam = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

            # Check if any beam has generated EOT token
            #if any(seq[0, -1] == self.model.embedding_model.eot_token for seq, _ in beam):
            #    break

        # Return the sequence with the best score
        best_seq, _ = min(beam, key=lambda x: x[1])
        return self.model.embedding_model.decode(best_seq.tolist())

    def apply_repetition_penalty(self, logits, sequence, penalty, window):
        # Get the most recent tokens within the window
        recent_tokens = sequence[0, -window:]
        
        # Count the occurrences of each token
        unique_tokens, counts = torch.unique(recent_tokens, return_counts=True)
        
        # Apply penalty to the logits of repeated tokens
        logits[0, unique_tokens] /= penalty ** counts.float()


class StandardGenerator(BaseGenerator):
    """Standard Generator Wrapper for GPT models"""

    def __init__(self, model, generate_cfg, device="cuda"):
        """Initialize the model and the configuration"""
        super().__init__(model)
        # self.model = model
        self.device = device 
        self.model = self.model.to(torch.device(device))
        self.generate_config = generate_cfg

    def default_generate(self, input_text):
        """
        Generate text using the default generation method
        """
        return self.generate(
            input_text,
            self.generate_config["max_new_tokens"],
            self.generate_config["temperature"],
            self.generate_config["top_k"],
        )

    @torch.no_grad()
    def generate(self, input_text, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        idx = self.model.embedding_model.tokenize_input(
            input_string=input_text,
            add_eot=False,
            truncate=True
        )
        # push to device
        idx = torch.tensor(idx).unsqueeze(0).to(torch.device(self.device))
        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits, _ = self.model.inference(idx)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature
            # logits might have shape (b,t,v) or (t,v)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                ## for batched logits
                if len(logits.shape) == 3:
                    logits[logits < v[:, :, [-1]]] = -float("Inf")
                ## for single logits
                else:
                    logits[logits < v[:, [-1]]] = -float("Inf")

            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)

            # check if done
            if idx_next == self.model.embedding_model.eot_token:
                break

            idx = torch.cat((idx, idx_next), dim=1)
        print(idx)
        return self.model.embedding_model.decode(idx.tolist())

    def forward(self, x):
        """Call the underlying model"""
        return self.model(x)

    def embed(self, x):
        """Embed the input"""
        return self.model.embed(x)



class StandardGenerator(BaseGenerator):
    """Standard Generator Wrapper for GPT models"""

    def __init__(self, model, generate_cfg, device="cuda"):
        """Initialize the model and the configuration"""
        super().__init__(model)
        self.device = device 
        self.model = self.model.to(torch.device(device))
        self.generate_config = generate_cfg

    def default_generate(self, input_text):
        """
        Generate text using the default generation method
        """
        return self.generate(
            input_text,
            self.generate_config["max_new_tokens"],
            self.generate_config["temperature"],
            self.generate_config["top_k"],
        )

    @torch.no_grad()
    def generate(self, input_text, max_new_tokens=100, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        idx = self.model.embedding_model.tokenize_input(
            input_string=input_text,
            add_eot=False,
            truncate=True
        )
        # push to device
        idx = torch.tensor(idx).unsqueeze(0).to(torch.device(self.device))
        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits, _ = self.model.inference(idx)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature
            # logits might have shape (b,t,v) or (t,v)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                ## for batched logits
                if len(logits.shape) == 3:
                    logits[logits < v[:, :, [-1]]] = -float("Inf")
                ## for single logits
                else:
                    logits[logits < v[:, [-1]]] = -float("Inf")

            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)

            # check if done
            if idx_next == self.model.embedding_model.eot_token:
                break

            idx = torch.cat((idx, idx_next), dim=1)

        return self.model.embedding_model.decode(idx.tolist())

    def forward(self, x):
        """Call the underlying model"""
        return self.model(x)

    def embed(self, x):
        """Embed the input"""
        return self.model.embed(x) 

GENERATOR_DICT = {
    "standard": lambda model, generate_cfg, device: StandardGenerator(model=model, generate_cfg=generate_cfg, device=device), 
    "beam_search": lambda model, generate_cfg, device: BeamSearchGenerator(model=model, generate_cfg=generate_cfg, device=device), 
    "entropy_temperature": lambda model, generate_cfg, device: EntropyTemperatureGenerator(model=model, generate_cfg=generate_cfg, device=device),
    "adaptive_sampler": lambda model, generate_cfg, device: AdaptiveGenerator(model=model, generate_cfg=generate_cfg, device=device)
    }

def build_generator(model, generate_cfg, device):
    """
    Build the generator
    """
    return GENERATOR_DICT[generate_cfg['generator_type']](model, generate_cfg, device)