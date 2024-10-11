"""
Generator Base Wrapper
"""

import torch
import torch.nn.functional as F

## libraries for entropy calculation
import math


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

        return self.model.embedding_model.decode(idx.tolist())

    def forward(self, x):
        """Call the underlying model"""
        return self.model(x)

    def embed(self, x):
        """Embed the input"""
        return self.model.embed(x)



class StandardGenerator(BaseGenerator):
    """Standard Generator Wrapper for GPT models"""

    def __init__(self, model):
        """Initialize the model and the configuration"""
        super().__init__(model)
        # self.model = model
        # self.device = model.device 

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
    "entropy_temperature": lambda model, generate_cfg, device: EntropyTemperatureGenerator(model=model, generate_cfg=generate_cfg, device=device)
    }

def build_generator(model, generate_cfg, device):
    """
    Build the generator
    """
    return GENERATOR_DICT[generate_cfg['generator_type']](model, generate_cfg, device)