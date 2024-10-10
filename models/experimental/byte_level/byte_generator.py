import torch 

def cross_entropy_loss_fn(logits, y):
    """Cross Entropy Loss Function"""
    logits = logits.view(-1, logits.size(-1))
    y = y.view(-1)
    return torch.nn.functional.cross_entropy(logits, y, ignore_index=-1)



class ByteGenerator(torch.nn.Module):
    """ Simplistic generator to test the byte-level embedding model (autoencoding)"""

    def __init__(self, model, generate_cfg, device="cuda"):
        """Initialize the model and the configuration"""
        super().__init__()
        self.model = model
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

        # pass through the model
        #logits, total_loss, info_dict = self.model(idx)
        logits, target_ids = self.model(idx)

        # split and decode individual blocks and their losses
        print(logits.size(), target_ids.size()) # torch.Size([1, 1024, 12, 259]) torch.Size([1, 1024, 12])
        input()

        for chunk_id in range(logits.size(1)):
            #print(logits[0, chunk_id], target_ids[0, chunk_id])
            target_decoded = self.model.embedding_model.decode(target_ids[0, chunk_id].view(-1, 1).tolist())
            #print("\n\n#####################")
            #print(target_decoded)
            # calculate and print the loss per byte
            chunk_logits = logits[0, chunk_id]
            chunk_targets = target_ids[0, chunk_id]
            #print(chunk_logits.size(), chunk_targets.size()) # torch.Size([12, 259]) torch.Size([12])
            chunk_loss = torch.nn.functional.cross_entropy(
                chunk_logits, #.unsqueeze(0), 
                chunk_targets.long(), #.unsqueeze(0), 
                ignore_index=self.model.embedding_model.pad_token_id, 
                reduce=False
            )
            softmax_probs = torch.nn.functional.softmax(chunk_logits, dim=-1)#[chunk_targets.view(-1,1)])#[chunk_targets])
            target_probs = softmax_probs.gather(dim=-1, index=chunk_targets.view(-1, 1))
            #print(target_probs)
            #print(chunk_loss)

            # sample and decode output
            output_ids = torch.argmax(softmax_probs, dim=-1)
            output_decoded = self.model.embedding_model.decode(output_ids.view(-1,1).tolist())
            #input(output_decoded)


            # remove all pad tokens from both source and target
            pad_mask = chunk_targets!=self.model.embedding_model.pad_token_id

            # mask out
            clean_targets = chunk_targets[pad_mask]
            clean_output = output_ids[pad_mask]

            # decode and print both
            clean_targets_decoded = self.model.embedding_model.decode(clean_targets.view(-1,1).tolist())
            clean_output_decoded = self.model.embedding_model.decode(clean_output.view(-1,1).tolist())
            print(f"Target bytes: \t{clean_targets_decoded}")
            print(f"Output bytes: \t{clean_output_decoded}")
            input()


        # try decoding the logits
        input(logits.view(-1).tolist())
        words = self.model.embedding_model.decode(logits.view(-1, 1).tolist())
        input(words)
        print(total_loss)
        input(info_dict)

        # sample logits 
        #input(logits.size())  # [1, 1024, 12, 259]

        # apply temperature
        logits = logits / temperature

        # flatten 1,2
        logits = logits.view(
            logits.size(0), 
            logits.size(1)*logits.size(2), 
            logits.size(3)
        ) # 1, (12*1024), 259


        # apply top-k
        if top_k is not None:
            v, _ = torch.topk(
                logits,
                min(top_k, logits.size(-1))
            )

            # patch the logits as necessary
            logits[logits < v[:, :, [-1]]] = -float("Inf")



        # apply softmax on dim = -1 and then flatten dim=1,2
        probs = torch.nn.functional.softmax(logits, dim=-1)

        #input(probs.size()) #1, 12288, 259

        # reshape for multinomial
        B, S, P = probs.size()

        probs = probs.view(B*S, P)

        idx_out = torch.multinomial(probs, num_samples=1)

        # reshape out
        idx_out = idx_out.view(B, S, 1)

        #input(idx_out.size())
        

        # sampled the actual tokens




        # decode and print
        output_decoded = self.model.embedding_model.decode(idx_out[0].tolist())
        #input(output_decoded)
        input_decoded = self.model.embedding_model.decode(idx.tolist())
        input(input_decoded)

        for i in range(0, 10000, 12):
            print(
                "".join(input_decoded[0][i:i+12]),
                "".join(output_decoded[i:i+12])
            )

        input()



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