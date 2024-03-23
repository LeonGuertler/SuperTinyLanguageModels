import torch
import numpy as np
import matplotlib.pyplot as plt

from models.the_10m_model import tokenizer


# load gpt2 tokenizer, iterate over all tokens and check length distribution
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
max_token_length = 0
longest_string = ""
long_dist = {}
for idx in range(50257):
    print(idx)
    token_length = len(tokenizer.decode([idx]))

    if token_length > 15:
        long_dist[token_length] = long_dist.get(token_length, 0) + 1


    if token_length > max_token_length:
        max_token_length = token_length
        longest_string = tokenizer.decode([idx])

print(max_token_length)
print(longest_string)

plt.bar(long_dist.keys(), long_dist.values())
plt.show()
exit()





# test tokenizer call
t = tokenizer.character_bpe_tokenizer({
    "arch": {
        "tokenizer": "character_basic",
        "tokenizer_model": {
            "vocab_size": 4096,
        }
    },
    "paths": {
        "data_path": "../../../data"
    },
    "training": {
        "dataset": "simple_en_wiki"
    }
})
# try encoding text
input_text = "This is a test sentence"
encoded = t.encode_text(input_text, "cpu")
t.fit()
exit()
# test input 
test_input = torch.rand(16, 100, 28)

bpe_tokens = []
for _ in range(16):
    attn = []
    i = 0
    while True:
        print(i)
        i1 = i + np.random.randint(1, 100-i)
        attn.append((i, i1))
        i = i1
        if i >= 99:
            break

    bpe_tokens.append(attn)




class CharEncTrans(torch.nn.Module):
    """
    This is a basic and not the final implementation. Single loop, to calculate the forward passes on tokens and then pool
    them one by one.
    """
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model=28, 
            nhead=4, 
            dim_feedforward=256, 
            dropout=0.1
        )

        # learned pad token
        self.pad_token = torch.nn.Parameter(torch.ones(28))





    def forward(self, emb, bpe_tokens):
        """
        emb: torch.Tensor
            The input tensor of shape (batch_size, seq_len, emb_dim)
        bpe_tokens: list
            A list of length batch_size, where each element is a list of tuples
            of the form (int, int) representing the start and end of the BPE tokens
        """
        # max bpe sequence length
        max_bpe_len = max([len(bpe) for bpe in bpe_tokens])
        pooled_emb = torch.zeros((emb.size(0), max_bpe_len, emb.size(2)))

        input(len(bpe_tokens))
        input(emb.size())

        # iterate over batch 
        for i, bpe in enumerate(bpe_tokens):
            # iterate over BPE sequence segments
            for ii, (start, end) in enumerate(bpe):
                # pass slice through transformer
                emb[i, start:end, :] = self.transformer(emb[i, start:end, :])

                # pool the tokens via average pooling
                pooled_emb[i, ii, :] = torch.mean(emb[i, start:end, :], dim=0)

            # fill the rest of the sequence with pad tokens
            for ii in range(len(bpe), max_bpe_len):
                pooled_emb[i, ii, :] = self.pad_token


        return pooled_emb





trans = CharEncTrans()
trans(test_input, bpe_tokens)























print(bpe_tokens)
input()



# random tensor of shape 20, 28



x = torch.rand(20, 28)
print(x.size())






# pool 1-4, 5-10, 11-20 via matrix matmul
# 20, 1 all zeros exectp values 1-4 which are 1/4 each
pool1 = torch.matmul(x.transpose(1,0).to("cpu"), torch.tensor([1
    if i < 4 else 0 for i in range(20)]).view(20, 1).to("cpu").float())
print(pool1.size())


