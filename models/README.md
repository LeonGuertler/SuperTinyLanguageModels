# Super Tiny Language Models / Models

In this folder we implement a set of basic components for tokenizers and transformer layers. Additionally we include an experimental folder with a set of models that we are currently working on.

## Interfaces
We provide the basic interface that all models must meet in [model_shell.py](model_shell.py).
The shell is assumed to contain three components:
1. **Embedder**: This component takes care of both tokenizing input, and also embedding the tokens into a dense, continuous representation that can be processed by the transformer layers. The embedder interface is given in [embedding_models.py](embedding_models.py).
2. **Transformer Core**: This component is the core of the model, and typically consists in a stack of transformer layers. We don't assume any particular interface for this, however we do implement a [`generic transformer'](core_models.py) that is intended to subsume most use cases.
3. **LM Head**: This component takes the output of the transformer core and maps it to the output space of the model. We define the interface in [model_heads.py](model_heads.py).

## Other Components
1. **Tokenizer**:
The tokenization interface can be found in [tokenizers/base_class.py](components/tokenizers/base_class.py). We additionally provide a [BPE tokenizer](components/tokenizers/bpe.py) and [GPT2 style Tokenizer](components/tokenizers/gpt2.py).
1. **Generator**:
In [generator.py](generator.py) we implement a simple generator which only implements top-$k$ sampling. Feel free to extend this, but if so add an interface from which generators can inherit.
N.B. while we have previously implemented kv-caching, it is not really worth it for the tiny models we are working with and thus was removed since it just adds complexity.
2. **Normalization**:
In [normalization.py](components/layers/normalization.py) we implement RMSNorm, LayerNorm, and a pass-through layer.
3. **Positional Encodings**:
In [positional_encodings.py](components/positional_encodings.py) we implement a variety of positional encodings, including the standard sinusoidal positional encodings, and the relative positional encodings from Shaw et al. (2018).
5. **Attention**:
Our [attention layer](components/layers/attention.py) implements, causal masks, groups, multi-head, and rotary embeddings.
6. **Feed Forward**:
In [feed_forward.py](components/layers/feedforward.py) we implement both the standard feedforward layer (with variable [activation](components/layers/activations.py) as well as the SwiGLU activation from [Shazeer et al. (2020)](
https://arxiv.org/abs/2002.05202).)

## Experimental Components
Our [experimental folder](experimental/README.md) includes a number of models that we are currently working on. These are obviously highly subject to change, and in general we expect most components to be implemented here first before being added to the main folders if at all.
1. **Byte Level** This includes components for building models that operate on the byte level
2. **Next Thought** This includes components that are intended to function on a latent - latent basis, rather than a token - token basis.
3. **Huggingface Interface** This just wraps the huggingface models in our interface so that we can test those models with our code, and compare them to our models fairly.
