# Resources for Getting Started with Language Models
## Important Papers
While academic papers can be difficult to read, don't worry about trying to understand everything. Oftentimes the fine details are not relevant, so try to focus on understanding the main ideas. Here are some papers that are important to understand the development of language models:
- [the original transformer paper](https://arxiv.org/abs/1706.03762): This paper is an *encoder-decoder* architecture, which is not popular these days. Main things to try to understand from the paper are the introduction of the attention mechanism.
- [BERT](https://arxiv.org/abs/1810.04805): This paper introduces a 'masked language modelling objective' - reconstructing a partially obscured version of the original text. It is an *encoder* only model
- [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): An example of a *decoder* only model. It uses a causal language modelling objective - predicting the next word in a sentence. More importantly, it shows that just scaling up the model size/data can lead to impressive zero-shot performance on a variety of tasks
- [GPT-3](https://arxiv.org/abs/2005.14165): This paper introduces a few-shot learning objective - predicting the next word in a sentence given a few examples of the task.

## Blogs
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/): A great blog post that explains the transformer architecture in a very visual way.
- [Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/): Another great blog post that explains the GPT-2 architecture in a very visual way.

## Huggingface
Huggingface is the de-facto standard for working with and sharing language models. It's great for downstream tasks using pre-trained models, and they provide helpful tools for understanding the concepts behind language models as well as tools for finetuning, a massive collection of datasets, and a large collection of pre-trained models. Here are some resources to get started with Huggingface:
- [Huggingface Transformers Documentation](https://huggingface.co/transformers/): The official documentation for the Huggingface Transformers library.
- [Huggingface Datasets Documentation](https://huggingface.co/docs/datasets/): The official documentation for the Huggingface Datasets library.
- [Huggingface Model Hub](https://huggingface.co/models): The official model hub for Huggingface. You can find a large collection of pre-trained models here.

## Videos
While we can't vouch for them [Stanford](https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4) and [MIT]() have some great free lectures on deep learning and NLP.

## Repositories
We highly recommend trying to train a language model from scratch to better understand the concepts. Here are some repositories that can help you get started:
- [MinGPT](https://github.com/karpathy/minGPT) - Very simple PyTorch implementation of GPT-2, also the basis of this repository

## Key Concepts
- **Tokenization**: The process of breaking up text into smaller pieces, usually words or subwords. These have integer representations that can be used as input to a model.
- **Embeddings**: A way to represent words as vectors. These vectors are learned during training and are used as input to the model. Additionally the final outputs of the model also act as embeddings.
- **Logits**: The raw output of a model before it is converted to probabilities.
- **Feed Forward Neural Network**: A neural network with (typically) multiple, fully connected layers that each perform a linear transformation followed by a non-linear activation function. These are applied to each token independently.
- **Attention**: The attention mechanism learns to update the representation of each token based on other tokens. A score is computed that determines how much influence a given token has on another token. Then the "value vectors" of the tokens are combined based on these scores to determine the update
- **Heads**: Attention heads basically divide the embedding space into multiple subspaces and then apply the attention mechanism to each subspace. This allows the model to learn different types of relationships between tokens.
- **Transformers**: A model that has interwoven attention and feed forward neural networks. The transformer architecture is composed of multiple layers of these blocks. The transformer architecture is the basis for many modern language models.
- **Weight Tying**: This is a technique where two layers of a neural network share the same weights -- this forces them to always compute the same function.
