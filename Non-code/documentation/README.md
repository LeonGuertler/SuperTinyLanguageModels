# Configuration Documentation

This documentation details the required inputs for configuring the model, trainer, and general settings. Note, experimental architectures are not included here, as they can be subject to frequent change

# Model Configuration (`model`)

Conceptually the model is split into three parts, the `core_model`[core_model.py], `embedder` and `lm_head`, where the `embedder` is responsible for converting strings to token-embeddings (i.e. includes both the tokenizer and the token embedder), the `core_model` does most of the 'thinking' (i.e. contains the actual transformer blocks), and the `lm_head` links the final hidden state back to the vocabulary. The model configuration should contain the following.

## Core Model `core_model_type`

The name of the core-model architecture to be used. Depending on the selected option, additional sub-configurations may be required. The following core-models are available:

- **generic (link the code)**: A basic transformer model with multiple attn+ffn blocks. This can be used for almost all classic architectures.
  - `num_layers`: The number of transformer blocks in the core model
  - `ffn`: Configuration for the Feedforward Network (FFN). [Link to the sub-config]
  - `attn`: Configuration for the Attention mechanism. [Link to the sub-config]
  - `ffn_weight_tying`: Whether to tie the weights of the FFN layers across the transformer blocks. _[default: false]_
    - `c_proj_weight_tying`: Whether to tie the weights of the c*proj linear layer in the attention block across the transformer blocks. *[default: false]\_
- **hf_core (link the code)**: This can be used to load huggingface models
  - `freeze_core`: Whether to freeze the core model. _[default: true]_
  - `todo`

## Embedding Model `embedding_model_type`

The name of the embedding model to be used. The available options are:

- **generic (link the code)**: A simple embedding model using torch.nn.Embedding to link the tokenizer indecies to learned embeddings.
  - `embedding_weight_tying`: Whether to tie the weights of the embedding with those of the language modeling head. _[default: true]_
  - `embedding_dropout`: The dropout rate to be used in the embedding layer. _[default: 0.0]_

## Tokenizer `tokenizer_type`

The name of the tokenizer to be used. Depending on the tokenizer selected, additional variables might be necessary. The options include:

- **GPT-2 (link the code)**: The GPT-2 tokenizer via tiktoken.
- **o200k_base (link the code)**: The GPT-40 tokenizer via tiktoken.
- **cl100k_base (link the code)**: The GPT-4 tokenizer via tiktoken.
- **p50k_base (link the code)**: The davinci tokenizer via tiktoken.
- **llama_32k (link the code)**: The old llama tokenizer via huggingface.
- **opt_50k (link the code)**: The OPT tokenizer via huggingface.
- **mistral_32k (link the code)**: The mistral tokenizer via huggingface.
- **bpe (link the code)**: A custom byte-pair encoding tokenizer using the hf library.
  - `vocab_size`: The size of the vocabulary
  - `tokenizer_dataset_name`: The name of the dataset used to train the tokenizer
  - `tokenizer_simplify`: Whether to simplify the tokenizer by removing foreign symbols and forcing digits to be tokenized at an individual level _[default: true]_

## Hidden Dimension Size `hidden_dim`

The dimensionality of the hidden layers in the model.

## Context Window `context_window`

The size of the context window (sequence length).

## Language Model Head `lm_head_type`

The name of the language model head to be used. The available options are:

- **generic (link the code)**: A simple linear layer to map the hidden state back to the vocabulary.
  - `lm_head_normalization`: The name of the normalization function to be used. The options are listed here _[TODO: link to normalization]_
  - `lm_head_bias`: true/false whether the bias units in the linear layers should be used _[defaul: false]_
  - `lm_head_dropout`: The dropout rate to be used in the language modeling head. _[default: 0.0]_

## Model Shell `model_shell_type`

The model shell combines the embedder, core_model and lm_head into a single model. The available options are:

- **standard (link the code)**: A simple model shell that combines the embedder, core_model and lm_head into a single model.

## Positional encoding `positional_encoding_type`

The type of positional encoding to be used. The options include:

- **learned**: A learned positional encoding applied right after token embedding in the embedder.
- **sincos**: A sinusoidal positional encoding applied right after token embedding in the embedder.
- **rope**: A rotary positional encoding applied in the attention blocks.
- **none**: No positional encoding is applied.

## The initialization function `initialization_fn`

The function to be used to initialize the model weights. Options include: _[default: kaiming]_

- **xavier**: Xavier initialization
  - `gain`: The gain value for the initialization _[default: 1.0]_
- **kaiming**: Kaiming initialization
  - `mode`: The mode of the initialization _[default: fan_in]_
  - `nonlinearity`: The nonlinearity function to be used _[default: gelu]_
- **none**: No initialization

# TODO: implement the initialization parameters

# Other sub-configs that might be necessary (as specified above)

### Feed Forward Network `ffn` [feedforward.py]

- **generic (link the code)**: A standard FFN block with two linear layers and an activation in-between.

  - `ffn_dim`: The feed-forward dimension (commonly 4x hidden-dim)
  - `bias`: true/false whether the bias units in the linear layers should be used _[defaul: false]_
  - `ffn_dropout`: The dropout rate to be used at the beginning of the feed-forward network. _[default: 0.0]_
  - `activation`: The name of the activation function to be used. Options include: _[defaul: gelu]_
    - _gelu_: https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    - _relu_: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    - _leakyrelu_: https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
    - _tanh_: https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html
    - _sigmoid_: https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
    - _silu_: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
    - _none_: https://pytorch.org/docs/stable/generated/torch.nn.Identity.html
  - `normalization`: The name of the normalization function to be used. The options are listed here _[TODO: link to normalization]_

- **silu_ffn (link the code)**: introduces a gating mechanism by applying the SiLU activation function to one linear transformation of the input, then multiplying it with another linear transformation. This allows for dynamic feature modulation, enhancing the network's expressiveness.
  - `ffn_dim`: The feed-forward dimension
  - `bias`: true/false whether the bias units in the linear layers should be used _[defaul: false]_
  - `normalization`: The name of the normalization function to be used. The options are listed here _[TODO: link to normalization]_

### Attention `attn` [attention.py]

- **causal (link the code)**: A standard MHA block.

  - `num_kv_heads`: The number of K and V heads.
  - `num_q_heads`: The number of Q heads. If GQA is not intended to be used, this can be skipped and will default to `num_kv_heads`. _[default: num_kv_heads]_
  - `bias`: true/false whether the bias units in the linear layers should be used _[defaul: false]_
  - `attn_dropout`: The dropout rate to be used at the beginning of the attention block. _[default: 0.0]_
  - `normalization`: The name of the normalization function to be used. The options are listed here _[TODO: link to normalization]_

- **bidirectional (link the code)**: A standard MHA block.
  - `num_kv_heads`: The number of K and V heads.
  - `num_q_heads`: The number of Q heads. If GQA is not intended to be used, this can be skipped and will default to `num_kv_heads`. _[default: num_kv_heads]_
  - `bias`: true/false whether the bias units in the linear layers should be used _[defaul: false]_

### Normalization `normalization` [normalization.py]

The name of the normalization to be used. Options include: _[defaul: rms_norm]_

- **rms_norm**: (Root Mean Square Normalization): Normalizes the input based on its root mean square (RMS) to stabilize training by controlling the variance of the activations.
- **layer_norm**: (Layer Normalization): Normalizes the input across features in each layer to ensure consistent activation distributions and improve convergence.
- **none**: No normalization is applied.

(TODO include an example .yaml file and link to it)

# Trainer Configuration (`trainer`)

The trainer configuration contains the settings for training the model. There are a number of main trainers to choose from, each with their own sub-configurations. The main trainers are:

## Base Trainer

The base trainer is a standard trainer that can be used for most standard language model training tasks (like autoregressive pre-training, MLM-based pre-training, fine-tuning, etc.). To use the base trainer, just set `trainer_type` to **base_trainer**. The following are the sub-configurations for the base trainer:

- `dataset`: The name of the dataset to be used for training.
- `batch_size`: The batch size to be used for training.
- `gradient_accumulation_steps`: The number of gradient accumulation steps.
- `max_iters`: The maximum number of iterations.
- `decay_lr`: Whether the learning rate should be decayed during training _[default: true]_
- `lr_decay_iters`: The number of iterations after which the learning rate decays _[default: max_iters]_
- `warmup_iters`: The number of warmup iterations.
- `eval_interval`: The number of steps after which the model goes through the inter-training evaluation. _[default: 2000]_
- `log_interval`: The number of steps after which the model logs the training progress. _[default: 10]_
- `checkpoint_interval`: The number of steps after which the model saves a checkpoint. _[default: 10000]_
- `lr_scheduler`: The learning rate scheduler to be used. Options include:
  - **cosine**: Cosine learning rate decay.
- `dataloader`: The dataloader to be used. Options include:
  - **standard**: Standard dataloader.
- `datasampling`: The data sampling strategy to be used. Options include:
  - **standard**: Standard dataloader.
- `loss_fn`: The loss function to be used. Options include:
  - **cross_entropy**: Cross-entropy loss.
  - **next_token_mlm**: The masked language modeling next token loss.
- `eval`: The evaluation sub-config _[TODO: link to the sub-config]_
- `optimizer`: The optimizer sub-config _[TODO: link to the sub-config]_

### Evaluation Sub-config (`eval`)

It can be important to estimate model performance during training. To that end, you can specify the evaluation sub-config.

- `mcq_benchmarks`: A list of benchmarks to evaluate on. The available benchmarks are:
  - **winogrande**:
  - **hellaswag**:
  - **arc_easy**:
  - **mmlu**:
  - **blimp**:
  - **truthful_qa**:
  - **piqa**:
  - **race_middle**:
  - **race_high**:
  - **boolq**:
  - **openbook_qa_closed**:
  - **openbook_qa_open**:
  - **copa**:
  - **commonsense_qa**:
  - **stlm_eval_arc_easy**:
  - **stlm_eval_hellaswag**:
  - **stlm_eval_truthful_qa**:
  - **stlm_eval_winogrande**:
- `num_samples`: The number of samples for evaluation. _[default: 1000]_
- `dataset_loss_eval_iters`: The number of iterations after which the dataset loss is evaluated. _[default: 1000]_
- `text_modeling_eval`: true/false whether to evaluate the text modeling capacity. _[default: true]_

### Optimizer Sub-config (`optimizer`)

Each optimizer requires a number of different values to be specified. Here are the optimizer options:

- **AdamW** (`optimizer_name`): The basic AdamW optimizer
  - `lr`: The learning rate _[default: 6e-4]_
  - `min_lr`: The minimum learning rate _[default: 6e-6]_
  - `weight_decay`: The weight decay factor _[default: 0.01]_
  - `beta1`: The beta1 value for the Adam optimizer _[default: 0.9]_
  - `beta2`: The beta2 value for the Adam optimizer _[default: 0.999]_
  - `grad_clip`: The gradient clipping value _[default: 1.0]_

# General Configuration (`general`)

## Logging (`logging`)

Defines logging options.

- **wandb_log**: Boolean flag to enable/disable logging to Weights & Biases. _[default: false]_
- **wandb_project**: Name of the Weights & Biases project. _[default: SuperTinyLanguageModels]_
- **wandb_run_name**: Custom name for the run. Highly encouraged. If not provided, will be as descriptive as possible. _[default: None]_

## Paths (`paths`)

Defines paths for saving outputs, data, checkpoints, and evaluations.

- **output_dir**: Directory for output files.
- **data_dir**: Directory for data files.
- **checkpoint_dir**: Directory for checkpoints.
- **eval_dir**: Directory for evaluation files.

## Seed (`seed`)

Defines the random seed used for reproducibility.

## Device (`device`)

Defines the device for training and evaluation _[default: cuda]_
