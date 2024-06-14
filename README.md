# Super Tiny Language Models

This GitHub repository presents our research on Super Tiny Language Models (STLMs), aimed at delivering high performance with significantly reduced parameter counts (90-95% smaller) compared to traditional large language models. We explore innovative techniques such as byte-level tokenization with pooling, weight tying, and efficient training strategies. The codebase covers various subproblems, including tokenizer-free models, self-play based training, and alternative training objectives, targeting models with 10M, 50M, and 100M parameters while maintaining competitive performance.

Our mission is to enhance the accessibility and practicality of high-performing language models across a wide range of applications by drastically reducing their computational and energy demands. We believe that our approach has the potential to contribute to more sustainable and inclusive AI development.

For a comprehensive understanding of our research methodology and initial findings, we strongly encourage you to read our paper: [Super Tiny Language Models](https://arxiv.org/abs/2405.14159)

Please note that this repository is an evolving work in progress, reflecting the ongoing nature of our research. It is subject to frequent updates and improvements as we continue to explore and refine our work on STLMs. We welcome the community's engagement with our work, value your feedback, and appreciate any contributions to this challenging but promising endeavor.


## Download
### Access Hugging Face
We aim to provide downloads on Hugging Face 🤗 in time shortly! 

## Quick Start
Experiments are a big part of what we do. However, there's presently no visibility on 1/ what experiments we are actively running, 2/ how have these experiments performed, and 3/ what are the intuitions behind them. We see this as an urgent issue. Hence, we will add a sunmmary view of our experiments as soon as possible. From which, you may better inform yourself before trying them out.

Till then, we are afraid we'll have to trouble you to scrappily explore our configs/full_configs folder. 

### Setup
Please install a linter, formatter, and pre-commit hooks before contributing. You can do this by running the following commands:
```bash
pip install -r requirements.txt
pre-commit install
```

We include a .pylintrc file in the repository. For VSCode users, you can install the Python extension and set the following configurations in your settings.json file:
```json
{
    "python.linting.pylintEnabled": true,
    "python.linting.enabled": true,
    "python.formatting.provider": "black",
    "python.linting.pylintArgs": [
        "--rcfile=.pylintrc"
    ]
}
```

### Execution
Subsequently, you may run them using:
```bash
python train.py --config-name full_configs/...
```
*(Note: You can omit the .yaml ending.)*

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 

## License
MIT
