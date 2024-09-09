# Super Tiny Language Models

[Model Interfaces](models/README.md) | [Full Configs](configs/full_configs/) | [Intro Paper](https://arxiv.org/abs/2405.14159) | [Discord](https://discord.gg/wwTruDPH)

This GitHub repository presents our research on Super Tiny Language Models (STLMs), aimed at delivering high performance with significantly reduced parameter counts (90-95% smaller) compared to traditional large language models. We explore innovative techniques such as byte-level tokenization with pooling, weight tying, and efficient training strategies. The codebase covers various subproblems, including tokenizer-free models, self-play based training, and alternative training objectives, targeting models with 10M, 50M, and 100M parameters while maintaining competitive performance.

Our mission is to enhance the accessibility and practicality of high-performing language models across a wide range of applications by drastically reducing their computational and energy demands. We believe that our approach has the potential to contribute to more sustainable and inclusive AI development.

For a comprehensive understanding of our research methodology and initial findings, we strongly encourage you to read our paper: [Super Tiny Language Models](https://arxiv.org/abs/2405.14159)

Please note that this repository is an evolving work in progress, reflecting the ongoing nature of our research. It is subject to frequent updates and improvements as we continue to explore and refine our work on STLMs. We welcome the community's engagement with our work, value your feedback, and appreciate any contributions to this challenging but promising endeavor.

## Research Reports

| Research Area               | Branch Name                    | Discord Channel     | Preregistration | Full Report  |
|-----------------------------|--------------------------------|---------------------|-----------------|--------------|
| Dropout | `dropout-sched-exp`   |                | [View preregistration](https://github.com/LeonGuertler/SuperTinyLanguageModels/blob/main/pre_reports/dropout_prereport.pdf) | In progress |
| Knowledge distillation  | `feature/knowledge-distillation/replace-teacher-tokenizer`|                  | [View preregistration](https://github.com/LeonGuertler/SuperTinyLanguageModels/blob/feature/knowledge-distillation/replace-teacher-tokenizer/reports/preregistration-knowledgedistillation.pdf)  | In progress  |
| Weight Tying | `ffn-sharing` |  | [View preregistration](pre_reports/weight_tying_prereport.pdf)| In progress |
| Byte Level             |        |                       | In progress  | In progress  |
| Self Play Evals        |            |   [Join the room](https://discord.gg/hgVhe6Hp)                 | In progress| In progress  |
| Optimizers   | `optimizers` | [Join the room](https://discord.gg/S5Q2ZmWY) | In progress | In progress

## Download
### Access to Hugging Face
We aim to provide downloads on Hugging Face 🤗 in time shortly!

## Quick Start
Experiments are a big part of what we do. However, there's presently no visibility on 1/ what experiments we are actively running, 2/ how have these experiments performed, and 3/ what are the intuitions behind them. We see this as an urgent issue. Hence, we will add a summary view of our experiments as soon as possible. From which, you may better inform yourself before trying them out.

Till then, we are afraid we'll have to trouble you to scrappily explore our [configs](configs/full_configs) folder. Here are steps you can take to use them locally.

### Setup
For basic usage, you can just install the requirements using:
```bash
pip install -r requirements.txt
```

### Execution & Scripts
The general way to execute a script is to add the config name as an argument. For example:
```bash
python train.py --config-name full_configs/...
```
*(Note: You can omit the .yaml ending.)*
We expose the following scripts:
- [train.py](train.py) - This script trains a model using the given configuration. It supports multi-gpu setups.
- [eval.py](eval.py) - Given a model_ckpt arg, this script evaluates the model on the test set. See the [confgs](configs/test.yaml) for the arguments used. Typically you just need to call it like `python eval.py model-ckpt=...`. The easiest way to run with huggingface models is to just comment the `model_ckpt` argument, and uncomment the ones with key `model`, changing the model_string as needed. (TODO: change this...)
- [generate.py](generate.py) - This script just builds a generator and allows you to interact with a model from the commandline putting in prompts and generating continuations. The interface for changing the model is the same as eval.py.

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please install a linter, formatter, and pre-commit hooks before contributing. You can do this by running the following command:
```bash
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
Note that our precommit hooks do not actually enforce the linting, although you can inspect the results in github actions.

### Experimental Contribution

If you want to add your own experiments please follow the following steps:
1. Open a PR and add a pre-report to the pre_reports folder. This should be a short document outlining the experiment you want to run, the expected results, and the motivation behind it. It should also include a pre-report disclosure statement stating the degree to which particular results are anticipated (e.g. by earlier experimentation).
2. Then add code to the same branch necessary to run your experiments. Usually this would be added to the [experimental folder under models](models/experimental/) although this will have to be integrated in e.g. the build functions to work properly
3. Finally run the experiments and write your final report. Please cite us the original pre-report in the final report!

For any questions around this please ask on the discord or open an issue, we are still establishing these guidelines

## Known Issues
The initial tokenization step setting up the dataset takes a long time. Do a run with a single gpu first to make sure these are initialized properly (or even use a debug run) when you are using a new dataset or tokenizer.

If it fails then it will always fail in the future due to some dodgy code. To fix this delete the contents of the data/{dataset_name} folder.

Currently this is not a properly packaged project, we plan to do this in the next refactoring after we have released our first set of experimental reports, after which point we aim to move towards a more stable release.

## License
[MIT](LICENSE)
