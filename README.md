# Super Tiny Language Models
This repository is WIP and highly subject to change

## Running & Recreating
See the full_configs folder and run using:
```bash
python train.py --config-name full_configs/...
```
You can omit the .yaml ending.

## Contribution & Setup
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

## Known Issues
The initial tokenization step setting up the dataset takes a long time. Do a run with a single gpu first to make sure these are initialized properly (or even use a debug run) when you are using a new dataset or tokenizer.
If it fails then it will always fail in the future due to some dodgy code. To fix this delete the contents of the data/{dataset_name} folder

## License
We should probably add a license...
