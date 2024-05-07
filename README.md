# Super Tiny Language Models
This repository is WIP and highly subject to change


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

## License
We should probably add a license...
