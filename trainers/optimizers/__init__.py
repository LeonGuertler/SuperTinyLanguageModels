import torch

OPTIMIZER_REGISTRY = {
    "adam": lambda model, optimizer_config: torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config.get("lr", 0.001),
        betas=(
            optimizer_config.get("beta1", 0.9),
            optimizer_config.get("beta2", 0.999),
        ),
        eps=optimizer_config.get("eps", 1e-8),
        weight_decay=optimizer_config.get("weight_decay", 0),
        amsgrad=optimizer_config.get("amsgrad", False),
    ),
    "adamW": lambda model, optimizer_config: torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.get("lr", 0.001),
        betas=(
            optimizer_config.get("beta1", 0.9),
            optimizer_config.get("beta2", 0.999),
        ),
        eps=optimizer_config.get("eps", 1e-8),
        weight_decay=optimizer_config.get("weight_decay", 0.01),
        amsgrad=optimizer_config.get("amsgrad", False),
    ),
    "sgd": lambda model, optimizer_config: torch.optim.SGD(
        model.parameters(),
        lr=optimizer_config["lr"],
        momentum=optimizer_config.get("momentum", 0),
        dampening=optimizer_config.get("dampening", 0),
        weight_decay=optimizer_config.get("weight_decay", 0),
        nesterov=optimizer_config.get("nesterov", False),
    ),
    "adadelta": lambda model, optimizer_config: torch.optim.Adadelta(
        model.parameters(),
        lr=optimizer_config.get("lr", 1.0),
        rho=optimizer_config.get("rho", 0.9),
        eps=optimizer_config.get("eps", 1e-6),
        weight_decay=optimizer_config.get("weight_decay", 0),
    ),
    "adagrad": lambda model, optimizer_config: torch.optim.Adagrad(
        model.parameters(),
        lr=optimizer_config.get("lr", 0.01),
        lr_decay=optimizer_config.get("lr_decay", 0),
        weight_decay=optimizer_config.get("weight_decay", 0),
        initial_accumulator_value=optimizer_config.get("initial_accumulator_value", 0),
        eps=optimizer_config.get("eps", 1e-10),
    ),
    "adamax": lambda model, optimizer_config: torch.optim.Adamax(
        model.parameters(),
        lr=optimizer_config.get("lr", 0.002),
        betas=(
            optimizer_config.get("beta1", 0.9),
            optimizer_config.get("beta2", 0.999),
        ),
        eps=optimizer_config.get("eps", 1e-8),
        weight_decay=optimizer_config.get("weight_decay", 0),
    ),
    "rmsprop": lambda model, optimizer_config: torch.optim.RMSprop(
        model.parameters(),
        lr=optimizer_config.get("lr", 0.01),
        alpha=optimizer_config.get("alpha", 0.99),
        eps=optimizer_config.get("eps", 1e-8),
        weight_decay=optimizer_config.get("weight_decay", 0),
        momentum=optimizer_config.get("momentum", 0),
        centered=optimizer_config.get("centered", False),
    ),
    "asgd": lambda model, optimizer_config: torch.optim.ASGD(
        model.parameters(),
        lr=optimizer_config.get("lr", 0.01),
        lambd=optimizer_config.get("lambd", 1e-4),
        alpha=optimizer_config.get("alpha", 0.75),
        t0=optimizer_config.get("t0", 1e6),
        weight_decay=optimizer_config.get("weight_decay", 0),
    ),
    "rprop": lambda model, optimizer_config: torch.optim.Rprop(
        model.parameters(),
        lr=optimizer_config.get("lr", 0.01),
        etas=optimizer_config.get("etas", (0.5, 1.2)),
        step_sizes=optimizer_config.get("step_sizes", (1e-6, 50)),
    ),

}

def build_optimizer(model, optimizer_config):
    return OPTIMIZER_REGISTRY[optimizer_config["optimizer_name"]](
        model, optimizer_config
    )
