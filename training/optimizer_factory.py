import torch.optim as optim


def build_optimizer(param_groups, config):
    opt_config = config["optimizer"]
    params = opt_config["params"]
    name = opt_config["name"].lower()

    if name == "adam":
        return optim.Adam(
            param_groups,
            **params
        )

    elif name == "adamw":
        return optim.AdamW(
            param_groups,
            **params
        )

    elif name == "radam":
        return optim.RAdam(
            param_groups,
            **params
        )

    elif name == "sgd":
        return optim.SGD(
            param_groups,
            momentum=0.9,
            **params
        )

    else:
        raise ValueError(f"Unknown optimizer: {name}")
    
