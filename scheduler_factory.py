import math

import torch.optim.lr_scheduler as lr_scheduler


class WarmupCosineClampLR(lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        total_epochs,
        warmup_epochs=5,
        t_max=None,
        min_lr_ratio=0.01,
    ):

        if t_max is None:
            t_max = total_epochs

        def lr_lambda(epoch):

            # Warmup
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs

            # Cosine phase
            cosine_epoch = epoch - warmup_epochs
            cosine_tmax = max(1, t_max - warmup_epochs)

            # Clamp to min_lr_ratio after t_max
            if cosine_epoch >= cosine_tmax:
                return min_lr_ratio

            progress = cosine_epoch / cosine_tmax

            cosine = 0.5 * (
                1 + math.cos(math.pi * progress)
            )

            return (
                min_lr_ratio
                + (1 - min_lr_ratio) * cosine
            )

        super().__init__(optimizer, lr_lambda)


def build_scheduler(optimizer, config):
    sched_config = config["scheduler"]
    name = sched_config["name"].lower()
    params = sched_config["params"]

    if name == "reduce_on_plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **params
        )

    elif name == "step":
        return lr_scheduler.StepLR(
            optimizer,
            **params
        )

    elif name == "cosine_annealing":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
            **params
        )
    
    elif name == "cosine_warmup":

        total_epochs = config["training"]["epochs"]
        warmup_epochs = params.get("warmup_epochs", 5)
        min_lr_ratio = params.get("min_lr_ratio", 0.01)

        def lr_lambda(epoch):

            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs

            progress = (
                epoch - warmup_epochs
            ) / max(1, total_epochs - warmup_epochs)

            cosine = 0.5 * (
                1 + math.cos(math.pi * progress)
            )

            return min_lr_ratio + (1 - min_lr_ratio) * cosine

        return lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda
        )

    elif name == "cosine_warmup_clamp":
        total_epochs = config["training"]["epochs"]
        warmup_epochs = params.get("warmup_epochs", 5)
        min_lr_ratio = params.get("min_lr_ratio", 0.01)
        t_max = params.get("t_max", None)

        return WarmupCosineClampLR(
            optimizer,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            min_lr_ratio=min_lr_ratio,
            t_max=t_max
        )

    elif name == "none":
        return None

    else:
        raise ValueError(f"Unknown scheduler: {name}")