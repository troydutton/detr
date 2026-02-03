import logging
from typing import Any, Dict, Union

Args = Dict[str, Union["Args", str, Any]]


def prepare_scheduler_arguments(args: Args, steps_per_epoch: int) -> Args:
    """
    Prepare learning rate scheduler arguments by adjusting step sizes based on steps per epoch.

    Args:
        args: Scheduler arguments.
        steps_per_epoch: Number of optimization steps in one epoch.

    Returns:
        args: Adjusted scheduler arguments.
    """

    name = args["_target_"].split(".")[-1]

    if name == "StepLR":
        args["step_size"] *= steps_per_epoch
    else:
        raise NotImplementedError(f"Scheduler '{name}' not supported yet.")

    logging.info(f"Prepared '{name}' scheduler with " + ", ".join([f"{k}={v:,}" for k, v in args.items() if k != "_target_"]) + ".")

    return args
