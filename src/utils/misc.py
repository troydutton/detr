import os
from contextlib import contextmanager, redirect_stdout
from typing import Callable, Dict, List, Tuple, TypeVar, Union

from torch import Tensor, device
from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


def take_annotation_from(this: Callable[P, T]) -> Callable[[Callable], Callable[P, T]]:
    def decorator(real_function: Callable[P, T]) -> Callable[P, T]:
        real_function.__doc__ = this.__doc__
        return real_function

    return decorator


def send_to_device(
    object: Union[Tensor, Dict, List, Tuple],
    device: device,
) -> Union[Tensor, Dict, List, Tuple]:
    """
    Send all tensors in an object to a device.

    Args:
        object: Object containing tensors.
        device: Device to send the tensors to.

    Returns:
        object: Object with tensors on the specified device.
    """

    if isinstance(object, Tensor):
        return object.to(device)
    elif isinstance(object, dict):
        return {k: send_to_device(v, device) for k, v in object.items()}
    elif isinstance(object, (list, tuple)):
        return [send_to_device(element, device) for element in object]
    else:
        return object


@contextmanager
def silence_stdout():
    """
    Context manager to silence stdout.
    """
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            yield
