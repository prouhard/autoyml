from functools import wraps
from typing import Any, Callable

from autoyml.abstract_model import AbstractModel
from autoyml.errors import NotFittedError


def require_fit(func: Callable[..., Any]) -> Callable[..., Any]:
    """Check that the underlying model has been fitted before executing `func`.

    Args:
        func: A method of a subclass of `AbstractModel`.

    Returns:
        The decorated method.

    Raises:
        NotFittedError: If the model is not ready for the method execution.

    """

    @wraps(func)
    def wrapper(self: AbstractModel, *args: Any, **kwargs: Any) -> Any:
        if self._model is None:
            raise NotFittedError(
                f"'{self.__class__.__name__}' instance must be fitted before calling '{func.__name__}'."
            )
        return func(self, *args, **kwargs)

    return wrapper
