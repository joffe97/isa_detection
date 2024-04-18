import inspect
from typing import Type


def get_object_variables(object_: object) -> list[tuple[str, object]]:
    return list(object_.__dict__.items())
