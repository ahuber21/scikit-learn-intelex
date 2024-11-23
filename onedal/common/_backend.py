# ==============================================================================
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
from contextlib import contextmanager
from types import MethodType
from typing import Any, Callable, Literal, Optional

from onedal import Backend, _default_backend, _spmd_backend
from onedal.common.policy_manager import PolicyManager

from .backend_manager import BackendManager

logger = logging.getLogger(__name__)

default_manager = BackendManager(_default_backend)
spmd_manager = BackendManager(_spmd_backend)

# define types for backend functions: default, dpc, spmd
BackendType = Literal["host", "dpc", "spmd"]

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class BackendFunction:
    """Wrapper around backend function to allow setting auxiliary information"""

    backend_type: BackendType

    def __init__(
        self,
        method: Callable[..., Any],
        backend: Any,
        name: str,
    ):
        self.method = method
        self.backend = backend
        self.name = name
        self.policy = None
        if backend.is_spmd:
            self.backend_type = "spmd"
        elif backend.is_dpc:
            self.backend_type = "dpc"
        else:
            self.backend_type = "host"

    def update_policy(self, *data) -> None:
        # load the correct policy for the given data
        self.policy = PolicyManager(self.backend).get_policy(None, *data)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.policy is not None:
            # inject the assigned policy into the method and dispatch to backend
            return self.method(self.policy, *args, **kwargs)
        else:
            try:
                return self.method(*args, **kwargs)
            except TypeError as e:
                raise TypeError(
                    "Call to module function raised TypeError. Did you forget to call 'update_policy' before calling the function?"
                ) from e

    def __repr__(self) -> str:
        return f"BackendFunction(<{self.backend_type}_backend>.{self.name})"


def inject_policy_manager(backend: Backend) -> Callable[..., Any]:
    def _get_policy(self, queue: Any, *data: Any) -> Any:
        policy_manager = PolicyManager(backend)
        return policy_manager.get_policy(queue, *data)

    return _get_policy


@contextmanager
def DefaultPolicyOverride(instance: Any):
    original_method = getattr(instance, "_get_policy", None)
    try:
        # Inject the new _get_policy method from _default_backend
        new_policy_method = inject_policy_manager(_default_backend)
        bound_method = MethodType(new_policy_method, instance)
        setattr(instance, "_get_policy", bound_method)
        yield
    finally:
        # Restore the original _get_policy method
        if original_method is not None:
            setattr(instance, "_get_policy", original_method)
        else:
            delattr(instance, "_get_policy")


def bind_default_backend(module_name: str, lookup_name: Optional[str] = None):
    def decorator(method: Callable[..., Any]):
        # grab the lookup_name from outer scope
        nonlocal lookup_name

        if lookup_name is None:
            lookup_name = method.__name__

        if _default_backend is None:
            logger.debug(
                f"Default backend unavailable, skipping decoration for '{method.__name__}'"
            )
            return method

        if lookup_name == "_get_policy":
            return inject_policy_manager(_default_backend)

        backend_method = default_manager.get_backend_component(module_name, lookup_name)
        wrapped_method = BackendFunction(
            backend_method,
            backend=_default_backend,
            name=f"{module_name}.{method.__name__}",
        )

        backend_name = "dpc" if _default_backend.is_dpc else "host"
        logger.debug(
            f"Assigned method '<{backend_name}_backend>.{module_name}.{lookup_name}' to '{method.__qualname__}'"
        )

        return wrapped_method

    return decorator


def bind_spmd_backend(module_name: str, lookup_name: Optional[str] = None):
    def decorator(method: Callable[..., Any]):
        # grab the lookup_name from outer scope
        nonlocal lookup_name

        if lookup_name is None:
            lookup_name = method.__name__

        if _spmd_backend is None:
            logger.debug(
                f"SPMD backend unavailable, skipping decoration for '{method.__name__}'"
            )
            return method

        if lookup_name == "_get_policy":
            return inject_policy_manager(_spmd_backend)

        backend_method = spmd_manager.get_backend_component(module_name, lookup_name)
        wrapped_method = BackendFunction(
            backend_method,
            backend=_spmd_backend,
            name=f"{module_name}.{method.__name__}",
        )

        logger.debug(
            f"Assigned method '<spmd_backend>.{module_name}.{lookup_name}' to '{method.__qualname__}' "
        )

        return wrapped_method

    return decorator
