import typing
import inspect
from brax.envs import create

if typing.TYPE_CHECKING:
    from brax.envs.base import Env


def get_env_and_source(env_name: str) -> tuple["Env", str, str]:
    """Create a Brax environment and return its module source code.

    Args:
        env_name: Name of the Brax environment to create.

    Returns:
        A tuple containing the environment instance, the source of the module
        defining its class, and the module name.
    """
    env = create(env_name)
    assert (mod := inspect.getmodule(env.__class__))
    env_source = inspect.getsource(mod)
    return env, env_source, mod.__name__
