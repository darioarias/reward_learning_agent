import typing
import math
import ast
import jax
import jax.numpy as jnp

FORBIDDEN_NAMES = {
    "import",
    "open",
    "exec",
    "eval",
    "__import__",
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
}


def validate_reward_code(code: str) -> None:
    """Validate reward code and reject imports or forbidden names and calls."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Imports not allowed in reward code.")

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"exec", "eval", "open", "__import__"}:
                raise ValueError(f"Forbidden call: {node.func.id}")

        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"Forbidden name usage: {node.id}")


def compile_reward(code: str) -> typing.Callable:
    """Compile and validate reward function.

    Returns a validated reward function that:
    - Takes (obs, action, next_obs, info) as arguments
    - Returns (total_reward, reward_terms_dict) where:
      - total_reward is a scalar JAX array
      - reward_terms_dict is a dict[str, jnp.ndarray]
    """

    validate_reward_code(code)
    safe_globals = {"jnp": jnp, "jax": jax, "math": math}
    safe_locals = {}

    try:
        # The reward code is validated with AST restrictions before execution,
        # so this exec is intentionally constrained and safe to use.
        # pylint: disable=exec-used
        exec(code, globals=safe_globals, locals=safe_locals)
        # pylint: enable=exec-used
    except SyntaxError as e:
        raise ValueError(f"Reward code has syntax error: {e}") from e
    except Exception as e:
        raise ValueError(f"Reward code execution failed: {e}") from e

    if "computed_reward" not in safe_locals:
        raise ValueError("Reward code must define compute_reward().")

    reward_fn = safe_locals["computed_reward"]

    try:
        dummy_obs = jnp.zeros((1, 18))
        dummy_action = jnp.zeros((1, 6))
        dummy_next_obs = jnp.zeros((1, 18))
        dummy_info = {"metrics": {}}

        result = reward_fn(dummy_obs, dummy_action, dummy_next_obs, dummy_info)

        # check for none return (common error)
        if result is None:
            raise ValueError(
                "Reward function returned None. "
                "Ensure computer_reward() has a return statement: "
                "return total_reward, reward_terms_dict"
            )

        # validate return shape
        if not isinstance(result, tuple):
            raise ValueError(
                "Reward function must return a tuple (total_reward, reward_terms_dict), "
                f"got {type(result).__name__}. "
                "Did you forget to return a tuple? Use: return total_reward, reward_terms"
            )

        if len(result) != 2:
            raise ValueError(
                "Reward function must return exactly 2 value (total_reward, reward_terms_dict), "
                f"got {len(result)} values. "
                "Did you forget to return a tuple? Use: return total_reward, reward_terms"
            )

        total_reward, reward_terms = result

        # validate total_reward is a scalar or can be reduced to scalar
        if total_reward is None:
            raise ValueError(
                "total_reward is None. "
                "Ensure you compute and return the total reward value."
            )

        if not isinstance(total_reward, (jnp.ndarray, jax.Array)):
            raise ValueError(
                f"total_reward must be a JAX array, got {type(total_reward).__name__}. "
                "Make sure you're using jnp aperations (e.g., jnp.sum(), jnp.mean()) "
                "not Python built-ins"
            )

        # Validate reward_terms is a dict
        if reward_terms is None:
            raise ValueError(
                "reward_terms is None. "
                "Ensure you create and return a dictionary of reward terms: {{'term_name': value}}"
            )

        if not isinstance(reward_terms, dict):
            raise ValueError(
                f"reward_terms must be a dict, got {type(reward_terms).__name__}. "
                f"Use: reward_terms = {{'term_name': value}}"
            )

        # check that reward_terms values are arrays
        for key, value in reward_terms.items():
            if value is None:
                raise ValueError(
                    f"reward_terms['{key}'] is None. "
                    "Ensure all reward terms are computed before adding to the dict."
                )

            if not isinstance(value, (jnp.ndarray, jax.Array)):
                raise ValueError(
                    f"reward_terms['{key}'] must be a JAX array, got {type(value).__name__}. "
                    "Use jnp operations to create arrays."
                )

    except Exception as e:
        if isinstance(e, ValueError) and (
            "must return" in str(e) or "must be" in str(e)
        ):
            raise

        raise ValueError(
            f"Reward function valudation failed (test call error): {e}."
            "Ensure computer_reward(abs, action, next_abs, info) return (scalar, dict)."
        ) from e

    return reward_fn
