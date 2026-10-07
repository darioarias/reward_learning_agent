import json
import typing
import pickle
from pathlib import Path
from datetime import datetime
import jax
import jax.numpy as jnp
from loguru import logger


def init_run_dir(base: str = "output") -> Path:
    """Create and return a unique output directory for a run.

    Args:
        base: Base directory under which the run directory is created.

    Returns:
        Path to the newly created run directory.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(base) / f"run_{run_id}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_json(path: Path, obj: object) -> None:
    """Save object as JSON with error handling."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, default=str))
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}")
        raise


def save_text(path: Path, text: str) -> None:
    """Save text to file with error handling."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    except Exception as e:
        logger.error(f"failed to save text to {path}: {e}")
        raise


def save_policy_params(path: Path, params: dict) -> None:
    """Save policy parameters using pickle.

    Extracts only the parameter arrays (no closures or function references)
    to ensure successful serialization. Uses JAX tree utilities to safely
    extract only array data.

    Note: The inference function is intentionally not saved because it is
    typically a closure and not pickle-serializable. To reload, recreate the
    environment and inference function, then load these params.
    """

    try:
        import numpy as np

        def is_array_like(obj: object) -> bool:
            """Check if x is a JAX or numpy array."""
            return isinstance(obj, (jnp.ndarray, jax.Array, np.ndarray))

        def extract_arrays_only(param: typing.Any):
            """Recursively extract only array data, converting to numpy.

            This function aggressively filters out any non-array objects
            including functions, closures, and other unpicklable objects.
            """

            try:
                if is_array_like(param):
                    # convert JAX array to numpy adn create a copy to avoid view issues
                    array = np.asarray(param)
                    return array.copy() if array.flags.writeable else array

                if isinstance(param, dict):
                    result = {}
                    for key, value in param.items():
                        try:
                            extracted = extract_arrays_only(value)
                            if extracted is not None:
                                result[key] = extracted
                        except (TypeError, AttributeError, ValueError):
                            # Skip entries that can't be processed (functions, closures, etc)
                            continue
                    return result if result else None

                if isinstance(param, (list, tuple)):
                    # Recursively process lists/tuple
                    extracted = []

                    for value in param:
                        try:
                            result = extract_arrays_only(value)
                            if result is not None:
                                extracted.append(result)
                        except (TypeError, AttributeError, ValueError):
                            # Skip items that can't be processed
                            continue

                    if extracted:
                        # Preserve tuple type if original was tuple
                        return (
                            type(param)(extracted)
                            if isinstance(param, tuple)
                            else extracted
                        )
                    return None

                # if callable(param) or hasattr(param, "__call__"):
                if callable(param):
                    # explicitly skip callable objects (function, closures)
                    return None

                return None  # skip everything else
            except (TypeError, AttributeError, ValueError, RuntimeError) as e:
                # if anything goes wrong during extraction, skip this node
                logger.debug(
                    "Skopping unpicklable object during "
                    f"param extraction: {type(param).__name__}: {e}"
                )
                return None

        # extract onlu arrays from the params structure
        clean_params = extract_arrays_only(params)
        if clean_params is None:
            raise ValueError(
                "No array data found in params structure. "
                "Params may be empty or contain only non-array objects."
            )

        # verify the extracted params are picklable before saving
        try:
            pickle.dumps(clean_params)
        except Exception as pickle_test_error:
            logger.warning(
                f"Extracted params are not directly picklable: {pickle_test_error}. "
                "Trying alternative serialization approach..."
            )

            # try using JAX's built-in serialization if available
            # otherwise, convert to a more basic structure
            def to_serializable(obj: object):
                """Convert to a structure that's definitely serializable."""
                if isinstance(obj, np.ndarray):
                    return (
                        obj.tolist() if obj.size < 10_000 else obj
                    )  # small arrays as lists, large as arrays

                if isinstance(obj, dict):
                    return {key: to_serializable(value) for key, value in obj.items()}

                if isinstance(obj, (list, tuple)):
                    converted = [to_serializable(value) for value in obj]
                    return type(obj)(converted) if isinstance(obj, tuple) else converted

                return obj

            clean_params = to_serializable(clean_params)

        policy_data = {
            "params": clean_params,
            "serialization": "pickle",
            "inference_fn_saved": False,
            "inference_fn_note": (
                "Inference function not saved. Recreate it using the same "
                "environment/reward setup and load tehse params."
            ),
        }

        with open(path, "wb") as policy_file:
            pickle.dump(policy_data, policy_file)

    except Exception as e:
        raise ValueError(
            f"Failed to save policy parameters: {e}. "
            "This usually means params contains unpicklable objects (closures/functions). "
            "Try ensuring params only contains arrays/dicts/lists."
        ) from e


def init_leaderboard(outdir: Path) -> Path:
    """Initialize the leaderboard CSV if it does not already exist.

    Args:
        outdir: Directory where the leaderboard file should be created.

    Returns:
        Path to the leaderboard CSV file.
    """
    leaderbord_path = outdir / "leaderboard.csv"
    if not leaderbord_path.exists():
        leaderbord_path.write_text(
            "iteration,candidate,score,avg_return,artifact_path\n"
        )
    return leaderbord_path


def append_leaderboard(
    leaderboard_path: Path,
    iteration: int,
    candidate: int,
    score: float,
    avg_return: float,
    artifact_path: str,
) -> None:
    """
    appends to the leaderboard file
    """
    try:
        leaderboard_path = Path(leaderboard_path)
        leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        with open(leaderboard_path, "a", encoding="utf-8") as file:
            file.write(
                f"{iteration},{candidate},{score},{avg_return},{artifact_path}\n"
            )
    except Exception as e:
        logger.error(f"Failed to append to leaderboard {leaderboard_path: {e}}")
        raise


def iteration_dir(outdir: Path, iteration: int) -> Path:
    """Create and return the directory for a specific training iteration."""
    path = outdir / f"iteration_{iteration:03d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def candidate_dir(outdir: Path, iteration: int, candidate: int) -> Path:
    """Create and return the directory for a specific candidate within an iteration."""
    path = iteration_dir(outdir, iteration) / f"candidate_{candidate:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_policy_params(path: Path) -> typing.Any:
    """Load policy parameters from a saved pickle file.

    Args:
        path: Path to the saved policy file (.pkl)

    Returns:
        dict with 'params' key containing the loaded parameters, and optionally
        'make_inference_fn' if it was successfully serialized
    """

    path = Path(path)

    if path.suffix != ".pkl":
        raise ValueError(f"expected .pkl file, got {path.suffix}")

    with open(path, "rb") as file:
        policy_data = pickle.load(file)
    return policy_data
