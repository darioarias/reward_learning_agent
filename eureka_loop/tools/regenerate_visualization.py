"""
Helper script to regenerate visualizations from saved artifacts.

This script allows you to regenerate HTML visualizations for saved candidates
by recreating the inference function directly from saved params and network config.
No retraining needed - avoids dimension mismatches!
"""

import json
from pathlib import Path
from loguru import logger

from .rl_runner import recreate_inference_fn, save_policy_rollout_html
from .reward_sandbox import compile_reward
from .artifacts import load_policy_params


def regenerate_visualization(
    candidate_dir: Path,
    run_root_dir: Path | None = None,
    horizon: int = 400,
    use_cpu: bool = True,
):
    """
    Regenerate HTML visualization for a saved candidate.

    This recreates the inference function directly from saved params and network config,
    avoiding the need to retrain. This ensures dimension matching and is much faster.

    Args:
        candidate_dir: Path to candidate directory (e.g., outputs/run_XXX/iteration_001/candidate_01)
        run_root_dir: Path to run root (e.g., outputs/run_XXX). If None, inferred from candidate_dir.
        horizon: Horizon for rollout visualization
        use_cpu: Whether to use CPU for rendering (avoids GPU JIT hangs)

    Returns:
        Path to generated HTML file
    """
    candidate_dir = Path(candidate_dir)
    logger.info(f"Regenerating visualization for {candidate_dir}")

    # Load reward function
    reward_code = (candidate_dir / "reward.py").read_text()
    reward_fn = compile_reward(reward_code)
    logger.info("Loaded reward function")

    # Load training metadata (includes network_config)
    metadata = json.loads((candidate_dir / "training_metadata.json").read_text())
    network_config = metadata.get("network_config")
    if not network_config:
        raise ValueError(
            "network_config not found in training_metadata.json. "
            "This candidate was saved with an older version. "
            "You may need to retrain or use the old regeneration method."
        )
    logger.info(
        "Loaded network config: obs_size="
        f"{network_config['observation_size']}, "
        f"action_size={network_config['action_size']}"
    )

    # Load policy params
    params_data = load_policy_params(candidate_dir / "policy_params.pkl")
    saved_params = params_data["params"]
    logger.info("Loaded policy parameters")

    # Load environment info
    if run_root_dir is None:
        run_root_dir = candidate_dir.parent.parent.parent
    run_root_dir = Path(run_root_dir)
    env_meta = json.loads((run_root_dir / "env_meta.json").read_text())
    env_name = env_meta["env_name"]
    logger.info(f"Loaded environment: {env_name}")

    # Recreate inference function directly from network config (NO RETRAINING!)
    logger.info(
        "Recreating inference function from network config (no retraining needed)..."
    )
    make_inf = recreate_inference_fn(
        env_name=env_name,
        reward_fn=reward_fn,
        network_config=network_config,
    )
    logger.info("Recreated inference function")

    # Convert saved params back to JAX arrays if needed
    import jax
    import jax.numpy as jnp
    import numpy as np

    def numpy_to_jax(obj):
        """Recursively convert numpy arrays to JAX arrays."""
        if isinstance(obj, np.ndarray):
            return jnp.array(obj)
        elif isinstance(obj, dict):
            return {k: numpy_to_jax(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            converted = [numpy_to_jax(v) for v in obj]
            return type(obj)(converted)
        else:
            return obj

    params = numpy_to_jax(saved_params)

    # Generate visualization
    logger.info("Generating HTML visualization...")
    save_policy_rollout_html(
        out_dir=candidate_dir,
        env_name=env_name,
        reward_fn=reward_fn,
        make_inference_fn=make_inf,
        params=params,
        seed=metadata["seed"],
        horizon=horizon,
        use_cpu_for_rendering=use_cpu,
    )

    html_path = candidate_dir / "rollout.html"
    logger.info(f"Visualization saved to {html_path}")
    return html_path
