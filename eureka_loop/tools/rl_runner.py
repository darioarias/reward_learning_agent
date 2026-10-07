import jax
import jax.numpy as jnp
import numpy as np
from brax.envs import create
from brax.io import html
from loguru import logger
from pathlib import Path

# Note: We use HTML viewer for visualizations (interactive 3D animation)
# No MP4 rendering needed - HTML viewer is better and doesn't cause GPU JIT hangs

# Try different import patterns for Brax PPO
# Handle case where 'train' might be a module or a function
import importlib
import types

try:
    from brax.training.agents.ppo import train as ppo_train_raw

    # Check if train is a module (not callable and has __file__ or __path__)
    if isinstance(ppo_train_raw, types.ModuleType):
        # train is a module, get the train function from it
        if hasattr(ppo_train_raw, "train") and callable(ppo_train_raw.train):
            ppo_train = ppo_train_raw.train
        else:
            # Try importing the train function from the module
            train_module = importlib.import_module("brax.training.agents.ppo.train")
            ppo_train = getattr(train_module, "train", None)
            if ppo_train is None or not callable(ppo_train):
                raise ImportError(
                    f"Could not find callable 'train' in brax.training.agents.ppo.train module. Available: {dir(train_module)}"
                )
    elif not callable(ppo_train_raw):
        # train is some other object, try to get train from it
        if hasattr(ppo_train_raw, "train") and callable(ppo_train_raw.train):
            ppo_train = ppo_train_raw.train
        else:
            raise ImportError(
                f"'train' from brax.training.agents.ppo is not callable and has no callable 'train' attribute. Type: {type(ppo_train_raw)}, Attributes: {dir(ppo_train_raw)}"
            )
    else:
        # train is directly callable (a function)
        ppo_train = ppo_train_raw
except ImportError as e:
    # If import fails, try alternative paths
    try:
        from brax.training.agents.ppo.train import train as ppo_train
    except ImportError:
        try:
            # Try to manually import the module
            train_module = importlib.import_module("brax.training.agents.ppo.train")
            ppo_train = getattr(train_module, "train")
            if not callable(ppo_train):
                raise ImportError(
                    f"'train' in brax.training.agents.ppo.train is not callable. Type: {type(ppo_train)}"
                )
        except (ImportError, AttributeError) as e2:
            raise ImportError(
                f"Could not import PPO train function. "
                f"First error: {e}. Second error: {e2}. "
                f"Please check your Brax installation and version."
            )


class RewardWrapper:
    """Wrapper for Brax environments to inject custom reward functions."""

    def __init__(self, env, reward_fn):
        self._env = env
        self._reward_fn = reward_fn

    def step(self, state, action):
        """Step the environment and replace reward with custom reward function."""
        next_state = self._env.step(state, action)
        obs, next_obs = state.obs, next_state.obs

        # Convert metrics to a simple dict to avoid JAX tracing issues
        # Extract only scalar values from metrics to avoid .items() on traced objects
        metrics_dict = {}
        if hasattr(next_state, "metrics") and next_state.metrics is not None:
            # Handle metrics safely - convert to dict if it's a dict-like object
            try:
                # If metrics is already a dict, use it directly
                if isinstance(next_state.metrics, dict):
                    metrics_dict = next_state.metrics
                else:
                    # If it's a pytree or other structure, try to convert
                    # Only extract what we need without iterating
                    metrics_dict = {"metrics": next_state.metrics}
            except Exception:
                # Fallback: empty dict if metrics access fails
                metrics_dict = {}

        info = {"metrics": metrics_dict}
        rew, _ = self._reward_fn(obs, action, next_obs, info)

        # Only modify the reward, not the metrics dict structure
        # This avoids pytree structure mismatches in JAX scan
        # Term metrics can be tracked separately if needed for logging
        next_state = next_state.replace(reward=rew)
        return next_state

    def reset(self, rng):
        """Reset the environment."""
        return self._env.reset(rng)

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped environment."""
        return getattr(self._env, name)


def make_env(env_name: str, reward_fn):
    """Create an environment wrapped with the custom reward function."""
    env = create(env_name)
    return RewardWrapper(env, reward_fn)


class BaselineEnvWrapper:
    """Wrapper for baseline environment to handle metrics issues."""

    def __init__(self, env):
        self._env = env

    def reset(self, rng):
        """Reset the environment."""
        state = self._env.reset(rng=rng)
        # Force metrics to be empty dict to avoid any .items() calls
        # Don't check isinstance() as it might trigger the error
        try:
            state = state.replace(metrics={})
        except Exception:
            # If replace fails, state is fine as-is
            pass
        return state

    def step(self, state, action):
        """Step the environment, handling metrics issues."""
        # Always force metrics to empty dict before stepping to avoid .items() errors
        try:
            state = state.replace(metrics={})
        except Exception:
            pass

        # Try to step - catch any .items() errors
        try:
            next_state = self._env.step(state, action)
        except (AttributeError, TypeError) as e:
            error_str = str(e).lower()
            if "items" in error_str or "arrayimpl" in error_str:
                # Metrics issue - force empty dict and retry once
                try:
                    state = state.replace(metrics={})
                    next_state = self._env.step(state, action)
                except Exception as e2:
                    # If retry fails, wrap the error
                    raise RuntimeError(
                        "Environment step failed due to metrics issue. "
                        f"Original: {e}, Retry: {e2}"
                    ) from e2
            else:
                raise

        # After stepping, always force metrics to empty dict
        try:
            next_state = next_state.replace(metrics={})
        except Exception:
            pass

        return next_state

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped environment."""
        return getattr(self._env, name)


def make_default_env(env_name: str):
    """Create environment with default/original reward function (no custom reward)."""
    return create(env_name)


def save_baseline_rollout_html(
    env_name: str, out_dir: Path, seed: int, horizon: int = 400
):
    """
    Save baseline rollout visualization using raw Brax environment (no wrappers, no custom rewards).

    Uses JIT-compiled step function for fast execution.
    Uses ZERO actions (not random) to show untrained behavior - halfcheetah should just fall/not move.
    This shows what the environment looks like before any training (baseline/untrained policy).

    Why it was slow before: Without JIT, each env.step() runs in Python (slow).
    With JIT: Step function compiled once, then runs fast on GPU/TPU.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[save_baseline_rollout_html] Creating baseline rollout for {env_name} (horizon={horizon})"
    )

    try:
        # Create raw Brax environment - no wrappers, no custom rewards
        env = create(env_name)

        # JIT-compile the step function for speed
        # This is much faster than non-JIT because:
        # 1. Without JIT: Each env.step() runs in Python (slow, ~100ms per step)
        # 2. With JIT: Step function compiled once, then runs on GPU/TPU (fast, ~1ms per step)
        # Use ZERO actions (not random) to show untrained behavior - halfcheetah should just fall/not move
        @jax.jit
        def jit_step(state, rng):
            """JIT-compiled step function - runs fast after compilation."""
            # Use zero actions to show untrained behavior (no learned policy)
            # This shows what happens when the agent doesn't act (baseline/untrained)
            action = jnp.zeros(shape=(env.action_size,))
            next_state = env.step(state, action)
            return next_state, rng

        # Reset environment
        rng = jax.random.PRNGKey(seed)
        state = env.reset(rng=rng)
        states = [state]  # Include initial state

        # Compile step function once (takes 1-2 minutes, but only once)
        logger.info(
            "[save_baseline_rollout_html] Compiling JIT step "
            "function (this may take 1-2 minutes, but only once)..."
        )
        try:
            warmup_state = env.reset(rng=jax.random.PRNGKey(0))
            warmup_rng = jax.random.PRNGKey(0)
            _, _ = jit_step(warmup_state, warmup_rng)
            logger.info(
                "[save_baseline_rollout_html] JIT compilation complete! "
                "Running rollout (fast now)..."
            )
        except Exception as e:
            logger.warning(
                f"[save_baseline_rollout_html] Compilation warning: {e}. Using non-JIT fallback (will be slow)..."
            )

            # Fallback to non-JIT if compilation fails - still use zero actions
            def jit_step(state, rng):
                action = jnp.zeros(shape=(env.action_size,))
                return env.step(state, action), rng

        # Run rollout with JIT-compiled steps (fast after compilation!)
        for step_num in range(horizon):
            state, rng = jit_step(state, rng)
            states.append(state)

            if (step_num + 1) % 100 == 0:
                logger.debug(
                    f"[save_baseline_rollout_html] Progress: {step_num + 1}/{horizon}"
                )

        logger.info(
            f"[save_baseline_rollout_html] Rollout complete, collected {len(states)} states"
        )

        # Render HTML
        html_path = out_dir / "rollout_baseline.html"
        logger.info(f"[save_baseline_rollout_html] Rendering HTML to {html_path}...")
        html_path.write_text(html.render(env.sys, [s.pipeline_state for s in states]))
        logger.info(
            f"[save_baseline_rollout_html] Baseline rollout saved to {html_path}"
        )

    except Exception as e:
        logger.error(f"[save_baseline_rollout_html] Failed: {e}")
        raise


def recreate_inference_fn(env_name: str, reward_fn, network_config: dict):
    """
    Recreate the inference function using the saved network architecture config.

    This allows us to reconstruct the exact same inference function without retraining,
    avoiding dimension mismatches when loading saved params.

    Args:
        env_name: Environment name
        reward_fn: Reward function
        network_config: Dict with 'observation_size' and 'action_size' from training

    Returns:
        make_inference_fn: Function that takes params and returns a policy function
    """
    try:
        from brax.training.agents.ppo import networks
    except ImportError as e:
        raise ImportError("Could not import brax.training.agents.ppo.networks") from e

    env = make_env(env_name, reward_fn)

    # Verify observation/action sizes match
    if env.observation_size != network_config["observation_size"]:
        raise ValueError(
            f"Observation size mismatch: env has {env.observation_size}, "
            f"but network_config has {network_config['observation_size']}"
        )
    if env.action_size != network_config["action_size"]:
        raise ValueError(
            f"Action size mismatch: env has {env.action_size}, "
            f"but network_config has {network_config['action_size']}"
        )

    # Create networks with Brax PPO defaults
    # Default architecture: policy_hidden_layer_sizes=(32, 32), value_hidden_layer_sizes=(256, 256)
    # These are the defaults used by brax.training.agents.ppo.train
    try:
        # Try with preprocess_observations if available
        if hasattr(networks, "preprocess_observations"):
            ppo_networks = networks.make_ppo_networks(
                observation_size=network_config["observation_size"],
                action_size=network_config["action_size"],
                preprocess_observations_fn=networks.preprocess_observations,
            )
        else:
            # Fallback: create without preprocess_observations
            ppo_networks = networks.make_ppo_networks(
                observation_size=network_config["observation_size"],
                action_size=network_config["action_size"],
            )
    except TypeError:
        # If make_ppo_networks doesn't accept preprocess_observations_fn, try without it
        ppo_networks = networks.make_ppo_networks(
            observation_size=network_config["observation_size"],
            action_size=network_config["action_size"],
        )

    # Create inference function factory (same as what ppo_train returns)
    def make_inference_fn(params):
        """Create inference function from params - matches Brax PPO structure."""

        def policy(observations, rng):
            """Policy function that takes observations and rng, returns actions."""
            return ppo_networks.policy_network.apply(params, observations, rng)

        return policy

    return make_inference_fn


def train_policy(
    env_name: str, reward_fn, seed: int, steps: int, episode_length: int = 1000
):
    """Train a Brax PPO policy for the given environment and reward function.

    Returns the inference factory, trained parameters, training metrics, and the
    environment network configuration needed to reconstruct the policy later.
    """
    env = make_env(env_name, reward_fn)
    make_inference_fn, params, train_metrics = ppo_train(
        environment=env,
        num_timesteps=steps,
        episode_length=episode_length,
        seed=seed,
    )

    # Extract network architecture info from environment for later reconstruction
    # This allows us to recreate the inference function without retraining
    # The observation_size and action_size are critical for matching dimensions
    network_config = {
        "observation_size": env.observation_size,
        "action_size": env.action_size,
        # Brax PPO uses default network architecture - capture it
        # Default: policy_hidden_layer_sizes=(32, 32), value_hidden_layer_sizes=(256, 256)
        # We'll use Brax defaults, but save the observation/action sizes which are critical
    }

    return make_inference_fn, params, train_metrics, network_config


def evaluate_policy(
    env_name: str,
    reward_fn,
    make_inference_fn,
    params,
    seed: int,
    episodes: int,
    horizon: int,
    use_cpu: bool = None,
):
    """Evaluate a trained policy.

    Args:
        use_cpu: If True, use CPU for evaluation. If False, use GPU/default device.
                 If None (default), auto-detect: use GPU if available, otherwise CPU.
                 Note: GPU has more memory and is recommended for Colab T4 instances.
    """
    if use_cpu is None:
        # Auto-detect: prefer GPU if available (more memory for LLVM compilation)
        try:
            gpu_devices = jax.devices("gpu")
            if gpu_devices:
                logger.info(
                    f"[evaluate_policy] GPU detected ({len(gpu_devices)} device(s)), using GPU for evaluation"
                )
                use_cpu = False
            else:
                logger.info(
                    "[evaluate_policy] No GPU detected, using CPU for evaluation"
                )
                use_cpu = True
        except Exception:
            logger.info(
                "[evaluate_policy] Could not detect devices, using default (GPU if available)"
            )
            use_cpu = False

    if use_cpu:
        logger.info("[evaluate_policy] Using CPU for evaluation")
        return _evaluate_policy_impl_cpu(
            env_name, reward_fn, make_inference_fn, params, seed, episodes, horizon
        )
    else:
        logger.info("[evaluate_policy] Using default device (GPU if available)")
        return _evaluate_policy_impl(
            env_name, reward_fn, make_inference_fn, params, seed, episodes, horizon
        )


def _evaluate_policy_impl_cpu(
    env_name: str,
    reward_fn,
    make_inference_fn,
    params,
    seed: int,
    episodes: int,
    horizon: int,
):
    """Internal implementation using CPU - faster compilation, slower execution."""
    logger.info(
        "[evaluate_policy] CPU mode: Using CPU (fast compilation, slower execution)..."
    )

    # Get CPU device
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        logger.warning(
            "[evaluate_policy] No CPU device found, falling back to default device"
        )
        return _evaluate_policy_impl(
            env_name, reward_fn, make_inference_fn, params, seed, episodes, horizon
        )

    cpu_device = cpu_devices[0]

    # Temporarily force CPU by using with_device_context
    # Note: This approach forces computation on CPU
    logger.info("[evaluate_policy] CPU mode: Compiling on CPU (this should be fast)...")
    env = make_env(env_name, reward_fn)

    # Move params to CPU
    params_cpu = jax.device_put(params, cpu_device)
    policy = make_inference_fn(params_cpu)

    # Warm up on CPU
    logger.info("[evaluate_policy] CPU mode: Warming up JIT compilation on CPU...")
    try:
        dummy_rng = jax.random.PRNGKey(0)
        dummy_state = env.reset(rng=dummy_rng)
        dummy_state = jax.device_put(dummy_state, cpu_device)
        dummy_action_rng = jax.random.split(dummy_rng)[0]
        dummy_action, _ = policy(dummy_state.obs, dummy_action_rng)
        dummy_action = jax.device_put(dummy_action, cpu_device)
        _ = env.step(dummy_state, dummy_action)
        logger.info(
            "[evaluate_policy] CPU mode: JIT warm-up complete (should be fast!)"
        )
    except Exception as e:
        logger.warning(
            f"[evaluate_policy] CPU mode: JIT warm-up failed (will compile during loop): {e}"
        )

    logger.info(
        "[evaluate_policy] CPU mode: Compilation complete. Running evaluation episodes..."
    )

    returns = []
    term_means = {}

    for ep in range(episodes):
        logger.info(f"[evaluate_policy] CPU mode: Episode {ep + 1}/{episodes}")
        rng = jax.random.PRNGKey(seed + ep)
        state = env.reset(rng=rng)
        state = jax.device_put(state, cpu_device)
        total = 0.0

        log_interval = max(1, horizon // 5)

        for step in range(horizon):
            rng, action_rng = jax.random.split(rng)
            action, _ = policy(state.obs, action_rng)
            action = jax.device_put(action, cpu_device)
            state = env.step(state, action)
            total += float(jnp.mean(state.reward))

            if (step + 1) % log_interval == 0 or step == horizon - 1:
                progress_pct = ((step + 1) / horizon) * 100
                logger.info(
                    f"[evaluate_policy] CPU mode: Episode {ep + 1}, Step {step + 1}/{horizon} ({progress_pct:.1f}%), Return so far: {total:.2f}"
                )

        returns.append(total)
        logger.info(
            f"[evaluate_policy] CPU mode: Episode {ep + 1} complete. Return: {total:.2f}"
        )

        for k, v in state.metrics.items():
            if k.startswith("term/"):
                term_means[k] = float(v)

    avg_return = float(np.mean(returns))
    logger.info(
        f"[evaluate_policy] CPU mode: Evaluation complete. Average return: {avg_return:.2f}"
    )

    return {"avg_return": avg_return, "returns": returns, "term_means": term_means}


def rollout_policy_states(
    env_name: str,
    reward_fn,
    make_inference_fn,
    params,
    seed: int,
    horizon: int,
    use_default_env: bool = False,
):
    """Rollout policy and return env + list of pipeline states for rendering.

    Uses JIT-compiled step function for speed. Uses the same device as params (typically GPU).

    Args:
        use_default_env: If True, use the original environment with default reward (no custom reward).
                        If False, use environment with custom reward function.
    """
    if use_default_env:
        env = make_default_env(env_name)
    else:
        env = make_env(env_name, reward_fn)
    policy = make_inference_fn(params)

    # For baseline rollouts (use_default_env=True), skip JIT entirely to avoid tracing issues
    # For regular rollouts, use JIT for the entire step function (like baseline does)
    use_jit = not use_default_env

    if use_jit:
        # JIT-compile the entire step function (action selection + env.step) in one go
        # This compiles once and then runs fast, avoiding repeated compilation during rollout
        @jax.jit
        def jit_step(state, rng):
            """JIT-compiled step function - compiles once, runs fast."""
            rng, action_rng = jax.random.split(rng)
            action, _ = policy(state.obs, action_rng)
            next_state = env.step(state, action)
            return next_state, rng

        logger.info(
            "[rollout_policy_states] Using JIT-compiled step function (will compile once, then fast)..."
        )

        # Warm-up compilation
        logger.info(
            "[rollout_policy_states] Compiling JIT step function (this may take 1-2 minutes, but only once)..."
        )
        try:
            warmup_rng = jax.random.PRNGKey(0)
            warmup_state = env.reset(rng=warmup_rng)
            _, _ = jit_step(warmup_state, warmup_rng)
            logger.info(
                "[rollout_policy_states] JIT compilation complete! Running rollout (fast now)..."
            )
        except Exception as e:
            logger.warning(
                f"[rollout_policy_states] JIT compilation warning: {e}. Using non-JIT fallback..."
            )
            use_jit = False

    if not use_jit:
        # No JIT for baseline - simple and avoids all tracing issues
        def jit_step(state, rng):
            """Non-JIT step function (for baseline rollouts or fallback)."""
            rng, action_rng = jax.random.split(rng)
            action, _ = policy(state.obs, action_rng)
            next_state = env.step(state, action)
            return next_state, rng

        logger.info(
            "[rollout_policy_states] Using non-JIT step function (baseline rollout or fallback)..."
        )

    # Run the rollout
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng=rng)
    states = [state]  # include initial frame

    for step_num in range(horizon):
        state, rng = jit_step(state, rng)
        states.append(state)

        # Log progress for long rollouts
        if (step_num + 1) % 100 == 0:
            logger.debug(
                f"[rollout_policy_states] Rollout progress: {step_num + 1}/{horizon}"
            )

    return env, states


def save_policy_rollout_html(
    out_dir: Path,
    env_name: str,
    reward_fn,
    make_inference_fn,
    params,
    seed: int,
    horizon: int = 400,
    filename: str = "rollout.html",
    use_default_env: bool = False,
):
    """
    Save rollout visualization as an interactive HTML viewer.

    The HTML viewer provides an interactive 3D animation that can be played, paused,
    scrubbed, and rotated. Uses the same device as params (typically GPU).

    Args:
        filename: Output filename (default: "rollout.html", can be "rollout_baseline.html" for default env)
        use_default_env: If True, use original environment with default reward (no custom reward).
                        If False, use environment with custom reward function.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[save_policy_rollout_html] Starting rollout (horizon={horizon}, filename={filename}, use_default_env={use_default_env})"
    )

    try:
        env, states = rollout_policy_states(
            env_name,
            reward_fn,
            make_inference_fn,
            params,
            seed,
            horizon,
            use_default_env=use_default_env,
        )
        logger.info(
            f"[save_policy_rollout_html] Rollout complete, collected {len(states)} states"
        )
    except Exception as e:
        logger.error(f"[save_policy_rollout_html] Rollout failed: {e}")
        raise

    # HTML viewer - interactive 3D animation (play, pause, scrub, rotate camera)
    try:
        html_path = out_dir / filename
        logger.info(
            f"[save_policy_rollout_html] Rendering interactive HTML viewer to {filename}..."
        )
        html_path.write_text(html.render(env.sys, [s.pipeline_state for s in states]))
        logger.info(f"[save_policy_rollout_html] HTML viewer saved to {html_path}")
        logger.info(
            "[save_policy_rollout_html] Open "
            f"{html_path} in a web browser to view the interactive 3D animation"
        )
    except Exception as e:
        logger.error(f"[save_policy_rollout_html] HTML rendering failed: {e}")
        raise


def _evaluate_policy_impl(
    env_name: str,
    reward_fn,
    make_inference_fn,
    params,
    seed: int,
    episodes: int,
    horizon: int,
):
    """Internal implementation using default device (GPU if available)."""
    logger.info("[evaluate_policy] GPU mode: Setting up environment and policy...")
    env = make_env(env_name, reward_fn)
    policy = make_inference_fn(params)

    # Create a JIT-compiled rollout function to avoid repeated compilations
    @jax.jit
    def rollout_episode(episode_rng):
        """JIT-compiled function to run a single episode."""
        state = env.reset(rng=episode_rng)
        total_reward = 0.0

        def step_fn(carry, _):
            state, rng, total = carry
            rng, action_rng = jax.random.split(rng)
            action, _ = policy(state.obs, action_rng)
            next_state = env.step(state, action)
            reward = jnp.mean(next_state.reward)
            total = total + reward
            return (next_state, rng, total), reward

        # Use scan to compile the entire episode at once
        (final_state, _, total_reward), _ = jax.lax.scan(
            step_fn, (state, episode_rng, 0.0), None, length=horizon
        )

        return total_reward, final_state

    # Compile the rollout function once
    logger.info(
        "[evaluate_policy] GPU mode: Compiling episode rollout (this may take 2-5 minutes, but only once)..."
    )
    try:
        # Warm-up compilation with a dummy key
        warmup_rng = jax.random.PRNGKey(0)
        _ = rollout_episode(warmup_rng)
        logger.info(
            "[evaluate_policy] GPU mode: Compilation complete! Running evaluation episodes..."
        )
    except Exception as e:
        logger.warning(
            f"[evaluate_policy] GPU mode: Compilation warning: {e}. Continuing anyway..."
        )

    returns = []
    term_means = {}

    for ep in range(episodes):
        logger.info(f"[evaluate_policy] Episode {ep + 1}/{episodes}")
        episode_rng = jax.random.PRNGKey(seed + ep)

        # Run the compiled rollout
        total, final_state = rollout_episode(episode_rng)
        total = float(total)

        returns.append(total)
        logger.info(f"[evaluate_policy] Episode {ep + 1} complete. Return: {total:.2f}")

        # Extract term metrics from final state
        for k, v in final_state.metrics.items():
            if k.startswith("term/"):
                term_means[k] = float(v)

    avg_return = float(np.mean(returns))
    logger.info(
        f"[evaluate_policy] Evaluation complete. Average return: {avg_return:.2f}"
    )

    return {"avg_return": avg_return, "returns": returns, "term_means": term_means}
