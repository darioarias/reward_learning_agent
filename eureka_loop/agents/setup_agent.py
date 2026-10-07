import yaml
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing import AsyncGenerator
from loguru import logger

from ..tools.brax_env import get_env_and_source
from ..tools.artifacts import init_run_dir, init_leaderboard, save_json


def _load_yaml(path):
    """Load YAML file with error handling."""
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            if data is None:
                raise ValueError(f"YAML file {path} is empty or contains no data")
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file {path}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading YAML file {path}: {e}")


class SetupAgent(BaseAgent):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if ctx.session.state.get("_setup_done", False):
            yield Event(author=self.name, content=None)
            return

        loop_cfg = _load_yaml("configs/loop_config.yaml")
        task_spec = _load_yaml("configs/task_spec_halfcheetah.yaml")

        # Validate configuration values
        required_loop_keys = [
            "env_name",
            "iterations",
            "candidates_per_iteration",
            "train_steps",
            "seed",
        ]
        for key in required_loop_keys:
            if key not in loop_cfg:
                raise ValueError(f"Missing required config key: {key}")

        if not isinstance(loop_cfg["iterations"], int) or loop_cfg["iterations"] < 1:
            raise ValueError(
                f"iterations must be a positive integer, got {loop_cfg['iterations']}"
            )
        if (
            not isinstance(loop_cfg["candidates_per_iteration"], int)
            or loop_cfg["candidates_per_iteration"] < 1
        ):
            raise ValueError(
                f"candidates_per_iteration must be a positive integer, got {loop_cfg['candidates_per_iteration']}"
            )
        if not isinstance(loop_cfg["train_steps"], int) or loop_cfg["train_steps"] < 1:
            raise ValueError(
                f"train_steps must be a positive integer, got {loop_cfg['train_steps']}"
            )

        if "evaluation" not in task_spec:
            raise ValueError("task_spec must contain 'evaluation' key")
        if (
            "episodes" not in task_spec["evaluation"]
            or "horizon" not in task_spec["evaluation"]
        ):
            raise ValueError(
                "task_spec.evaluation must contain 'episodes' and 'horizon' keys"
            )
        if (
            not isinstance(task_spec["evaluation"]["episodes"], int)
            or task_spec["evaluation"]["episodes"] < 1
        ):
            raise ValueError(
                f"evaluation.episodes must be a positive integer, got {task_spec['evaluation']['episodes']}"
            )
        if (
            not isinstance(task_spec["evaluation"]["horizon"], int)
            or task_spec["evaluation"]["horizon"] < 1
        ):
            raise ValueError(
                f"evaluation.horizon must be a positive integer, got {task_spec['evaluation']['horizon']}"
            )

        outdir = init_run_dir()
        leaderboard_path = init_leaderboard(outdir)

        _, env_source, mod_name = get_env_and_source(loop_cfg["env_name"])

        (outdir / "env_code.py").write_text(env_source)
        save_json(outdir / "task_spec.json", task_spec)
        save_json(outdir / "loop_config.json", loop_cfg)
        save_json(
            outdir / "env_meta.json",
            {"module": mod_name, "env_name": loop_cfg["env_name"]},
        )

        ctx.session.state.update(
            {
                "_setup_done": True,
                "outdir": str(outdir),
                "leaderboard_path": str(leaderboard_path),
                "env_name": loop_cfg["env_name"],
                "env_code": env_source,
                "task_spec": task_spec,
                "train_steps": loop_cfg["train_steps"],
                "seed": loop_cfg["seed"],
                "K": loop_cfg["candidates_per_iteration"],
                "eval_episodes": task_spec["evaluation"]["episodes"],
                "eval_horizon": task_spec["evaluation"]["horizon"],
                "iteration": 1,
                "best_reward_code": "",
                "reflection": "",
                "human_reflection": "",  # Store human feedback separately
                "return_threshold": loop_cfg.get("return_threshold", 2500),
                "plateau_delta": loop_cfg.get("plateau_delta", 100),
                "plateau_patience": loop_cfg.get("plateau_patience", 2),
                "plateau_count": 0,
                "prev_best_return": None,
            }
        )

        # Generate baseline (default environment) rollout once at the start
        # This shows the original environment behavior before any custom reward
        try:
            logger.info(
                "[SetupAgent] Generating baseline (default environment) rollout visualization..."
            )
            from ..tools.rl_runner import save_baseline_rollout_html

            env_name = loop_cfg["env_name"]
            seed = loop_cfg["seed"]
            video_horizon = min(400, task_spec["evaluation"]["horizon"])

            # Use simple baseline function - no wrappers, no custom rewards, just raw environment
            save_baseline_rollout_html(
                env_name=env_name,
                out_dir=outdir,
                seed=seed,
                horizon=video_horizon,
            )
            logger.info(
                "[SetupAgent] Saved baseline rollout (rollout_baseline.html) - shows original environment behavior"
            )
        except Exception as baseline_err:
            logger.warning(
                f"[SetupAgent] Baseline rollout failed (non-critical): {baseline_err}. Continuing..."
            )

        yield Event(author=self.name, content=None)
