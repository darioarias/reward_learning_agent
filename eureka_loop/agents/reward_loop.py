import yaml
from google.adk.agents import LoopAgent
from .llm_agents import RewardDesignerOpenAIAgent, RewardReflectorOpenAIAgent
from .evaluator_agent import CandidateEvaluatorAgent
from .selector_agent import SelectorAgent
from .human_reflection_agent import HumanReflectionAgent
from .increment_iteration_agent import IncrementIterationAgent
from .exit_checker_agent import ExitCheckerAgent


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_reward_loop():
    loop_cfg = _load_yaml("configs/loop_config.yaml")
    model = loop_cfg.get("openai_model", "gpt-4o-mini")

    return LoopAgent(
        name="RewardEvolutionLoop",
        sub_agents=[
            RewardDesignerOpenAIAgent(model=model),
            CandidateEvaluatorAgent(),
            SelectorAgent(name="SelectorAgent"),
            RewardReflectorOpenAIAgent(model=model),
            HumanReflectionAgent(
                name="HumanReflectionAgent"
            ),  # Human-in-the-loop after LLM reflection
            ExitCheckerAgent(name="ExitCheckerAgent"),
            IncrementIterationAgent(name="IncrementIterationAgent"),
        ],
        max_iterations=loop_cfg["iterations"],
    )
