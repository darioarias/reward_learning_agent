"""ADK entrypoint for `adk run eureka_loop`"""

from google.adk.agents import SequentialAgent
from .agents.setup_agent import SetupAgent
from .agents.reward_loop import build_reward_loop

root_agent = SequentialAgent(
    name="EurekaRewardEvolutionPipeline",
    sub_agents=[
        SetupAgent(name="SetupAgent"),
        build_reward_loop(),
    ],
    description="Initializes env/task state and runs an Eureka-like reward evolution loop.",
)
