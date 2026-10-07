import os
import json
from typing import AsyncGenerator

from openai import OpenAI
from pydantic import PrivateAttr
from loguru import logger

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CANDIDATE_DELIM = "### CANDIDATE ###"
DESIGNER_SYSTEM = (
    "You are a precise reward-function code generator for JAX/Brax environments."
)


def _designer_prompt(
    task_spec,
    env_code,
    best_reward_code,
    reflection,
    K,
    candidate_results: str | None = None,
):
    """
    Build the prompt for the Reward Designer.

    From iteration 2 onward, we include an explicit "Query with Feedback" section
    (policy training/eval results + reflection) similar to the Eureka paper diagram.
    """
    if best_reward_code:
        improvement_instruction = f"""IMPORTANT: You MUST generate {K} IMPROVED versions of the BEST REWARD SO FAR below.
- Each candidate should be a PROGRESSIVE IMPROVEMENT or VARIATION of the best reward
- Build upon the successful aspects identified in the feedback/reflection
- Try different approaches to address issues mentioned in the feedback/reflection
- Do NOT generate completely new rewards from scratch - they must be based on the best reward below"""

        feedback_block = f"""
QUERY WITH FEEDBACK (from the last iteration):
We trained RL policies using reward functions. Here are the results for ALL candidates from the last iteration.

POLICY RESULTS (JSON) - Contains results for ALL candidates:
{candidate_results if candidate_results else "(No candidate_results provided)"}

REFLECTION - Analysis of what worked, what failed, and HOW TO IMPROVE:
{reflection if reflection else "(No reflection available)"}

NOTE: This reflection may include both LLM-generated analysis and human feedback (if provided).
Pay special attention to any "HUMAN FEEDBACK" section, as it contains direct human insights.

CRITICAL INSTRUCTIONS:
1. The "BEST REWARD SO FAR" below is the ONLY reward you should use as a base for improvement.
2. The candidate_results JSON contains results for BOTH candidates (best and worse).
3. The reflection contains THREE critical sections you MUST use:
   a) Analysis of what worked (preserve and enhance these aspects)
   b) Analysis of what failed (avoid these approaches)
   c) **IMPROVEMENT SUGGESTIONS** - The reflection includes "recommended_edits" and "prompt_seed" fields with SPECIFIC suggestions for how to improve the best reward
4. Generate {K} improved versions based ONLY on the "BEST REWARD SO FAR" below.
5. Do NOT use the worse candidate's reward code as a base - only learn from its failures.
6. The worse candidate's results are provided for learning (what NOT to do), not for copying.

INSTRUCTION:
Carefully analyze the policy results and reflection above. The reflection contains:
- What worked in the best candidate (preserve and enhance)
- What failed in worse candidates (avoid)
- **SPECIFIC IMPROVEMENT SUGGESTIONS** in the "recommended_edits" and "prompt_seed" fields - USE THESE!

Generate {K} improved reward function candidates that:
- Are based ONLY on the "BEST REWARD SO FAR" below
- Preserve what worked in the best candidate (from reflection's "what_worked" and "strengths")
- Fix issues identified in the reflection (from reflection's "what_failed" and "weaknesses")
- **FOLLOW the specific improvement suggestions** from reflection's "recommended_edits" field
- Use the insights from reflection's "prompt_seed" field to guide your improvements
- Avoid the mistakes that caused the worse candidate to fail
- CRITICAL: If reflection mentions stability issues (falling, losing balance, becoming unstable), ensure your rewards include stability/balance terms:
  * Reward maintaining upright posture (body orientation angles)
  * Reward maintaining body height (prevent falling)
  * Penalize large orientation deviations
  * Balance speed and stability - the task requires BOTH"""
    else:
        improvement_instruction = f"""Generate {K} initial reward candidates for this task.
- These are the first candidates, so explore different reward shaping approaches
- IMPORTANT: The task requires running fast WHILE staying stable - ensure your rewards balance both:
  * Forward velocity reward (for speed)
  * Stability/balance rewards (for maintaining upright posture, preventing falls)
  * Control smoothness (for energy efficiency)
- Do NOT over-prioritize speed at the expense of stability"""
        feedback_block = ""

    return f"""Return exactly {K} reward candidates.

{improvement_instruction}

CRITICAL FORMAT RULES:
- Output MUST be plain Python ONLY (no markdown, no backticks, no commentary).
- Each candidate MUST start with: def compute_reward(
- Separate candidates using EXACT delimiter line:
{CANDIDATE_DELIM}
- No imports.
- Only use: jnp, jax, math.
- compute_reward must return: (total_reward, reward_terms_dict)
- reward_terms_dict must be a dict[str, jnp.ndarray] for each term.
- CRITICAL: Every compute_reward function MUST end with: return total_reward, reward_terms
- NEVER return None or forget the return statement - this will cause validation to fail.

TASK SPEC:
{json.dumps(task_spec, indent=2)}

ENV CODE:
{env_code}

BEST REWARD SO FAR (may be empty - if empty, this is the first iteration):
{best_reward_code if best_reward_code else "(No previous best reward - generate initial candidates)"}
{feedback_block}

EXAMPLE OUTPUT SHAPE (for 2 candidates):
def compute_reward(obs, action, next_obs, info):
    obs = jnp.asarray(obs)
    next_obs = jnp.asarray(next_obs)
    action = jnp.asarray(action)
    
    # Compute reward components
    v_x = next_obs[..., 8]  # forward velocity
    total = v_x  # total reward
    terms = {{"velocity": v_x}}  # reward terms dict
    
    # CRITICAL: Always return a tuple
    return total, terms
{CANDIDATE_DELIM}
def compute_reward(obs, action, next_obs, info):
    obs = jnp.asarray(obs)
    next_obs = jnp.asarray(next_obs)
    action = jnp.asarray(action)
    
    # Compute reward components
    v_x = next_obs[..., 8]
    ctrl_cost = -0.1 * jnp.sum(action ** 2)
    total = v_x + ctrl_cost
    terms = {{"velocity": v_x, "ctrl": ctrl_cost}}
    
    # CRITICAL: Always return a tuple
    return total, terms
"""


class RewardDesignerOpenAIAgent(BaseAgent):
    _model: str = PrivateAttr()

    def __init__(self, model="gpt-4o-mini", **kwargs):
        super().__init__(name="RewardDesignerOpenAIAgent", **kwargs)
        self._model = model

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        iteration = ctx.session.state.get("iteration", 1)
        best_reward_code = ctx.session.state.get("best_reward_code", "")
        reflection = ctx.session.state.get("reflection", "")
        K = ctx.session.state["K"]

        logger.info(
            f"[RewardDesignerOpenAIAgent] Starting reward design for iteration {iteration}"
        )
        logger.info(f"[RewardDesignerOpenAIAgent] Inputs:")
        logger.info(f"  - K (candidates to generate): {K}")

        # Log task specification
        task_spec = ctx.session.state["task_spec"]
        logger.info(f"  - task_spec:")
        logger.info(f"      task_name: {task_spec.get('task_name', 'N/A')}")
        logger.info(f"      objective: {task_spec.get('objective', 'N/A')}")
        logger.info(f"      success_metric: {task_spec.get('success_metric', 'N/A')}")
        if "constraints" in task_spec:
            logger.info(
                f"      constraints: {len(task_spec['constraints'])} constraint(s)"
            )
            for i, constraint in enumerate(task_spec["constraints"], 1):
                logger.info(f"        {i}. {constraint}")
        if "evaluation" in task_spec:
            logger.info(f"      evaluation: {task_spec['evaluation']}")

        # Log environment code
        env_code = ctx.session.state["env_code"]
        logger.info(f"  - env_code length: {len(env_code)} chars")
        env_code_preview_lines = env_code.split("\n")[:10]
        logger.info(f"  - env_code preview (first 10 lines):")
        for i, line in enumerate(env_code_preview_lines, 1):
            logger.info(f"      {i}: {line[:100]}...")

        logger.info(f"  - best_reward_code length: {len(best_reward_code)} chars")
        logger.info(f"  - reflection length: {len(reflection)} chars")

        if best_reward_code:
            # Log first few lines of best reward code
            first_lines = best_reward_code.split("\n")[:5]
            logger.info(f"  - best_reward_code preview (first 5 lines):")
            for i, line in enumerate(first_lines, 1):
                logger.info(f"      {i}: {line[:80]}...")
        else:
            logger.info(
                "  - best_reward_code: EMPTY (this is the first iteration or no previous best candidate)"
            )

        if reflection:
            logger.info(f"  - reflection length: {len(reflection)} chars")
            logger.info("  - Complete reflection:")
            logger.info("    " + "=" * 76)
            # Log the complete reflection with indentation for readability
            for line in reflection.split("\n"):
                logger.info(f"    {line}")
            logger.info("    " + "=" * 76)
        else:
            logger.info("  - reflection: EMPTY (no previous reflection available)")

        candidate_results = None
        # Only include "query with feedback" after we have results (i.e., from iteration 2 onward).
        if iteration > 1:
            candidate_results = ctx.session.state.get("candidate_results", None)

        prompt = _designer_prompt(
            ctx.session.state["task_spec"],
            ctx.session.state["env_code"],
            best_reward_code,
            reflection,
            K,
            candidate_results=candidate_results,
        )

        from tenacity import (
            retry,
            stop_after_attempt,
            wait_exponential,
            retry_if_exception_type,
        )

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((Exception,)),  # Retry on any exception
            reraise=True,
        )
        def call_llm():
            return client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": DESIGNER_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )

        try:
            resp = call_llm()
            generated_text = resp.choices[0].message.content
            ctx.session.state["reward_candidates_text"] = generated_text

            # Count how many candidates were generated
            candidate_count = generated_text.count(CANDIDATE_DELIM) + 1
            logger.info(
                f"[RewardDesignerOpenAIAgent] Generated {candidate_count} reward candidate(s) (expected {K})"
            )
            logger.debug(
                f"[RewardDesignerOpenAIAgent] Generated text length: {len(generated_text)} chars"
            )
        except Exception as e:
            logger.error(
                f"[RewardDesignerOpenAIAgent] Failed to get LLM response after retries: {e}"
            )
            raise ValueError(f"LLM API call failed: {e}")
        yield Event(author=self.name, content=None)


class RewardReflectorOpenAIAgent(BaseAgent):
    _model: str = PrivateAttr()

    def __init__(self, model="gpt-4o-mini", **kwargs):
        super().__init__(name="RewardReflectorOpenAIAgent", **kwargs)
        self._model = model

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        iteration = ctx.session.state.get("iteration", 1)
        task_spec = ctx.session.state["task_spec"]
        candidate_results = ctx.session.state.get("candidate_results", "")
        best_candidate_index = ctx.session.state.get("best_candidate_index", 0)

        # Validate required state
        if not candidate_results:
            logger.warning(
                "[RewardReflectorOpenAIAgent] No candidate_results found. This may be the first iteration."
            )
            # For first iteration, we might not have results yet - skip reflection
            ctx.session.state["reflection"] = (
                "No candidate results available for reflection (first iteration)."
            )
            yield Event(author=self.name, content=None)
            return

        logger.info(
            f"[RewardReflectorOpenAIAgent] Starting reflection for iteration {iteration}"
        )
        logger.info(
            f"[RewardReflectorOpenAIAgent] Best candidate index: {best_candidate_index}"
        )
        logger.debug(
            f"[RewardReflectorOpenAIAgent] Candidate results length: {len(candidate_results)} chars"
        )

        prompt = f"""Analyze the candidate reward functions and provide detailed feedback for improvement.

TASK SPECIFICATION:
{json.dumps(task_spec, indent=2)}

CANDIDATE RESULTS (JSON):
{candidate_results}

BEST CANDIDATE INDEX: {best_candidate_index}

IMPORTANT: The candidate_results contains results for ALL candidates from this iteration.
- Candidate at index {best_candidate_index} is the BEST performing one (selected for next iteration)
- The other candidate(s) performed WORSE and will NOT be used as a base for next iteration
- Your analysis should help the designer understand what to preserve (from best) and what to avoid (from worse)

ANALYSIS INSTRUCTIONS:
For EACH candidate (both best and worse), analyze the following behavioral metrics from train_metrics:
1. Forward Velocity (eval/episode_x_velocity): 
   - Positive = moving forward (good), Negative = moving backward (bad)
   - Higher absolute value = faster movement
   - Compare velocities across candidates
   - WARNING: Very high velocity may indicate instability or falling - check stability metrics

2. Stability and Balance (CRITICAL):
   - Check if the policy maintains upright posture throughout the episode
   - Look for signs of falling: sudden drops in height, loss of balance, episode termination
   - Analyze body orientation/angles - should remain stable and balanced
   - If velocity increases but stability decreases, this is a RED FLAG - reward may be over-prioritizing speed
   - The task requires "staying stable" - speed without stability is a failure

3. Distance Traveled (eval/episode_x_position):
   - Final x_position indicates how far the cheetah traveled
   - Positive = forward progress, Negative = backward movement
   - Higher absolute value = better performance
   - If distance is high but episode ends early (falling), stability is compromised

4. Control Smoothness (eval/episode_reward_ctrl):
   - More negative = higher control penalty (jerky/inefficient actions)
   - Less negative = smoother, more efficient control
   - Compare control penalties across candidates
   - Jerky actions may indicate instability or over-aggressive speed pursuit

5. Running Reward Component (eval/episode_reward_run):
   - Component of reward related to forward motion
   - Positive = reward for running, Negative = penalty
   - Analyze if this component is working as intended
   - If this is too high relative to stability terms, it may cause instability

6. Episode Length / Termination:
   - Check if episodes complete the full horizon or terminate early
   - Early termination often indicates falling or instability
   - Compare episode lengths across candidates

7. Overall Performance (score/avg_return):
   - Compare final scores across all candidates
   - Identify what makes the best candidate successful
   - Consider: Is high score due to speed OR stability? Both are required.

8. Reward Code Analysis:
   - Examine the actual reward function code
   - Identify which reward terms are contributing positively/negatively
   - Check for numerical stability issues
   - Verify reward shaping aligns with task objectives
   - CRITICAL: Check if reward has stability/balance terms (upright posture, height maintenance, orientation)
   - If reward only emphasizes velocity without stability terms, this will cause falling

Based on this analysis, provide a structured analysis:

For the BEST candidate (index {best_candidate_index}):
- what_worked: Specific aspects that led to good performance (e.g., "forward velocity reward term was effective", "control penalty weight was well-balanced")
- strengths: Key strengths to preserve and build upon

For the WORSE candidate(s):
- what_failed: Specific issues that caused poor performance (e.g., "control penalty too high causing freezing", "negative velocity indicates backward movement", "reward shaping was insufficient", "policy falls at end due to instability", "over-prioritized speed without stability")
- weaknesses: Key weaknesses to avoid in future iterations

Overall recommendations:
- diagnosis: Overall assessment comparing best vs worse candidates
  - Pay special attention to speed vs stability tradeoff
  - If best candidate has high velocity but falls/becomes unstable, this is a critical issue
  - The task requires BOTH speed AND stability - one without the other is insufficient
- recommended_edits: Concrete suggestions for improving the BEST reward function:
  - If stability is an issue: "add stability bonus for maintaining upright posture", "penalize large body orientation angles", "reward maintaining body height", "add balance term based on center of mass"
  - If speed-stability tradeoff: "reduce forward velocity reward weight and increase stability reward weight", "add penalty for falling or losing balance", "ensure stability terms are weighted appropriately relative to velocity"
  - Examples: "reduce control penalty weight from 0.1 to 0.05", "add stability bonus: reward = 0.5 * velocity + 0.3 * stability - 0.1 * control_cost", "increase forward velocity reward scaling"
- prompt_seed: Key insights to guide next reward generation, emphasizing:
  * What to preserve from the best candidate
  * What to avoid based on the worse candidate's failures
  * How to improve the best candidate further
  * CRITICAL: If stability issues are detected, emphasize the need for stability/balance reward terms
  * Balance speed and stability - the task requires "staying stable" while running fast

CRITICAL: Make it clear that the designer should ONLY use the best candidate's reward code as a base, and use the worse candidate's results only as learning (what NOT to do).

Return ONLY valid JSON (no markdown, no code blocks) with these fields.
"""

        from tenacity import (
            retry,
            stop_after_attempt,
            wait_exponential,
            retry_if_exception_type,
        )

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((Exception,)),
            reraise=True,
        )
        def call_llm():
            return client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

        try:
            resp = call_llm()
            reflection_text = resp.choices[0].message.content
            ctx.session.state["reflection"] = reflection_text
            logger.info(
                f"[RewardReflectorOpenAIAgent] Generated reflection (length: {len(reflection_text)} chars)"
            )
            logger.info("=" * 80)
            logger.info("[RewardReflectorOpenAIAgent] COMPLETE REFLECTION:")
            logger.info("=" * 80)
            # Log the complete reflection, splitting by lines for better readability
            for line in reflection_text.split("\n"):
                logger.info(line)
            logger.info("=" * 80)
        except Exception as e:
            logger.error(
                f"[RewardReflectorOpenAIAgent] Failed to get LLM response after retries: {e}"
            )
            # Don't fail the whole loop if reflection fails - just use empty reflection
            ctx.session.state["reflection"] = f"Reflection generation failed: {e}"
        yield Event(author=self.name, content=None)
