from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing import AsyncGenerator
from loguru import logger


class HumanReflectionAgent(BaseAgent):
    """
    Agent that prompts for human feedback/reflection after LLM reflection.

    This allows human-in-the-loop refinement of the reflection before it's used
    to generate the next iteration's reward candidates.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        iteration = ctx.session.state.get("iteration", 1)
        llm_reflection = ctx.session.state.get("reflection", "")
        best_reward_code = ctx.session.state.get("best_reward_code", "")
        best_return = ctx.session.state.get("best_return", None)

        logger.info(
            f"[HumanReflectionAgent] Starting human reflection for iteration {iteration}"
        )

        if not llm_reflection:
            logger.warning(
                "[HumanReflectionAgent] No LLM reflection found. Skipping human reflection."
            )
            yield Event(author=self.name, content=None)
            return

        # Display summary for human review
        logger.info("=" * 80)
        logger.info("[HumanReflectionAgent] LLM-GENERATED REFLECTION SUMMARY:")
        logger.info("=" * 80)
        logger.info(f"Iteration: {iteration}")
        logger.info(f"Best Return: {best_return}")
        logger.info(f"Best Reward Code Length: {len(best_reward_code)} chars")
        logger.info("")
        logger.info("COMPLETE LLM REFLECTION:")
        logger.info("-" * 80)
        # Log the complete reflection
        for line in llm_reflection.split("\n"):
            logger.info(line)
        logger.info("-" * 80)
        logger.info("")
        logger.info("=" * 80)
        logger.info("[HumanReflectionAgent] HUMAN FEEDBACK PROMPT:")
        logger.info("=" * 80)
        logger.info(
            "You can provide additional feedback/reflection to improve the next iteration."
        )
        logger.info("This will be combined with the LLM reflection above.")
        logger.info("")
        logger.info("Options:")
        logger.info("  1. Press ENTER to skip (use only LLM reflection)")
        logger.info(
            "  2. Type your feedback and press ENTER (will be combined with LLM reflection)"
        )
        logger.info("  3. Type 'skip' to skip human reflection")
        logger.info("")

        # Get human input - use a more reliable method that works in ADK CLI
        try:
            import sys
            import asyncio

            # Check if stdin is available and interactive
            if not sys.stdin.isatty():
                logger.info(
                    "[HumanReflectionAgent] stdin is not a TTY (non-interactive). Skipping human feedback."
                )
                human_feedback = ""
            else:
                # Flush all output streams to ensure prompt is visible
                sys.stdout.flush()
                sys.stderr.flush()

                # Use a helper function that tries multiple input methods
                def read_user_input():
                    """Read input using the most reliable method available."""
                    try:
                        # Method 1: Try input() - standard Python input
                        result = input()
                        return result
                    except (EOFError, OSError):
                        # Method 2: Fallback to sys.stdin.readline()
                        try:
                            line = sys.stdin.readline()
                            return line.rstrip("\n\r") if line else ""
                        except (EOFError, OSError):
                            return ""

                # Use asyncio to run input in executor with a timeout
                loop = asyncio.get_event_loop()

                # Try to get input with a 30-second timeout
                # If timeout expires or input fails, skip human feedback
                try:
                    # Print prompt directly to stdout (not through logger) for better visibility
                    # This ensures the prompt appears immediately and is interactive
                    print(
                        "[HumanReflectionAgent] >>> Your feedback (optional, 30s timeout): ",
                        end="",
                        flush=True,
                    )
                    sys.stdout.flush()

                    # Use executor to run input in a separate thread
                    # This prevents blocking the async event loop
                    human_feedback = await asyncio.wait_for(
                        loop.run_in_executor(None, read_user_input), timeout=30.0
                    )
                    human_feedback = human_feedback.strip() if human_feedback else ""
                    print()  # New line after input
                except asyncio.TimeoutError:
                    print()  # New line after timeout
                    logger.info(
                        "[HumanReflectionAgent] Input timeout (30s). Auto-skipping human feedback. Using LLM reflection only."
                    )
                    human_feedback = ""
                except (EOFError, OSError, KeyboardInterrupt, RuntimeError) as e:
                    print()  # New line after error
                    # Handle cases where stdin is not available or interrupted
                    logger.info(
                        f"[HumanReflectionAgent] Input not available or interrupted: {e}. Using LLM reflection only."
                    )
                    human_feedback = ""

            if not human_feedback or human_feedback.lower() == "skip":
                logger.info(
                    "[HumanReflectionAgent] No human feedback provided. Using LLM reflection only."
                )
                # Keep LLM reflection as-is
                combined_reflection = llm_reflection
            else:
                logger.info(
                    f"[HumanReflectionAgent] Received human feedback (length: {len(human_feedback)} chars)"
                )

                # Combine LLM reflection with human feedback
                combined_reflection = f"""{llm_reflection}

=== HUMAN FEEDBACK ===
{human_feedback}
"""
                logger.info(
                    "[HumanReflectionAgent] Combined LLM reflection with human feedback"
                )

            # Update reflection in session state
            ctx.session.state["reflection"] = combined_reflection
            ctx.session.state["human_reflection"] = (
                human_feedback
                if human_feedback and human_feedback.lower() != "skip"
                else ""
            )

            logger.info(
                f"[HumanReflectionAgent] Final reflection length: {len(combined_reflection)} chars"
            )

        except Exception as e:
            logger.warning(
                f"[HumanReflectionAgent] Failed to get human input (non-critical): {e}. Using LLM reflection only."
            )
            # Keep LLM reflection as-is if human input fails
            ctx.session.state["reflection"] = llm_reflection
            ctx.session.state["human_reflection"] = ""

        yield Event(author=self.name, content=None)
