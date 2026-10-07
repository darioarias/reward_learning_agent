from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing import AsyncGenerator
import json
import numpy as np
from loguru import logger


class SelectorAgent(BaseAgent):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info("[SelectorAgent] Starting candidate selection...")

        # Check if candidate_results exists
        if "candidate_results" not in ctx.session.state:
            raise ValueError(
                "[SelectorAgent] 'candidate_results' not found in session state. "
                "CandidateEvaluatorAgent may not have run successfully."
            )

        try:
            results = json.loads(ctx.session.state["candidate_results"])
            logger.info(f"[SelectorAgent] Loaded {len(results)} candidate results")
        except json.JSONDecodeError as e:
            logger.error(
                f"[SelectorAgent] Failed to parse candidate_results as JSON: {e}"
            )
            logger.error(
                "[SelectorAgent] candidate_results content (first 500 chars): "
                f"{ctx.session.state['candidate_results'][:500]}"
            )
            raise ValueError(
                f"[SelectorAgent] Invalid JSON in candidate_results: {e}"
            ) from e

        if not results:
            # Check for different error scenarios
            llm_error = ctx.session.state.get("llm_output_error", "")
            eval_errors = ctx.session.state.get("evaluation_errors", "")

            error_parts = []
            if llm_error:
                error_parts.append(f"LLM parsing error: {llm_error}")
            if eval_errors:
                error_parts.append(
                    f"Evaluation errors (all candidates failed):\n{eval_errors}"
                )
            if not error_parts:
                error_parts.append("No candidate results were produced.")

            error_msg = "\n".join(error_parts)
            raise ValueError(f"[SelectorAgent] Empty candidate results. {error_msg}")

        # Extract scores, handling NaN and invalid values
        scores = []
        for i, r in enumerate(results):
            score = r.get("score", -float("inf"))
            # Convert NaN to -inf so they're not selected
            if isinstance(score, (int, float)) and not np.isnan(score):
                scores.append(float(score))
                logger.debug(
                    f"[SelectorAgent] Candidate {i+1}: score={float(score):.2f}"
                )
            else:
                scores.append(-float("inf"))
                logger.warning(
                    f"[SelectorAgent] Candidate {i+1}: invalid score={score}, using -inf"
                )

        if not scores or all(s == -float("inf") for s in scores):
            logger.error(
                f"[SelectorAgent] All candidate scores are invalid. Scores: {scores}"
            )
            raise ValueError(
                "[SelectorAgent] All candidate scores are invalid (NaN or missing). "
                "Cannot select best candidate."
            )

        best_idx = int(np.argmax(scores))
        best = results[best_idx]
        logger.info(
            f"[SelectorAgent] Selected candidate {best_idx + 1} "
            f"(0-indexed: {best_idx}) with score={scores[best_idx]:.2f}"
        )

        # Validate required keys exist
        if "reward_code" not in best:
            raise ValueError(
                f"[SelectorAgent] Best candidate {best_idx} missing 'reward_code' key"
            )
        if "eval" not in best or "avg_return" not in best["eval"]:
            raise ValueError(
                f"[SelectorAgent] Best candidate {best_idx} missing 'eval.avg_return' key"
            )

        ctx.session.state["best_candidate_index"] = best_idx
        ctx.session.state["best_reward_code"] = best["reward_code"]
        ctx.session.state["best_candidate_summary"] = json.dumps(
            best, indent=2, default=str
        )
        ctx.session.state["best_return"] = float(best["eval"]["avg_return"])

        logger.info(
            "[SelectorAgent] Selection complete. "
            f"Best return: {ctx.session.state['best_return']:.2f}"
        )
        logger.debug(
            "[SelectorAgent] Best candidate summary: "
            f"{ctx.session.state['best_candidate_summary'][:200]}..."
        )

        yield Event(author=self.name, content=None)
