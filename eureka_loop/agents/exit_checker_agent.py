from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing import AsyncGenerator
from google.adk.tools.tool_context import ToolContext
from .exit_tool import exit_loop


class ExitCheckerAgent(BaseAgent):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        best_return = ctx.session.state.get("best_return", None)
        if best_return is None:
            yield Event(author=self.name, content=None)
            return

        threshold = ctx.session.state["return_threshold"]
        delta = ctx.session.state["plateau_delta"]
        patience = ctx.session.state["plateau_patience"]

        prev = ctx.session.state.get("prev_best_return", None)

        if prev is not None:
            improvement = best_return - prev
            if improvement < delta:
                ctx.session.state["plateau_count"] = (
                    ctx.session.state.get("plateau_count", 0) + 1
                )
            else:
                ctx.session.state["plateau_count"] = 0

        ctx.session.state["prev_best_return"] = best_return
        plateau_count = ctx.session.state.get("plateau_count", 0)

        # Exit if we've reached the threshold OR if we've plateaued after reaching a good performance
        # The original logic required both conditions, which seems too restrictive
        # New logic: exit if threshold reached OR if we've plateaued (with some minimum performance)
        should_exit = False
        exit_reason = ""

        if best_return >= threshold:
            if plateau_count >= patience:
                should_exit = True
                exit_reason = f"Reached threshold ({best_return:.2f} >= {threshold}) and plateaued ({plateau_count} >= {patience})"
            # Also exit immediately if we're well above threshold (no need to wait for plateau)
            elif best_return >= threshold * 1.1:  # 10% above threshold
                should_exit = True
                exit_reason = f"Significantly exceeded threshold ({best_return:.2f} >= {threshold * 1.1:.2f})"
        elif plateau_count >= patience and best_return > 0:
            # Exit if we've plateaued and have positive performance (avoid exiting on negative plateaus)
            should_exit = True
            exit_reason = f"Performance plateaued ({plateau_count} >= {patience}) with return {best_return:.2f}"

        if should_exit:
            logger.info(f"[ExitCheckerAgent] Exiting loop: {exit_reason}")
            tc = ToolContext(agent_name=self.name)
            exit_loop(tc)
            yield Event(author=self.name, actions=tc.actions)
            return

        yield Event(author=self.name, content=None)
