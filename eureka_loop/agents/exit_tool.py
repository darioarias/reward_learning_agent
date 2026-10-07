from google.adk.tools.tool_context import ToolContext


def exit_loop(tool_context: ToolContext):
    """Call to stop the LoopAgent when criteria is met."""
    print(f"[Tool Call] exit_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    return {}
