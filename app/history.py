MAX_HISTORY_TURNS = 5


def format_history(history: list[dict] | None) -> str:
    """Render prior conversation turns as plain text for a prompt.

    Only the most recent MAX_HISTORY_TURNS are kept so the prompt doesn't grow unbounded over a
    long conversation.
    """
    if not history:
        return "(This is the first question in the conversation — no prior turns.)"

    recent = history[-MAX_HISTORY_TURNS:]
    lines = []
    for turn in recent:
        lines.append(f"User asked: {turn.get('query', '')}")
        lines.append(f"Assistant answered: {turn.get('answer', '')}")
    return "\n".join(lines)
