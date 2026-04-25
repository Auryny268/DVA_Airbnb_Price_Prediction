"""Chat tab: ask Claude questions about the NYC Airbnb dataset.

The model uses tool calls (defined in chat_tools.py) to query the data. Each
assistant turn runs a manual tool-use loop until Claude returns end_turn.

This tab is self-contained — it does not honour the dashboard's global
Month/Borough/Room filters, since its tools need access to the full dataset to
answer arbitrary questions.
"""

from __future__ import annotations

import json
import os

import anthropic
import streamlit as st

from chat_data import SCHEMA_SUMMARY, load_data
from chat_tools import TOOL_SCHEMAS, execute_tool

MODEL_OPTIONS = {
    "Haiku 4.5 (fast, cheap)": "claude-haiku-4-5",
    "Sonnet 4.6 (smart)": "claude-sonnet-4-6",
}
DEFAULT_MESSAGE_WINDOW = 6
MAX_TOOL_ITERATIONS = 8
MAX_TOKENS = 4096

SYSTEM_PROMPT = f"""You are a data analyst assistant for an NYC Airbnb dataset. You answer user questions by calling tools — never invent numbers.

{SCHEMA_SUMMARY}

Guidelines:
- Always use a tool to get concrete numbers. Don't guess.
- Pick the most specific tool. For proximity questions use `location_search`; for attribute filters use `query_listings`; for "what affects X" questions use `correlate`, `amenity_impact`, or `location_impact`.
- When the user asks "how many", "average", "top", "compare" — those map to specific tools, not a single generic query.
- Be concise. Cite concrete numbers from tool results. When a result set is small, list a few example listings.
- If a user question is ambiguous, make a reasonable assumption using the semantic defaults above and state the assumption in your answer.
"""

STARTER_PROMPTS = [
    "Which borough has the highest median listing price?",
    "How many superhost entire-home listings in Manhattan rent for under $250?",
    "Does having a pool affect nightly price?",
    "Show me the top 5 highest-revenue listings near a subway (within 0.5 km).",
    "How does average price change across the months in the data?",
    "What's the correlation between review score and revenue?",
]


def _secret(name: str, default=None):
    """Read a secret from .streamlit/secrets.toml, falling back to env then default."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
    return os.environ.get(name, default)


def _message_window_size() -> int:
    raw = _secret("MESSAGE_WINDOW", DEFAULT_MESSAGE_WINDOW)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MESSAGE_WINDOW


def _windowed_history(messages: list[dict], n: int) -> list[dict]:
    """Take the last n messages but ensure the slice begins with a user turn."""
    sliced = messages[-n:]
    while sliced and sliced[0]["role"] != "user":
        sliced = sliced[1:]
    return sliced


def _run_turn(client: anthropic.Anthropic, user_text: str, model: str) -> None:
    """Append the user turn, run the tool-use loop, render everything."""
    st.session_state.chat_messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    windowed = _windowed_history(st.session_state.chat_messages, _message_window_size())
    api_messages: list[dict] = [{"role": m["role"], "content": m["content"]} for m in windowed]

    assistant_placeholder = st.chat_message("assistant")
    rendered_text_blocks: list[str] = []
    rendered_tool_calls: list[dict] = []  # {name, input, result}

    with assistant_placeholder:
        for iteration in range(MAX_TOOL_ITERATIONS):
            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=[
                    {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                ],
                tools=TOOL_SCHEMAS,
                messages=api_messages,
            )

            api_messages.append({"role": "assistant", "content": response.content})

            tool_uses = []
            for block in response.content:
                if block.type == "text" and block.text:
                    rendered_text_blocks.append(block.text)
                    st.markdown(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if response.stop_reason == "end_turn" or not tool_uses:
                break

            tool_results = []
            for tu in tool_uses:
                result = execute_tool(tu.name, tu.input)
                rendered_tool_calls.append({"name": tu.name, "input": tu.input, "result": result})
                if st.session_state.get("chat_show_tool_calls", True):
                    with st.expander(f"Tool call: `{tu.name}`"):
                        st.json({"input": tu.input, "result": result})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result),
                })

            api_messages.append({"role": "user", "content": tool_results})
        else:
            st.warning(f"Stopped after {MAX_TOOL_ITERATIONS} tool iterations.")

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": "\n\n".join(rendered_text_blocks) if rendered_text_blocks else "(no text output)",
        "tool_calls": rendered_tool_calls,
    })


def render_chat_tab() -> None:
    """Render the chat tab. Safe to call once per run inside a `with tab:` block."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    st.markdown("#### Ask the Data")
    st.caption(
        "Chat with Claude about the full dataset. The assistant calls tools to query "
        "the data — no guessing. This tab ignores the global sidebar filters."
    )

    api_key = _secret("ANTHROPIC_API_KEY")
    if not api_key:
        st.error(
            "Missing `ANTHROPIC_API_KEY`. Add it to `.streamlit/secrets.toml` "
            "(see `.streamlit/secrets.toml.example`) or set the environment variable."
        )
        return
    client = anthropic.Anthropic(api_key=api_key)

    with st.expander("Chat settings", expanded=False):
        df = load_data()
        st.markdown(
            f"**Dataset**: {df['id'].nunique():,} listings × "
            f"{df['month'].nunique()} months = {len(df):,} rows."
        )

        model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0, key="chat_model_label")
        selected_model = MODEL_OPTIONS[model_label]
        st.caption(f"`{selected_model}` · history window: last {_message_window_size()} msgs")

        st.divider()
        st.markdown("**Try asking**")
        cols = st.columns(2)
        for i, p in enumerate(STARTER_PROMPTS):
            if cols[i % 2].button(p, use_container_width=True, key=f"chat_starter_{hash(p)}"):
                st.session_state.chat_pending_prompt = p
                st.rerun()

        st.divider()
        st.checkbox("Show tool calls", value=True, key="chat_show_tool_calls")
        if st.button("Clear chat", use_container_width=True, key="chat_clear"):
            st.session_state.chat_messages = []
            st.rerun()

    selected_model = MODEL_OPTIONS[st.session_state.get("chat_model_label", list(MODEL_OPTIONS.keys())[0])]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], str):
                st.markdown(msg["content"])
            else:
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        st.markdown(block["text"])
            if msg["role"] == "assistant" and st.session_state.get("chat_show_tool_calls", True):
                for tc in msg.get("tool_calls", []):
                    with st.expander(f"Tool call: `{tc['name']}`"):
                        st.json({"input": tc["input"], "result": tc["result"]})

    prompt = st.chat_input("Ask about the data...", key="chat_input")
    if "chat_pending_prompt" in st.session_state:
        prompt = st.session_state.pop("chat_pending_prompt")

    if prompt:
        _run_turn(client, prompt, selected_model)
