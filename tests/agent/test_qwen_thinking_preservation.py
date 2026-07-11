"""Tests for Qwen3.6 thinking/reasoning_content preservation with llama.cpp."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from run_agent import AIAgent
from agent.chat_completion_helpers import build_assistant_message
from agent.agent_runtime_helpers import copy_reasoning_content_for_api, reapply_reasoning_echo_for_provider
from hermes_state import SessionDB


def _strip_think_blocks_mock(content: str) -> str:
    """Minimal mock of strip_think_blocks that properly removes tag contents."""
    if not content:
        return ""
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)
    return content.strip()


def _make_agent(
    provider: str = "custom",
    model: str = "qwen3.6",
    base_url: str = "http://127.0.0.1:8080/v1",
    preserve_thinking: bool = True
) -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    agent.base_url = base_url
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    agent._thinking_prefill_retries = 0
    agent.reasoning_replay = False
    agent.session_id = "session_1"
    agent.reasoning_config = None
    agent.api_mode = "chat_completions"
    agent.verbose = False
    agent.max_iterations = 90
    agent._budget_exhausted_injected = False
    agent._budget_grace_call = False

    # Setup request_overrides to mimic resolver output
    agent.request_overrides = {
        "extra_body": {
            "chat_template_kwargs": {
                "preserve_thinking": preserve_thinking
            }
        }
    }

    # Bind real AIAgent methods for reasoning extraction and replay detection
    agent._should_replay_reasoning_content = AIAgent._should_replay_reasoning_content.__get__(agent, AIAgent)
    agent._needs_thinking_reasoning_pad = AIAgent._needs_thinking_reasoning_pad.__get__(agent, AIAgent)
    agent._needs_deepseek_tool_reasoning = AIAgent._needs_deepseek_tool_reasoning.__get__(agent, AIAgent)
    agent._needs_kimi_tool_reasoning = AIAgent._needs_kimi_tool_reasoning.__get__(agent, AIAgent)
    agent._needs_mimo_tool_reasoning = AIAgent._needs_mimo_tool_reasoning.__get__(agent, AIAgent)
    agent._extract_reasoning = AIAgent._extract_reasoning.__get__(agent, AIAgent)

    # Use a proper mock for _strip_think_blocks that removes tag contents
    agent._strip_think_blocks = _strip_think_blocks_mock

    agent._custom_providers = []

    return agent


# テスト1：非ストリーミング取得
def test_non_streaming_extraction():
    agent = _make_agent()
    # Response message mock
    assistant_message = SimpleNamespace(
        role="assistant",
        content="answer-A",
        reasoning_content="thought-A"
    )
    msg = build_assistant_message(agent, assistant_message, "stop")
    assert msg.get("reasoning_content") == "thought-A"
    assert msg.get("content") == "answer-A"


# テスト2：ストリーミング取得
def test_streaming_extraction_simulation():
    agent = _make_agent()
    mock_message = SimpleNamespace(
        role="assistant",
        content="answer-B",
        reasoning_content="thought-B",
        tool_calls=None
    )
    msg = build_assistant_message(agent, mock_message, "stop")
    assert msg.get("reasoning_content") == "thought-B"
    assert msg.get("content") == "answer-B"


# テスト3：インラインthinkのフォールバック
def test_inline_think_fallback():
    agent = _make_agent()
    # Response with raw <think> tags in content
    assistant_message = SimpleNamespace(
        role="assistant",
        content="<think>\nthought-C\n</think>\n\nanswer-C",
    )
    msg = build_assistant_message(agent, assistant_message, "stop")
    assert msg.get("reasoning") == "thought-C"
    assert msg.get("content") == "answer-C"


# テスト4：セッション往復 (SQLite)
def test_session_roundtrip(tmp_path):
    # Using SessionDB in a temporary directory
    db_file = tmp_path / "test_state.db"
    db = SessionDB(db_path=Path(db_file))
    db.create_session("session_1", "cli")

    # Store messages using append_message
    db.append_message("session_1", "user", content="question-D")
    db.append_message(
        "session_1", "assistant",
        content="answer-D",
        reasoning_content="thought-D",
        reasoning="thought-D",
        finish_reason="stop",
        reasoning_details=[{"type": "thinking", "text": "details"}],
    )

    # Load back using get_messages
    history = db.get_messages("session_1")
    loaded_msg = history[-1]

    assert loaded_msg.get("reasoning_content") == "thought-D"
    assert loaded_msg.get("content") == "answer-D"


# テスト5：次回リクエストへの再送
def test_next_request_replay():
    agent = _make_agent(preserve_thinking=True)

    # 1ターン目のassistantレスポンス
    source_msg = {
        "role": "assistant",
        "content": "first-answer",
        "reasoning_content": "thought-XYZ"
    }

    api_msg = source_msg.copy()
    copy_reasoning_content_for_api(agent, source_msg, api_msg)

    assert api_msg.get("role") == "assistant"
    assert api_msg.get("reasoning_content") == "thought-XYZ"
    assert api_msg.get("content") == "first-answer"


# テスト6：ツール呼び出し
def test_tool_call_preservation():
    agent = _make_agent(preserve_thinking=True)

    source_msg = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "need-to-read-readme",
        "tool_calls": [
            {
                "id": "call_read",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}'
                }
            }
        ]
    }

    api_msg = source_msg.copy()
    copy_reasoning_content_for_api(agent, source_msg, api_msg)

    assert api_msg.get("reasoning_content") == "need-to-read-readme"
    assert api_msg.get("tool_calls") == source_msg["tool_calls"]


# テスト7：空文字列の維持
def test_empty_string_preservation():
    agent = _make_agent(preserve_thinking=True)

    source_msg = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "",
        "tool_calls": []
    }

    api_msg = source_msg.copy()
    copy_reasoning_content_for_api(agent, source_msg, api_msg)

    assert "reasoning_content" in api_msg
    assert api_msg["reasoning_content"] == ""


# テスト8：無効時の非送信
def test_disabled_no_replay():
    agent = _make_agent(preserve_thinking=False)

    source_msg = {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "thought-process"
    }

    api_msg = source_msg.copy()
    copy_reasoning_content_for_api(agent, source_msg, api_msg)

    assert "reasoning_content" not in api_msg


# テスト9：プロバイダー切り替えによる誤昇格防止
def test_provider_switch_no_accidental_promotion():
    agent = _make_agent(preserve_thinking=False)

    source_msg = {
        "role": "assistant",
        "content": "answer",
        "reasoning": "openai-thought"
    }

    api_msg = source_msg.copy()
    copy_reasoning_content_for_api(agent, source_msg, api_msg)

    assert "reasoning_content" not in api_msg


# テスト10：gateway再読み込み
def test_gateway_reload_preservation(tmp_path):
    db_file = tmp_path / "gateway_state.db"
    db = SessionDB(db_path=Path(db_file))
    db.create_session("gateway_session", "gateway")

    # Store message using append_message
    db.append_message(
        "gateway_session", "assistant",
        content="answer-E",
        reasoning_content="thought-E",
        finish_reason="stop",
    )

    # Reload from a fresh DB handle
    db2 = SessionDB(db_path=Path(db_file))
    history = db2.get_messages("gateway_session")
    loaded_msg = history[-1]

    agent = _make_agent(preserve_thinking=True)
    api_msg = loaded_msg.copy()
    copy_reasoning_content_for_api(agent, loaded_msg, api_msg)

    assert api_msg.get("reasoning_content") == "thought-E"


# テスト11：フォールバック時のAPIメッセージ再構築
def test_fallback_rebuild():
    agent = _make_agent(preserve_thinking=True)

    api_messages = [
        {
            "role": "assistant",
            "content": "answer-F",
            "reasoning_content": "thought-F"
        }
    ]

    # When active provider does NOT require replay (preserve_thinking=False)
    agent.request_overrides["extra_body"]["chat_template_kwargs"]["preserve_thinking"] = False
    changed = reapply_reasoning_echo_for_provider(agent, api_messages)
    assert changed == 1
    assert "reasoning_content" not in api_messages[0]

    # When active provider DOES require replay (preserve_thinking=True)
    agent.request_overrides["extra_body"]["chat_template_kwargs"]["preserve_thinking"] = True
    api_messages = [
        {
            "role": "assistant",
            "content": "answer-F",
            "reasoning": "thought-F"
        }
    ]
    changed = reapply_reasoning_echo_for_provider(agent, api_messages)
    assert changed == 1
    assert api_messages[0].get("reasoning_content") == "thought-F"


# テスト12：build_assistant_message → copy_reasoning_content_for_api ラウンドトリップ
def test_build_and_replay_roundtrip():
    """Verify the full path: API response → build_assistant_message → history →
    copy_reasoning_content_for_api → next request payload."""
    agent = _make_agent(preserve_thinking=True)

    # Simulate API response with reasoning_content
    api_response_msg = SimpleNamespace(
        role="assistant",
        content="first-answer",
        reasoning_content="remember-me-123",
        tool_calls=None,
    )

    # Step 1: Build assistant message from API response
    assistant_msg = build_assistant_message(agent, api_response_msg, "stop")
    assert assistant_msg.get("reasoning_content") == "remember-me-123"
    assert assistant_msg.get("content") == "first-answer"

    # Step 2: Build conversation history for next turn
    messages = [
        {"role": "user", "content": "question-1"},
        assistant_msg,
        {"role": "user", "content": "question-2"},
    ]

    # Step 3: Assemble API payload for next turn
    api_messages = []
    for m in messages:
        api_msg = m.copy()
        copy_reasoning_content_for_api(agent, m, api_msg)
        api_messages.append(api_msg)

    # Check the assistant message in the payload still has reasoning_content
    assistant_turn = api_messages[1]
    assert assistant_turn.get("role") == "assistant"
    assert assistant_turn.get("reasoning_content") == "remember-me-123"

    # User messages should NOT have reasoning_content
    assert "reasoning_content" not in api_messages[0]
    assert "reasoning_content" not in api_messages[2]
