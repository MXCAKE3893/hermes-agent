"""Tests for Nextcloud Talk standalone send_message delivery."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from tools.send_message_tool import _send_to_platform


def test_nextcloud_talk_media_uses_adapter_helper(tmp_path):
    document = tmp_path / "report.pdf"
    document.write_bytes(b"%PDF-1.4 test")
    helper = AsyncMock(
        return_value={
            "success": True,
            "platform": "nextcloud_talk",
            "chat_id": "room-token",
            "message_id": "42",
        }
    )

    with patch("tools.send_message_tool._send_nextcloud_talk", helper):
        result = asyncio.run(
            _send_to_platform(
                Platform.NEXTCLOUD_TALK,
                SimpleNamespace(enabled=True, extra={"secret": "shared-secret"}),
                "room-token",
                "here is the report",
                thread_id="42",
                media_files=[(str(document), False)],
            )
        )

    assert result["success"] is True
    helper.assert_awaited_once()
    call = helper.await_args
    assert call.args[1:] == ("room-token", "here is the report")
    assert call.kwargs["thread_id"] == "42"
    assert call.kwargs["media_files"] == [(str(document), False)]
