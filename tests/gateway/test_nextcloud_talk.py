"""Tests for the Nextcloud Talk bots-v1 gateway adapter."""

import asyncio
import hmac
import hashlib
import json

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.nextcloud_talk import NextcloudTalkAdapter


SECRET = "super-secret"


def _make_adapter(**extra_overrides) -> NextcloudTalkAdapter:
    extra = {"secret": SECRET, "base_url": "https://nextcloud.example.com"}
    extra.update(extra_overrides)
    return NextcloudTalkAdapter(PlatformConfig(enabled=True, extra=extra))


def _body(payload: dict) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _signature(random_header: str, body: bytes, secret: str = SECRET) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        random_header.encode("utf-8") + body,
        hashlib.sha256,
    ).hexdigest()


def _message_signature(random_header: str, message: str, secret: str = SECRET) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        random_header.encode("utf-8") + message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _create_payload(**overrides) -> dict:
    payload = {
        "type": "Create",
        "actor": {"type": "Person", "id": "users/ada", "name": "Ada"},
        "object": {
            "type": "Note",
            "id": "1567",
            "name": "message",
            "content": json.dumps(
                {"message": "hello {mention-call1}", "parameters": {"mention-call1": {"type": "call"}}}
            ),
            "mediaType": "text/markdown",
        },
        "target": {"type": "Collection", "id": "room-token", "name": "Room"},
    }
    payload.update(overrides)
    return payload


class _FakeRequest:
    def __init__(self, *, body: bytes = b"{}", headers: dict | None = None):
        self._body = body
        self.headers = headers or {}

    async def read(self):
        return self._body


class TestNextcloudTalkConfig:
    def test_gateway_config_accepts_nextcloud_talk_platform(self):
        config = GatewayConfig.from_dict(
            {
                "platforms": {
                    "nextcloud_talk": {
                        "enabled": True,
                        "extra": {"secret": "expected"},
                    }
                }
            }
        )

        assert Platform.NEXTCLOUD_TALK in config.platforms
        assert Platform.NEXTCLOUD_TALK in config.get_connected_platforms()

    def test_nextcloud_talk_requires_secret_to_be_connected(self):
        config = GatewayConfig.from_dict(
            {"platforms": {"nextcloud_talk": {"enabled": True, "extra": {}}}}
        )

        assert Platform.NEXTCLOUD_TALK in config.platforms
        assert Platform.NEXTCLOUD_TALK not in config.get_connected_platforms()

    def test_env_overrides_apply_to_nextcloud_talk_platform(self, monkeypatch):
        config = GatewayConfig(platforms={})

        monkeypatch.setenv("NEXTCLOUD_TALK_ENABLED", "true")
        monkeypatch.setenv("NEXTCLOUD_TALK_BOT_SECRET", "env-secret")
        monkeypatch.setenv("NEXTCLOUD_TALK_BASE_URL", "https://nc.example")
        monkeypatch.setenv("NEXTCLOUD_TALK_HOST", "127.0.0.1")
        monkeypatch.setenv("NEXTCLOUD_TALK_PORT", "8650")
        monkeypatch.setenv("NEXTCLOUD_TALK_WEBHOOK_PATH", "talk-hook")

        _apply_env_overrides(config)

        cfg = config.platforms[Platform.NEXTCLOUD_TALK]
        assert cfg.enabled is True
        assert cfg.extra["secret"] == "env-secret"
        assert cfg.extra["base_url"] == "https://nc.example"
        assert cfg.extra["host"] == "127.0.0.1"
        assert cfg.extra["port"] == 8650
        assert cfg.extra["webhook_path"] == "talk-hook"


class TestNextcloudTalkSignatures:
    def test_inbound_signature_verifies_random_plus_raw_body(self):
        adapter = _make_adapter()
        body = _body(_create_payload())
        random_header = "abc123"
        headers = {
            "X-Nextcloud-Talk-Random": random_header,
            "X-Nextcloud-Talk-Signature": _signature(random_header, body),
        }

        assert adapter._verify_inbound(headers, body) is True

    def test_inbound_signature_fails_when_body_changes(self):
        adapter = _make_adapter()
        original = _body(_create_payload())
        changed = _body(_create_payload(object={"content": "tampered"}))
        random_header = "abc123"
        headers = {
            "X-Nextcloud-Talk-Random": random_header,
            "X-Nextcloud-Talk-Signature": _signature(random_header, original),
        }

        assert adapter._verify_inbound(headers, changed) is False


class TestNextcloudTalkParsing:
    def test_create_payload_builds_message_event(self):
        adapter = _make_adapter()
        event = adapter._build_message_event(
            _create_payload(),
            "https://nextcloud.example.com",
        )

        assert event is not None
        assert event.source.platform == Platform.NEXTCLOUD_TALK
        assert event.source.chat_id == "room-token"
        assert event.source.chat_name == "Room"
        assert event.source.chat_type == "group"
        assert event.source.user_id == "users/ada"
        assert event.source.user_name == "Ada"
        assert event.message_id == "1567"
        assert event.text == "hello {mention-call1}"
        assert event.raw_message["nextcloud_talk"]["backend"] == "https://nextcloud.example.com"
        assert event.raw_message["nextcloud_talk"]["parameters"] == {"mention-call1": {"type": "call"}}
        assert adapter._backend_by_conversation["room-token"] == "https://nextcloud.example.com"

    def test_reply_payload_builds_reply_context(self):
        adapter = _make_adapter()
        payload = _create_payload()
        payload["object"]["inReplyTo"] = {
            "actor": {"type": "Person", "id": "users/grace", "name": "Grace"},
            "object": {
                "type": "Note",
                "id": "99",
                "content": json.dumps({"message": "parent text", "parameters": {}}),
            },
        }

        event = adapter._build_message_event(payload)

        assert event.reply_to_message_id == "99"
        assert event.reply_to_text == "parent text"

    @pytest.mark.parametrize("hook_type", ["Join", "Leave", "Like", "Undo"])
    def test_non_create_hooks_are_ignored(self, hook_type):
        adapter = _make_adapter()
        payload = _create_payload(type=hook_type)

        assert adapter._build_message_event(payload) is None

    def test_bot_actors_are_ignored(self):
        adapter = _make_adapter()
        payload = _create_payload(actor={"type": "Application", "id": "bots/bot-1", "name": "Hermes"})

        assert adapter._build_message_event(payload) is None


class TestNextcloudTalkWebhook:
    @pytest.mark.anyio
    async def test_webhook_rejects_bad_signature(self):
        adapter = _make_adapter()
        body = _body(_create_payload())
        request = _FakeRequest(
            body=body,
            headers={
                "X-Nextcloud-Talk-Random": "abc123",
                "X-Nextcloud-Talk-Signature": "0" * 64,
            },
        )

        resp = await adapter._handle_webhook(request)

        assert resp.status == 401

    @pytest.mark.anyio
    async def test_webhook_schedules_valid_create_payload(self):
        adapter = _make_adapter()
        scheduled: list[tuple[dict, object]] = []

        async def _capture(payload, event):
            scheduled.append((payload, event))

        adapter.set_message_scheduler(_capture)
        payload = _create_payload()
        body = _body(payload)
        random_header = "abc123"
        request = _FakeRequest(
            body=body,
            headers={
                "X-Nextcloud-Talk-Random": random_header,
                "X-Nextcloud-Talk-Signature": _signature(random_header, body),
                "X-Nextcloud-Talk-Backend": "https://nextcloud.example.com",
            },
        )

        resp = await adapter._handle_webhook(request)
        await asyncio.sleep(0.05)

        assert resp.status == 202
        assert len(scheduled) == 1
        scheduled_payload, event = scheduled[0]
        assert scheduled_payload["type"] == "Create"
        assert event.text == "hello {mention-call1}"

    @pytest.mark.anyio
    async def test_webhook_acks_ignored_hooks_without_scheduling(self):
        adapter = _make_adapter()
        scheduled = []
        adapter.set_message_scheduler(lambda payload, event: scheduled.append(event))
        payload = _create_payload(type="Like")
        body = _body(payload)
        random_header = "abc123"
        request = _FakeRequest(
            body=body,
            headers={
                "X-Nextcloud-Talk-Random": random_header,
                "X-Nextcloud-Talk-Signature": _signature(random_header, body),
            },
        )

        resp = await adapter._handle_webhook(request)

        assert resp.status == 202
        assert scheduled == []


class _FakeResponse:
    status = 201

    async def text(self):
        return json.dumps({"ocs": {"data": {"id": "2001"}}})

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, *, data=None, headers=None):
        self.calls.append({"url": url, "data": data, "headers": headers})
        return _FakeResponse()


class TestNextcloudTalkOutbound:
    @pytest.mark.anyio
    async def test_outbound_uses_bot_message_endpoint_and_signed_message_field(self):
        adapter = _make_adapter()
        fake_session = _FakeSession()
        adapter._session = fake_session

        result = await adapter.send("room-token", "hello", reply_to="1567")

        assert result.success is True
        assert result.message_id == "2001"
        assert len(fake_session.calls) == 1
        call = fake_session.calls[0]
        assert call["url"] == "https://nextcloud.example.com/ocs/v2.php/apps/spreed/api/v1/bot/room-token/message"
        assert call["headers"]["OCS-APIRequest"] == "true"
        assert call["headers"]["X-Nextcloud-Talk-Bot-Random"]
        assert call["headers"]["X-Nextcloud-Talk-Bot-Signature"] == _message_signature(
            call["headers"]["X-Nextcloud-Talk-Bot-Random"],
            "hello",
        )
        sent_payload = json.loads(call["data"].decode("utf-8"))
        assert sent_payload["message"] == "hello"
        assert sent_payload["replyTo"] == 1567
        assert sent_payload["referenceId"]

    @pytest.mark.anyio
    async def test_outbound_can_use_backend_learned_from_webhook(self):
        adapter = _make_adapter(base_url="")
        fake_session = _FakeSession()
        adapter._session = fake_session
        adapter._backend_by_conversation["room-token"] = "https://learned.example"

        result = await adapter.send("room-token", "hello")

        assert result.success is True
        assert fake_session.calls[0]["url"] == "https://learned.example/ocs/v2.php/apps/spreed/api/v1/bot/room-token/message"
