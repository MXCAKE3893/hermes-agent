"""Nextcloud Talk bots-v1 webhook adapter."""

from __future__ import annotations

import asyncio
import hmac
import hashlib
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional
from urllib.parse import quote

try:
    from aiohttp import BasicAuth, ClientSession, web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    BasicAuth = None  # type: ignore[assignment]
    ClientSession = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8647
DEFAULT_WEBHOOK_PATH = "/nextcloud-talk"
DEFAULT_HEALTH_PATH = "/health"
DEFAULT_UPLOAD_DIR = "/Hermes Agent"
MAX_MESSAGE_LENGTH = 32_000
_TALK_API_PATH = "/ocs/v2.php/apps/spreed/api/v1"

MessageScheduler = Callable[[Dict[str, Any], MessageEvent], Awaitable[None] | None]


def check_nextcloud_talk_requirements() -> bool:
    """Return whether required webhook dependencies are available."""
    return AIOHTTP_AVAILABLE


class NextcloudTalkAdapter(BasePlatformAdapter):
    """Receive Nextcloud Talk bot webhooks and send bots-v1 replies."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.NEXTCLOUD_TALK)
        extra = config.extra or {}
        self._secret = str(extra.get("secret") or config.token or "").strip()
        self._base_url = str(extra.get("base_url") or "").strip().rstrip("/")
        self._host = str(extra.get("host", DEFAULT_HOST))
        self._port = int(extra.get("port", DEFAULT_PORT))
        self._webhook_path = self._normalize_path(
            extra.get("webhook_path", DEFAULT_WEBHOOK_PATH)
        )
        self._health_path = self._normalize_path(extra.get("health_path", DEFAULT_HEALTH_PATH))
        self._file_user = str(
            extra.get("file_user")
            or os.getenv("NEXTCLOUD_TALK_FILE_USER")
            or os.getenv("NEXTCLOUD_USERNAME")
            or ""
        ).strip()
        self._file_app_password = str(
            extra.get("file_app_password")
            or os.getenv("NEXTCLOUD_TALK_FILE_APP_PASSWORD")
            or os.getenv("NEXTCLOUD_PASSWORD")
            or ""
        )
        self._upload_dir = self._normalize_nextcloud_path(
            extra.get("upload_dir")
            or os.getenv("NEXTCLOUD_TALK_UPLOAD_DIR")
            or DEFAULT_UPLOAD_DIR
        )
        self._max_upload_bytes = self._resolve_max_upload_bytes(extra.get("max_upload_mb"))
        self._runner = None
        self._session: Optional["ClientSession"] = None
        self._message_scheduler: Optional[MessageScheduler] = None
        self._backend_by_conversation: dict[str, str] = {}
        self._accepted_count = 0
        self._ignored_count = 0

    @staticmethod
    def _normalize_path(path: Any) -> str:
        raw = str(path or "").strip() or "/"
        return raw if raw.startswith("/") else f"/{raw}"

    @classmethod
    def _normalize_nextcloud_path(cls, path: Any) -> str:
        raw = str(path or DEFAULT_UPLOAD_DIR).strip() or DEFAULT_UPLOAD_DIR
        normalized = cls._normalize_path(raw).rstrip("/")
        return normalized or "/"

    @staticmethod
    def _resolve_max_upload_bytes(config_value: Any) -> int:
        raw = config_value if config_value not in (None, "") else os.getenv(
            "NEXTCLOUD_TALK_MAX_UPLOAD_MB",
            "100",
        )
        try:
            mb = float(raw)
        except (TypeError, ValueError):
            mb = 100.0
        return max(int(mb * 1024 * 1024), 1)

    @staticmethod
    def _header(headers: Any, name: str) -> str:
        if not headers:
            return ""
        value = headers.get(name)
        if value is not None:
            return str(value)
        lower_name = name.lower()
        for key, candidate in headers.items():
            if str(key).lower() == lower_name:
                return str(candidate)
        return ""

    @staticmethod
    def _parse_content(value: Any) -> tuple[str, dict[str, Any]]:
        if isinstance(value, dict):
            parsed = value
        elif isinstance(value, str) and value:
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                return value, {}
        else:
            return "", {}
        if not isinstance(parsed, dict):
            return str(value or ""), {}
        message = parsed.get("message")
        parameters = parsed.get("parameters")
        return str(message or ""), parameters if isinstance(parameters, dict) else {}

    def set_message_scheduler(self, scheduler: Optional[MessageScheduler]) -> None:
        self._message_scheduler = scheduler

    async def connect(self) -> bool:
        if not self._secret:
            logger.error("[nextcloud_talk] shared bot secret is required")
            return False

        app = web.Application()
        app.router.add_get(self._health_path, self._handle_health)
        app.router.add_post(self._webhook_path, self._handle_webhook)

        self._session = ClientSession(trust_env=True)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._mark_connected()
        logger.info(
            "[nextcloud_talk] Listening on %s:%d%s",
            self._host,
            self._port,
            self._webhook_path,
        )
        return True

    async def disconnect(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        base_url = self._resolve_base_url(chat_id)
        if not base_url:
            return SendResult(
                success=False,
                error="Nextcloud Talk base_url is not configured and no backend was learned for this conversation",
            )
        if not self._secret:
            return SendResult(success=False, error="Nextcloud Talk bot secret is not configured")

        session = self._session or ClientSession(trust_env=True)
        close_session = self._session is None
        message_ids: list[str] = []
        raw_response: Any = None
        try:
            chunks = self.truncate_message(content or "", MAX_MESSAGE_LENGTH)
            for chunk in chunks:
                payload: dict[str, Any] = {
                    "message": chunk,
                    "referenceId": secrets.token_hex(32),
                }
                if reply_to:
                    try:
                        payload["replyTo"] = int(reply_to)
                    except (TypeError, ValueError):
                        logger.debug(
                            "[nextcloud_talk] Ignoring non-integer reply_to=%r",
                            reply_to,
                        )
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                random_header = secrets.token_hex(32)
                signature = self._sign_outbound_message(random_header, chunk)
                headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "OCS-APIRequest": "true",
                    "X-Nextcloud-Talk-Bot-Random": random_header,
                    "X-Nextcloud-Talk-Bot-Signature": signature,
                }
                url = self._message_url(base_url, chat_id)
                async with session.post(url, data=body, headers=headers) as response:
                    text = await response.text()
                    try:
                        raw_response = json.loads(text) if text else None
                    except ValueError:
                        raw_response = text
                    if response.status < 200 or response.status >= 300:
                        return SendResult(
                            success=False,
                            error=f"Nextcloud Talk send failed with HTTP {response.status}: {text[:500]}",
                            raw_response=raw_response,
                            retryable=response.status >= 500,
                        )
                    message_id = self._extract_response_message_id(raw_response)
                    if message_id:
                        message_ids.append(message_id)
            return SendResult(
                success=True,
                message_id=message_ids[-1] if message_ids else None,
                raw_response=raw_response,
                continuation_message_ids=tuple(message_ids[:-1]),
            )
        except Exception as exc:
            logger.exception("[nextcloud_talk] Failed to send message")
            return SendResult(success=False, error=str(exc), retryable=True)
        finally:
            if close_session:
                await session.close()

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "group"}

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Share a local image through Nextcloud Files into Talk."""
        return await self.send_document(
            chat_id=chat_id,
            file_path=image_path,
            caption=caption,
            file_name=kwargs.get("file_name"),
            reply_to=reply_to,
            metadata=metadata,
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Share a local video through Nextcloud Files into Talk."""
        return await self.send_document(
            chat_id=chat_id,
            file_path=video_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Share a local audio file through Nextcloud Files into Talk."""
        return await self.send_document(
            chat_id=chat_id,
            file_path=audio_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Upload a local file to Nextcloud Files and share it into Talk."""
        base_url = self._resolve_base_url(chat_id)
        if not base_url:
            return SendResult(
                success=False,
                error="Nextcloud Talk base_url is required for file sharing",
            )
        if not self._file_user or not self._file_app_password:
            return SendResult(
                success=False,
                error=(
                    "Nextcloud Talk file sharing requires NEXTCLOUD_TALK_FILE_USER "
                    "and NEXTCLOUD_TALK_FILE_APP_PASSWORD"
                ),
            )

        local_path = Path(file_path).expanduser()
        if not local_path.is_file():
            return SendResult(success=False, error=f"File not found: {file_path}")
        size = local_path.stat().st_size
        if size > self._max_upload_bytes:
            return SendResult(
                success=False,
                error=(
                    f"File exceeds NEXTCLOUD_TALK_MAX_UPLOAD_MB "
                    f"({size} > {self._max_upload_bytes} bytes): {file_path}"
                ),
            )

        session = self._session or ClientSession(trust_env=True)
        close_session = self._session is None
        auth = BasicAuth(self._file_user, self._file_app_password)
        try:
            nextcloud_path = await self._upload_file(
                session=session,
                auth=auth,
                base_url=base_url,
                local_path=local_path,
                file_name=file_name,
            )
            return await self._share_file(
                session=session,
                auth=auth,
                base_url=base_url,
                chat_id=chat_id,
                nextcloud_path=nextcloud_path,
                caption=caption,
                reply_to=reply_to,
                metadata=metadata,
            )
        except Exception as exc:
            logger.exception("[nextcloud_talk] Failed to share file")
            return SendResult(success=False, error=str(exc), retryable=True)
        finally:
            if close_session:
                await session.close()

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response(
            {
                "status": "ok",
                "platform": self.platform.value,
                "webhook_path": self._webhook_path,
                "accepted": self._accepted_count,
                "ignored": self._ignored_count,
            }
        )

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        try:
            body = await request.read()
        except Exception:
            return web.Response(status=400)

        if not self._verify_inbound(request.headers, body):
            return web.Response(status=401)

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return web.Response(status=400)
        if not isinstance(payload, dict):
            return web.Response(status=400)

        backend = self._header(request.headers, "X-Nextcloud-Talk-Backend").strip().rstrip("/")
        event = self._build_message_event(payload, backend)
        if event is None:
            self._ignored_count += 1
            return web.Response(status=202)

        self._accepted_count += 1
        self._schedule_message(payload, event)
        return web.Response(status=202)

    def _verify_inbound(self, headers: Any, body: bytes) -> bool:
        random_header = self._header(headers, "X-Nextcloud-Talk-Random")
        signature = self._header(headers, "X-Nextcloud-Talk-Signature").lower()
        if not self._secret or not random_header or not signature:
            return False
        expected = self._sign(random_header, body)
        return hmac.compare_digest(signature, expected)

    def _sign(self, random_header: str, body: bytes) -> str:
        digest = hmac.new(
            self._secret.encode("utf-8"),
            random_header.encode("utf-8") + body,
            hashlib.sha256,
        )
        return digest.hexdigest()

    def _sign_outbound_message(self, random_header: str, message: str) -> str:
        """Sign outbound Talk bot messages.

        Nextcloud's bots-v1 send endpoint validates the HMAC over the random
        header plus the message field, while inbound webhooks are signed over
        the random header plus the raw request body.
        """
        digest = hmac.new(
            self._secret.encode("utf-8"),
            random_header.encode("utf-8") + message.encode("utf-8"),
            hashlib.sha256,
        )
        return digest.hexdigest()

    def _build_message_event(
        self,
        payload: Dict[str, Any],
        backend: str = "",
    ) -> Optional[MessageEvent]:
        actor = payload.get("actor") if isinstance(payload.get("actor"), dict) else {}
        actor_type = str(actor.get("type") or "")
        actor_id = str(actor.get("id") or "")
        if actor_type == "Application" or actor_id.startswith("bots/"):
            return None
        if payload.get("type") != "Create":
            return None

        obj = payload.get("object") if isinstance(payload.get("object"), dict) else {}
        target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
        chat_id = str(target.get("id") or "").strip()
        if not chat_id:
            return None
        if backend:
            self._backend_by_conversation[chat_id] = backend

        text, parameters = self._parse_content(obj.get("content"))
        if not text.strip():
            return None

        reply_to_message_id = None
        reply_to_text = None
        in_reply_to = obj.get("inReplyTo") if isinstance(obj.get("inReplyTo"), dict) else {}
        parent_obj = in_reply_to.get("object") if isinstance(in_reply_to.get("object"), dict) else {}
        if parent_obj:
            parent_id = parent_obj.get("id")
            if parent_id is not None:
                reply_to_message_id = str(parent_id)
            parent_text, _ = self._parse_content(parent_obj.get("content"))
            reply_to_text = parent_text or None

        source = self.build_source(
            chat_id=chat_id,
            chat_name=str(target.get("name") or chat_id),
            chat_type="group",
            user_id=actor_id or None,
            user_name=str(actor.get("name") or actor_id or ""),
            message_id=str(obj.get("id")) if obj.get("id") is not None else None,
        )
        raw_message = dict(payload)
        raw_message["nextcloud_talk"] = {
            "backend": backend,
            "parameters": parameters,
        }
        return MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=raw_message,
            message_id=str(obj.get("id")) if obj.get("id") is not None else None,
            reply_to_message_id=reply_to_message_id,
            reply_to_text=reply_to_text,
        )

    def _schedule_message(self, payload: Dict[str, Any], event: MessageEvent) -> None:
        scheduler = self._message_scheduler
        if scheduler is not None:
            result = scheduler(payload, event)
            if asyncio.iscoroutine(result):
                task = asyncio.create_task(result)
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            return

        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _resolve_base_url(self, chat_id: str) -> str:
        return (self._base_url or self._backend_by_conversation.get(str(chat_id), "")).rstrip("/")

    @staticmethod
    def _site_base_url(base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith(_TALK_API_PATH):
            return base[: -len(_TALK_API_PATH)].rstrip("/")
        return base

    @staticmethod
    def _quote_nextcloud_path(path: str) -> str:
        return "/".join(quote(part, safe="") for part in path.strip("/").split("/") if part)

    def _dav_files_url(self, base_url: str, nextcloud_path: str = "") -> str:
        site = self._site_base_url(base_url)
        user = quote(self._file_user, safe="")
        suffix = self._quote_nextcloud_path(nextcloud_path)
        url = f"{site}/remote.php/dav/files/{user}"
        return f"{url}/{suffix}" if suffix else url

    def _share_url(self, base_url: str) -> str:
        return f"{self._site_base_url(base_url)}/ocs/v2.php/apps/files_sharing/api/v1/shares"

    async def _ensure_upload_dir(self, session: "ClientSession", auth: "BasicAuth", base_url: str) -> None:
        parts = [part for part in self._upload_dir.strip("/").split("/") if part]
        current = ""
        for part in parts:
            current = f"{current}/{part}" if current else f"/{part}"
            async with session.request("MKCOL", self._dav_files_url(base_url, current), auth=auth) as response:
                if response.status in {201, 405}:
                    continue
                text = await response.text()
                raise RuntimeError(
                    f"Failed to create Nextcloud upload directory {current}: HTTP {response.status}: {text[:500]}"
                )

    async def _upload_file(
        self,
        *,
        session: "ClientSession",
        auth: "BasicAuth",
        base_url: str,
        local_path: Path,
        file_name: Optional[str],
    ) -> str:
        await self._ensure_upload_dir(session, auth, base_url)
        display_name = file_name or local_path.name
        safe_name = display_name.replace("/", "_").replace("\\", "_").strip() or local_path.name
        remote_name = f"{secrets.token_hex(4)}-{safe_name}"
        nextcloud_path = f"{self._upload_dir}/{remote_name}" if self._upload_dir != "/" else f"/{remote_name}"
        with local_path.open("rb") as fh:
            async with session.put(
                self._dav_files_url(base_url, nextcloud_path),
                data=fh,
                auth=auth,
                headers={"OCS-APIRequest": "true"},
            ) as response:
                if response.status not in {200, 201, 204}:
                    text = await response.text()
                    raise RuntimeError(
                        f"Nextcloud WebDAV upload failed with HTTP {response.status}: {text[:500]}"
                    )
        return nextcloud_path

    async def _share_file(
        self,
        *,
        session: "ClientSession",
        auth: "BasicAuth",
        base_url: str,
        chat_id: str,
        nextcloud_path: str,
        caption: Optional[str],
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> SendResult:
        talk_metadata: dict[str, Any] = {}
        if caption:
            talk_metadata["caption"] = caption[:MAX_MESSAGE_LENGTH]
        if reply_to:
            try:
                talk_metadata["replyTo"] = int(reply_to)
            except (TypeError, ValueError):
                logger.debug("[nextcloud_talk] Ignoring non-integer file reply_to=%r", reply_to)
        data: dict[str, str] = {
            "shareType": "10",
            "shareWith": str(chat_id),
            "path": nextcloud_path,
            "referenceId": secrets.token_hex(32),
        }
        if talk_metadata:
            data["talkMetaData"] = json.dumps(talk_metadata, ensure_ascii=False, separators=(",", ":"))

        headers = {"Accept": "application/json", "OCS-APIRequest": "true"}
        async with session.post(
            self._share_url(base_url),
            data=data,
            auth=auth,
            headers=headers,
        ) as response:
            text = await response.text()
            try:
                raw_response = json.loads(text) if text else None
            except ValueError:
                raw_response = text
            if response.status < 200 or response.status >= 300:
                return SendResult(
                    success=False,
                    error=f"Nextcloud Talk file share failed with HTTP {response.status}: {text[:500]}",
                    raw_response=raw_response,
                    retryable=response.status >= 500,
                )
            return SendResult(
                success=True,
                message_id=self._extract_response_message_id(raw_response),
                raw_response=raw_response,
            )

    def _message_url(self, base_url: str, token: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith(_TALK_API_PATH):
            api_base = base
        else:
            api_base = f"{base}{_TALK_API_PATH}"
        return f"{api_base}/bot/{quote(str(token), safe='')}/message"

    @staticmethod
    def _extract_response_message_id(raw_response: Any) -> Optional[str]:
        if not isinstance(raw_response, dict):
            return None
        candidates = [
            raw_response.get("id"),
            raw_response.get("messageId"),
            raw_response.get("ocs", {}).get("data", {}).get("id")
            if isinstance(raw_response.get("ocs"), dict)
            and isinstance(raw_response.get("ocs", {}).get("data"), dict)
            else None,
        ]
        for candidate in candidates:
            if candidate is not None:
                return str(candidate)
        return None
