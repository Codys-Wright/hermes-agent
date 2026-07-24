"""Nextcloud Talk gateway adapter.

Connects to a Nextcloud instance via its Talk Bot API (webhook-based).
Inbound messages arrive via webhook POST; outbound messages use the
OCS Bot API with HMAC-SHA256 signature authentication.

Each Hermes profile can auto-create and manage its own Nextcloud Talk
team (group conversation) for multi-agent collaboration.

Environment variables:
    NEXTCLOUD_TALK_URL              Nextcloud server URL (e.g. https://cloud.example.com)
    NEXTCLOUD_TALK_BOT_SECRET       Shared HMAC secret registered in Talk bot settings
    NEXTCLOUD_TALK_WEBHOOK_PORT     Port for inbound webhook server (default: 8789)
    NEXTCLOUD_TALK_WEBHOOK_PATH     URL path for webhook endpoint (default: /webhook/nc-talk)
    NEXTCLOUD_TALK_API_USER         Nextcloud username for room management API calls
    NEXTCLOUD_TALK_API_PASSWORD     Password for room management API calls
    NEXTCLOUD_TALK_ALLOWED_USERS    Comma-separated user IDs
    NEXTCLOUD_TALK_HOME_CHANNEL     Room token for cron/notification delivery
    NEXTCLOUD_TALK_AUTO_TEAM        Auto-create a team conversation per profile (default: true)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# Nextcloud Talk message size limit (OCS API enforces 32000 chars, but
# 4000 is the practical limit for readable chat messages).
MAX_MESSAGE_LENGTH = 4000

# Room type codes from the Nextcloud Talk API.
_ROOM_TYPE_MAP = {
    1: "dm",       # one-on-one
    2: "group",    # group conversation
    3: "channel",  # public channel
    4: "channel",  # password-protected public
    5: "dm",       # "changelog" user — treated as dm
    6: "dm",       # former one-on-one (left)
}

# Webhook server defaults.
_DEFAULT_WEBHOOK_PORT = 8789
_DEFAULT_WEBHOOK_PATH = "/webhook/nc-talk"

# Reconnect / cache parameters.
_ROOM_CACHE_TTL = 300  # 5 minutes
_ROOM_CACHE_ERR_TTL = 30  # 30 seconds for failed lookups


def check_nextcloud_talk_requirements() -> bool:
    """Return True if the Nextcloud Talk adapter can be used."""
    url = os.getenv("NEXTCLOUD_TALK_URL", "")
    secret = os.getenv("NEXTCLOUD_TALK_BOT_SECRET", "")
    if not url:
        logger.debug("Nextcloud Talk: NEXTCLOUD_TALK_URL not set")
        return False
    if not secret:
        logger.debug("Nextcloud Talk: NEXTCLOUD_TALK_BOT_SECRET not set")
        return False
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        logger.warning("Nextcloud Talk: aiohttp not installed")
        return False


class NextcloudTalkAdapter(BasePlatformAdapter):
    """Gateway adapter for Nextcloud Talk (self-hosted)."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.NEXTCLOUD_TALK)

        extra = config.extra or {}
        self._base_url: str = (
            extra.get("url", "")
            or os.getenv("NEXTCLOUD_TALK_URL", "")
        ).rstrip("/")
        self._bot_secret: str = (
            extra.get("bot_secret", "")
            or os.getenv("NEXTCLOUD_TALK_BOT_SECRET", "")
        )
        self._api_user: str = (
            extra.get("api_user", "")
            or os.getenv("NEXTCLOUD_TALK_API_USER", "")
        )
        self._api_password: str = (
            extra.get("api_password", "")
            or os.getenv("NEXTCLOUD_TALK_API_PASSWORD", "")
        )

        # Webhook server config.
        self._webhook_port: int = int(
            extra.get("webhook_port", "")
            or os.getenv("NEXTCLOUD_TALK_WEBHOOK_PORT", str(_DEFAULT_WEBHOOK_PORT))
        )
        self._webhook_path: str = (
            extra.get("webhook_path", "")
            or os.getenv("NEXTCLOUD_TALK_WEBHOOK_PATH", _DEFAULT_WEBHOOK_PATH)
        )

        # Auto-team creation per profile.
        self._auto_team: bool = (
            extra.get("auto_team", "")
            or os.getenv("NEXTCLOUD_TALK_AUTO_TEAM", "true")
        ).lower() in ("true", "1", "yes")
        self._team_room_token: Optional[str] = extra.get("team_room_token")

        # Runtime state.
        self._session: Any = None  # aiohttp.ClientSession
        self._app: Any = None      # aiohttp web.Application
        self._runner: Any = None   # aiohttp web.AppRunner
        self._site: Any = None     # aiohttp web.TCPSite
        self._running = False

        # Caches.
        self._dedup = MessageDeduplicator()
        self._room_cache: Dict[str, tuple] = {}  # token -> (info_dict, expiry_ts)


    # ------------------------------------------------------------------
    # HMAC-SHA256 signature helpers
    # ------------------------------------------------------------------

    def _sign(self, body: str) -> tuple[str, str]:
        """Generate HMAC-SHA256 signature for outbound requests.

        Returns (random_hex, signature_hex).
        """
        random_hex = secrets.token_hex(32)
        message = (random_hex + body).encode("utf-8")
        sig = hmac.new(
            self._bot_secret.encode("utf-8"),
            message,
            hashlib.sha256,
        ).hexdigest()
        return random_hex, sig

    def _verify_signature(
        self, body: bytes, signature: str, random_hex: str
    ) -> bool:
        """Verify inbound webhook HMAC-SHA256 signature."""
        message = random_hex.encode("utf-8") + body
        expected = hmac.new(
            self._bot_secret.encode("utf-8"),
            message,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(signature, expected)

    # ------------------------------------------------------------------
    # HTTP helpers (OCS API)
    # ------------------------------------------------------------------

    def _ocs_headers(self) -> Dict[str, str]:
        """Common headers for OCS API calls."""
        return {
            "OCS-APIRequest": "true",
            "Accept": "application/json",
        }

    def _basic_auth(self):
        """Return aiohttp.BasicAuth for room management calls."""
        import aiohttp
        if self._api_user and self._api_password:
            return aiohttp.BasicAuth(self._api_user, self._api_password)
        return None

    async def _ocs_get(self, path: str) -> Dict[str, Any]:
        """GET an OCS endpoint with basic auth."""
        import aiohttp
        url = f"{self._base_url}/{path.lstrip('/')}"
        try:
            async with self._session.get(
                url,
                headers=self._ocs_headers(),
                auth=self._basic_auth(),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("NC Talk GET %s → %s: %s", path, resp.status, body[:200])
                    return {}
                data = await resp.json()
                return data.get("ocs", {}).get("data", data)
        except Exception as exc:
            logger.error("NC Talk GET %s error: %s", path, exc)
            return {}

    async def _ocs_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to an OCS endpoint with basic auth."""
        import aiohttp
        url = f"{self._base_url}/{path.lstrip('/')}"
        try:
            async with self._session.post(
                url,
                headers={**self._ocs_headers(), "Content-Type": "application/json"},
                auth=self._basic_auth(),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("NC Talk POST %s → %s: %s", path, resp.status, body[:200])
                    return {}
                data = await resp.json()
                return data.get("ocs", {}).get("data", data)
        except Exception as exc:
            logger.error("NC Talk POST %s error: %s", path, exc)
            return {}

    # ------------------------------------------------------------------
    # Room management
    # ------------------------------------------------------------------

    async def _get_room_info(self, room_token: str) -> Dict[str, Any]:
        """Get room info with caching."""
        now = time.monotonic()
        cached = self._room_cache.get(room_token)
        if cached:
            info, expiry = cached
            if now < expiry:
                return info

        info = await self._ocs_get(
            f"ocs/v2.php/apps/spreed/api/v4/room/{room_token}"
        )
        if info:
            self._room_cache[room_token] = (info, now + _ROOM_CACHE_TTL)
        else:
            self._room_cache[room_token] = ({}, now + _ROOM_CACHE_ERR_TTL)
        return info

    async def _create_team_room(self, name: str) -> Optional[str]:
        """Create a group conversation (team) and return its token.

        Used for per-profile agent teams.
        """
        if not self._api_user:
            logger.warning("NC Talk: cannot create team — NEXTCLOUD_TALK_API_USER not set")
            return None

        data = await self._ocs_post(
            "ocs/v2.php/apps/spreed/api/v4/room",
            {
                "roomType": 2,  # group conversation
                "roomName": name,
            },
        )
        token = data.get("token")
        if token:
            logger.info("NC Talk: created team room '%s' (token: %s)", name, token)
        else:
            logger.error("NC Talk: failed to create team room '%s'", name)
        return token

    async def _ensure_team_room(self) -> Optional[str]:
        """Ensure a team room exists for this profile. Creates one if needed."""
        if self._team_room_token:
            return self._team_room_token

        if not self._auto_team:
            return None

        # Derive team name from profile name.
        from hermes_cli.config import get_hermes_home
        hermes_home = get_hermes_home()
        profile_name = os.path.basename(hermes_home) or "default"
        team_name = f"Hermes — {profile_name}"

        token = await self._create_team_room(team_name)
        if token:
            self._team_room_token = token
            # Persist for next startup.
            env_path = os.path.join(hermes_home, ".env")
            try:
                with open(env_path, "a") as f:
                    f.write(f"\nNEXTCLOUD_TALK_TEAM_ROOM={token}\n")
                logger.info("NC Talk: saved team room token to .env")
            except OSError as exc:
                logger.warning("NC Talk: could not save team room token: %s", exc)
        return token

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Start webhook server and optionally create team room."""
        import aiohttp
        from aiohttp import web

        if not self._base_url or not self._bot_secret:
            logger.error("Nextcloud Talk: URL or bot secret not configured")
            return False

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self._running = True

        # Set up webhook server.
        self._app = web.Application()
        self._app.router.add_post(self._webhook_path, self._handle_webhook)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", self._webhook_port)
        await self._site.start()
        logger.info(
            "Nextcloud Talk: webhook server listening on 127.0.0.1:%d%s",
            self._webhook_port,
            self._webhook_path,
        )

        # Auto-create team room if configured.
        if self._auto_team and self._api_user:
            asyncio.create_task(self._ensure_team_room())

        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        """Shut down webhook server and HTTP session."""
        self._running = False

        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        if self._session and not self._session.closed:
            await self._session.close()

        self._mark_disconnected()
        logger.info("Nextcloud Talk: disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message to a Nextcloud Talk room as the NC user.

        Uses the standard Talk chat API with Basic Auth (not the Bot API).
        Messages appear from the user account, so they're @mentionable and
        show as a real participant — not a "Bot" badge.
        """
        if not content:
            return SendResult(success=True)

        import aiohttp
        from urllib.parse import urlencode

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, MAX_MESSAGE_LENGTH)

        last_id = None
        for chunk in chunks:
            url = (
                f"{self._base_url}/ocs/v2.php/apps/spreed/api/v1"
                f"/chat/{chat_id}"
            )
            headers = {
                **self._ocs_headers(),
                "Content-Type": "application/x-www-form-urlencoded",
            }
            body = urlencode({"message": chunk})

            try:
                async with self._session.post(
                    url,
                    headers=headers,
                    auth=self._basic_auth(),
                    data=body,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status >= 400:
                        resp_body = await resp.text()
                        logger.error(
                            "NC Talk send → %s: %s", resp.status, resp_body[:200]
                        )
                        return SendResult(
                            success=False,
                            error=f"HTTP {resp.status}: {resp_body[:200]}",
                        )
                    data = await resp.json()
                    ocs_data = data.get("ocs", {}).get("data") or {}
                    last_id = str(ocs_data.get("id", ""))
            except Exception as exc:
                logger.error("NC Talk send error: %s", exc)
                return SendResult(success=False, error=str(exc))

        return SendResult(success=True, message_id=last_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return room name and type."""
        info = await self._get_room_info(chat_id)
        if not info:
            return {"name": chat_id, "type": "channel"}

        room_type_code = info.get("type", 3)
        chat_type = _ROOM_TYPE_MAP.get(room_type_code, "channel")
        display_name = info.get("displayName") or info.get("name") or chat_id
        return {"name": display_name, "type": chat_type}

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    async def send_typing(
        self, chat_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """NC Talk typing indicators require the signaling server (HPB)
        WebSocket — no REST API available. No-op for now."""
        pass

    async def edit_message(
        self, chat_id: str, message_id: str, content: str
    ) -> SendResult:
        """Edit an existing message via the Talk chat API."""
        import aiohttp
        from urllib.parse import urlencode

        formatted = self.format_message(content)
        # Truncate to max length for a single edit
        if len(formatted) > MAX_MESSAGE_LENGTH:
            formatted = formatted[:MAX_MESSAGE_LENGTH - 3] + "..."

        url = (
            f"{self._base_url}/ocs/v2.php/apps/spreed/api/v1"
            f"/chat/{chat_id}/{message_id}"
        )
        headers = {
            **self._ocs_headers(),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        body = urlencode({"message": formatted})

        try:
            async with self._session.put(
                url,
                headers=headers,
                auth=self._basic_auth(),
                data=body,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status >= 400:
                    resp_body = await resp.text()
                    logger.error("NC Talk edit → %s: %s", resp.status, resp_body[:200])
                    return SendResult(success=False, error=f"HTTP {resp.status}")
                return SendResult(success=True, message_id=message_id)
        except Exception as exc:
            logger.error("NC Talk edit error: %s", exc)
            return SendResult(success=False, error=str(exc))

    def format_message(self, content: str) -> str:
        """Nextcloud Talk supports a subset of markdown."""
        return content

    # ------------------------------------------------------------------
    # Webhook handler
    # ------------------------------------------------------------------

    async def _handle_webhook(self, request) -> Any:
        """Process an inbound webhook POST from Nextcloud Talk."""
        from aiohttp import web

        # Extract security headers.
        signature = request.headers.get("X-Nextcloud-Talk-Signature", "")
        random_hex = request.headers.get("X-Nextcloud-Talk-Random", "")

        if not signature or not random_hex:
            return web.Response(status=400, text="Missing signature headers")

        # Read and verify body.
        body = await request.read()
        if not self._verify_signature(body, signature, random_hex):
            logger.warning("NC Talk: invalid webhook signature")
            return web.Response(status=401, text="Invalid signature")

        # Parse payload.
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return web.Response(status=400, text="Invalid JSON")

        # Acknowledge immediately — agent sessions are long-running.
        # Process message asynchronously.
        asyncio.create_task(self._process_webhook_message(payload))
        return web.Response(status=200, text="OK")

    async def _process_webhook_message(self, payload: Dict[str, Any]) -> None:
        """Parse webhook payload and dispatch as MessageEvent."""
        try:
            actor = payload.get("actor", {})
            obj = payload.get("object", {})
            target = payload.get("target", {})

            user_id = actor.get("id", "")
            user_name = actor.get("name", "") or user_id
            raw_content = obj.get("content", "")
            # Content may be a JSON string with {message, parameters} structure
            if isinstance(raw_content, str) and raw_content.startswith("{"):
                try:
                    parsed = json.loads(raw_content)
                    message_text = parsed.get("message", raw_content).strip()
                except (json.JSONDecodeError, TypeError):
                    message_text = raw_content.strip()
            else:
                message_text = str(raw_content).strip()
            room_token = target.get("id", "")
            room_name = target.get("displayName", "") or target.get("name", room_token)

            # Ignore own messages — when sending as a NC user, the webhook
            # fires for our own messages too. Filter by API user ID.
            own_user = self._api_user
            if own_user and (
                user_id == own_user
                or user_id == f"users/{own_user}"
                or user_id.endswith(f"/{own_user}")
            ):
                return

            if not message_text or not room_token:
                return

            # Ignore system messages (contain {placeholder} tokens like
            # "{actor} added {user}", "{file}", etc.)
            import re
            if re.search(r'\{[a-z_]+\}', message_text):
                return

            # Skip non-message event types (Join, Leave, etc.)
            event_type = payload.get("type", "")
            if event_type and event_type not in ("Create",):
                return

            # Generate a message ID from payload if not provided.
            msg_id = obj.get("id", f"nc-{room_token}-{int(time.time() * 1000)}")
            msg_id = str(msg_id)

            # Dedup.
            if self._dedup.is_duplicate(msg_id):
                return

            # Determine room type — query the NC API, but if that fails
            # (bot user may lack permissions), fall back to counting
            # participants from the target metadata.
            room_info = await self._get_room_info(room_token)
            room_type_code = room_info.get("type", 0) if room_info else 0
            if room_type_code == 0:
                # API failed — assume group unless the room name looks like
                # a 1:1 DM (format: ["user1","user2"])
                room_name_str = target.get("name", "")
                if room_name_str.startswith("[") and room_name_str.count(",") == 1:
                    room_type_code = 1  # likely a DM
                else:
                    room_type_code = 2  # assume group — require @mention
            chat_type = _ROOM_TYPE_MAP.get(room_type_code, "channel")

            # Mention-gating for group conversations:
            # In DMs (type 1) → always respond
            # In groups/channels (type 2,3,4) → respond if @mentioned OR
            #   if the user replied directly to one of our messages
            if room_type_code != 1:  # not a 1:1 DM
                own_user = self._api_user
                addressed = False

                # Check 1: direct reply to our message
                in_reply_to = obj.get("inReplyTo", {})
                if in_reply_to:
                    reply_actor = in_reply_to.get("actor", {})
                    reply_actor_id = reply_actor.get("id", "")
                    if (reply_actor_id == own_user
                        or reply_actor_id == f"users/{own_user}"
                        or reply_actor_id.endswith(f"/{own_user}")):
                        addressed = True

                # Check 2: @mention in message text
                if not addressed:
                    mention_patterns = [
                        f"@{own_user}",
                        f"@\"{own_user}\"",
                    ]
                    addressed = any(
                        p.lower() in message_text.lower()
                        for p in mention_patterns
                    )

                # Check 3: structured mention parameters
                if not addressed and isinstance(raw_content, str):
                    try:
                        parsed_params = json.loads(raw_content)
                        params = parsed_params.get("parameters", {})
                        mentions = params.values() if isinstance(params, dict) else (params if isinstance(params, list) else [])
                        for param in mentions:
                            if (isinstance(param, dict)
                                and param.get("type") == "user"
                                and param.get("id") == own_user):
                                addressed = True
                                break
                    except (json.JSONDecodeError, TypeError):
                        pass

                if not addressed:
                    logger.debug(
                        "NC Talk: skipping group message — not @%s and not a reply (room=%s)",
                        own_user, room_token,
                    )
                    return

                # Keep the @mention in the message so the agent knows
                # it's being addressed directly

            # Build event.
            source = self.build_source(
                chat_id=room_token,
                chat_type=chat_type,
                user_id=user_id,
                user_name=user_name,
            )

            msg_type = MessageType.TEXT
            if message_text.startswith("/"):
                msg_type = MessageType.COMMAND

            event = MessageEvent(
                text=message_text,
                message_type=msg_type,
                source=source,
                raw_message=payload,
                message_id=msg_id,
            )

            await self.handle_message(event)

        except Exception as exc:
            logger.error("NC Talk: error processing webhook message: %s", exc)

    # ------------------------------------------------------------------
    # Team management helpers (public API for multi-agent use)
    # ------------------------------------------------------------------

    async def add_user_to_team(self, user_id: str) -> bool:
        """Add a Nextcloud user to this profile's team room."""
        token = self._team_room_token
        if not token:
            logger.warning("NC Talk: no team room — cannot add user")
            return False

        data = await self._ocs_post(
            f"ocs/v2.php/apps/spreed/api/v4/room/{token}/participants",
            {"newParticipant": user_id, "source": "users"},
        )
        return bool(data)

    async def send_to_team(self, message: str) -> SendResult:
        """Send a message to this profile's team room."""
        token = self._team_room_token
        if not token:
            return SendResult(success=False, error="No team room configured")
        return await self.send(token, message)

    @property
    def team_room_token(self) -> Optional[str]:
        """Return the team room token, if available."""
        return self._team_room_token
