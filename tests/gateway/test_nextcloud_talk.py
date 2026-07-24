"""Tests for the Nextcloud Talk gateway adapter."""

import asyncio
import hashlib
import hmac
import json
import os
import secrets
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is importable
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.nextcloud_talk import (
    NextcloudTalkAdapter,
    check_nextcloud_talk_requirements,
    MAX_MESSAGE_LENGTH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Return a PlatformConfig for Nextcloud Talk."""
    return PlatformConfig(
        enabled=True,
        extra={
            "url": "https://cloud.example.com",
            "bot_secret": "test-secret-key-1234",
            "api_user": "admin",
            "api_password": "app-password-1234",
            "webhook_port": "9999",
            "webhook_path": "/webhook/test",
            "auto_team": "false",
        },
    )


@pytest.fixture
def adapter(config):
    """Return an adapter instance (not connected)."""
    return NextcloudTalkAdapter(config)


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

class TestRequirements:
    def test_missing_url(self, monkeypatch):
        monkeypatch.delenv("NEXTCLOUD_TALK_URL", raising=False)
        monkeypatch.delenv("NEXTCLOUD_TALK_BOT_SECRET", raising=False)
        assert check_nextcloud_talk_requirements() is False

    def test_missing_secret(self, monkeypatch):
        monkeypatch.setenv("NEXTCLOUD_TALK_URL", "https://example.com")
        monkeypatch.delenv("NEXTCLOUD_TALK_BOT_SECRET", raising=False)
        assert check_nextcloud_talk_requirements() is False

    def test_all_set(self, monkeypatch):
        monkeypatch.setenv("NEXTCLOUD_TALK_URL", "https://example.com")
        monkeypatch.setenv("NEXTCLOUD_TALK_BOT_SECRET", "secret")
        assert check_nextcloud_talk_requirements() is True


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------

class TestAdapterInit:
    def test_config_from_extra(self, adapter):
        assert adapter._base_url == "https://cloud.example.com"
        assert adapter._bot_secret == "test-secret-key-1234"
        assert adapter._api_user == "admin"
        assert adapter._webhook_port == 9999
        assert adapter._webhook_path == "/webhook/test"
        assert adapter._auto_team is False

    def test_config_from_env(self, monkeypatch):
        monkeypatch.setenv("NEXTCLOUD_TALK_URL", "https://env.example.com")
        monkeypatch.setenv("NEXTCLOUD_TALK_BOT_SECRET", "env-secret")
        config = PlatformConfig(enabled=True, extra={})
        adapter = NextcloudTalkAdapter(config)
        assert adapter._base_url == "https://env.example.com"
        assert adapter._bot_secret == "env-secret"


# ---------------------------------------------------------------------------
# HMAC signature
# ---------------------------------------------------------------------------

class TestSignature:
    def test_sign_roundtrip(self, adapter):
        body = '{"message": "hello"}'
        random_hex, sig = adapter._sign(body)
        assert len(random_hex) == 64  # 32 bytes hex
        assert len(sig) == 64  # SHA-256 hex

        # Verify the signature
        assert adapter._verify_signature(
            body.encode(), sig, random_hex
        ) is True

    def test_bad_signature_rejected(self, adapter):
        body = b'{"message": "hello"}'
        assert adapter._verify_signature(body, "bad-sig", "bad-random") is False

    def test_tampered_body_rejected(self, adapter):
        body = '{"message": "hello"}'
        random_hex, sig = adapter._sign(body)
        tampered = b'{"message": "evil"}'
        assert adapter._verify_signature(tampered, sig, random_hex) is False


# ---------------------------------------------------------------------------
# Message event building
# ---------------------------------------------------------------------------

class TestMessageProcessing:
    @pytest.mark.asyncio
    async def test_webhook_payload_parsing(self, adapter):
        """Verify _process_webhook_message builds a correct MessageEvent."""
        events = []
        adapter.handle_message = AsyncMock(side_effect=lambda e: events.append(e))
        # Stub room info
        # Type 1 = one-to-one DM: always answered (groups require an @mention).
        adapter._get_room_info = AsyncMock(return_value={"type": 1, "displayName": "Alice"})

        payload = {
            "actor": {"id": "alice", "name": "Alice"},
            "object": {"id": "42", "content": "Hello Hermes"},
            "target": {"id": "abcd1234", "displayName": "Team Room"},
        }
        await adapter._process_webhook_message(payload)

        assert len(events) == 1
        event = events[0]
        assert event.text == "Hello Hermes"
        assert event.message_id == "42"
        assert event.source.chat_id == "abcd1234"
        assert event.source.user_id == "alice"

    @pytest.mark.asyncio
    async def test_empty_content_ignored(self, adapter):
        events = []
        adapter.handle_message = AsyncMock(side_effect=lambda e: events.append(e))
        adapter._get_room_info = AsyncMock(return_value={})

        payload = {
            "actor": {"id": "alice", "name": "Alice"},
            "object": {"content": ""},
            "target": {"id": "room1"},
        }
        await adapter._process_webhook_message(payload)
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------

class TestSend:
    @pytest.mark.asyncio
    async def test_send_empty_content(self, adapter):
        result = await adapter.send("room1", "")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_as_user_uses_ocs_api(self, adapter):
        """send() posts via the user chat API: OCS headers + BasicAuth,
        no Bot-API HMAC headers (messages appear from the account)."""
        import aiohttp

        mock_resp = AsyncMock()
        mock_resp.status = 201
        mock_resp.json = AsyncMock(return_value={
            "ocs": {"data": {"id": 99}}
        })

        mock_session = MagicMock()
        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_post_ctx)
        adapter._session = mock_session

        result = await adapter.send("abcd1234", "Hello!")
        assert result.success is True
        assert result.message_id == "99"

        # Verify the user-API contract: OCS header present, no bot-HMAC.
        call_kwargs = mock_session.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert headers.get("OCS-APIRequest") == "true"
        assert "X-Nextcloud-Talk-Bot-Random" not in headers
        assert call_kwargs.kwargs.get("auth") is not None


# ---------------------------------------------------------------------------
# Chat info
# ---------------------------------------------------------------------------

class TestChatInfo:
    @pytest.mark.asyncio
    async def test_group_room(self, adapter):
        adapter._get_room_info = AsyncMock(return_value={
            "type": 2, "displayName": "My Team"
        })
        info = await adapter.get_chat_info("token123")
        assert info["name"] == "My Team"
        assert info["type"] == "group"

    @pytest.mark.asyncio
    async def test_dm_room(self, adapter):
        adapter._get_room_info = AsyncMock(return_value={
            "type": 1, "displayName": "Alice"
        })
        info = await adapter.get_chat_info("token456")
        assert info["type"] == "dm"

    @pytest.mark.asyncio
    async def test_unknown_room(self, adapter):
        adapter._get_room_info = AsyncMock(return_value={})
        info = await adapter.get_chat_info("unknown")
        assert info["name"] == "unknown"
        assert info["type"] == "channel"
