"""The HTTP backend: providers, retries and what counts as a usable answer.

No network. Every test injects a fake transport, so the request that would go
out is inspected directly -- which is the only way to check the things that
matter here (that the page text is actually in the body, that a 429 is waited
out rather than dropped) without a key and without a free tier's rate limit.
"""

from __future__ import annotations

import json

import pytest

from bt import translate
from bt.translate import (
    PROVIDERS,
    OpenAICompatTranslator,
    TranslationError,
    resolve_provider,
)


def reply(content: str, finish: str = "stop", status: int = 200):
    """One OpenAI-shaped chat completion."""
    body = {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }
    return status, json.dumps(body).encode()


class FakeTransport:
    """Records requests and replays a scripted list of responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, url, headers, body, timeout):
        del timeout
        self.requests.append(
            {"url": url, "headers": headers, "body": json.loads(body.decode())}
        )
        # A response is (status, payload) or (status, payload, headers), which
        # is how the transport hands Retry-After back without raising.
        return self.responses.pop(0) if self.responses else reply("ok")


@pytest.fixture
def no_sleep(monkeypatch):
    """Backoff is real seconds; the test should not spend them."""
    slept: list[float] = []
    monkeypatch.setattr(translate.time, "sleep", slept.append)
    return slept


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------
def test_local_provider_needs_no_key():
    """The point of the local preset: nothing to sign up for.

    llama.cpp is already a hard requirement of this project -- marker 2.0
    spawns llama-server itself -- so the binary is present on any machine that
    can run the OCR at all.
    """
    provider = resolve_provider("local")
    assert provider.key_env is None
    assert provider.base_url.startswith("http://")


def test_every_provider_has_a_default_model():
    """A preset that still needs a model id is not a preset."""
    for name, provider in PROVIDERS.items():
        assert provider.model, f"{name} has no default model"
        assert provider.base_url.endswith("/v1") or provider.base_url.endswith("/")


def test_overrides_beat_the_preset():
    provider = resolve_provider("openrouter", model="some/other:free", base_url="http://x/v1")
    assert provider.model == "some/other:free"
    assert provider.base_url == "http://x/v1"


def test_unknown_provider_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="local"):
        resolve_provider("nonesuch")


def test_a_hosted_provider_without_its_key_fails_before_the_run(monkeypatch):
    """Fail at construction, not on page 1 of 582.

    The failure is identical either way, but one costs a launch and a look at
    the log to understand.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(TranslationError, match="OPENROUTER_API_KEY"):
        OpenAICompatTranslator(resolve_provider("openrouter"))


# --------------------------------------------------------------------------
# the request
# --------------------------------------------------------------------------
def test_the_page_and_the_instructions_both_reach_the_model():
    transport = FakeTransport(reply("una página"))
    tr = OpenAICompatTranslator(resolve_provider("local"), transport=transport)

    assert tr("a page of text", "English") == "una página"

    sent = transport.requests[0]
    assert sent["url"].endswith("/chat/completions")
    roles = {m["role"]: m["content"] for m in sent["body"]["messages"]}
    assert "English" in roles["system"]
    assert roles["user"] == "a page of text"
    assert sent["body"]["model"] == PROVIDERS["local"].model


def test_no_authorization_header_without_a_key():
    """A local server is not asked for credentials it does not want."""
    transport = FakeTransport(reply("x"))
    tr = OpenAICompatTranslator(resolve_provider("local"), transport=transport)
    tr("page", "English")
    assert "Authorization" not in transport.requests[0]["headers"]


def test_the_key_is_sent_as_a_bearer_token(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    transport = FakeTransport(reply("x"))
    tr = OpenAICompatTranslator(resolve_provider("openrouter"), transport=transport)
    tr("page", "English")
    assert transport.requests[0]["headers"]["Authorization"] == "Bearer sk-test-123"


def test_requests_carry_a_real_user_agent():
    """Groq sits behind Cloudflare, which bans urllib's default agent.

    With no User-Agent set, urllib sends ``Python-urllib/3.x`` and Cloudflare
    refuses the request at the edge with ``HTTP 403: error code: 1010`` -- a
    body that never came from the API, so neither the key nor the model id was
    ever looked at. Reported as a run that "stopped early", it reads exactly
    like a retired model id, which is what it was first diagnosed as.
    """
    transport = FakeTransport(reply("x"))
    tr = OpenAICompatTranslator(resolve_provider("local"), transport=transport)
    tr("page", "English")

    agent = transport.requests[0]["headers"].get("User-Agent", "")
    assert agent, "no User-Agent set: urllib supplies a banned one"
    assert "urllib" not in agent.lower()


def test_an_edge_block_is_named_rather_than_reported_as_a_bad_request(no_sleep):
    """``error code: 1010`` is Cloudflare's, not the provider's.

    The provider's own refusals are JSON (``model_not_found``,
    ``invalid_api_key``); a bare ``error code: NNNN`` is the CDN in front of it
    and means the request never arrived. Saying only "HTTP 403" sends the
    reader to check their key and their model id, neither of which is wrong.
    """
    transport = FakeTransport((403, b"error code: 1010"))
    tr = OpenAICompatTranslator(resolve_provider("local"), transport=transport, rpm=0)

    with pytest.raises(TranslationError, match="(?i)cloudflare"):
        tr("page", "English")
    assert len(transport.requests) == 1  # not retried: it fails identically


def test_a_retired_model_id_says_how_to_find_a_live_one(no_sleep):
    """The other half of the 1010 incident: the preset id was also dead.

    `llama-3.3-70b-versatile` was groq's default here and had been retired;
    the CDN block hid that until it was fixed. A free tier retires ids on its
    own schedule, so the message names the endpoint that lists the current
    ones rather than leaving the reader to search for a changelog.
    """
    body = b'{"error":{"message":"The model does not exist","code":"model_not_found"}}'
    transport = FakeTransport((404, body))
    tr = OpenAICompatTranslator(resolve_provider("local"), transport=transport, rpm=0)

    with pytest.raises(TranslationError) as exc:
        tr("page", "English")
    assert "/models" in str(exc.value)  # where the live ids are listed
    assert "--model" in str(exc.value)  # and how to pass one
    assert len(transport.requests) == 1


# --------------------------------------------------------------------------
# the response, before it is trusted
# --------------------------------------------------------------------------
def test_a_truncated_answer_is_an_error_not_a_page():
    """finish_reason 'length' means the page was cut off mid-sentence.

    It reads exactly like a page that ended there, so nothing downstream can
    tell. Free-tier models often have small output caps, which makes this the
    likeliest silent failure of the whole stage.
    """
    transport = FakeTransport(reply("half a transl", finish="length"))
    tr = OpenAICompatTranslator(resolve_provider("local"), transport=transport)
    with pytest.raises(TranslationError, match="truncat"):
        tr("page", "English")


def test_an_empty_answer_is_an_error():
    transport = FakeTransport(reply("   "))
    tr = OpenAICompatTranslator(resolve_provider("local"), transport=transport)
    with pytest.raises(TranslationError, match="empty"):
        tr("page", "English")


def test_an_unparseable_body_says_so():
    transport = FakeTransport((200, b"<html>gateway</html>"))
    tr = OpenAICompatTranslator(resolve_provider("local"), transport=transport)
    with pytest.raises(TranslationError, match="not JSON"):
        tr("page", "English")


# --------------------------------------------------------------------------
# rate limits -- the defining constraint of a free tier
# --------------------------------------------------------------------------
def test_a_rate_limit_is_waited_out_not_dropped(no_sleep):
    transport = FakeTransport(
        (429, b'{"error": "slow down"}'),
        (429, b'{"error": "slow down"}'),
        reply("done"),
    )
    tr = OpenAICompatTranslator(
        resolve_provider("local"), transport=transport, rpm=0
    )
    assert tr("page", "English") == "done"
    assert len(transport.requests) == 3
    assert no_sleep and no_sleep[1] > no_sleep[0]  # backs off further each time


def test_retry_after_is_honoured_over_the_backoff(no_sleep):
    """Free tiers say how long to wait; guessing longer wastes the window."""
    transport = FakeTransport(
        (429, b"", {"Retry-After": "7"}),
        reply("done"),
    )
    tr = OpenAICompatTranslator(resolve_provider("local"), transport=transport, rpm=0)
    tr("page", "English")
    assert no_sleep[0] == pytest.approx(7)


def test_a_persistent_rate_limit_eventually_stops_the_run(no_sleep):
    """Giving up must be loud: the chunk is not written and the run stops."""
    transport = FakeTransport(*[(429, b"") for _ in range(10)])
    tr = OpenAICompatTranslator(
        resolve_provider("local"), transport=transport, rpm=0, max_retries=3
    )
    with pytest.raises(TranslationError, match="429"):
        tr("page", "English")
    assert len(transport.requests) == 4  # the first try plus three retries


def test_a_4xx_that_is_not_a_rate_limit_is_not_retried(no_sleep):
    """A bad model id fails the same way 582 times; retrying just delays it."""
    transport = FakeTransport((400, b'{"error": "unknown model"}'), reply("x"))
    tr = OpenAICompatTranslator(resolve_provider("local"), transport=transport, rpm=0)
    with pytest.raises(TranslationError, match="400"):
        tr("page", "English")
    assert len(transport.requests) == 1


def test_requests_are_paced_to_stay_under_the_limit(no_sleep):
    """Better to wait 3s by choice than to be refused and wait 60s.

    A free tier of 20 requests a minute is one every 3 seconds; sending faster
    only converts into 429s and backoff.
    """
    transport = FakeTransport(reply("a"), reply("b"))
    tr = OpenAICompatTranslator(resolve_provider("local"), transport=transport, rpm=20)
    tr("page one", "English")
    tr("page two", "English")
    assert any(s > 0 for s in no_sleep)


def test_pacing_is_off_for_a_local_server(no_sleep):
    """Nothing to rate limit: the only cost is the machine's own time."""
    assert PROVIDERS["local"].rpm == 0
    transport = FakeTransport(reply("a"), reply("b"))
    tr = OpenAICompatTranslator(resolve_provider("local"), transport=transport)
    tr("one", "English")
    tr("two", "English")
    assert not any(s > 0 for s in no_sleep)


def test_token_usage_is_accumulated():
    transport = FakeTransport(reply("a"), reply("b"))
    tr = OpenAICompatTranslator(resolve_provider("local"), transport=transport)
    tr("one", "English")
    tr("two", "English")
    assert tr.input_tokens == 20 and tr.output_tokens == 40
