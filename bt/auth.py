"""Naive per-account login, for the public deployment only.

This is deliberately not real authentication. Signup asks for an email but
never verifies it belongs to the signer, and "forgot your code" reissues a
fresh one to anyone who can name an existing username + email pair -- there
is no inbox in the loop at any point, by design (see README "Deploy your own
copy"). What this buys, on a single shared free container, is not security:
it is (1) a private upload/output folder per visitor, so one person's PDF
does not collide with or leak to another's, and (2) a place to hang the
per-account page cap so one visitor cannot monopolise the only CPU this app
has. Local, personal use of ``app.py`` never imports this module.

Accounts are stored in ``output/accounts.json``, next to ``queue.json`` --
same convention, same lack of file locking (a lost signup under a genuine
race is an acceptable, retry-and-it-works failure mode here, not a silent
data-corruption one). On the free Hugging Face Spaces tier this file does
NOT survive a sleep/restart cycle, since only the Docker image itself is
persistent there -- accounts are only as durable as one container's uptime.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "output"
ACCOUNTS_FILE = OUTPUT_ROOT / "accounts.json"

CODE_LENGTH = 6
# Bounds per-document OCR cost on the shared free CPU container -- see
# README "Deploy your own copy" for why this exists and why a document over
# the cap is rejected outright rather than truncated to the first N pages.
MAX_PAGES_PER_DOCUMENT = 5

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_username(username: str) -> str:
    return username.strip().lower()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _hash_code(code: str, salt: str) -> str:
    return hashlib.sha256((salt + code).encode("utf-8")).hexdigest()


def _generate_code() -> str:
    return f"{secrets.randbelow(10**CODE_LENGTH):0{CODE_LENGTH}d}"


def _load() -> dict:
    if not ACCOUNTS_FILE.exists():
        return {}
    try:
        return json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _save(data: dict) -> None:
    ACCOUNTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ACCOUNTS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(ACCOUNTS_FILE)


def valid_username(username: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z0-9_-]{3,32}", username.strip()))


def valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email.strip()))


def signup(username: str, email: str) -> str:
    """Create a new account and return its one-time code.

    Raises ValueError (with a message safe to show the user) if the username
    or email fails basic validation or the username is already taken.
    """
    username = normalize_username(username)
    email = normalize_email(email)
    if not valid_username(username):
        raise ValueError(
            "Username must be 3-32 characters: letters, digits, - or _."
        )
    if not valid_email(email):
        raise ValueError("That doesn't look like an email address.")

    data = _load()
    if username in data:
        raise ValueError("That username is already taken.")

    code = _generate_code()
    salt = secrets.token_hex(8)
    data[username] = {"email": email, "code_hash": _hash_code(code, salt), "salt": salt}
    _save(data)
    return code


def login(username: str, code: str) -> bool:
    """True if `code` is the current one-time code for `username`."""
    entry = _load().get(normalize_username(username))
    if not entry:
        return False
    return secrets.compare_digest(_hash_code(code.strip(), entry["salt"]), entry["code_hash"])


def reset_code(username: str, email: str) -> str | None:
    """Issue a fresh code if `username` + `email` match an existing account.

    The old code stops working immediately. Returns None (rather than
    raising) on any mismatch, so callers can show one generic "no matching
    account" message without distinguishing a bad username from a bad email.
    """
    username = normalize_username(username)
    email = normalize_email(email)
    data = _load()
    entry = data.get(username)
    if not entry or entry["email"] != email:
        return None

    code = _generate_code()
    salt = secrets.token_hex(8)
    entry["code_hash"] = _hash_code(code, salt)
    entry["salt"] = salt
    _save(data)
    return code


def account_dir(username: str, root: Path | None = None) -> Path:
    """Where this account's uploads live -- also the namespace for its output.

    ``root`` defaults to this module's ``ROOT`` looked up at call time, not
    baked into the signature -- a default of ``root: Path = ROOT`` would bind
    the real project root once at import time, so tests monkeypatching
    ``auth.ROOT`` to a tmp dir would be silently ignored and app.py's public
    mode would write into this repo's actual uploads/ during a test run.
    """
    return (root if root is not None else ROOT) / "uploads" / normalize_username(username)
