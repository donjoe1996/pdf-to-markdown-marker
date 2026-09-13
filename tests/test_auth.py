"""bt.auth: naive per-account login for the public deployment.

Not testing "is this secure" -- it deliberately isn't (see the module
docstring). These pin the behaviour the login UI depends on: a code works
exactly once it's issued, a wrong code or unknown user is rejected, and
resetting requires the matching email, not just the username.
"""

from __future__ import annotations

import pytest

from bt import auth


def test_signup_returns_a_working_code():
    code = auth.signup("alice", "alice@example.com")
    assert len(code) == auth.CODE_LENGTH
    assert auth.login("alice", code)


def test_login_rejects_wrong_code():
    auth.signup("alice", "alice@example.com")
    assert not auth.login("alice", "000000")


def test_login_rejects_unknown_username():
    assert not auth.login("nobody", "123456")


def test_username_and_email_are_case_and_space_insensitive():
    code = auth.signup("  Alice  ", "  Alice@Example.com  ")
    assert auth.login("ALICE", code)
    # reset must match the same normalized email
    new_code = auth.reset_code("alice", "alice@example.com")
    assert new_code is not None
    assert auth.login("alice", new_code)


def test_signup_rejects_duplicate_username():
    auth.signup("alice", "alice@example.com")
    with pytest.raises(ValueError, match="already taken"):
        auth.signup("alice", "someone-else@example.com")


@pytest.mark.parametrize(
    "username,email",
    [
        ("ab", "alice@example.com"),  # too short
        ("alice!!", "alice@example.com"),  # bad characters
        ("alice", "not-an-email"),
    ],
)
def test_signup_rejects_invalid_input(username, email):
    with pytest.raises(ValueError):
        auth.signup(username, email)


def test_reset_code_requires_matching_email():
    auth.signup("alice", "alice@example.com")
    assert auth.reset_code("alice", "wrong@example.com") is None
    assert auth.reset_code("nobody", "alice@example.com") is None


def test_reset_code_invalidates_the_old_code():
    old_code = auth.signup("alice", "alice@example.com")
    new_code = auth.reset_code("alice", "alice@example.com")
    assert new_code is not None
    assert new_code != old_code
    assert not auth.login("alice", old_code)
    assert auth.login("alice", new_code)


def test_account_dir_is_namespaced_and_normalized(tmp_path):
    d1 = auth.account_dir("Alice", root=tmp_path)
    d2 = auth.account_dir("alice", root=tmp_path)
    assert d1 == d2 == tmp_path / "uploads" / "alice"
