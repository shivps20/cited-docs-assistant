"""Local users and their access groups, read from users.yaml.

The API identifies a user only by name (X-KB-User header); the access groups always come from
this file, never from the client. Every user also belongs to the public group "all".
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from kb.retrieve.search import PUBLIC_GROUP

_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class UsersError(Exception):
    """users.yaml is missing or invalid; the message lists every problem."""

    def __init__(self, errors: list[str]):
        """Keep the list of problems and build a message listing them all."""
        super().__init__(f"{len(errors)} users.yaml error(s):\n" + "\n".join(errors))
        self.errors = errors


class UnknownUser(Exception):
    """The requested user is not in users.yaml."""


@dataclass(frozen=True)
class User:
    """One configured user: id, display name and access groups (without the public group)."""

    user_id: str
    name: str
    groups: tuple[str, ...] = ()

    @property
    def search_groups(self) -> list[str]:
        """Groups used in retrieval filters: the user's groups plus the public group."""
        return sorted({PUBLIC_GROUP, *self.groups})


@dataclass
class UserDirectory:
    """All configured users and the one used when a request names none."""

    users: dict[str, User] = field(default_factory=dict)
    default_user: str = ""

    def resolve(self, user_id: str | None) -> User:
        """The named user, or the default user when no name is given; UnknownUser otherwise."""
        key = (user_id or "").strip() or self.default_user
        if key not in self.users:
            raise UnknownUser(f"unknown user {key!r}; known users: {', '.join(sorted(self.users))}")
        return self.users[key]


def load_users(path: Path) -> UserDirectory:
    """Read and validate users.yaml; raises UsersError listing every problem found."""
    if not path.exists():
        raise UsersError([f"{path} not found"])
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise UsersError([f"{path.name} is not valid YAML: {e}"]) from e

    errors: list[str] = []
    raw_users = data.get("users") if isinstance(data, dict) else None
    if not isinstance(raw_users, dict) or not raw_users:
        raise UsersError(["'users' must be a mapping with at least one user"])

    users: dict[str, User] = {}
    for user_id, spec in raw_users.items():
        user_id = str(user_id)
        if not _ID.match(user_id):
            errors.append(f"user id {user_id!r} must be lowercase letters, digits, '-' or '_'")
            continue
        spec = spec or {}
        if not isinstance(spec, dict):
            errors.append(f"user {user_id!r}: expected a mapping with 'name' and 'groups'")
            continue
        groups = spec.get("groups") or []
        if not isinstance(groups, list) or not all(isinstance(g, str) and _ID.match(g) for g in groups):
            errors.append(f"user {user_id!r}: 'groups' must be a list of group names (lowercase, '-', '_')")
            continue
        users[user_id] = User(user_id, str(spec.get("name") or user_id),
                              tuple(sorted({g for g in groups if g != PUBLIC_GROUP})))

    default = str(data.get("default_user") or "")
    if default not in users:
        errors.append(f"default_user {default!r} is not one of the users")
    if errors:
        raise UsersError(errors)
    return UserDirectory(users, default)
