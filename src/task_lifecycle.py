"""Small, UI-independent contracts for asynchronous task result admission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import uuid


@dataclass
class TaskToken:
    """Identity captured when a background task starts.

    A worker may finish after the user has changed the project, model, or
    upstream artifact.  The token lets the GUI reject that late result without
    inspecting the worker implementation.
    """

    task_type: str
    project_generation: int
    source_sha256: str = ""
    model_sha256: str = ""
    input_fingerprint: str = ""
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    invalidated: bool = False

    def invalidate(self) -> None:
        self.invalidated = True

    def belongs_to(self, state: Any) -> bool:
        """Check ownership for cleanup, without admitting a result.

        Cleanup may clear a running flag only in the project that started the
        task.  ``invalidated`` is intentionally ignored here; it is an
        admission-only guard and must never allow a cancelled task to commit.
        """
        return (
            int(getattr(state, "project_generation", 0) or 0) == int(self.project_generation)
            and (
                not self.source_sha256
                or str(getattr(state, "source_sha256", "") or "") == self.source_sha256
            )
            and (
                not self.model_sha256
                or str(getattr(state, "model_sha256", "") or "") == self.model_sha256
            )
        )

    def matches(self, state: Any, *, input_fingerprint: Optional[str] = None) -> bool:
        if self.invalidated:
            return False
        if int(getattr(state, "project_generation", 0) or 0) != int(
            self.project_generation
        ):
            return False
        if self.source_sha256 and str(getattr(state, "source_sha256", "") or "") != self.source_sha256:
            return False
        if self.model_sha256 and str(getattr(state, "model_sha256", "") or "") != self.model_sha256:
            return False
        if input_fingerprint is not None and self.input_fingerprint != input_fingerprint:
            return False
        return True


def state_identity(state: Any) -> Dict[str, str | int]:
    """Return the current project identity used by task tokens."""
    return {
        "project_generation": int(getattr(state, "project_generation", 0) or 0),
        "source_sha256": str(getattr(state, "source_sha256", "") or ""),
        "model_sha256": str(getattr(state, "model_sha256", "") or ""),
    }


def make_task_token(state: Any, task_type: str, *, input_fingerprint: str = "") -> TaskToken:
    identity = state_identity(state)
    return TaskToken(task_type=task_type, input_fingerprint=input_fingerprint, **identity)


def result_is_admissible(
    token: Optional[TaskToken],
    state: Any,
    *,
    input_fingerprint: Optional[str] = None,
) -> bool:
    """Fail closed when a worker has no token or no longer matches state."""
    return bool(token is not None and token.matches(state, input_fingerprint=input_fingerprint))


def admit_task_result(
    token: Optional[TaskToken],
    state: Any,
    *,
    input_fingerprint: Optional[str] = None,
    require_token: bool = True,
) -> bool:
    """Unified result-admission gate for every asynchronous task."""
    if token is None:
        return not require_token
    return result_is_admissible(token, state, input_fingerprint=input_fingerprint)

