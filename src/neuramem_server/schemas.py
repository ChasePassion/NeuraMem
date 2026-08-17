"""Request/response contracts — identical to the legacy /v1/* API.

user_id is validated against ^[A-Za-z0-9_-]{1,64}$ at the schema level
(architecture_target.md #16): it flows into store filters, so a hostile
value must fail with 422 instead of breaking expressions downstream.
"""

import re
from typing import List, Optional

from pydantic import BaseModel, field_validator

USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _validate_user_id(value):
    if value is None:  # Optional user_id (consolidate-all)
        return value
    if not USER_ID_PATTERN.match(value):
        raise ValueError(
            "user_id must match ^[A-Za-z0-9_-]{1,64}$"
        )
    return value


# -- chat -------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user_id: str
    chat_id: str = ""
    message: str
    history: List[ChatMessage] = []

    _user_id = field_validator("user_id")(_validate_user_id)


# -- memories -----------------------------------------------------------------


class ManageRequest(BaseModel):
    user_id: str
    chat_id: str = ""
    user_text: str
    assistant_text: str

    _user_id = field_validator("user_id")(_validate_user_id)


class ManageResponse(BaseModel):
    added_ids: List[int]
    success: bool = True


class SearchRequest(BaseModel):
    user_id: str
    query: str

    _user_id = field_validator("user_id")(_validate_user_id)


class MemoryResponse(BaseModel):
    id: int
    user_id: str
    memory_type: str
    ts: int
    chat_id: str
    text: str
    group_id: int = -1


class SearchResponse(BaseModel):
    episodic: List[MemoryResponse]
    semantic: List[MemoryResponse]


class DeleteResponse(BaseModel):
    success: bool
    deleted_count: int = 0


class ResetRequest(BaseModel):
    user_id: str

    _user_id = field_validator("user_id")(_validate_user_id)


class ResetResponse(BaseModel):
    success: bool
    deleted_count: int


class ConsolidateRequest(BaseModel):
    user_id: Optional[str] = None

    _user_id = field_validator("user_id")(_validate_user_id)


class ConsolidateResponse(BaseModel):
    memories_processed: int
    semantic_created: int


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
