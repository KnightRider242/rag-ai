"""Small typed contract for local visual interpretation."""

from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

ShortObservation = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=240)]


VisionFailureCategory = Literal[
    "configuration", "generation_limit", "timeout", "transport",
    "http_status", "invalid_envelope", "invalid_output", "resource_limit",
]


class VisionError(ValueError):
    """Sanitized local vision failure with a safe diagnostic category."""

    def __init__(self, message: str, *, category: VisionFailureCategory = "invalid_envelope") -> None:
        super().__init__(message)
        self.category = category


class VisionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=1200)
    observations: list[ShortObservation] = Field(default_factory=list, max_length=8)
    limitations: list[ShortObservation] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def require_description(self) -> "VisionOutput":
        if not self.description.strip():
            raise ValueError("Visual description must not be blank.")
        return self


class VisionProvider(Protocol):
    model_name: str

    def analyze(self, *, image: bytes, task_prompt: str) -> VisionOutput: ...

    def close(self) -> None: ...
