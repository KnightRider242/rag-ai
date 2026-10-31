"""Focused contract for one structured local generation request."""

from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class GenerationConfig(BaseModel):
    num_ctx: int = Field(default=4096, ge=2048, le=8192)
    num_predict: int = Field(default=512, ge=64, le=1024)
    temperature: float = Field(default=0, ge=0, le=1)
    max_response_bytes: int = Field(default=32768, ge=4096, le=131072)


class GeneratedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=600)
    source_ids: list[str] = Field(min_length=1, max_length=5)


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "insufficient_evidence"]
    claims: list[GeneratedClaim] = Field(max_length=5)

    @model_validator(mode="after")
    def validate_status(self) -> "GeneratedAnswer":
        if self.status == "answered" and not self.claims:
            raise ValueError("An answered response requires claims.")
        if self.status == "insufficient_evidence" and self.claims:
            raise ValueError("An abstention cannot contain claims.")
        return self


class _AnsweredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["answered"]
    claims: list[GeneratedClaim] = Field(min_length=1, max_length=5)


class _InsufficientOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["insufficient_evidence"]
    claims: list[GeneratedClaim] = Field(max_length=0)


GENERATION_OUTPUT_ADAPTER = TypeAdapter(
    Annotated[_AnsweredOutput | _InsufficientOutput, Field(discriminator="status")]
)


class GenerationProvider(Protocol):
    def generate(
        self, *, system_prompt: str, user_prompt: str, config: GenerationConfig,
    ) -> GeneratedAnswer: ...


class GenerationError(RuntimeError):
    """Local generation failed or returned an invalid response."""
