"""Data models for the json2lean pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExerciseStatus(Enum):
    PENDING = "pending"
    PREPROCESSED = "preprocessed"
    TRANSLATED = "translated"
    VALID = "valid"
    REPAIR_FAILED = "repair_failed"
    ERROR = "error"


@dataclass
class Exercise:
    """One exercise extracted from the input JSON."""

    raw: Dict[str, Any]
    index: int
    label: str = ""
    problem: str = ""
    status: ExerciseStatus = ExerciseStatus.PENDING

    # Populated after preprocessing
    preprocessed_problem: str = ""

    # Populated after translation
    lean_code: str = ""

    # Populated after validation
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    compile_returncode: int = -1

    # Tracking
    repair_attempts: int = 0

    def __post_init__(self) -> None:
        if not self.label:
            self.label = str(
                self.raw.get("source_idx")
                or self.raw.get("index")
                or self.index
            )
        if not self.problem:
            self.problem = self.raw.get("problem", "")

    @property
    def is_valid(self) -> bool:
        return self.compile_returncode == 0 and len(self.errors) == 0


@dataclass
class CompileResult:
    """Result of compiling a single Lean file."""

    filename: str
    stdout: str
    returncode: int
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TokenUsage:
    """Token usage for a single API call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_type: str = ""  # "preprocess", "translate", "recover"
    exercise_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_type": self.call_type,
            "exercise_label": self.exercise_label,
        }


@dataclass
class PipelineConfig:
    """Parsed configuration for the pipeline."""

    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 180.0

    # Preprocessing
    preprocessing_enabled: bool = True
    preprocessing_max_tokens: int = 4096
    preprocessing_max_attempts: int = 8

    # Translation
    translation_max_tokens: int = 4096
    translation_max_attempts: int = 3

    # Recovery
    recovery_max_tokens: int = 4096
    recovery_max_retries: int = 5

    # Lean
    lean_toolchain_dir: str = "lean"
    lean_timeout_seconds: int = 120

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        pre = data.get("preprocessing", {})
        trans = data.get("translation", {})
        rec = data.get("recovery", {})
        lean = data.get("lean", {})

        return cls(
            api_key=data["api_key"],
            base_url=data["base_url"],
            model=data["model"],
            timeout_seconds=data.get("timeout_seconds", 180.0),
            preprocessing_enabled=pre.get("enabled", True),
            preprocessing_max_tokens=pre.get("max_tokens", 4096),
            preprocessing_max_attempts=pre.get("max_attempts", 8),
            translation_max_tokens=trans.get("max_tokens", 4096),
            translation_max_attempts=trans.get("max_attempts", 3),
            recovery_max_tokens=rec.get("max_tokens", 4096),
            recovery_max_retries=rec.get("max_retries", 5),
            lean_toolchain_dir=lean.get("toolchain_dir", "lean"),
            lean_timeout_seconds=lean.get("timeout_seconds", 120),
        )
