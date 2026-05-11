from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


HAND_PIECES = ("HI", "KA", "KI", "GI", "KE", "KY", "FU")
COLORS = ("black", "white")


def empty_hands() -> dict[str, dict[str, int]]:
    return {color: {piece: 0 for piece in HAND_PIECES} for color in COLORS}


@dataclass(frozen=True)
class RecognitionOptions:
    model_path: Path | None = None
    screenshots_dir: Path | None = None
    labels_dir: Path | None = None
    calibration_dir: Path | None = None
    include_hands: bool = False
    backend: str = "auto"
    train_if_missing: bool = True
    exclude_sample: str | None = None
    out_dir: Path | None = None


@dataclass(frozen=True)
class RecognitionResult:
    image: str
    board: list[list[str]]
    hands: dict[str, dict[str, int]]
    confidence: list[dict[str, Any]]
    raw_report: dict[str, Any]
    output_path: str | None = None
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "image": self.image,
            "board": self.board,
            "hands": self.hands,
            "confidence": self.confidence,
            "raw_report": self.raw_report,
            "output_path": self.output_path,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecognitionResult":
        return cls(
            image=str(data.get("image") or ""),
            board=list(data.get("board") or []),
            hands=dict(data.get("hands") or empty_hands()),
            confidence=list(data.get("confidence") or []),
            raw_report=dict(data.get("raw_report") or {}),
            output_path=data.get("output_path"),
            needs_review=bool(data.get("needs_review", False)),
            review_reasons=list(data.get("review_reasons") or []),
        )
