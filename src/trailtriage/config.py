from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Input(Section):
    recursive: bool = True
    extensions: list[str] = ["jpg", "jpeg", "png", "webp"]


class Detection(Section):
    model: str = "~/.local/share/trailtriage/models/yolo11s.pt"
    device: str = "auto"
    confidence_threshold: float = Field(0.15, gt=0, le=1)
    batch_size: int | str = "auto"

    @model_validator(mode="after")
    def valid_batch(self):
        if self.batch_size != "auto" and (
            not isinstance(self.batch_size, int) or not 1 <= self.batch_size <= 256
        ):
            raise ValueError("batch_size must be auto or an integer between 1 and 256")
        return self


class Interest(Section):
    human_weight: float = Field(100, ge=0)
    vehicle_weight: float = Field(70, ge=0)
    animal_weight: float = Field(60, ge=0)
    unknown_weight: float = Field(30, ge=0)
    interesting_threshold: float = Field(60, ge=0)
    uncertain_threshold: float = Field(20, ge=0)
    category_groups: dict[str, list[str]] = {
        "humans": ["person"],
        "vehicles": ["car", "truck", "bus", "motorcycle", "bicycle"],
        "animals": ["dog", "cat", "bird", "horse", "cow", "sheep", "bear", "deer"],
    }

    @model_validator(mode="after")
    def valid_thresholds(self):
        if set(self.category_groups) - {"humans", "vehicles", "animals"}:
            raise ValueError("Category group names must be humans, vehicles or animals")
        if self.uncertain_threshold > self.interesting_threshold:
            raise ValueError("uncertain_threshold must not exceed interesting_threshold")
        return self


class Bursts(Section):
    enabled: bool = True
    max_gap_seconds: float = Field(10, ge=0)


class Deduplication(Section):
    enabled: bool = True
    phash_distance: int = Field(6, ge=0, le=64)
    representatives_per_cluster: int = Field(1, ge=1)
    comparison_window: int = Field(16, ge=1, le=256)


class Output(Section):
    mode: str = "symlink"
    generate_gallery: bool = True

    @model_validator(mode="after")
    def valid_mode(self):
        if self.mode not in {"symlink", "copy"}:
            raise ValueError("output.mode must be symlink or copy")
        return self


class Config(Section):
    input: Input = Field(default_factory=Input)
    detection: Detection = Field(default_factory=Detection)
    interest: Interest = Field(default_factory=Interest)
    bursts: Bursts = Field(default_factory=Bursts)
    deduplication: Deduplication = Field(default_factory=Deduplication)
    output: Output = Field(default_factory=Output)


def load_config(path: Path | None = None) -> Config:
    try:
        return Config.model_validate(yaml.safe_load(path.read_text()) or {}) if path else Config()
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {exc}") from exc
