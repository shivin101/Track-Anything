from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_CHECKPOINTS_PATH = Path(__file__).parent / "checkpoints"


@dataclass
class TrackAnythingCheckpointDetails:
    path: Path
    """
    The path to the checkpoint.
    """
    model_resolution: int
    """
    The image resolution the model performs inference with.
    """


class TrackAnythingCheckpoint(Enum):
    """
    Represents a checkpoint for TrackAnything and contains its path.
    """

    # Ref: .project.bc/.source/config.yaml

    BASE = TrackAnythingCheckpointDetails(
        path=_CHECKPOINTS_PATH / "inspyrenet-base.pth",
        model_resolution=1024,
    )
    """
    The base checkpoint.
    """

    FAST = TrackAnythingCheckpointDetails(
        path=_CHECKPOINTS_PATH / "inspyrenet-fast.pth",
        model_resolution=384,
    )
    """
    A checkpoint for faster inference time and lower memory usage, but lower accuracy.
    """

    NIGHTLY = TrackAnythingCheckpointDetails(
        path=_CHECKPOINTS_PATH / "inspyrenet-nightly.pth",
        model_resolution=1024,
    )
    """
    The latest checkpoint, which may be unstable.
    """