try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = version("chatterbox-tts")


from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .audio_utils import (
    load_and_condition_reference,
    load_source_audio,
    loudness_normalise,
    trim_silence,
)

__all__ = [
    "ChatterboxTTS",
    "ChatterboxVC",
    "load_and_condition_reference",
    "load_source_audio",
    "loudness_normalise",
    "trim_silence",
]
