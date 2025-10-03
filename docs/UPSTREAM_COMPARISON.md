# Upstream Comparison Snapshot

This note captures the main differences between our working branch (`work`) and the current `resemble-ai/chatterbox` `master` branch as of 2025-10-01. It highlights upstream additions that we should consider porting, along with fork-specific functionality that we should preserve.

## Divergence at a Glance

- **Fork-first additions.** Our branch ships a guided voice cloning Gradio app, shared audio-conditioning helpers, and a LoRA fine-tuning script to support cloning workflows end to end. These are not present upstream and remain differentiators we should keep while merging other changes.
- **Upstream leap to multilingual.** The upstream repository has moved to a multilingual release (`v0.1.4`) with 23 supported languages, a dedicated multilingual inference class, and new UI/packaging assets to showcase the expanded coverage.

## Upstream Enhancements Worth Porting

### Multilingual inference stack

The upstream release introduces `ChatterboxMultilingualTTS`, which wraps multilingual checkpoints, a language-aware tokenizer, and pre-baked conditional embeddings. The class exposes the same conditioning interface as the English model while adding a `language_id` switch and built-in support for 23 locale codes:

```
from .models.tokenizers import MTLTokenizer
...
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  ...
  "tr": "Turkish",
  "zh": "Chinese",
}
...
class ChatterboxMultilingualTTS:
    def generate(self, text: str, language_id: str = "en", ...):
        ...
```

Porting this module (and its checkpoint loader) would let our fork ship the multilingual weights alongside the existing English flow.

### Multilingual Gradio workflow

Upstream now includes `multilingual_app.py`, a Gradio Blocks UI that mirrors our voice-cloning helper but layers in language presets, default reference clips, and helper copy for each locale. This script provides a ready-made UX to demonstrate multilingual zero-shot cloning without having to juggle manual prompt selection.

### Tokenizer and dependency updates

The `pyproject.toml` upstream was bumped to `v0.1.4`, pinning new text processing dependencies (`spacy-pkuseg`, `pykakasi`, `russian-text-stresser`) and aligning the runtime to Python 3.10+. The tokenizer module gained multilingual-aware logic to service the new model. We should align our package metadata and dependency set when ingesting the multilingual changes to avoid runtime mismatches.

### Model configuration tweaks

Supporting the multilingual checkpoints required updates in the `T3` config, alignment stream analyzer, and flow utilities to accept wider vocabularies and different sample rates. These structural tweaks arrive alongside the new tokenizer and should be merged to keep our inference stack compatible with future upstream releases.

## Fork Functionality to Preserve

- **Guided voice clone UI.** `gradio_voice_clone_app.py` delivers a novice-friendly workflow for recording/uploading reference audio, tuning inference knobs, and exporting cloned speech in one panel.
- **Shared conditioning utilities.** `src/chatterbox/audio_utils.py` centralises loudness normalisation, silence trimming, and multi-file prompt handling used by both TTS and VC pipelines.
- **LoRA experimentation script.** `scripts/train_lora.py` gives power users an entry point for continuing training on custom datasets, which remains absent upstream.

## Suggested Integration Path

1. **Sync core libraries.** Mirror upstream changes in the tokenizer, model configs, and new multilingual modules before reconciling app-layer code. This ensures we adopt the architecture required for the new checkpoints.
2. **Reconcile apps.** Merge upstream's `multilingual_app.py` alongside our existing voice cloning Blocks UI, sharing common helper components where practical.
3. **Update packaging.** Bump our `pyproject.toml` version and dependency pins in step with upstream once the multilingual runtime is incorporated.
4. **Regression test cloning workflows.** After merging upstream code, validate that our conditioning utilities and Gradio helpers still behave correctly with both English and multilingual checkpoints.
