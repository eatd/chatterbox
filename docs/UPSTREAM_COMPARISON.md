# Upstream Comparison Snapshot

This note captures the main differences between our working branch (`work`) and the current `resemble-ai/chatterbox` `master` branch as of 2025-10-01. It highlights upstream additions we are intentionally deferring, along with fork-specific functionality that we should preserve.

## Divergence at a Glance

- **Fork-first additions.** Our branch ships a guided voice cloning Gradio app, shared audio-conditioning helpers, and a LoRA fine-tuning script to support cloning workflows end to end. These are not present upstream and remain differentiators we should keep while merging other changes.
- **Upstream leap to multilingual.** The upstream repository has moved to a multilingual release (`v0.1.4`) with 23 supported languages, a dedicated multilingual inference class, and new UI/packaging assets to showcase the expanded coverage, but we are deferring these additions to keep the fork English-only and dependency-light.

## Upstream Enhancements We Are Deferring

### Multilingual inference stack (out of scope)

The upstream branch ships a `ChatterboxMultilingualTTS` implementation, tokenizer updates, and pre-baked multilingual checkpoints. Shipping those assets would force us to depend on the `spacy` ecosystem (`spacy-pkuseg`, `pykakasi`, `russian-text-stresser`) whose Typer pin (`<0.10`) conflicts with the Typer version required by Gradio 5.x. Because our fork is intentionally English-only, we will not chase the multilingual modules or their dependencies for now.

### Multilingual Gradio workflow (not planned)

Upstream's `multilingual_app.py` Blocks UI demonstrates the new locale options. Pulling it across would drag in the same dependency graph and testing overhead as the multilingual inference stack, so we are skipping it alongside the underlying model changes.

### Tokenizer / config churn (blocked on multilingual scope)

The tokenizer and model-config updates upstream primarily exist to service the multilingual release. Without adopting those checkpoints we would be carrying unused code and heavy dependencies. We will revisit once there is a pressing need for non-English voices.

## Fork Functionality to Preserve

- **Guided voice clone UI.** `gradio_voice_clone_app.py` delivers a novice-friendly workflow for recording/uploading reference audio, tuning inference knobs, and exporting cloned speech in one panel.
- **Shared conditioning utilities.** `src/chatterbox/audio_utils.py` centralises loudness normalisation, silence trimming, and multi-file prompt handling used by both TTS and VC pipelines.
- **LoRA experimentation script.** `scripts/train_lora.py` gives power users an entry point for continuing training on custom datasets, which remains absent upstream.

## Suggested Integration Path

1. **Sync core libraries.** Mirror upstream changes in the tokenizer, model configs, and new multilingual modules before reconciling app-layer code. This ensures we adopt the architecture required for the new checkpoints.
2. **Reconcile apps.** Merge upstream's `multilingual_app.py` alongside our existing voice cloning Blocks UI, sharing common helper components where practical.
3. **Update packaging.** Bump our `pyproject.toml` version and dependency pins in step with upstream once the multilingual runtime is incorporated.
4. **Regression test cloning workflows.** After merging upstream code, validate that our conditioning utilities and Gradio helpers still behave correctly with both English and multilingual checkpoints.
