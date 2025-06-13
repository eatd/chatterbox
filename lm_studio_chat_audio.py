import json
import logging
import requests
import sounddevice as sd
import numpy as np
import torch

from chatterbox.tts import ChatterboxTTS

LLM_ENDPOINT = "http://localhost:1234/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

logger = logging.getLogger(__name__)


def stream_chat(prompt: str):
    payload = {
        "model": "local-model",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    with requests.post(LLM_ENDPOINT, json=payload, headers=HEADERS, stream=True) as r:
        for line in r.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                line = line[len(b"data: ") :]
            if line.strip() == b"[DONE]":
                break
            data = json.loads(line.decode("utf-8"))
            delta = data["choices"][0]["delta"]
            yield delta.get("content", "")


def speak_stream(token_stream, tts_model: ChatterboxTTS):
    buffer = ""
    for token in token_stream:
        buffer += token
        if token.endswith((".", "?", "!")):
            logger.info("Synthesizing: %s", buffer)
            audio = tts_model.generate(buffer).squeeze(0).numpy()
            sd.play(audio, tts_model.sr)
            sd.wait()
            buffer = ""


def main():
    logging.basicConfig(level=logging.INFO)
    model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
    prompt = "Hello! How do you feel today?"
    speak_stream(stream_chat(prompt), model)


if __name__ == "__main__":
    main()
