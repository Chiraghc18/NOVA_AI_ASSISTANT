"""
NOVA visual interface server.

Wraps Brain.py with a small local HTTP API for the web interface.
Run with:  python server.py
Then open: http://127.0.0.1:8000
"""

import json
import os
import re
import threading

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import Brain  # loads Whisper, Silero VAD, TTS, embeddings, FAISS memory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = FastAPI(title="NOVA")

# Only one voice/text turn at a time (single mic + single speaker).
turn_lock = threading.Lock()

state = {
    "status": "idle",   # idle | listening | transcribing | thinking | speaking
    "note": "",         # short message such as "No speech detected"
    "messages": [
        {"role": m["role"], "content": m["content"]}
        for m in Brain.conversation_history
    ],
}

# ---------- Model ----------
# Brain.py reads nova_model.txt at startup. The UI dropdown writes to it too.
MODEL_FILE = os.path.join(BASE_DIR, "nova_model.txt")

if os.path.exists(MODEL_FILE):
    print(f"[NOVA] Model from nova_model.txt: {Brain.OLLAMA_MODEL!r}")
else:
    print(f"[NOVA] nova_model.txt not found, using default model: {Brain.OLLAMA_MODEL}")

# ---------- Audio devices ----------
# Device numbers in Brain.py only suit one PC (and can shift when Bluetooth
# devices connect). Here they are validated at startup, matched by name if the
# user picked devices in Settings, and otherwise fall back to system defaults.
AUDIO_FILE = os.path.join(BASE_DIR, "nova_audio.json")


def _channel_key(kind):
    return "max_input_channels" if kind == "input" else "max_output_channels"


def device_ok(index, kind):
    try:
        return Brain.sd.query_devices(index)[_channel_key(kind)] > 0
    except Exception:
        return False


def device_name(index):
    try:
        return Brain.sd.query_devices(index)["name"]
    except Exception:
        return "Not available"


def find_device_by_name(name, kind):
    try:
        for i, d in enumerate(Brain.sd.query_devices()):
            if d["name"] == name and d[_channel_key(kind)] > 0:
                return i
    except Exception:
        pass
    return None


def system_default_device(kind):
    try:
        api = Brain.sd.query_hostapis(Brain.sd.default.hostapi)
        idx = api["default_input_device" if kind == "input" else "default_output_device"]
        if idx is not None and idx >= 0 and device_ok(idx, kind):
            return idx
    except Exception:
        pass
    try:
        for i, d in enumerate(Brain.sd.query_devices()):
            if d[_channel_key(kind)] > 0:
                return i
    except Exception:
        pass
    return None


def apply_devices(inp, out):
    Brain.INPUT_DEVICE = inp
    Brain.OUTPUT_DEVICE = out
    try:
        Brain.sd.default.device = (inp, out)
    except Exception:
        pass


def init_audio_devices():
    try:
        with open(AUDIO_FILE, "r", encoding="utf-8-sig") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    if not isinstance(saved, dict):
        saved = {}

    chosen = {}
    for kind, current in (("input", Brain.INPUT_DEVICE), ("output", Brain.OUTPUT_DEVICE)):
        idx = None
        name = saved.get(kind)
        if name:
            idx = find_device_by_name(name, kind)
        elif device_ok(current, kind):
            idx = current
        if idx is None:
            idx = system_default_device(kind)
        chosen[kind] = idx

    apply_devices(
        chosen["input"] if chosen["input"] is not None else Brain.INPUT_DEVICE,
        chosen["output"] if chosen["output"] is not None else Brain.OUTPUT_DEVICE,
    )
    print(f"[NOVA] Microphone: {device_name(Brain.INPUT_DEVICE)}")
    print(f"[NOVA] Speaker: {device_name(Brain.OUTPUT_DEVICE)}")


def list_devices(kind, current):
    """Devices of the default host API (plus the current one), for the UI."""
    items = []
    try:
        default_api = Brain.sd.default.hostapi
        apis = Brain.sd.query_hostapis()
        for i, d in enumerate(Brain.sd.query_devices()):
            if d[_channel_key(kind)] <= 0:
                continue
            if d["hostapi"] != default_api and i != current:
                continue
            label = d["name"]
            if d["hostapi"] != default_api:
                label += f" ({apis[d['hostapi']]['name']})"
            items.append({"index": i, "label": label})
    except Exception:
        pass
    return items


init_audio_devices()

CLEAR_COMMANDS = {
    "clear conversation history",
    "clear my conversation history",
    "clear chat history",
    "reset conversation history",
    "reset chat history",
}


def add_message(role, content):
    state["messages"].append({"role": role, "content": content})


def run_turn(text=None):
    """Runs one full turn using the Brain functions."""
    try:
        Brain.clear_stop()
        state["note"] = ""

        if text is None:
            if not device_ok(Brain.INPUT_DEVICE, "input"):
                state["note"] = "Microphone not available. Choose one in Settings."
                return

            state["status"] = "listening"
            audio_file = Brain.record_audio()
            if audio_file is None:
                if Brain.stop_requested():
                    state["note"] = "Stopped."
                else:
                    state["note"] = "No speech detected. Try again."
                return

            state["status"] = "transcribing"
            text = Brain.transcribe_audio(audio_file)

            if Brain.stop_requested():
                state["note"] = "Stopped."
                return

            if not text:
                state["note"] = "Couldn't recognize any speech."
                return

        add_message("user", text)
        state["status"] = "thinking"

        normalized = re.sub(r"[^a-z\s]", "", text.lower()).strip()
        if normalized in CLEAR_COMMANDS:
            Brain.clear_conversation_history()
            response = (
                "Recent conversation history cleared. "
                "Your long-term memories are still saved."
            )
            state["messages"].clear()
            add_message("user", text)
        else:
            response = Brain.get_ollama_response(text)
            if response is None:
                state["note"] = "Stopped."
                return

        add_message("assistant", response)

        if not Brain.stop_requested():
            if not device_ok(Brain.OUTPUT_DEVICE, "output"):
                state["note"] = "Speaker not available. Choose one in Settings."
            else:
                state["status"] = "speaking"
                Brain.speak_response(response)

    except Exception as e:
        state["note"] = f"Error: {e}"
    finally:
        state["status"] = "idle"
        turn_lock.release()


def start_turn(text=None):
    if not turn_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="NOVA is busy")
    threading.Thread(target=run_turn, args=(text,), daemon=True).start()


class TextIn(BaseModel):
    text: str


class ModelIn(BaseModel):
    key: str


class AudioIn(BaseModel):
    input: int
    output: int


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/api/state")
def get_state():
    return state


@app.post("/api/talk")
def talk():
    start_turn(None)
    return {"ok": True}


@app.post("/api/text")
def text_turn(body: TextIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message")
    start_turn(text)
    return {"ok": True}


@app.post("/api/stop")
def stop_everything():
    Brain.request_stop()
    return {"ok": True}


@app.get("/api/memory")
def get_memory():
    return {
        "count": len(Brain.memory_texts),
        "recent": list(reversed(Brain.memory_texts[-30:])),
    }


@app.get("/api/settings")
def get_settings():
    return {
        "user": Brain.USER_NAME,
        "whisper": Brain.WHISPER_MODEL,
        "model": Brain.OLLAMA_MODEL,
        "models": Brain.OLLAMA_MODELS,
        "microphone": device_name(Brain.INPUT_DEVICE),
        "speaker": device_name(Brain.OUTPUT_DEVICE),
    }


@app.get("/api/audio")
def get_audio():
    return {
        "inputs": list_devices("input", Brain.INPUT_DEVICE),
        "outputs": list_devices("output", Brain.OUTPUT_DEVICE),
        "input": Brain.INPUT_DEVICE,
        "output": Brain.OUTPUT_DEVICE,
    }


@app.post("/api/audio")
def set_audio(body: AudioIn):
    if turn_lock.locked():
        raise HTTPException(status_code=409, detail="NOVA is busy")
    if not device_ok(body.input, "input") or not device_ok(body.output, "output"):
        raise HTTPException(status_code=400, detail="That device is not available")
    apply_devices(body.input, body.output)
    try:
        with open(AUDIO_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "input": device_name(body.input),
                    "output": device_name(body.output),
                },
                f,
            )
    except OSError:
        pass
    return {"ok": True}


@app.post("/api/model")
def set_model(body: ModelIn):
    if body.key not in Brain.OLLAMA_MODELS:
        raise HTTPException(status_code=400, detail="Unknown model")
    Brain.OLLAMA_MODEL = Brain.OLLAMA_MODELS[body.key]["model"]
    try:
        with open(MODEL_FILE, "w", encoding="utf-8") as f:
            f.write(Brain.OLLAMA_MODEL)
    except OSError:
        pass
    return {"model": Brain.OLLAMA_MODEL}


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)