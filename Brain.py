import whisper
import requests
from TTS.api import TTS
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import msvcrt
import threading
import time
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import re
import json
from collections import deque

import torch
from silero_vad import load_silero_vad

# ============================================================
# SETTINGS
# ============================================================

USER_NAME = "Chirag"

WHISPER_MODEL = "small"
OLLAMA_MODELS = {
    "1": {
        "name": "TinyLlama",
        "model": "tinyllama:latest",
        "storage": "637 MB"
    },
    "2": {
        "name": "Llama 3.2 (3B)",
        "model": "llama3.2:3b",
        "storage": "2.0 GB"
    },
    "3": {
        "name": "Phi-3 Mini",
        "model": "phi3:mini",
        "storage": "2.2 GB"
    },
    "4": {
        "name": "Mistral",
        "model": "mistral:latest",
        "storage": "4.4 GB"
    },
    "5": {
        "name": "Llama 3",
        "model": "llama3:latest",
        "storage": "4.7 GB"
    },
    "6": {
        "name": "Qwen 3.5 (4B)",
        "model": "qwen3.5:4b",
        "storage": "3.3 GB"
    },
    "7": {
        "name": "Gemma 4 (E4B)",
        "model": "gemma4:e4b",
        "storage": "7 GB+"
    },
    "8": {
        "name": "Gemma 4 (12B)",
        "model": "gemma4:12b",
        "storage": "7.7 GB"
    }
}

# Default model, overridden by nova_model.txt (saved by Setup_NOVA.bat or the UI)
OLLAMA_MODEL = OLLAMA_MODELS["5"]["model"]
try:
    _model_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "nova_model.txt"
    )
    with open(_model_file, "r", encoding="utf-8-sig") as _f:
        _saved_model = _f.read().strip()
    if _saved_model:
        OLLAMA_MODEL = _saved_model
except OSError:
    pass

SAMPLE_RATE = 16000

# Silero VAD recording settings
VAD_CHUNK_SAMPLES = 512       # Required frame size for Silero at 16 kHz
VAD_THRESHOLD = 0.50          # Speech probability threshold
VAD_START_FRAMES = 2          # Consecutive speech frames to start recording
VAD_SILENCE_SECONDS = 0.80    # Stop after this much continuous silence
VAD_PRE_ROLL_SECONDS = 0.25   # Preserve audio just before speech detection
VAD_START_TIMEOUT = 10        # Give up if no speech starts within 10 seconds
VAD_MAX_RECORD_SECONDS = 30   # Safety limit for one utterance
VAD_MIN_SPEECH_SECONDS = 0.20 # Reject very short accidental detections

MEMORY_DISTANCE_LIMIT = 1.0
MAX_HISTORY_MESSAGES = 12       # 6 user/assistant exchanges
MAX_MEMORY_RESULTS = 3

# Audio device configuration
INPUT_DEVICE = 2       # Intel Microphone Array
OUTPUT_DEVICE = 5      # Boult Bluetooth Headphones

# Audio files
INPUT_FILE = "input.wav"
RESPONSE_FILE = "response.wav"

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/chat"

WHISPER_HALLUCINATIONS = [
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subtitles by",
    "you",
    "bye",
]

# ============================================================
# STOP CONTROL
# ============================================================
# request_stop() cancels whatever NOVA is doing right now:
#   - recording      -> recording ends and is discarded
#   - the LLM reply  -> generation is cancelled (checked as the reply streams in)
#   - speech         -> playback is cut off, or never starts
# The web interface calls request_stop() from its Stop button and
# clear_stop() at the start of every turn.

stop_event = threading.Event()


def request_stop():
    stop_event.set()

    try:
        sd.stop()
    except Exception:
        pass


def clear_stop():
    stop_event.clear()


def stop_requested():
    return stop_event.is_set()


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

sd.default.device = (INPUT_DEVICE, OUTPUT_DEVICE)

print("\n========== NOVA AUDIO CONFIGURATION ==========")

try:
    input_info = sd.query_devices(INPUT_DEVICE, "input")
    output_info = sd.query_devices(OUTPUT_DEVICE, "output")

    print("Microphone:", input_info["name"])
    print("Headphones:", output_info["name"])

except Exception as e:
    print("Audio device configuration error:", e)

# ============================================================
# LOAD MODELS
# ============================================================

print("\n🔄 Loading Whisper model...")

whisper_model = whisper.load_model(WHISPER_MODEL)

print("✅ Whisper Loaded")

print("\n🔄 Loading Silero VAD model...")

vad_model = load_silero_vad(onnx=False)
vad_model.eval()

print("✅ Silero VAD Loaded")

print("\n🔄 Loading TTS model...")

tts = TTS(model_name="tts_models/en/ljspeech/vits")

print("✅ TTS Loaded")

print("\n🔄 Loading Embedding Model...")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("✅ Embedding Model Loaded\n")

# ============================================================
# VECTOR MEMORY SETUP
# ============================================================

dimension = 384

memory_file = "nova_memory.pkl"
index_file = "nova_faiss.index"
conversation_file = "nova_conversation.json"

# Recent conversation is separate from long-term FAISS memory.
# It is loaded at startup and the most recent messages are kept in context.
conversation_history = []


def load_conversation_history():
    """Load recent chat messages without affecting long-term FAISS memory."""
    global conversation_history

    if not os.path.exists(conversation_file):
        conversation_history = []
        return

    try:
        with open(conversation_file, "r", encoding="utf-8") as f:
            saved_history = json.load(f)

        if not isinstance(saved_history, list):
            raise ValueError("Conversation history must be a list")

        valid_messages = []
        for item in saved_history:
            if (
                isinstance(item, dict)
                and item.get("role") in ("user", "assistant")
                and isinstance(item.get("content"), str)
                and item["content"].strip()
            ):
                valid_messages.append({
                    "role": item["role"],
                    "content": item["content"].strip()
                })

        conversation_history = valid_messages[-MAX_HISTORY_MESSAGES:]
        print(f"🧠 Loaded {len(conversation_history)} recent conversation messages.")

    except (OSError, json.JSONDecodeError, ValueError) as e:
        print("⚠️ Could not load conversation history:", e)
        conversation_history = []


def save_conversation_history():
    """Save chat history atomically to reduce the risk of a partial file."""
    temporary_file = conversation_file + ".tmp"

    try:
        with open(temporary_file, "w", encoding="utf-8") as f:
            json.dump(conversation_history, f, ensure_ascii=False, indent=2)
        os.replace(temporary_file, conversation_file)
    except OSError as e:
        print("⚠️ Could not save conversation history:", e)


def add_to_conversation(role, content):
    """Append one message and enforce the recent-history limit."""
    if role not in ("user", "assistant") or not content.strip():
        return

    conversation_history.append({
        "role": role,
        "content": content.strip()
    })
    del conversation_history[:-MAX_HISTORY_MESSAGES]
    save_conversation_history()


def clear_conversation_history():
    """Clear recent chat context; keep long-term FAISS memories intact."""
    conversation_history.clear()
    save_conversation_history()
    print("🧹 Recent conversation history cleared. Long-term memory was kept.")


load_conversation_history()

if os.path.exists(index_file) and os.path.exists(memory_file):

    print("📂 Loading existing memory...")

    index = faiss.read_index(index_file)

    with open(memory_file, "rb") as f:
        memory_texts = pickle.load(f)

    # Protect against mismatched memory files
    if index.ntotal != len(memory_texts):
        print("⚠️ Memory index mismatch. Creating new index.")

        index = faiss.IndexFlatL2(dimension)
        memory_texts = []

else:

    print("🆕 Creating new memory...")

    index = faiss.IndexFlatL2(dimension)
    memory_texts = []


def save_memory():

    faiss.write_index(index, index_file)

    with open(memory_file, "wb") as f:
        pickle.dump(memory_texts, f)


def store_memory(text):

    embedding = embed_model.encode(
        [text]
    ).astype("float32")

    index.add(embedding)

    memory_texts.append(text)

    save_memory()


def retrieve_memory(query, k=3):

    if len(memory_texts) == 0:
        return ""

    query_embedding = embed_model.encode(
        [query]
    ).astype("float32")

    D, I = index.search(
        query_embedding,
        min(k, len(memory_texts))
    )

    retrieved = []

    for dist, idx in zip(D[0], I[0]):

        if (
            idx != -1
            and idx < len(memory_texts)
            and dist < MEMORY_DISTANCE_LIMIT
        ):

            retrieved.append(memory_texts[idx])

    return "\n".join(retrieved)


# ============================================================
# RECORD AUDIO USING SILERO VAD
# ============================================================

def record_audio(filename=INPUT_FILE):
    """
    Records until Silero VAD detects speech followed by enough silence.

    Audio is captured in 512-sample frames at 16 kHz, the expected
    frame size for Silero VAD at this sample rate.

    Returns None if nothing usable was recorded or if Stop was requested.
    """

    print("\n🎤 Listening... Start speaking.")

    try:
        device_info = sd.query_devices(INPUT_DEVICE, "input")

        if device_info["max_input_channels"] < 1:
            print("❌ Selected device has no microphone input.")
            return None

        pre_roll_frames = max(
            1,
            int(VAD_PRE_ROLL_SECONDS * SAMPLE_RATE / VAD_CHUNK_SAMPLES)
        )
        silence_frames_required = max(
            1,
            int(VAD_SILENCE_SECONDS * SAMPLE_RATE / VAD_CHUNK_SAMPLES)
        )

        pre_roll = deque(maxlen=pre_roll_frames)
        recorded_frames = []
        speech_started = False
        consecutive_speech_frames = 0
        voiced_frames = 0
        silence_frames = 0
        start_wait = time.monotonic()
        recording_start = None

        # Reset recurrent state between utterances.
        vad_model.reset_states()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=VAD_CHUNK_SAMPLES,
            device=INPUT_DEVICE,
            channels=1,
            dtype="float32"
        ) as stream:

            while True:

                if stop_event.is_set():
                    print("⏹ Recording cancelled.\n")
                    return None

                chunk, overflowed = stream.read(VAD_CHUNK_SAMPLES)

                if overflowed:
                    print("⚠️ Audio input overflow; a small section may be lost.")

                audio_chunk = np.asarray(
                    chunk[:, 0],
                    dtype=np.float32
                ).copy()

                audio_tensor = torch.from_numpy(audio_chunk)

                with torch.no_grad():
                    speech_probability = vad_model(
                        audio_tensor,
                        SAMPLE_RATE
                    ).item()

                now = time.monotonic()

                if not speech_started:
                    pre_roll.append(audio_chunk)

                    if speech_probability >= VAD_THRESHOLD:
                        consecutive_speech_frames += 1
                    else:
                        consecutive_speech_frames = 0

                    if consecutive_speech_frames >= VAD_START_FRAMES:
                        speech_started = True
                        recording_start = now
                        recorded_frames.extend(list(pre_roll))
                        voiced_frames = consecutive_speech_frames
                        print("🗣 Speech detected. Recording...")
                        continue

                    if now - start_wait >= VAD_START_TIMEOUT:
                        print("⚠️ No speech detected. Try again.\n")
                        return None

                else:
                    recorded_frames.append(audio_chunk)

                    if speech_probability >= VAD_THRESHOLD:
                        voiced_frames += 1
                        silence_frames = 0
                    else:
                        silence_frames += 1

                    elapsed = now - recording_start

                    if silence_frames >= silence_frames_required:
                        print("🔇 Silence detected. Recording stopped.")
                        break

                    if elapsed >= VAD_MAX_RECORD_SECONDS:
                        print("⏱ Maximum recording duration reached.")
                        break

        speech_duration = voiced_frames * VAD_CHUNK_SAMPLES / SAMPLE_RATE

        if speech_duration < VAD_MIN_SPEECH_SECONDS:
            print("⚠️ Speech was too short. Please try again.\n")
            return None

        audio = np.concatenate(recorded_frames).astype(np.float32)

        if audio.size == 0:
            print("⚠️ No audio captured.\n")
            return None

        # Avoid peak-normalizing, which can amplify background noise.
        sf.write(filename, audio, SAMPLE_RATE)

        print(
            f"✅ Recording saved ({len(audio) / SAMPLE_RATE:.1f} seconds).\n"
        )

        return filename

    except Exception as e:
        print("❌ Microphone/VAD Error:", e)

        try:
            vad_model.reset_states()
        except Exception:
            pass

        return None


# ============================================================
# TRANSCRIBE AUDIO
# ============================================================

def transcribe_audio(filename):

    print("🧠 Transcribing using Whisper...")

    try:

        result = whisper_model.transcribe(
            filename,
            fp16=False,
            language="en",
            initial_prompt=(
                f"My name is {USER_NAME}. "
                "Nova is my voice assistant."
            ),
            condition_on_previous_text=False,
        )

        text = result["text"].strip()

        # Remove punctuation for comparison
        cleaned = re.sub(
            r"[^\w\s]",
            "",
            text
        ).lower().strip()

        # Filter hallucinations
        if (
            cleaned in WHISPER_HALLUCINATIONS
            or "thank you for watching" in cleaned
        ):

            print(
                "🗣 Ignored likely hallucination:",
                text,
                "\n"
            )

            return ""

        if not cleaned:

            print("⚠️ No recognizable speech.\n")

            return ""

        print("🗣 You said:", text, "\n")

        return text

    except Exception as e:

        print("❌ Whisper Error:", e)

        return ""


# ============================================================
# CLEAN OLLAMA RESPONSE
# ============================================================

# Sentences like "I remember..." or "we had a fun chat earlier" are dropped
# from replies, even if the model ignores the instruction not to say them.
MEMORY_TALK = re.compile(
    r"(?i)\b(?:i remember|i recall|as we discussed|as we talked|"
    r"we had a (?:fun|great|nice|good|lovely)|"
    r"we (?:talked|chatted|spoke) (?:about|earlier|before)|"
    r"earlier (?:we|you))"
)


def remove_memory_talk(reply):

    sentences = re.split(r"(?<=[.!?])\s+", reply)

    kept = [s for s in sentences if not MEMORY_TALK.search(s)]

    return " ".join(kept) if kept else reply


def clean_reply(reply):

    reply = re.split(
        r"Instruction\s*\d*",
        reply
    )[0]

    reply = re.sub(
        r"\n{2,}",
        "\n",
        reply
    )

    return remove_memory_talk(reply.strip())


# ============================================================
# OLLAMA RESPONSE WITH MEMORY
# ============================================================

def get_ollama_response(prompt):
    """
    Returns the reply text, or None if Stop was requested while the
    reply was being generated.
    """

    print("🤖 Asking Ollama...\n")

    try:

        system_prompt = (
            f"You are Nova, a helpful voice assistant. "
            f"The user's name is {USER_NAME}.\n"
            "Reply in 1-2 short, clear sentences with a touch of light humor.\n"
            "Use recent conversation history to understand follow-up questions.\n"
            "Use long-term memory only when it is relevant to the current question.\n"
            "Never say things like 'I remember', 'I recall', 'as we discussed', "
            "'earlier we talked', or 'we had a fun conversation'. Use what you know "
            "naturally, as if it were common knowledge between you.\n"
            "Do not mention past conversations unless the user asks about them.\n"
            "Do not say the user's name in every reply. Use it only occasionally.\n"
            "For a simple greeting, reply with a short greeting and ask how you can help.\n"
            "If you do not know a personal detail, say so instead of inventing it.\n"
            "Never invent extra instructions, exercises, or follow-up tasks."
        )

        past_memory = retrieve_memory(prompt, k=MAX_MEMORY_RESULTS)

        # Keep the system instructions, relevant long-term memory, recent
        # conversation, and current question as separate chat messages.
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        if past_memory:
            messages.append({
                "role": "system",
                "content": (
                    "Background facts about the user from earlier chats. "
                    "Use only details that help answer the current question, "
                    "and never mention that you remember or recall anything:\n"
                    f"{past_memory}"
                )
            })

        messages.extend(conversation_history[-MAX_HISTORY_MESSAGES:])
        messages.append({
            "role": "user",
            "content": prompt
        })

        print(
            "Conversation messages:", len(messages) - 1,
            "| Prompt characters:", sum(len(m["content"]) for m in messages)
        )

        start = time.time()

        # The reply is streamed so that Stop can cancel it part-way through.
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": True,
                "options": {
                    "num_predict": 100,
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "stop": [
                        "Instruction",
                        "\n\n\n"
                    ]
                }
            },
            timeout=90,
            stream=True
        )

        if response.status_code != 200:

            print(
                "Ollama API Error:",
                response.status_code,
                response.text[:500]
            )

            response.close()

            if response.status_code == 404:
                print(
                    f"💡 Model '{OLLAMA_MODEL}' is not downloaded. "
                    f"Run: ollama pull {OLLAMA_MODEL}"
                )
                return (
                    "That model isn't downloaded yet. "
                    "Please download it with Ollama first."
                )

            return "Sorry, I couldn't process that."

        reply_parts = []
        stopped = False
        api_error = None

        try:

            for line in response.iter_lines():

                if stop_event.is_set():
                    stopped = True
                    break

                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if chunk.get("error"):
                    api_error = chunk["error"]
                    break

                reply_parts.append(
                    chunk.get("message", {}).get("content", "")
                )

                if chunk.get("done"):
                    break

        finally:
            response.close()

        print(
            "LLM time:",
            round(time.time() - start, 2),
            "seconds"
        )

        if stopped:
            print("⏹ Response cancelled.\n")
            return None

        if api_error:
            print("Ollama API Error:", api_error)
            return "Sorry, I couldn't process that."

        reply = clean_reply("".join(reply_parts))

        if not reply:
            return "Sorry, I didn't catch a good answer for that."

        # Save the exchange only after Ollama returns a valid answer.
        add_to_conversation("user", prompt)
        add_to_conversation("assistant", reply)

        # FAISS is the longer-term semantic memory. Store the complete
        # exchange so retrieval can recover both the question and answer.
        if len(prompt.split()) >= 3:
            store_memory(
                f"User: {prompt}\nNOVA: {reply}"
            )

        return reply

    except requests.exceptions.ConnectionError:
        print("❌ Ollama connection failed.")
        return "Ollama is not running."

    except requests.exceptions.Timeout:
        print("❌ Ollama request timed out.")
        return "Ollama is taking too long to respond. Please try again."

    except Exception as e:
        print("❌ Ollama Error:", e)
        return "Sorry, something went wrong."


# ============================================================
# SPEAK RESPONSE
# ============================================================

def speak_response(text):

    # Clean formatting and emojis
    text = re.sub(
        r"[^\x00-\x7F]+",
        "",
        text
    )

    text = re.sub(
        r"-{2,}",
        "",
        text
    )

    text = re.sub(
        r"[*_`#]",
        "",
        text
    )

    text = re.sub(
        r"\n+",
        " ",
        text
    ).strip()

    if not text:
        return

    if stop_event.is_set():
        return

    try:

        print("🔊 Nova is speaking...")

        # Generate speech
        tts.tts_to_file(
            text=text,
            file_path=RESPONSE_FILE
        )

        # Stop may have been pressed while the voice was being generated
        if stop_event.is_set():
            print("⏹ Speech cancelled.\n")
            return

        # Load generated audio
        audio, fs = sf.read(
            RESPONSE_FILE,
            dtype="float32"
        )

        # Play through selected headphones
        sd.play(
            audio,
            samplerate=fs,
            device=OUTPUT_DEVICE
        )

        # If Stop arrived just before playback began, cut it off now
        if stop_event.is_set():
            sd.stop()
            print("⏹ Speech cancelled.\n")
            return

        sd.wait()

        if stop_event.is_set():
            print("⏹ Speech stopped.\n")
            return

        print("✅ Speech completed.\n")

    except Exception as e:

        print("❌ TTS Playback Error:", e)


# ============================================================
# TIMED INPUT (1 MINUTE)
# ============================================================

def timed_input(prompt, timeout=60):

    print(
        prompt,
        end="",
        flush=True
    )

    start_time = time.time()

    user_input = ""

    while True:

        if msvcrt.kbhit():

            char = msvcrt.getwche()

            if char == "\r":

                print()

                return user_input.strip()

            elif char == "\b":

                if user_input:

                    user_input = user_input[:-1]

            else:

                user_input += char

        if time.time() - start_time > timeout:

            print(
                "\n⏰ No input for 60 seconds. "
                "Exiting Nova AI..."
            )

            return None

        time.sleep(0.01)


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    print("\n🚀 NOVA AI ASSISTANT READY!")
    print("=" * 45)

    while True:

        user_choice = timed_input(
            "\nDo you want to ask a question? (y/n): ",
            timeout=60
        )

        if user_choice is None:
            break

        user_choice = user_choice.lower()

        if user_choice == "n":

            print("\n👋 Exiting Nova AI...")

            break

        if user_choice != "y":

            print(
                "⚠️ Please type 'y' or 'n'\n"
            )

            continue

        # Record
        audio_file = record_audio()

        if audio_file is None:
            continue

        # Transcribe
        user_query = transcribe_audio(audio_file)

        if not user_query:

            print(
                "⚠️ No speech detected. Try again.\n"
            )

            continue

        # Simple local memory command. This clears recent chat context only;
        # it does not delete the long-term FAISS memory files.
        normalized_query = re.sub(r"[^a-z\s]", "", user_query.lower()).strip()
        clear_commands = {
            "clear conversation history",
            "clear my conversation history",
            "clear chat history",
            "reset conversation history",
            "reset chat history"
        }

        if normalized_query in clear_commands:
            clear_conversation_history()
            assistant_response = (
                "Recent conversation history cleared. "
                "Your long-term memories are still saved."
            )
        else:
            # Generate response
            assistant_response = get_ollama_response(user_query)

        if assistant_response is None:
            continue

        print("\n🤖 Nova Response:\n")

        print(assistant_response, "\n")

        # Speak
        speak_response(assistant_response)

        print("-" * 45)


# ============================================================
# START NOVA
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\n\n👋 NOVA stopped by user.")

    finally:

        sd.stop()