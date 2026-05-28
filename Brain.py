import whisper
import requests
from TTS.api import TTS
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import msvcrt
import time
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import re
sd.default.device = 13
# ------------------------------
# LOAD MODELS
# ------------------------------

print("🔄 Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("✅ Whisper Loaded")

print("🔄 Loading XTTS model...")
tts = TTS(model_name="tts_models/en/ljspeech/vits")
print("✅ XTTS Loaded")

print("🔄 Loading Embedding Model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding Model Loaded\n")

# ------------------------------
# VECTOR MEMORY SETUP
# ------------------------------

dimension = 384
memory_file = "nova_memory.pkl"
index_file = "nova_faiss.index"

if os.path.exists(index_file) and os.path.exists(memory_file):
    print("📂 Loading existing memory...")
    index = faiss.read_index(index_file)
    with open(memory_file, "rb") as f:
        memory_texts = pickle.load(f)
else:
    print("🆕 Creating new memory...")
    index = faiss.IndexFlatL2(dimension)
    memory_texts = []

def save_memory():
    faiss.write_index(index, index_file)
    with open(memory_file, "wb") as f:
        pickle.dump(memory_texts, f)

def store_memory(text):
    embedding = embed_model.encode([text]).astype("float32")
    index.add(embedding)
    memory_texts.append(text)
    save_memory()

def retrieve_memory(query, k=3):
    if len(memory_texts) == 0:
        return ""

    query_embedding = embed_model.encode([query]).astype("float32")
    D, I = index.search(query_embedding, min(k, len(memory_texts)))

    retrieved = []
    for idx in I[0]:
        if idx < len(memory_texts):
            retrieved.append(memory_texts[idx])

    return "\n".join(retrieved)

# ------------------------------
# RECORD AUDIO
# ------------------------------

def record_audio(filename="input.wav", duration=4, fs=48000):
    print("🎤 Please speak your query...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    sf.write(filename, audio, fs)
    print("✅ Recording complete.\n")
    return filename

# ------------------------------
# TRANSCRIBE
# ------------------------------

def transcribe_audio(filename):
    print("🧠 Transcribing using Whisper...")
    result = whisper_model.transcribe(
    filename,
    fp16=False,
    language="en"
)
    text = result["text"].strip()
    print("🗣 You said:", text, "\n")
    return text

# ------------------------------
# OLLAMA RESPONSE WITH MEMORY
# ------------------------------

def get_ollama_response(prompt):
    print("🤖 Asking Ollama...\n")

    try:
        system_prompt = (
            "You are Nova, a helpful AI assistant.\n"
            "Use relevant past memory if helpful.\n"
            "Give short, clear answers.\n"
            "Maximum 3 short lines.\n"
            "try to be smart and funny .\n"
        )

        past_memory = retrieve_memory(prompt)

        full_prompt = (
            system_prompt +
            "\n\nRelevant Past Memory:\n" +
            past_memory +
            "\n\nCurrent Question:\n" +
            prompt +
            "\nAnswer:"
        )
        print("Prompt length:", len(full_prompt))
        start = time.time()
        response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "phi3:mini",      #llama3.2:3b phi3:mini
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "num_predict": 60,     # limit tokens
            "temperature": 0.4,    # stable
            "top_p": 0.9
        }
    },
    timeout=60
)
        
        print("LLM time:", time.time() - start)

        if response.status_code == 200:
            reply = response.json().get("response", "").strip()

            # Store conversation in vector memory
            store_memory("User: " + prompt)
            store_memory("Assistant: " + reply)

            return reply
        else:
            print("Ollama API Error:", response.text)
            return "Sorry, I couldn't process that."

    except Exception as e:
        print("Connection Error:", e)
        return "Ollama is not running."

# ------------------------------
# SPEAK RESPONSE
# ------------------------------

def speak_response(text):
    # Remove separators and weird formatting
    text = re.sub(r"-{2,}", "", text)
    text = re.sub(r"\n+", " ", text)
    text = text.strip()

    if not text:
        return

    tts.tts_to_file(
        text=text,
        file_path="response.wav"
    )

    os.system("start response.wav")

# ------------------------------
# TIMED INPUT (1 MIN)
# ------------------------------

def timed_input(prompt, timeout=60):
    print(prompt, end="", flush=True)
    start_time = time.time()
    user_input = ""

    while True:
        if msvcrt.kbhit():
            char = msvcrt.getwche()

            if char == "\r":
                print()
                return user_input.strip()
            elif char == "\b":
                user_input = user_input[:-1]
            else:
                user_input += char

        if time.time() - start_time > timeout:
            print("\n⏰ No input for 60 seconds. Exiting Nova AI...")
            return None

# ------------------------------
# MAIN LOOP
# ------------------------------

print("🚀 Nova AI Assistant Ready!\n")

while True:
    user_choice = timed_input("Do you want to ask a question? (y/n): ", timeout=60)

    if user_choice is None:
        break

    user_choice = user_choice.lower()

    if user_choice == "n":
        print("👋 Exiting Nova AI...")
        break

    if user_choice != "y":
        print("⚠️ Please type 'y' or 'n'\n")
        continue

    audio_file = record_audio()

    user_query = transcribe_audio(audio_file)

    if not user_query:
        print("⚠️ No speech detected. Try again.\n")
        continue

    assistant_response = get_ollama_response(user_query)

    print("🤖 Ollama Response:\n")
    print(assistant_response, "\n")

    speak_response(assistant_response)

    print("-" * 40)