import streamlit as st
import librosa
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline

# --------------------------
# Load Models
# --------------------------
asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-small")
text_emotion_model = pipeline("text-classification",
                              model="bhadresh-savani/bert-base-go-emotion",
                              return_all_scores=True)

# --------------------------
# Helper Functions
# --------------------------
def get_text_emotions(transcript):
    emotions = text_emotion_model(transcript)[0]
    emotions_sorted = sorted(emotions, key=lambda x: x['score'], reverse=True)
    return emotions_sorted[0]  # top emotion

def get_voice_emotion(y, sr):
    # Dummy simulation – replace with your trained SER model later
    emotions = ["happy", "sad", "angry", "neutral", "fear", "surprise"]
    probs = np.random.dirichlet(np.ones(len(emotions)), size=1)[0]  # random probs
    max_idx = np.argmax(probs)
    return {"label": emotions[max_idx], "score": float(probs[max_idx])}

def fusion_logic(text_emotion, voice_emotion):
    # Weighted fusion: 70% voice, 30% text
    if voice_emotion['score'] > 0.7:
        return voice_emotion
    else:
        return text_emotion

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="🎤 Voice-to-Text Emotion Analysis", layout="wide")
st.title("🎤 Voice-to-Text Emotion Analysis")
st.markdown("Analyze speech & detect emotions from both **text** and **voice tone**")

# Upload audio file
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    # --------------------------
    # Audio Preprocessing
    # --------------------------
    y, sr = librosa.load(audio_file, sr=16000, mono=True)

    # --------------------------
    # Step 1: Speech-to-Text
    # --------------------------
    transcript = asr_model(audio_file)["text"]
    st.subheader("📝 Transcript")
    st.markdown(f"**{transcript}**")

    # --------------------------
    # Step 2: Text Emotion
    # --------------------------
    text_emotion = get_text_emotions(transcript)

    # --------------------------
    # Step 3: Voice Emotion
    # --------------------------
    voice_emotion = get_voice_emotion(y, sr)

    # --------------------------
    # Step 4: Fusion
    # --------------------------
    final_emotion = fusion_logic(text_emotion, voice_emotion)

    # --------------------------
    # Results Display
    # --------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📝 Text Emotion", text_emotion['label'], f"{text_emotion['score']:.2f}")

    with col2:
        st.metric("🎤 Voice Emotion", voice_emotion['label'], f"{voice_emotion['score']:.2f}")

    with col3:
        st.metric("✅ Final Emotion", final_emotion['label'], f"{final_emotion['score']:.2f}")

    # --------------------------
    # Chart (Timeline Simulation)
    # --------------------------
    st.subheader("📊 Emotion Confidence Timeline")
    fig, ax = plt.subplots()
    timeline = np.linspace(0, len(y)/sr, 10)  # 10 segments
    conf_scores = np.random.rand(10)  # dummy confidence simulation
    ax.plot(timeline, conf_scores, marker="o")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Confidence")
    ax.set_title("Emotion Confidence Over Time")
    st.pyplot(fig)
