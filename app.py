import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import speech_recognition as sr
from gtts import gTTS
import os,html
import base64
from datetime import datetime

# ── Page config ──
st.set_page_config(
    page_title="LinguaAI Translator",
    page_icon="🌍",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

* { font-family: 'DM Sans', sans-serif; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e2e;
}

/* Chat message — user */
.user-msg {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #2d2d4e;
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px;
    margin: 8px 0 8px 40px;
    color: #c8c8e8;
    font-size: 15px;
    line-height: 1.6;
    position: relative;
}

/* Chat message — bot */
.bot-msg {
    background: linear-gradient(135deg, #0d1f0d, #0a2010);
    border: 1px solid #1a3d1a;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    margin: 8px 40px 8px 0;
    color: #a8e6a8;
    font-size: 15px;
    line-height: 1.6;
}

.msg-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
    opacity: 0.5;
}

.msg-time {
    font-size: 10px;
    opacity: 0.4;
    margin-top: 6px;
    text-align: right;
}

/* Input area */
.stTextInput input {
    background: #0f0f1a !important;
    border: 1px solid #2d2d4e !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
}

.stTextInput input:focus {
    border-color: #4a9eff !important;
    box-shadow: 0 0 0 2px rgba(74, 158, 255, 0.15) !important;
}

/* Buttons */
.stButton button {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #4a9eff, #6b5bef) !important;
    border: none !important;
    color: white !important;
}

.stButton button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(74, 158, 255, 0.3) !important;
}

/* Voice button */
.voice-active {
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255, 80, 80, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(255, 80, 80, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 80, 80, 0); }
}

/* Stats cards */
.stat-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #4a9eff;
}

.stat-label {
    font-size: 12px;
    color: #6868a8;
    margin-top: 4px;
}

/* Mode tabs */
.mode-tab {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 10px 20px;
    cursor: pointer;
    text-align: center;
    transition: all 0.2s;
}

.mode-tab.active {
    background: linear-gradient(135deg, #1a2a4a, #1a1a3a);
    border-color: #4a9eff;
    color: #4a9eff;
}

/* Divider */
hr {
    border-color: #1e1e2e !important;
    margin: 16px 0 !important;
}

/* Scrollable chat area */
.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 8px 0;
}

/* Audio player */
audio {
    width: 100%;
    margin-top: 8px;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.badge-green {
    background: rgba(74, 222, 128, 0.1);
    color: #4ade80;
    border: 1px solid rgba(74, 222, 128, 0.2);
}

.badge-blue {
    background: rgba(74, 158, 255, 0.1);
    color: #4a9eff;
    border: 1px solid rgba(74, 158, 255, 0.2);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# MODEL LOADING — WEIGHTS APPROACH
# No more version conflicts!
# ══════════════════════════════════════════
@st.cache_resource
def load_model_and_vocab():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # ── File paths ──
        weights_path = os.path.join(base_dir, "translation_weights.weights.h5")
        en_pkl_path  = os.path.join(base_dir, "text_vec_en.pkl")
        es_pkl_path  = os.path.join(base_dir, "text_vec_es.pkl")

        # ── Load vocabularies ──
        with open(en_pkl_path, "rb") as f:
            vocab_en = pickle.load(f)
        with open(es_pkl_path, "rb") as f:
            vocab_es = pickle.load(f)

        # ── Rebuild TextVectorization layers ──
        vocab_size = 10000
        max_length = 50

        text_vec_layer_en = tf.keras.layers.TextVectorization(
            max_tokens=vocab_size,
            output_sequence_length=max_length,
            vocabulary=vocab_en
        )
        text_vec_layer_es = tf.keras.layers.TextVectorization(
            max_tokens=vocab_size,
            output_sequence_length=max_length,
            vocabulary=vocab_es
        )

        # ── Rebuild model architecture (exact same as training) ──
        embed_size = 128

        # Inputs
        encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
        decoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)

        # Vectorize
        encoder_input_ids = text_vec_layer_en(encoder_inputs)
        decoder_input_ids = text_vec_layer_es(decoder_inputs)

        # Embeddings
        encoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
        decoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)

        encoder_embeddings = encoder_embedding_layer(encoder_input_ids)
        decoder_embeddings = decoder_embedding_layer(decoder_input_ids)

        # Encoder — Bidirectional LSTM
        encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, return_state=True, return_sequences=True)
        )
        encoder_outputs, *encoder_states = encoder(encoder_embeddings)

        # Concat forward + backward states
        encoder_states = [
            tf.keras.layers.Concatenate()([encoder_states[0], encoder_states[2]]),
            tf.keras.layers.Concatenate()([encoder_states[1], encoder_states[3]])
        ]

        # Decoder — LSTM(512)
        decoder = tf.keras.layers.LSTM(512, return_sequences=True)
        decoder_outputs = decoder(decoder_embeddings, initial_state=encoder_states)

        # Attention
        attention_layer  = tf.keras.layers.Attention()
        attention_outputs = attention_layer([decoder_outputs, encoder_outputs])

        # Output
        output_layer = tf.keras.layers.Dense(vocab_size, activation="softmax")
        y_probs = output_layer(attention_outputs)

        # Build model
        model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[y_probs])

        # ── Load weights ──
        model.load_weights(weights_path)

        return model, text_vec_layer_en, text_vec_layer_es, True

    except Exception as e:
        st.error(f"EXACT ERROR: {e}")
        return None, None, None, False


model, text_vec_layer_en, text_vec_layer_es, model_loaded = load_model_and_vocab()


# ── Core functions ──
def translate(sentence):
    if not model_loaded:
        return "⚠️ Model not loaded"
    translation = ""
    for word_idx in range(50):
        X = tf.constant([sentence])
        X_dec = tf.constant(["startofseq " + translation])
        y_prob = model.predict((X, X_dec), verbose=0)[0, word_idx]
        predicted_word_id = np.argmax(y_prob)
        predicted_word = text_vec_layer_es.get_vocabulary()[predicted_word_id]
        if predicted_word == "endofseq":
            break
        translation += " " + predicted_word
    return translation.strip()


def record_voice():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=12)
        return recognizer.recognize_google(audio)
    except sr.WaitTimeoutError:
        st.error("⏱️ Timeout — no speech detected. Speak faster after clicking.")
        return None
    except sr.UnknownValueError:
        st.error("🔇 Speech detected but couldn't understand. Speak clearly.")
        return None
    except sr.RequestError as e:
        st.error(f"🌐 Google API error — check internet connection: {e}")
        return None
    except OSError as e:
        st.error(f"🎤 Microphone error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None


def get_audio_b64(text, lang="es"):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save("_tmp_audio.mp3")
        with open("_tmp_audio.mp3", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        os.remove("_tmp_audio.mp3")
        return b64
    except:
        return None


def play_audio(text, lang="es"):
    b64 = get_audio_b64(text, lang)
    if b64:
        st.markdown(
            f'<audio autoplay controls style="width:100%;margin-top:6px">'
            f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
            unsafe_allow_html=True
        )


# ── Session state ──
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "total_translations" not in st.session_state:
    st.session_state.total_translations = 0
if "mode" not in st.session_state:
    st.session_state.mode = "chat"


# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px'>
        <div style='font-family:Syne,sans-serif; font-size:22px; font-weight:800; 
                    background: linear-gradient(135deg,#4a9eff,#6b5bef); 
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent'>
            🌍 LinguaAI
        </div>
        <div style='font-size:12px; color:#6868a8; margin-top:4px'>
            English → Spanish Translator
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Model status
    status = "✅ Model Ready" if model_loaded else "⚠️ Model Not Found"
    color = "#4ade80" if model_loaded else "#f87171"
    st.markdown(f"<div style='font-size:13px; color:{color}'>{status}</div>",
                unsafe_allow_html=True)

    st.divider()

    # Stats
    st.markdown("**Session Stats**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{st.session_state.total_translations}</div>
            <div class='stat-label'>Translated</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{len(st.session_state.chat_history)}</div>
            <div class='stat-label'>Messages</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Mode selector
    st.markdown("**Mode**")
    mode_chat = st.button("💬 Chat Mode",
                          use_container_width=True,
                          type="primary" if st.session_state.mode == "chat" else "secondary")
    mode_voice = st.button("🎤 Voice Mode",
                           use_container_width=True,
                           type="primary" if st.session_state.mode == "voice" else "secondary")

    if mode_chat:
        st.session_state.mode = "chat"
        st.rerun()
    if mode_voice:
        st.session_state.mode = "voice"
        st.rerun()

    st.divider()

    # Example sentences
    st.markdown("**Quick Examples**")
    examples = [
        "I am happy",
        "Thank you very much",
        "Good morning",
        "I love you",
        "Where is the library",
        "The cat is sleeping",
    ]
    for ex in examples:
        if st.button(ex, key=f"sb_{ex}", use_container_width=True):
            with st.spinner("Translating..."):
                result = translate(ex)
            st.session_state.chat_history.append({
                "user": ex, "bot": result,
                "time": datetime.now().strftime("%H:%M"),
                "mode": "text"
            })
            st.session_state.total_translations += 1
            st.rerun()

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("""
    <div style='font-size:11px; color:#3a3a5a; text-align:center; margin-top:20px'>
        Built with Bidirectional LSTM + Attention<br>
        Trained on 118k sentence pairs
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════
st.markdown("""
<h1 style='font-family:Syne,sans-serif; font-size:32px; font-weight:800; 
           background:linear-gradient(135deg,#4a9eff,#6b5bef,#a855f7);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
           margin-bottom:4px'>
    English → Spanish Translator
</h1>
<p style='color:#6868a8; font-size:14px; margin-bottom:24px'>
    Type or speak English — get instant Spanish translation with audio playback
</p>
""", unsafe_allow_html=True)

# ── Chat History ──
if st.session_state.chat_history:
    st.markdown("### Conversation")
    for msg in st.session_state.chat_history:
        mode_icon = "🎤" if msg.get("mode") == "voice" else "⌨️"

        # User message
        safe_user = html.escape(msg['user'])
        st.markdown(f"""
        <div class='user-msg'>
            <div class='msg-label'>You {mode_icon}</div>
            {safe_user}
            <div class='msg-time'>{msg['time']}</div>
        </div>""", unsafe_allow_html=True)

        # Bot message
        safe_bot = html.escape(msg['bot'])
        st.markdown(f"""
        <div class='bot-msg'>
            <div class='msg-label'>🌍 Translation</div>
            {safe_bot}
            <div class='msg-time'>{msg['time']}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

# ══════════════════════════════════════════
# CHAT MODE
# ══════════════════════════════════════════
if st.session_state.mode == "chat":
    st.markdown("### 💬 Chat Mode")

    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Type your sentence",
            placeholder="e.g. I like soccer and going to the beach...",
            key="chat_input",
            label_visibility="collapsed"
        )
    with col2:
        send_btn = st.button("Send →", type="primary", use_container_width=True)

    # Options row
    col1, col2, col3 = st.columns(3)
    with col1:
        auto_play = st.checkbox("🔊 Auto-play Spanish audio", value=True)
    with col2:
        play_english = st.checkbox("🔊 Play English audio", value=False)
    with col3:
        show_confidence = st.checkbox("📊 Show word count", value=False)

    if send_btn and user_input.strip():
        with st.spinner("Translating..."):
            result = translate(user_input)

        # Add to history
        st.session_state.chat_history.append({
            "user": user_input,
            "bot": result,
            "time": datetime.now().strftime("%H:%M"),
            "mode": "text"
        })
        st.session_state.total_translations += 1

        # Show result
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🇬🇧 English**")
            st.info(user_input)
            if show_confidence:
                st.caption(f"Words: {len(user_input.split())}")
            if play_english:
                play_audio(user_input, lang="en")

        with col2:
            st.markdown("**🇪🇸 Spanish**")
            st.success(result)
            if show_confidence:
                st.caption(f"Words: {len(result.split())}")
            if auto_play:
                play_audio(result, lang="es")

        st.rerun()


# ══════════════════════════════════════════
# VOICE MODE
# ══════════════════════════════════════════
elif st.session_state.mode == "voice":
    st.markdown("### 🎤 Voice Mode")

    st.markdown("""
    <div style='background:#0f0f1a; border:1px solid #1e1e2e; border-radius:16px; 
                padding:24px; text-align:center; margin-bottom:20px'>
        <div style='font-size:48px; margin-bottom:12px'>🎤</div>
        <div style='font-family:Syne,sans-serif; font-size:18px; font-weight:600; 
                    color:#e8e8f0; margin-bottom:8px'>
            Speak in English
        </div>
        <div style='font-size:13px; color:#6868a8'>
            Click the button below and speak clearly into your microphone
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        record_btn = st.button(
            "🎤 Start Recording",
            type="primary",
            use_container_width=True,
            key="record_btn"
        )

    if record_btn:
        with st.spinner("🎤 Listening... speak now!"):
            detected_text = record_voice()

        if detected_text:
            st.markdown(f"""
            <div style='background:#0d1a2d; border:1px solid #1a3a5a; border-radius:12px;
                        padding:16px; margin:12px 0'>
                <div style='font-size:11px; color:#4a9eff; font-weight:600; 
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px'>
                    Detected Speech
                </div>
                <div style='font-size:16px; color:#c8d8f0'>{detected_text}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Translating..."):
                result = translate(detected_text)

            st.session_state.chat_history.append({
                "user": detected_text,
                "bot": result,
                "time": datetime.now().strftime("%H:%M"),
                "mode": "voice"
            })
            st.session_state.total_translations += 1

            # Show result
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🇬🇧 English (detected)**")
                st.info(detected_text)
            with col2:
                st.markdown("**🇪🇸 Spanish (translation)**")
                st.success(result)
                st.markdown("**🔊 Playing...**")
                play_audio(result, lang="es")

            st.rerun()

        else:
            st.error("❌ Could not detect speech. Please check your microphone and try again.")

    # Manual input fallback in voice mode
    st.divider()
    st.markdown("**Or type manually:**")
    manual_input = st.text_input(
        "Manual input",
        placeholder="Type if microphone is unavailable...",
        label_visibility="collapsed",
        key="voice_manual"
    )
    if st.button("Translate Text", key="voice_manual_btn"):
        if manual_input.strip():
            with st.spinner("Translating..."):
                result = translate(manual_input)
            st.session_state.chat_history.append({
                "user": manual_input,
                "bot": result,
                "time": datetime.now().strftime("%H:%M"),
                "mode": "text"
            })
            st.session_state.total_translations += 1
            st.success(f"**Spanish:** {result}")
            play_audio(result, lang="es")
            st.rerun()