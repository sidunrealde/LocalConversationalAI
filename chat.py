"""
Local Conversational AI Agent - Main Streamlit Application

Features:
- Voice input via microphone (ASR)
- Text or speech input options
- Customizable personality and voice
- Voice output (TTS) with avatar animation support
- Conversation history

Phase 1: ASR + LLM + TTS (Voice Chat Loop)
"""

import streamlit as st
import numpy as np
import sounddevice as sd
import base64
import threading
import scipy.signal
import re
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import webrtcvad
import streamlit.components.v1 as components
from PIL import Image
from modules import setup_logger, TTSEngine, OllamaChat, ASREngine

# Initialize logger
logger = setup_logger(__name__)


def autoplay_audio(audio_bytes: bytes, mime_type: str = "audio/mpeg"):
    """Autoplay audio bytes in the browser."""
    if not audio_bytes:
        return 0
    
    # Estimate duration: rough heuristic
    estimated_duration = (len(audio_bytes) / 1024) * 0.05  # ~50ms per KB for MP3
    
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    html = f"""
    <audio autoplay="true" controls>
        <source src="data:{mime_type};base64,{b64}" type="{mime_type}">
    </audio>
    """
    components.html(html, height=50)
    
    return estimated_duration


def play_audio_sequence(audio_chunks: list[bytes], mime_type: str = "audio/mpeg"):
    """Play a list of audio chunks sequentially in a single player."""
    if not audio_chunks:
        return
    sources = [base64.b64encode(b).decode("utf-8") for b in audio_chunks if b]
    if not sources:
        return
    
    # Calculate total duration to delay rerun (rough estimate: 1KB ≈ 0.05s for MP3)
    total_bytes = sum(len(b) for b in audio_chunks if b)
    estimated_duration = (total_bytes / 1024) * 0.05  # seconds
    
    items_js = ",".join([f"'data:{mime_type};base64,{s}'" for s in sources])
    html = f"""
    <audio id="seq_player" controls autoplay></audio>
    <script>
      const sources = [{items_js}];
      const player = document.getElementById('seq_player');
      let idx = 0;
      const playNext = () => {{
        if (idx >= sources.length) return;
        player.src = sources[idx++];
        player.play();
      }};
      player.addEventListener('ended', playNext);
      playNext();
    </script>
    """
    components.html(html, height=60)
    
    return estimated_duration


def pop_sentence_chunk(text_buffer: str):
    """Extract a complete sentence from text buffer."""
    match = list(re.finditer(r"[.!?]\s+", text_buffer))
    if not match:
        return "", text_buffer
    last = match[-1].end()
    sentence = text_buffer[:last].strip()
    remaining = text_buffer[last:]
    return sentence, remaining


def build_rtc_config(stun_urls: list[str], turn_url: str | None, turn_user: str | None, turn_pass: str | None) -> dict:
    ice_servers = []
    if stun_urls:
        ice_servers.append({"urls": stun_urls})
    if turn_url and turn_user and turn_pass:
        ice_servers.append({
            "urls": [turn_url],
            "username": turn_user,
            "credential": turn_pass,
        })
    return {"iceServers": ice_servers} if ice_servers else {}


class RealtimeMicProcessor(AudioProcessorBase):
    """Always-on mic processor with VAD and buffering."""

    def __init__(self):
        self.vad = webrtcvad.Vad(1)  # 0=least aggressive, 3=most aggressive (changed from 2 to 1)
        self.sample_rate = 16000
        self.frame_ms = 30  # Changed from 20 to 30 for better VAD stability
        self.frame_size = int(self.sample_rate * self.frame_ms / 1000)
        self.buffer = bytearray()
        self.in_speech = False
        self.silence_frames = 0
        self.latest_pcm = None
        self.lock = threading.Lock()
        self.frame_count = 0
        self.speech_frame_count = 0

    def recv(self, frame):
        self.frame_count += 1
        audio = frame.to_ndarray()
        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        audio = audio.astype(np.float32)
        input_rate = frame.sample_rate
        if input_rate != self.sample_rate:
            audio = scipy.signal.resample_poly(audio, self.sample_rate, input_rate)

        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767).astype(np.int16).tobytes()

        bytes_per_frame = self.frame_size * 2
        for i in range(0, len(pcm), bytes_per_frame):
            chunk = pcm[i : i + bytes_per_frame]
            if len(chunk) < bytes_per_frame:
                continue

            try:
                is_speech = self.vad.is_speech(chunk, self.sample_rate)
                if is_speech:
                    self.speech_frame_count += 1
            except Exception:
                is_speech = False
                
            if is_speech:
                self.in_speech = True
                self.silence_frames = 0
                self.buffer.extend(chunk)
            else:
                if self.in_speech:
                    self.silence_frames += 1
                    if self.silence_frames <= 15:  # Increased from 5 to 15 for more tolerance
                        self.buffer.extend(chunk)
                    if self.silence_frames >= 30:  # Increased from 10 to 30 for longer pauses
                        with self.lock:
                            if len(self.buffer) > self.sample_rate * 2 * 0.3:  # Reduced min length from 0.5s to 0.3s
                                self.latest_pcm = bytes(self.buffer)
                        self.buffer.clear()
                        self.in_speech = False
                        self.silence_frames = 0

        return frame

    def pop_audio(self):
        with self.lock:
            data = self.latest_pcm
            self.latest_pcm = None
        return data

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Voice AI Agent - RTX 3090",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🗣️🎥 Voice AI Agent - RTX 3090")
st.markdown("*Conversational AI with voice I/O and custom personality*")

# ============================================================================
# SIDEBAR - INITIALIZATION & CONFIG
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Initialize session state for engines
    if "engines_initialized" not in st.session_state:
        logger.info("Initializing AI engines...")
        st.info("🚀 Initializing AI engines (first run, may take 30-60s)...")
        
        try:
            st.session_state.tts_engine = TTSEngine(voice_name="lessac")
            st.session_state.ollama_chat = OllamaChat(model="qwen2.5:7b")
            st.session_state.asr_engine = ASREngine(model_size="medium")
            
            st.session_state.engines_initialized = True
            logger.info("All engines initialized successfully")
            st.success("✅ Engines ready!")
        
        except Exception as e:
            logger.error(f"Failed to initialize engines: {e}")
            st.error(f"❌ Initialization failed: {e}")
            st.error("Please check logs and try again.")
            st.stop()
    
    # Get engines from session state
    tts_engine = st.session_state.tts_engine
    ollama_chat = st.session_state.ollama_chat
    asr_engine = st.session_state.asr_engine
    
    # --- Personality Config ---
    st.subheader("👤 Personality")
    personality = st.selectbox(
        "Agent Personality:",
        [
            "Witty Unreal Engine expert from Bengaluru",
            "Helpful coding assistant",
            "Casual friend",
            "Custom",
        ],
        key="personality_select",
    )

    # --- Realtime Output ---
    st.subheader("⚡ Realtime")
    realtime_output = st.checkbox(
        "Stream responses in realtime",
        value=True,
        key="realtime_output",
    )
    live_tts = st.checkbox(
        "Live TTS streaming (speak while generating)",
        value=True,
        key="live_tts",
    )

    # --- WebRTC Settings ---
    with st.expander("🎧 WebRTC Settings"):
        stun_list = st.text_area(
            "STUN servers (one per line)",
            value="stun:stun.l.google.com:19302\nstun:stun1.l.google.com:19302",
            key="stun_list",
        )
        turn_url = st.text_input("TURN URL (optional)", key="turn_url")
        turn_user = st.text_input("TURN Username", key="turn_user")
        turn_pass = st.text_input("TURN Password", type="password", key="turn_pass")
    
    if personality == "Custom":
        custom_personality = st.text_area(
            "Define custom personality:",
            value="You are a helpful AI assistant.",
            height=100,
            key="custom_personality",
        )
        system_prompt = custom_personality
    else:
        system_prompt = f"You are {personality}. Provide helpful, concise responses."
    
    logger.debug(f"System prompt: {system_prompt[:50]}...")
    
    # --- Voice Config ---
    st.subheader("🔊 Voice Settings")
    voice_options = tts_engine.get_available_voices()
    selected_voice = st.selectbox(
        "TTS Voice:",
        voice_options,
        index=0,
        key="voice_select",
    )
    
    if selected_voice != tts_engine.voice_name:
        tts_engine.set_voice(selected_voice)
        logger.info(f"Voice switched to: {selected_voice}")
    
    # Speech recognition settings
    st.subheader("🎤 Speech Recognition")
    asr_model_size = st.selectbox(
        "ASR Model Size:",
        asr_engine.get_available_models(),
        index=2,  # medium
        key="asr_model",
        help="Larger models = better accuracy but slower. tiny=fast, large=best",
    )
    
    if asr_model_size != asr_engine.model_size:
        asr_engine.set_model_size(asr_model_size)
        logger.info(f"ASR model switched to: {asr_model_size}")
    
    # --- Avatar Upload ---
    st.subheader("🎬 Avatar (Optional)")
    uploaded_file = st.file_uploader(
        "Upload character image (for future video avatar):",
        type=["png", "jpg", "jpeg"],
    )
    
    if uploaded_file:
        st.session_state.avatar_image = uploaded_file
        st.image(uploaded_file, caption="Talking Avatar", width=150)
        logger.debug(f"Avatar uploaded: {uploaded_file.name}")
    else:
        st.session_state.avatar_image = None
    
    # --- Advanced Options ---
    with st.expander("🔧 Advanced Options"):
        temperature = st.slider(
            "LLM Temperature (creativity):",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="temperature",
        )
        
        top_p = st.slider(
            "LLM Top-P (diversity):",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.1,
            key="top_p",
        )
    
    # --- Session Controls ---
    st.subheader("📋 Session")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        logger.info("Conversation cleared")
        st.success("Conversation cleared!")
    
    if st.button("📊 Show Debug Info", use_container_width=True):
        st.session_state.show_debug = not st.session_state.get("show_debug", False)

# ============================================================================
# MAIN CHAT AREA
# ============================================================================

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.debug("Conversation history initialized")

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    content = msg["content"]
    
    with st.chat_message(role):
        st.markdown(content)
        
        # Show audio player for assistant messages with audio
        if role == "assistant" and "audio" in msg:
            st.audio(msg["audio"], format="audio/mp3")

# ============================================================================
# INPUT HANDLING
# ============================================================================

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    input_method = st.radio(
        "Input Mode:",
        ["Text", "Voice (Mic)", "Voice (Local Continuous)"],
        horizontal=True,
        key="input_mode",
    )

with col2:
    if input_method == "Voice (Mic)":
        record_seconds = st.slider(
            "Record seconds:",
            min_value=1,
            max_value=15,
            value=5,
            step=1,
            key="record_seconds",
        )
        mic_button = st.button("🎤 Record & Transcribe", type="primary", use_container_width=True)
    elif input_method == "Voice (Local Continuous)":
        if "local_listening" not in st.session_state:
            st.session_state.local_listening = False
        
        if st.session_state.local_listening:
            if st.button("⏹️ Stop Listening", type="secondary", use_container_width=True):
                st.session_state.local_listening = False
        else:
            if st.button("🎤 Start Listening", type="primary", use_container_width=True):
                st.session_state.local_listening = True
        mic_button = False
    else:
        mic_button = False

with col3:
    if input_method == "Text":
        user_input = st.chat_input("Type your message...")
    else:
        user_input = None

# ============================================================================
# VOICE INPUT PROCESSING
# ============================================================================

if input_method == "Voice (Local Continuous)":
    st.caption("🎧 Local continuous listening using system mic + VAD")
    
    if st.session_state.get("local_listening", False):
        if st.session_state.get("ai_speaking", False):
            st.info("🔊 AI is speaking... (speak to interrupt)")
        else:
            st.success("🎤 Listening continuously... (speak anytime)")
        
        # Continuously listen in a loop (even during AI speaking to enable interruption)
        try:
            # Capture audio from system mic
            duration = 1.5  # Shorter chunks for faster interrupt detection
            sample_rate = 16000
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            
            audio_data = recording.squeeze()
            
            # Simple energy-based VAD with different thresholds
            energy = np.sqrt(np.mean(audio_data ** 2))
            
            # Use MUCH higher threshold during AI speaking to avoid self-interruption
            # User speaking to interrupt should be significantly louder/closer than speaker output
            is_ai_speaking = st.session_state.get("ai_speaking", False)
            
            if is_ai_speaking:
                # During AI speech: require very high energy (loud, close speech only)
                # This prevents picking up speaker output
                threshold = 0.08  # Very high threshold
                
                if energy > threshold:
                    with st.spinner("📝 Transcribing..."):
                        user_input = asr_engine.transcribe(
                            audio_data,
                            sample_rate=sample_rate,
                            language="en",
                        )
                    
                    if user_input and len(user_input.strip()) > 0:
                        # Require at least 3 words and very high energy for interrupt
                        word_count = len(user_input.split())
                        if word_count >= 3 and energy > 0.09:
                            # Likely a real interruption - user speaking loudly/closely
                            st.session_state.interrupt = True
                            st.warning("⏸️ Interrupting AI...")
                            logger.info(f"User interrupted AI (energy={energy:.4f}): {user_input[:50]}")
                        else:
                            # Still might be AI echo, ignore
                            logger.debug(f"Ignoring during AI speech: energy={energy:.4f}, words={word_count}, text='{user_input[:30]}'")
                            user_input = None
                            st.rerun()
                    else:
                        user_input = None
                        st.rerun()
                else:
                    user_input = None
                    st.rerun()
            else:
                # Normal listening mode: use higher threshold to avoid background noise
                threshold = 0.04  # Increased to reduce false positives from ambient noise
                
                if energy > threshold:
                    # Show energy level for debugging
                    logger.debug(f"Energy detected: {energy:.4f} (threshold: {threshold})")
                    
                    with st.spinner("📝 Transcribing..."):
                        user_input = asr_engine.transcribe(
                            audio_data,
                            sample_rate=sample_rate,
                            language="en",
                        )
                    
                    if user_input and len(user_input.strip()) > 0:
                        # Filter out common Whisper hallucinations and noise
                        word_count = len(user_input.split())
                        text_lower = user_input.strip().lower()
                        
                        # Common Whisper hallucination phrases (appears on silence/noise)
                        hallucination_phrases = [
                            "thank you", "thanks for watching", "subscribe",
                            "see you", "bye", "goodbye", "thank you for watching",
                            "like and subscribe", "don't forget to subscribe",
                            "thanks for listening", "you", "okay", "um", "uh",
                            "hmm", "ah", "oh", "...", ".", "", "the", "and",
                            "music", "applause", "silence", "[music]", "[applause]",
                            "i'm sorry", "sorry", "what", "huh", "yeah", "yes", "no",
                            "right", "so", "well", "like", "just", "i", "a", "the",
                            "foreign", "[foreign]", "subtitle", "captions",
                        ]
                        
                        is_hallucination = (
                            word_count < 2 or  # Single words are usually noise
                            text_lower in hallucination_phrases or
                            any(text_lower.startswith(p) for p in ["thank", "subscribe", "like and", "don't forget"]) or
                            any(text_lower.endswith(p) for p in ["for watching", "for listening", "next time"]) or
                            (word_count < 4 and energy < 0.06)  # Short + low energy = likely noise
                        )
                        
                        if is_hallucination:
                            logger.debug(f"Filtered hallucination: '{user_input}' (energy={energy:.4f}, words={word_count})")
                            user_input = None
                            st.rerun()  # Continue listening
                        else:
                            st.success(f"✅ Heard: {user_input}")
                            logger.info(f"Local mic transcription (energy={energy:.4f}): {user_input[:80]}...")
                            # Continue to process this input below
                    else:
                        user_input = None
                        st.rerun()  # Continue listening
                else:
                    user_input = None
                    st.rerun()  # Continue listening
                
        except Exception as e:
            logger.error(f"Local mic error: {e}", exc_info=True)
            st.error(f"❌ Mic error: {e}")
            user_input = None
            st.rerun()  # Continue listening despite error

if input_method == "Voice (Always Listening)":
    st.caption("🎧 Always listening. Start talking to interrupt the AI.")
    stun_urls = [s.strip() for s in st.session_state.get("stun_list", "").splitlines() if s.strip()]
    rtc_config = build_rtc_config(
        stun_urls,
        st.session_state.get("turn_url"),
        st.session_state.get("turn_user"),
        st.session_state.get("turn_pass"),
    )
    webrtc_ctx = webrtc_streamer(
        key="always_listening",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=RealtimeMicProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "audio": {"echoCancellation": True, "noiseSuppression": True},
            "video": False,
        },
        async_processing=True,
    )

    if webrtc_ctx.state.playing is False:
        st.warning("WebRTC not connected. Ensure mic permissions and network access to STUN/TURN.")

    with st.expander("🔍 WebRTC Diagnostics"):
        st.write(f"**State:** {webrtc_ctx.state}")
        st.write(f"**Playing:** {webrtc_ctx.state.playing}")
        st.write(f"**Audio Processor:** {webrtc_ctx.audio_processor is not None}")
        if webrtc_ctx.audio_processor:
            st.write(f"**In Speech:** {webrtc_ctx.audio_processor.in_speech}")
            st.write(f"**Buffer Size:** {len(webrtc_ctx.audio_processor.buffer)}")
            st.write(f"**Total Frames:** {webrtc_ctx.audio_processor.frame_count}")
            st.write(f"**Speech Frames:** {webrtc_ctx.audio_processor.speech_frame_count}")
            st.write(f"**Silence Frames:** {webrtc_ctx.audio_processor.silence_frames}")
            if webrtc_ctx.audio_processor.in_speech:
                st.success("🎤 Detecting speech!")
            elif webrtc_ctx.audio_processor.speech_frame_count > 0:
                st.info("✅ Audio detected (waiting for speech end)")
            else:
                st.warning("🔇 No speech detected yet (try speaking louder)")

    if webrtc_ctx.audio_processor:
        pcm_bytes = webrtc_ctx.audio_processor.pop_audio()
        if pcm_bytes:
            st.session_state.interrupt = True
            audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            with st.spinner("📝 Transcribing..."):
                user_input = asr_engine.transcribe(
                    audio_np,
                    sample_rate=16000,
                    language="en",
                )

            if not user_input:
                st.warning("No speech detected. Please try again.")
                logger.warning("ASR returned empty transcription")
            else:
                st.success("✅ Transcription complete")
                logger.info(f"Mic transcription: {user_input[:80]}...")

if input_method == "Voice (Push-to-Talk)":
    st.caption("🎧 Push-to-talk. Start/Stop to control recording.")
    stun_urls = [s.strip() for s in st.session_state.get("stun_list", "").splitlines() if s.strip()]
    rtc_config = build_rtc_config(
        stun_urls,
        st.session_state.get("turn_url"),
        st.session_state.get("turn_user"),
        st.session_state.get("turn_pass"),
    )
    webrtc_ctx = webrtc_streamer(
        key="push_to_talk",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=RealtimeMicProcessor,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration=rtc_config,
        desired_playing_state=st.session_state.get("ptt_active", False),
        async_processing=True,
    )

    if st.session_state.get("ptt_active", False) and webrtc_ctx.state.playing is False:
        st.warning("WebRTC not connected. Ensure mic permissions and network access to STUN/TURN.")

    with st.expander("🔍 WebRTC Diagnostics"):
        st.write(f"**State:** {webrtc_ctx.state}")
        st.write(f"**Playing:** {webrtc_ctx.state.playing}")
        st.write(f"**PTT Active:** {st.session_state.get('ptt_active', False)}")
        st.write(f"**Audio Processor:** {webrtc_ctx.audio_processor is not None}")
        if webrtc_ctx.audio_processor:
            st.write(f"**In Speech:** {webrtc_ctx.audio_processor.in_speech}")
            st.write(f"**Buffer Size:** {len(webrtc_ctx.audio_processor.buffer)}")
            st.write(f"**Total Frames:** {webrtc_ctx.audio_processor.frame_count}")
            st.write(f"**Speech Frames:** {webrtc_ctx.audio_processor.speech_frame_count}")
            st.write(f"**Silence Frames:** {webrtc_ctx.audio_processor.silence_frames}")
            if webrtc_ctx.audio_processor.in_speech:
                st.success("🎤 Detecting speech!")
            elif webrtc_ctx.audio_processor.speech_frame_count > 0:
                st.info("✅ Audio detected (waiting for speech end)")
            else:
                st.warning("🔇 No speech detected yet (try speaking louder)")

    if webrtc_ctx.audio_processor:
        pcm_bytes = webrtc_ctx.audio_processor.pop_audio()
        if pcm_bytes:
            st.session_state.interrupt = True
            audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            with st.spinner("📝 Transcribing..."):
                user_input = asr_engine.transcribe(
                    audio_np,
                    sample_rate=16000,
                    language="en",
                )

            if not user_input:
                st.warning("No speech detected. Please try again.")
                logger.warning("ASR returned empty transcription")
            else:
                st.success("✅ Transcription complete")
                logger.info(f"Mic transcription: {user_input[:80]}...")

if mic_button:
    try:
        sample_rate = 16000
        duration = st.session_state.get("record_seconds", 5)
        st.info(f"🎙️ Recording for {duration} seconds...")
        logger.info(f"Recording mic audio for {duration}s at {sample_rate} Hz")

        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        audio_data = recording.squeeze()
        if audio_data.size == 0:
            st.error("No audio captured. Please try again.")
            logger.warning("Mic recording captured no audio")
        else:
            with st.spinner("📝 Transcribing..."):
                user_input = asr_engine.transcribe(
                    audio_data,
                    sample_rate=sample_rate,
                    language="en",
                )

            if not user_input:
                st.warning("No speech detected. Please try again.")
                logger.warning("ASR returned empty transcription")
            else:
                st.success("✅ Transcription complete")
                logger.info(f"Mic transcription: {user_input[:80]}...")

    except Exception as e:
        logger.error(f"Mic recording failed: {e}", exc_info=True)
        st.error(f"❌ Mic error: {e}")

# ============================================================================
# MESSAGE PROCESSING & RESPONSE GENERATION
# ============================================================================

if user_input:
    st.session_state.interrupt = False
    st.session_state.ai_speaking = True  # Mark AI as speaking to pause listening
    logger.info(f"User input received: {user_input[:50]}...")
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                logger.debug("Sending to Ollama LLM...")
                
                # Prepare messages for LLM
                lm_messages = [{"role": "user", "content": user_input}]
                
                response_text = ""
                tts_pending = ""
                tts_chunks = []
                if st.session_state.get("realtime_output", True):
                    # Stream response tokens in realtime
                    stream_container = st.empty()
                    for token in ollama_chat.stream_chat(
                        messages=lm_messages,
                        system_prompt=system_prompt,
                        temperature=st.session_state.get("temperature", 0.7),
                        top_p=st.session_state.get("top_p", 0.9),
                    ):
                        if st.session_state.get("interrupt", False):
                            logger.info("Response interrupted by user")
                            st.warning("⏹️ Response interrupted")
                            break
                        response_text += token
                        stream_container.markdown(response_text)

                        if st.session_state.get("live_tts", True):
                            tts_pending += token
                            sentence, tts_pending = pop_sentence_chunk(tts_pending)
                            if sentence:
                                audio_chunk = tts_engine.synthesize(sentence)
                                tts_chunks.append(audio_chunk)
                else:
                    # Get full response from Ollama
                    response_text = ollama_chat.chat(
                        messages=lm_messages,
                        system_prompt=system_prompt,
                        temperature=st.session_state.get("temperature", 0.7),
                        top_p=st.session_state.get("top_p", 0.9),
                    )
                    st.markdown(response_text)
                
                logger.debug(f"LLM response: {response_text[:100]}...")
                
                # Generate audio response
                audio_bytes = None
                audio_duration = 0
                if not st.session_state.get("interrupt", False):
                    with st.spinner("🔊 Generating voice..."):
                        logger.debug("Synthesizing speech...")
                        if st.session_state.get("realtime_output", True) and st.session_state.get("live_tts", True):
                            # Speak any leftover text not yet spoken
                            leftover = tts_pending.strip()
                            if leftover:
                                audio_chunk = tts_engine.synthesize(leftover)
                                tts_chunks.append(audio_chunk)
                            audio_bytes = None
                            audio_duration = play_audio_sequence(tts_chunks, mime_type="audio/mpeg") or 0
                        else:
                            audio_bytes = tts_engine.synthesize(response_text)
                            audio_duration = autoplay_audio(audio_bytes, mime_type="audio/mpeg")
                        if audio_bytes is not None:
                            logger.debug(f"Audio generated: {len(audio_bytes)} bytes")
                
                # Wait for audio to finish playing before resuming listening
                if audio_duration > 0:
                    import time
                    time.sleep(audio_duration + 0.5)  # Add 0.5s buffer
                
                # Save to conversation history
                message_payload = {
                    "role": "assistant",
                    "content": response_text,
                }
                if not st.session_state.get("interrupt", False) and audio_bytes:
                    message_payload["audio"] = audio_bytes

                st.session_state.messages.append(message_payload)
                
                logger.info("Response generated and added to history")
                
                # Mark AI as done speaking and resume listening
                st.session_state.ai_speaking = False
                
                # Continue listening if in continuous mode
                if input_method == "Voice (Local Continuous)" and st.session_state.get("local_listening", False):
                    st.rerun()
            
            except Exception as e:
                # Clear speaking flag on error too
                st.session_state.ai_speaking = False
                logger.error(f"Error generating response: {e}", exc_info=True)
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg)

# ============================================================================
# DEBUG INFO
# ============================================================================

if st.session_state.get("show_debug", False):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🐛 Debug Info")
    
    with st.sidebar.expander("Session State"):
        debug_state = {
            "Messages count": len(st.session_state.messages),
            "Current personality": personality,
            "Current voice": tts_engine.voice_name,
            "ASR model": asr_engine.model_size,
            "LLM model": ollama_chat.model,
            "Temperature": st.session_state.get("temperature", 0.7),
            "Top-P": st.session_state.get("top_p", 0.9),
        }
        st.json(debug_state)
    
    with st.sidebar.expander("Recent Logs"):
        import os
        log_files = sorted(
            [f for f in os.listdir("logs") if f.endswith(".log")],
            reverse=True
        )
        if log_files:
            latest_log = f"logs/{log_files[0]}"
            with open(latest_log, "r") as f:
                log_content = f.read()
                # Show last 500 chars
                st.code(log_content[-1000:], language="log")

st.markdown("---")
st.caption("🚀 Phase 1: Voice Chat with Personality. Future: Knowledge Scoping + Avatar Video")
