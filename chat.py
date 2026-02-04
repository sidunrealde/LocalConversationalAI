"""
Local Conversational AI Agent - Main Streamlit Application

Features:
- Voice input via microphone (ASR) with WebRTC
- Text or speech input options
- Customizable personality and voice
- Voice output (TTS) with avatar animation support
- Conversation history
- Echo cancellation to prevent AI voice feedback

Phase 1: ASR + LLM + TTS (Voice Chat Loop)
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import base64
import time
import threading
import scipy.signal
import re
import queue
import sounddevice as sd
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from PIL import Image
from modules import setup_logger, TTSEngine, OllamaChat, ASREngine

# Initialize logger
logger = setup_logger(__name__)

# Shared state for audio level monitoring (thread-safe)
class AudioState:
    def __init__(self):
        self._lock = threading.Lock()
        self._energy = 0.0
        self._is_ai_speaking = False
        self._gate_threshold = 0.08
        self._speech_threshold = 0.03
        self._frame_count = 0
        
    def update_energy(self, energy: float):
        with self._lock:
            self._energy = energy
            self._frame_count += 1
            
    def get_energy(self) -> float:
        with self._lock:
            return self._energy
    
    def get_frame_count(self) -> int:
        with self._lock:
            return self._frame_count
            
    def set_ai_speaking(self, speaking: bool):
        with self._lock:
            self._is_ai_speaking = speaking
            
    def is_ai_speaking(self) -> bool:
        with self._lock:
            return self._is_ai_speaking
            
    def set_thresholds(self, speech: float, gate: float):
        with self._lock:
            self._speech_threshold = speech
            self._gate_threshold = gate
            
    def get_effective_threshold(self) -> float:
        with self._lock:
            return self._gate_threshold if self._is_ai_speaking else self._speech_threshold

# Global audio state
if "audio_state" not in st.session_state:
    st.session_state.audio_state = AudioState()

audio_state = st.session_state.audio_state


def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    """
    Callback to process each audio frame from WebRTC.
    Updates the shared audio state with energy levels.
    Returns silence to prevent echo.
    """
    try:
        arr = frame.to_ndarray()
        
        # Handle different array shapes
        if arr.ndim > 1:
            # Take first channel or average
            if arr.shape[0] <= arr.shape[1]:
                arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
            else:
                arr = arr[:, 0] if arr.shape[1] == 1 else arr.mean(axis=1)
        
        # Normalize to float32 [-1, 1]
        arr_float = arr.astype(np.float32)
        max_val = np.abs(arr_float).max()
        if max_val > 1.0:
            arr_float = arr_float / 32768.0
        
        # Calculate RMS energy
        energy = float(np.sqrt(np.mean(arr_float ** 2)))
        
        # Update shared state
        audio_state.update_energy(energy)
        
    except Exception as e:
        logger.error(f"Audio callback error: {e}")
    
    # Return silence to prevent echo
    try:
        arr = frame.to_ndarray()
        arr.fill(0)
        new_frame = av.AudioFrame.from_ndarray(arr, layout=frame.layout.name)
        new_frame.rate = frame.rate
        new_frame.sample_rate = frame.sample_rate
        new_frame.time_base = frame.time_base
        new_frame.pts = frame.pts
        return new_frame
    except:
        return frame


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


def play_audio_sequence(audio_chunks: list, mime_type: str = "audio/mpeg"):
    """Play a list of audio chunks sequentially in a single player."""
    if not audio_chunks:
        return 0
    sources = [base64.b64encode(b).decode("utf-8") for b in audio_chunks if b]
    if not sources:
        return 0
    
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
            st.session_state.asr_engine = ASREngine(model_size="small")
            
            # Preload ASR model
            st.info("⏳ Loading Whisper ASR model (first run takes ~60s)...")
            st.session_state.asr_engine.load_model()
            logger.info("ASR model loaded")
            
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
    
    # --- Echo Cancellation Settings ---
    st.subheader("🔇 Echo Cancellation")
    speech_threshold = st.slider(
        "Speech Detection Threshold",
        0.001, 0.2, 0.02, 0.001,
        key="speech_thresh_slider",
        help="Lower = more sensitive, Higher = less false positives"
    )
    gate_threshold = st.slider(
        "Echo Gate Threshold (when AI speaking)",
        0.01, 0.3, 0.05, 0.01,
        key="gate_thresh_slider",
        help="Audio below this level is suppressed during AI speech"
    )
    
    # Update audio state thresholds
    audio_state.set_thresholds(speech_threshold, gate_threshold)
    
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
# WEBRTC AUDIO CONNECTION (in sidebar)
# ============================================================================

with st.sidebar:
    st.header("🔌 Audio Connection")
    st.caption("WebRTC with Browser Echo Cancellation")
    
    # Create WebRTC streamer - using SENDONLY for mic input only
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,  # Only receive audio from browser
        audio_frame_callback=audio_frame_callback,
        media_stream_constraints={
            "video": False,
            "audio": {
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True,
                "sampleRate": 16000,
            }
        },
        rtc_configuration=RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }),
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing:
        st.success("✅ Mic Connected")
        st.caption("Browser AEC enabled")
    else:
        st.warning("🛑 Click START to connect mic")

# ============================================================================
# MAIN CHAT AREA
# ============================================================================

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.debug("Conversation history initialized")

# Initialize AI speaking state
if "ai_speaking" not in st.session_state:
    st.session_state.ai_speaking = False

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
        ["Text", "Voice (Mic)", "Voice (WebRTC Continuous)"],
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
    elif input_method == "Voice (WebRTC Continuous)":
        if "local_listening" not in st.session_state:
            st.session_state.local_listening = False
        
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("🎤 Listen", type="primary", use_container_width=True, disabled=st.session_state.local_listening):
                st.session_state.local_listening = True
                st.rerun()
        with col_stop:
            if st.button("⏹️ Stop", type="secondary", use_container_width=True, disabled=not st.session_state.local_listening):
                st.session_state.local_listening = False
                st.rerun()
        mic_button = False
    else:
        mic_button = False

with col3:
    if input_method == "Text":
        user_input = st.chat_input("Type your message...")
    else:
        user_input = None

# ============================================================================
# REAL-TIME AUDIO LEVEL DISPLAY (Always visible when WebRTC connected)
# ============================================================================

if input_method == "Voice (WebRTC Continuous)":
    st.markdown("---")
    
    # Audio level meter
    audio_debug_container = st.container()
    
    with audio_debug_container:
        col_level, col_status = st.columns([3, 1])
        
        with col_level:
            # Show current audio level
            current_energy = audio_state.get_energy()
            threshold = audio_state.get_effective_threshold()
            frame_count = audio_state.get_frame_count()
            
            # Create a visual level meter
            level_pct = min(current_energy * 50, 1.0)  # Scale for visibility
            
            st.metric(
                label="🎤 Mic Level",
                value=f"{current_energy:.4f}",
                delta=f"Threshold: {threshold:.3f}"
            )
            st.progress(level_pct, text=f"Energy: {current_energy:.4f} | Frames: {frame_count}")
            
        with col_status:
            if webrtc_ctx.state.playing:
                if current_energy > threshold:
                    st.success("🗣️ SPEECH")
                else:
                    st.info("🔇 Silence")
            else:
                st.warning("❌ No Mic")
    
    # Auto-refresh to update levels
    if webrtc_ctx.state.playing:
        time.sleep(0.1)  # Small delay to allow audio processing
        st.rerun()

# ============================================================================
# VOICE INPUT PROCESSING - WebRTC Continuous Mode
# ============================================================================

if input_method == "Voice (WebRTC Continuous)" and st.session_state.get("local_listening", False):
    st.caption("🎧 WebRTC continuous listening with echo cancellation")
    
    # Check WebRTC connection
    if not webrtc_ctx.state.playing:
        st.warning("⚠️ Please click START in the 'Audio Connection' sidebar to enable the microphone.")
        st.session_state.local_listening = False
    else:
        is_ai_speaking = st.session_state.get("ai_speaking", False)
        audio_state.set_ai_speaking(is_ai_speaking)
        
        if is_ai_speaking:
            st.warning("🔊 AI speaking... (speak loudly to interrupt)")
        else:
            st.success("🎤 Listening... (speak now, will auto-detect)")
        
        status_placeholder = st.empty()
        
        # Parameters for voice activity detection
        sample_rate = 16000
        silence_duration_to_stop = 1.5  # Seconds of silence to stop recording
        max_duration = 15.0  # Maximum recording time
        min_speech_duration = 0.3  # Minimum speech duration to accept
        
        # Collect audio using the audio receiver
        audio_chunks = []
        speech_detected = False
        silence_start = None
        recording_start = None
        
        try:
            # Use audio receiver to get frames
            if hasattr(webrtc_ctx, 'audio_receiver') and webrtc_ctx.audio_receiver:
                timeout_count = 0
                max_timeout = 150  # ~15 seconds at 0.1s per iteration
                
                while st.session_state.get("local_listening", False) and timeout_count < max_timeout:
                    try:
                        # Try to get audio frames
                        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=0.1)
                        
                        if not audio_frames:
                            timeout_count += 1
                            continue
                        
                        timeout_count = 0  # Reset on successful frame
                        
                        for frame in audio_frames:
                            # Convert frame to numpy array
                            arr = frame.to_ndarray()
                            
                            # Handle stereo to mono
                            if arr.ndim > 1:
                                if arr.shape[0] <= arr.shape[1]:
                                    arr = arr.mean(axis=0)
                                else:
                                    arr = arr.mean(axis=1)
                            
                            # Normalize
                            arr_float = arr.astype(np.float32)
                            if np.abs(arr_float).max() > 1.0:
                                arr_float = arr_float / 32768.0
                            
                            # Resample to 16kHz if needed
                            if frame.sample_rate != sample_rate:
                                num_samples = int(len(arr_float) * sample_rate / frame.sample_rate)
                                if num_samples > 0:
                                    arr_float = scipy.signal.resample(arr_float, num_samples)
                            
                            # Calculate energy
                            energy = float(np.sqrt(np.mean(arr_float ** 2)))
                            audio_state.update_energy(energy)
                            
                            # Get threshold
                            threshold = audio_state.get_effective_threshold()
                            
                            # Voice activity detection
                            if energy > threshold:
                                if not speech_detected:
                                    speech_detected = True
                                    recording_start = time.time()
                                    logger.info(f"Speech detected! Energy: {energy:.4f}")
                                
                                silence_start = None
                                audio_chunks.append(arr_float)
                                
                                elapsed = time.time() - recording_start
                                status_placeholder.success(f"🎤 Recording... {elapsed:.1f}s")
                                
                                # Check max duration
                                if elapsed >= max_duration:
                                    status_placeholder.info("⏱️ Max duration reached")
                                    break
                                    
                            elif speech_detected:
                                # Still recording, but silence now
                                audio_chunks.append(arr_float)
                                
                                if silence_start is None:
                                    silence_start = time.time()
                                
                                silence_elapsed = time.time() - silence_start
                                status_placeholder.info(f"🔇 Silence... {silence_elapsed:.1f}s")
                                
                                if silence_elapsed >= silence_duration_to_stop:
                                    status_placeholder.success("✓ Processing speech...")
                                    break
                            else:
                                status_placeholder.info(f"👂 Waiting... (level: {energy:.4f})")
                        
                        # Check if we should stop
                        if speech_detected and silence_start and (time.time() - silence_start) >= silence_duration_to_stop:
                            break
                        if speech_detected and recording_start and (time.time() - recording_start) >= max_duration:
                            break
                            
                    except queue.Empty:
                        timeout_count += 1
                        continue
                    except Exception as e:
                        logger.error(f"Frame processing error: {e}")
                        continue
                
                # Process collected audio
                if speech_detected and len(audio_chunks) > 5:
                    combined_audio = np.concatenate(audio_chunks)
                    total_duration = len(combined_audio) / sample_rate
                    
                    if total_duration >= min_speech_duration:
                        logger.info(f"Processing {total_duration:.1f}s of audio...")
                        
                        with st.spinner("📝 Transcribing..."):
                            user_input = asr_engine.transcribe(
                                combined_audio,
                                sample_rate=sample_rate,
                                language="en",
                            )
                        
                        if user_input and len(user_input.strip()) > 0:
                            st.success(f"✅ Heard: {user_input}")
                            logger.info(f"Transcription: {user_input}")
                        else:
                            st.warning("⚠️ No speech recognized")
                            user_input = None
                    else:
                        st.warning(f"⚠️ Audio too short ({total_duration:.1f}s)")
                        user_input = None
                else:
                    if timeout_count >= max_timeout:
                        st.warning("⏱️ Timeout - no audio received")
                    user_input = None
            else:
                st.error("❌ Audio receiver not available. Try refreshing the page.")
                user_input = None
                
        except Exception as e:
            logger.error(f"WebRTC error: {e}", exc_info=True)
            st.error(f"❌ Error: {e}")
            user_input = None
        
        # Stop listening after processing
        st.session_state.local_listening = False

# ============================================================================
# VOICE INPUT PROCESSING - Simple Mic Recording
# ============================================================================

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
        user_input = None

# ============================================================================
# MESSAGE PROCESSING & RESPONSE GENERATION
# ============================================================================

if user_input:
    st.session_state.interrupt = False
    st.session_state.ai_speaking = True
    audio_state.set_ai_speaking(True)
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
                    time.sleep(audio_duration + 0.5)
                
                # Save to conversation history
                message_payload = {
                    "role": "assistant",
                    "content": response_text,
                }
                if not st.session_state.get("interrupt", False) and audio_bytes:
                    message_payload["audio"] = audio_bytes

                st.session_state.messages.append(message_payload)
                
                logger.info("Response generated and added to history")
                
                # Mark AI as done speaking
                st.session_state.ai_speaking = False
                audio_state.set_ai_speaking(False)
                
                # Continue listening if in continuous mode
                if input_method == "Voice (WebRTC Continuous)":
                    st.session_state.local_listening = True
                    st.rerun()
            
            except Exception as e:
                st.session_state.ai_speaking = False
                audio_state.set_ai_speaking(False)
                logger.error(f"Error generating response: {e}", exc_info=True)
                st.error(f"❌ Error: {str(e)}")

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
            "AI Speaking": st.session_state.get("ai_speaking", False),
            "Continuous Listening": st.session_state.get("local_listening", False),
            "Audio Energy": audio_state.get_energy(),
            "Frame Count": audio_state.get_frame_count(),
        }
        st.json(debug_state)
    
    with st.sidebar.expander("Recent Logs"):
        import os
        log_dir = "logs"
        if os.path.exists(log_dir):
            log_files = sorted(
                [f for f in os.listdir(log_dir) if f.endswith(".log")],
                reverse=True
            )
            if log_files:
                latest_log = f"{log_dir}/{log_files[0]}"
                with open(latest_log, "r") as f:
                    log_content = f.read()
                    st.code(log_content[-1000:], language="log")

st.markdown("---")
st.caption("🚀 Phase 1: Voice Chat with Personality + Echo Cancellation. Future: Knowledge Scoping + Avatar Video")
