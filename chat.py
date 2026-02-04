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
import base64
import time
import threading
import scipy.signal
import re
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, AudioProcessorBase
import av
import queue
from PIL import Image
from modules import setup_logger, TTSEngine, OllamaChat, ASREngine

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_queue = queue.Queue()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
             self.audio_queue.put(frame)
        except Exception as e:
             logger.error(f"Queue put error: {e}")
             
        # Return silence to prevent echo
        try:
            # Create silence from existing frame properties
            arr = frame.to_ndarray()
            arr[:] = 0 # Mute
            new_frame = av.AudioFrame.from_ndarray(arr, layout=frame.layout.name)
            new_frame.rate = frame.rate
            new_frame.sample_rate = frame.sample_rate
            new_frame.time_base = frame.time_base
            new_frame.pts = frame.pts
            return new_frame
        except Exception as e:
            logger.error(f"Silence generation error: {e}")
            return frame # Fallback to echo if silence fails (better than breaking pipeline)

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



# Note: build_rtc_config and RealtimeMicProcessor removed (WebRTC cleanup)

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
            
            # Preload ASR model (this can take 30-60s on first run)
            st.info("⏳ Loading Whisper ASR model (first run takes ~60s)...")
            st.session_state.asr_engine.load_model()
            logger.info("ASR model loaded")
            
            # Preload ASR model (this can take 30-60s on first run)
            st.info("⏳ Loading Whisper ASR model (first run takes ~60s)...")
            st.session_state.asr_engine.load_model()
            logger.info("ASR model loaded")
            
            # Defer WASAPI initialization to manual user action to prevent startup crashes
            st.session_state.wasapi_capture = None
            st.session_state.echo_canceller = None
            
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
    wasapi_capture = st.session_state.get("wasapi_capture")
    
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
    
    # --- Audio Device Selection ---
    # --- WebRTC Audio Connection ---
    with st.sidebar:
        st.header("🔌 Audio Connection")
        
        # This uses the browser's native audio engine + AEC
        # This uses the browser's native audio engine + AEC
        ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDRECV,
            audio_receiver_size=1024,
            media_stream_constraints={"video": False, "audio": True},
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            async_processing=True,
        )
        
        # Javascript to MUTE the browser audio element (Prevent Echo)
        st.markdown(
            """
            <script>
            function muteStreamlitAudio() {
                const audios = document.querySelectorAll("audio");
                audios.forEach(audio => {
                    audio.muted = true;
                });
            }
            // Run periodically to catch the element
            setInterval(muteStreamlitAudio, 1000);
            </script>
            """,
            unsafe_allow_html=True
        )
        
        if ctx.state.playing:
            st.success("✅ Connected (Browser Audio)")
            st.caption("Using Chrome/Edge Echo Cancellation")
        else:
            st.warning("🛑 Click START to connect audio")
            
    # Legacy variables (mocked to Nones/Default)
    wasapi = None
    allow_interruptions = True # Always reliable with WebRTC AEC
    speech_threshold = st.slider("Speech Threshold", 0.01, 0.5, 0.03, 0.01, key="speech_thresh_slider")

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
    
    # Debug panel
    with st.expander("🔍 Audio Debug", expanded=True):
        debug_placeholder = st.empty()
        energy_bar = st.empty()
    
    if st.session_state.get("local_listening", False):
        is_ai_speaking = st.session_state.get("ai_speaking", False)
        
        if is_ai_speaking:
            # Listening during AI speech with echo cancellation
            st.warning("🔊 AI speaking... (listening with echo cancellation - speak to interrupt)")
        else:
            st.success("🎤 Listening... (speak, will auto-detect when you stop)")
        
        # Dynamic VAD-based recording with WASAPI Echo Cancellation
        try:
            sample_rate = 16000
            
            # Parameters for dynamic recording
            chunk_duration = 0.3  # 300ms chunks
            chunk_samples = int(chunk_duration * sample_rate)
            max_duration = 15.0  # Maximum recording time
            silence_threshold = 0.02  # Energy threshold for silence
            silence_duration_to_stop = 1.0  # Seconds of silence to stop
            
            # Thresholds
            speech_threshold = 0.04  # Same threshold whether AI speaking or not
            
            logger.info(f"Starting recording loop. AI speaking: {is_ai_speaking}")
            
            audio_chunks = []
            silence_chunks = 0
            speech_detected = False
            chunks_for_silence_stop = int(silence_duration_to_stop / chunk_duration)
            max_chunks = int(max_duration / chunk_duration)
            
            status_text = st.empty()
            
            # Ensure WASAPI capture is initialized
            # Ensure WebRTC is connected
            if not ctx.state.playing:
                st.warning("Please click START in the 'Audio Connection' sidebar to enable the microphone.")
                st.stop()
            
            status_text = st.empty()
            
            frame_count = 0
                # Loop while connected
                while ctx.state.playing:
                    if not ctx.audio_receiver:
                        status_text.text("⏳ Waiting for audio stream...")
                        time.sleep(0.1)
                        continue
                        
                    try:
                        audio_frames = ctx.audio_receiver.get_frames(timeout=0.1)
                    except queue.Empty:
                        status_text.text("⏳ No audio frames...")
                        time.sleep(0.01)
                        continue
                        
                    if not audio_frames:
                        time.sleep(0.01)
                        continue
                    
                    frame_count += 1
                        
                    for frame in audio_frames:
                        # Convert to numpy
                        sound = frame.to_ndarray()
                        
                        # Stereo to Mono if needed
                        if sound.ndim > 1:
                            sound = sound.mean(axis=1)
                            
                        # Resample if needed (WebRTC usually 48k, we want 16k)
                        if frame.sample_rate and frame.sample_rate != sample_rate:
                            # Calculate number of samples
                            num_samples = int(len(sound) * sample_rate / frame.sample_rate)
                            if num_samples <= 0:
                                continue
                            sound = scipy.signal.resample(sound, num_samples)
                        
                        # Normalize float32
                        if sound.dtype != np.float32:
                            sound = sound.astype(np.float32) / 32768.0
                            
                        # Accumulate logic (unchanged mostly)
                        # We process frame-by-frame (typically 10-20ms)
                        # But existing logic expects "chunks" of 300ms?
                        # Actually existing logic just calculates energy of "chunk".
                        # If "chunk" is small, energy is still valid.
                        # But we should accumulate 300ms worth?
                        # Let's just use the frame as the chunk. It works fine for VAD.
                        
                        chunk = sound
                        
                        # Debug info
                        # ...
                        
                        # Calculate energy
                        energy = np.sqrt(np.mean(chunk ** 2))
                        
                        # Dynamic Thresholding
                        effective_threshold = speech_threshold
                        if is_ai_speaking:
                            # WebRTC AEC is perfect, but if user has speakers, maybe bump slightly?
                            # effective_threshold *= 1.5
                            pass 

                        # Update debug display
                        energy_pct = float(min(energy * 20, 1.0))
                        energy_bar.progress(energy_pct, text=f"Energy: {energy:.4f} / {effective_threshold:.4f}")
                        
                        if frame_count % 50 == 0:
                             logger.debug(f"Frame {frame_count}: Energy={energy:.5f} (Threshold={effective_threshold:.4f})")
                        
                        if energy > effective_threshold:
                            speech_detected = True
                            silence_chunks = 0
                            audio_chunks.append(chunk)
                            recorded_time = len(audio_chunks) * (len(chunk)/sample_rate) # Approximate
                            
                            if is_ai_speaking:
                                status_text.text(f"🎤 Interrupting... {recorded_time:.1f}s")
                            else:
                                status_text.text(f"🎤 Recording... {recorded_time:.1f}s")
                            debug_placeholder.markdown(f"**Recording:** `{recorded_time:.1f}s`")
                        elif speech_detected:
                            # Speech ended logic
                            audio_chunks.append(chunk)
                            silence_chunks += 1
                            # Silence duration check...
                            # Since frame duration varies, we use count? frame is usually 10ms. 
                            # 1.0s = 100 chunks.
                            if silence_chunks > 50: # Approx 0.5-1s
                                status_text.text("✓ Speech ended, processing...")
                                raise StopIteration # Break inner loop to process
                        else:
                            # Waiting...
                            # If AI speaking, check interrupts
                             pass

                    # End for frame
                # End while
                
            except StopIteration:
                 # Processing block (unchanged)
                 pass
            except Exception as e:
                logger.error(f"Audio loop error: {e}", exc_info=True)
                st.error(f"Error: {e}")
                
            if speech_detected and len(audio_chunks) > 10:
                # Process Audio
                 # Combine all chunks
                audio_data = np.concatenate(audio_chunks)
                
                # If AI was speaking, stop it (interruption)
                if is_ai_speaking:
                    st.session_state.ai_speaking = False
                    logger.info("AI speech interrupted by user!")
                    
                    debug_placeholder.markdown(f"""
                    **Captured Audio:**
                    - Duration: `{total_duration:.1f}s`
                    - Samples: `{len(audio_data)}`
                    - Echo Cancelled: `{is_ai_speaking}`
                    """)
                    
                    logger.info(f"Captured {total_duration:.1f}s of speech, transcribing...")
                    
                    with st.spinner("📝 Transcribing..."):
                        user_input = asr_engine.transcribe(
                            audio_data,
                            sample_rate=sample_rate,
                            language="en",
                        )
                    
                    logger.info(f"Transcription result: '{user_input}'")
                    
                    if user_input and len(user_input.strip()) > 0:
                        st.success(f"✅ Heard: {user_input}")
                        logger.info(f"Transcription accepted: {user_input}")
                        # Will be processed below
                    else:
                        logger.debug("Empty transcription result")
                        user_input = None
                        st.rerun()
                        

        except Exception as e:
            logger.error(f"Local mic error: {e}", exc_info=True)
            st.error(f"❌ Mic error: {e}")
            user_input = None
            st.rerun()

# Note: WebRTC-based voice modes removed to eliminate mic conflicts with sounddevice

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
                        
                        # Mark AI as speaking BEFORE playback starts
                        st.session_state.ai_speaking = True
                        
                        if st.session_state.get("realtime_output", True) and st.session_state.get("live_tts", True):
                            # Speak any leftover text not yet spoken
                            leftover = tts_pending.strip()
                            if leftover:
                                audio_chunk = tts_engine.synthesize(leftover)
                                tts_chunks.append(audio_chunk)
                            audio_bytes = None
                            
                            # Note: WASAPI echo cancellation captures loopback automatically
                            # No need to manually set reference signal anymore
                            
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
