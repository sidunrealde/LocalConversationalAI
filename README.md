# 🗣️🎥 Local Conversational AI Agent

A fully customizable conversational AI agent running locally on RTX 3090 with:
- 🎤 Voice input (speech-to-text via Whisper)
- 🧠 LLM responses (Ollama)
- 🔊 Voice output (TTS via Piper)
- 👤 Customizable personality & voice
- 📝 Conversation history
- 📊 Full debugging & logging

## 🎯 Features (Phase 1 - Complete)

✅ **Text & Voice Chat Loop**
- Type messages or use voice input (placeholder in Phase 1)
- Get responses from local LLM (qwen2.5:7b)
- Hear responses via TTS

✅ **Customizable Personality**
- 3 preset personalities (Unreal Engine expert, Coding assistant, Casual friend)
- Custom personality text editor
- Different response tones based on selected personality

✅ **Voice Customization**
- Multiple TTS voices: lessac, bryce, kristin, amy
- Real-time voice switching
- Audio quality controlled

✅ **Advanced Controls**
- Temperature & Top-P adjustable for creativity
- Conversation history persistence
- Debug info panel with logs

## 🔜 Planned Features (Phase 2+)

🔄 **Phase 2**: Knowledge Scoping
- Upload documents (PDF, TXT, DOCX)
- Vector embedding with semantic search
- Agent only answers from scoped knowledge
- Graceful refusal on out-of-scope questions

🎬 **Phase 3**: Talking Avatar Video
- Upload character image
- Auto-generate animated talking video
- Lip-sync to TTS audio
- Choice of LivePortrait (fast) or SadTalker (high-quality)

🎙️ **Phase 4**: Real Microphone Input
- WebRTC-based voice recording in Streamlit
- Real-time transcription
- No need for placeholder text

## 📂 Project Structure

```
LocalConversationalAI/
├── chat.py                    # Main Streamlit application
├── requirements.txt           # Python dependencies
├── SETUP.md                   # Installation & setup guide
├── TESTING.md                 # Testing checklist (Phase 1)
├── README.md                  # This file
│
├── modules/                   # Modular components
│   ├── __init__.py
│   ├── logger.py              # Logging configuration
│   ├── tts.py                 # Text-to-Speech (Piper)
│   ├── asr.py                 # Speech-to-Text (Whisper)
│   └── ollama_client.py       # LLM client wrapper
│
├── data/                      # Data storage
│   ├── voices/                # Cached TTS models
│   ├── avatars/               # Generated avatar videos (Phase 3)
│   └── documents/             # Uploaded documents (Phase 2)
│
└── logs/                      # Application logs
    └── agent_YYYYMMDD_HHMMSS.log
```

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Ollama (https://ollama.ai)
- RTX 3090 or similar GPU
- 16+ GB RAM, 20+ GB disk

### 2. Setup
```bash
# Clone/navigate to project
cd LocalConversationalAI

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Start Services
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Download model
ollama pull qwen2.5:7b

# Terminal 3: Run app
streamlit run chat.py
```

### 4. Access App
Open `http://localhost:8501` in browser

**First run takes 2-5 minutes** for model downloads. Watch for "✅ Engines ready!" message.

## 📖 Documentation

- **[SETUP.md](SETUP.md)**: Detailed installation & configuration
- **[TESTING.md](TESTING.md)**: Phase 1 testing checklist & benchmarks
- **[chat.py](chat.py)**: Main application (well-commented)
- **[modules/](modules/)**: Component documentation in docstrings

## 🧪 Testing

Follow the **[TESTING.md](TESTING.md)** checklist for comprehensive testing:

```
✅ Checkpoint 1: Initialization (0-2 min)
✅ Checkpoint 2: Text Input Loop (2-5 min)
✅ Checkpoint 3: Personality Switching (5-8 min)
✅ Checkpoint 4: Voice Selection (8-10 min)
✅ Checkpoint 5: Conversation History (10-12 min)
✅ Checkpoint 6: Advanced Options (12-14 min)
✅ Checkpoint 7: Debug Info (14-15 min)
✅ Checkpoint 8: Error Handling (15-20 min)
✅ Checkpoint 9: Avatar Upload (20-22 min)
✅ Checkpoint 10: Custom Personality (22-25 min)
```

**Expected total**: ~25 minutes for full Phase 1 test suite

## 📊 Performance (RTX 3090)

| Component | VRAM | Latency | Notes |
|-----------|------|---------|-------|
| LLM (qwen2.5:7b) | 6-8 GB | 3-8s/msg | Fast, good quality |
| ASR (Whisper medium) | 2-4 GB | 5-15s/min audio | GPU accelerated |
| TTS (Piper) | <1 GB | 2-5s | Fast synthesis |
| **Total** | **~10 GB** | **5-13s/msg** | Comfortable fit |

## 🔧 Configuration

### Environment Variables (Optional)
Create `.env` file:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
ASR_MODEL_SIZE=medium
TTS_VOICE=lessac
```

### Model Selection
- **LLM**: Edit `chat.py` line ~140 or use sidebar dropdown (Phase 2)
- **ASR**: Sidebar "ASR Model Size" - choose: tiny, base, small, medium, large
- **TTS**: Sidebar "Voice Settings" - choose: lessac, bryce, kristin, amy

## 🐛 Debugging

**View latest logs**:
```bash
# In app: Sidebar → Show Debug Info → Recent Logs

# Or in terminal:
tail -50 logs/agent_*.log
```

**Enable debug output**:
- Logger automatically creates detailed logs in `logs/` folder
- Both file (DEBUG level) and console (INFO level) output

**Check system health**:
```bash
# Monitor GPU
nvidia-smi -l 1

# Check Ollama models
ollama list

# Test connection
ollama list
```

## 🤝 Contributing

Phase 1 is production-ready. Next priorities:
1. Real mic input (Phase 2 prep)
2. Knowledge base integration (Phase 2)
3. Avatar animation (Phase 3)

## 📝 License

See [LICENSE](LICENSE) file

## 🆘 Troubleshooting

**"Ollama server is not running"**
```bash
# Start Ollama in another terminal
ollama serve
```

**GPU out of memory**
- Use smaller model: `neural-chat:7b` instead of `qwen2.5:7b`
- Use smaller ASR: `tiny` or `base`

**Slow transcription**
- Reduce ASR model size (tiny → base → medium)
- Check `nvidia-smi` for other GPU usage

**Voice download fails**
- Check internet connection
- Logs will show which voice failed
- Models are cached automatically after first download

See [SETUP.md](SETUP.md) for more troubleshooting

---

**Status**: Phase 1 ✅ Ready | Phase 2 🔄 Next | Phase 3+ 🔜 Planned

Last updated: February 2025
