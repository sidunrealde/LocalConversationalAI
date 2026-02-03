# Setup & Installation Guide

## Phase 1: Voice Chat with Personality (ASR + LLM + TTS)

### Prerequisites

- **Python**: 3.10 or 3.11 (3.12 may have compatibility issues)
- **Ollama**: Running locally (download from https://ollama.ai)
- **CUDA**: For GPU acceleration (optional but recommended for RTX 3090)
- **FFmpeg**: For audio processing (optional)

### Step 1: Environment Setup

#### 1a. Create Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Or on Linux/Mac:
python3 -m venv venv
source venv/bin/activate
```

#### 1b. Install Dependencies
```bash
pip install -r requirements.txt
```

**Expected time**: 2-5 minutes (longer on first run due to package downloads)

### Step 2: Start Ollama Server

```bash
ollama serve
```

This starts the Ollama API server on `http://localhost:11434`

**Check if running**: Open another terminal and run:
```bash
ollama list
```

### Step 3: Pull LLM Model

In another terminal:
```bash
ollama pull qwen2.5:7b
```

This downloads the 7B parameter model (~4.5 GB). 
- **Time**: 5-15 minutes depending on internet speed
- Alternative models: `llama2:7b`, `mistral:7b`, `neural-chat:7b`

### Step 4: Run the Application

```bash
streamlit run chat.py
```

This opens the Streamlit app at `http://localhost:8501`

**First run**:
- Will download Whisper ASR model (~3 GB) - takes 2-5 minutes
- Will download Piper TTS voice model (~100 MB) - takes 30-60 seconds
- Shows "✅ Engines ready!" when done

---

## Testing Checklist (Phase 1)

### ✅ Checkpoint 1: Initialization (0-2 min)

- [ ] No errors in terminal output
- [ ] Streamlit app opens at `http://localhost:8501`
- [ ] Sidebar shows "✅ Engines ready!"
- [ ] Debug Info section shows all engines initialized

**If stuck**: Check logs in `logs/` folder

### ✅ Checkpoint 2: Text Input (2-5 min)

1. Select **Text** input mode
2. Type in chat: *"Hello, who are you?"*
3. Click **Send**

**Expected output**:
- User message appears in chat
- "🤔 Thinking..." spinner shows
- LLM response appears
- "🔊 Generating voice..." spinner shows
- Audio player appears with response

**If fails**:
- Check Ollama is running: `ollama list` in terminal
- Check logs for error messages

### ✅ Checkpoint 3: Personality Switching (5-8 min)

1. Sidebar → Change "Agent Personality"
2. Try with each option:
   - "Witty Unreal Engine expert"
   - "Helpful coding assistant"
   - "Casual friend"
3. Type same question: *"What is AI?"*

**Expected**: Responses have different tone/style for each personality

**Debug**: Check "Show Debug Info" → "Session State" for current personality

### ✅ Checkpoint 4: Voice Selection (8-10 min)

1. Sidebar → "Voice Settings" → Change TTS voice
2. Regenerate a response (type new question)

**Expected**: Audio sounds different

**Available voices**: lessac, bryce, kristin, amy

**Troubleshooting**: If voice doesn't change, check logs for download errors

### ✅ Checkpoint 5: Conversation History (10-12 min)

1. Have a multi-turn conversation (3-5 exchanges)
2. Refresh the page (F5)

**Expected**: Conversation history persists

3. Sidebar → "Clear Conversation"

**Expected**: Chat clears, history gone

### ✅ Checkpoint 6: Advanced Options (12-14 min)

1. Sidebar → Expand "Advanced Options"
2. Adjust Temperature slider (0.0 to 1.0)
   - Low (0.3): More deterministic, repetitive
   - High (0.9): More creative, varied
3. Ask: *"Give me 3 creative ideas for a game"*

**Expected**: Varying levels of creativity in responses

### ✅ Checkpoint 7: Debug Info (14-15 min)

1. Sidebar → "Show Debug Info"
2. Expand "Session State"

**Expected**: Shows:
- Messages count
- Current personality
- Current voice
- ASR model
- LLM model
- Temperature/Top-P values

### ✅ Checkpoint 8: Error Handling (15-20 min)

**Test graceful error handling:**

1. Stop Ollama server (Ctrl+C in terminal)
2. Try to send a message in chat
3. Should see error: "Ollama server is not running"

**Expected**: App doesn't crash, error is logged

**Test recovery**:
4. Restart Ollama: `ollama serve`
5. Try sending message again
6. Should work normally

---

## Log Files

Logs are saved to `logs/agent_YYYYMMDD_HHMMSS.log`

**View latest logs**:
```bash
# Windows
Get-Content (Get-Item logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName -Tail 50

# Linux/Mac
tail -50 logs/$(ls -t logs/ | head -1)
```

**Or in app**: Debug Info → "Recent Logs"

---

## Performance Tuning (RTX 3090)

### Memory Management
- **Ollama** (qwen2.5:7b): ~6-8 GB VRAM
- **Whisper** (medium): ~2-4 GB VRAM
- **Piper** (TTS): <1 GB
- **Embeddings** (future): ~1-2 GB

Total: ~8-10 GB (fits comfortably on RTX 3090's 24 GB)

### Speed Optimization

**For faster responses**:
1. Use smaller ASR model: `tiny` or `base` (trades accuracy for speed)
2. Reduce LLM temperature (more deterministic = faster)
3. Use smaller LLM: `neural-chat:7b` instead of `qwen2.5:7b`

**For better quality**:
1. Use larger ASR model: `large-v3` (slower but better accuracy)
2. Use larger LLM: `llama2:13b` (slower but better responses)

---

## Troubleshooting

### Issue: "Ollama server is not running"
**Solution**: Start Ollama in another terminal
```bash
ollama serve
```

### Issue: GPU out of memory (OOM)
**Solution**: 
- Use smaller models
- Close other GPU applications
- Check VRAM: `nvidia-smi`

### Issue: Slow transcription
**Solution**:
- Use smaller ASR model: medium → base → tiny
- Sidebar → "ASR Model Size" → Select "tiny" or "base"

### Issue: Voice model download fails
**Solution**:
- Check internet connection
- Try again (downloads are cached)
- Check logs for specific error

### Issue: Streamlit keeps rerunning
**Solution**:
- This is normal on first run (many components loading)
- App will stabilize after initial setup
- Close Sidebar after testing if distracting

---

## Next Steps (Phase 2+)

- ✅ **Phase 1 (Current)**: ASR + LLM + TTS voice chat
- 🔜 **Phase 2**: Knowledge scoping (RAG)
- 🔜 **Phase 3**: Talking avatar video generation
- 🔜 **Phase 4**: Real-time mic input via WebRTC

---

## Support

Check logs in `logs/` folder for detailed error messages
