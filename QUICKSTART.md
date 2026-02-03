# Quick Start Guide - Phase 1 Voice Chat

## 🎯 5-Minute Overview

```
Your Code                 Your Computer
───────────────          ─────────────────
chat.py         ────→    Streamlit UI
                         (http://localhost:8501)
                              ↓
                         [You type message]
                              ↓
modules/ollama_client.py ────→ Ollama LLM
                             (qwen2.5:7b)
                              ↓
modules/tts.py          ────→ Piper TTS
                             (audio synthesis)
                              ↓
                         [You hear response]
```

## 📋 Pre-Start Checklist

- [ ] Python 3.10+ installed
- [ ] Ollama downloaded (https://ollama.ai)
- [ ] RTX 3090 available (or similar GPU)
- [ ] 20+ GB free disk space
- [ ] Internet connection (for model downloads)

## ⚡ Quick Start (3 Steps)

### Step 1️⃣: Environment Setup (5 min)

```bash
# Open PowerShell in project folder

cd f:\Projects\LLM\LocalConversationalAI

# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install all packages (grab coffee ☕)
pip install -r requirements.txt
```

**Expected**: No errors, lots of downloading

### Step 2️⃣: Start Ollama (3 min)

**Open 2 new PowerShell windows**

**PowerShell #1** - Start Ollama:
```bash
ollama serve
```

Expected output:
```
Listening on 127.0.0.1:11434
```

**PowerShell #2** - Download model:
```bash
ollama pull qwen2.5:7b
```

Expected output:
```
pulling manifest...
downloading model...
(waits 5-15 min)
```

### Step 3️⃣: Run App (2 min)

**PowerShell #3** (with venv activated):
```bash
streamlit run chat.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

**Browser**: Opens automatically to `http://localhost:8501`

**Wait for**: "✅ Engines ready!" message (1-5 min on first run)

## ✅ First Test (Text Input)

1. **Sidebar Check**: Should show configuration options
2. **Type Message**: `"Hello, who are you?"`
3. **Click Send** (or press Enter)
4. **See Response**:
   - Text appears
   - Audio player shows below
   - Click play 🎵 to hear response

**Congrats!** 🎉 You have a working voice chat agent!

---

## 📊 What Just Happened

```
INPUT: "Hello, who are you?"
   ↓
[Streamlit App]
   ↓
[Ollama LLM] "I'm a helpful AI assistant..."
   ↓
[Piper TTS] Generates audio from text
   ↓
OUTPUT: Audio plays + text shown
```

---

## 🧪 Quick Tests (15 min)

### Test 1: Change Personality

1. **Sidebar** → "Agent Personality" dropdown
2. Change to: "Witty Unreal Engine expert from Bengaluru"
3. Type: `"What is machine learning?"`

**Expected**: Response sounds different (more witty/casual)

### Test 2: Change Voice

1. **Sidebar** → "Voice Settings" 
2. Change voice to: "bryce"
3. Type: `"Say hello in a friendly way"`
4. Click play on audio

**Expected**: Voice sounds different (smoother, different tone)

### Test 3: See History

1. Have 3-4 exchanges
2. Refresh page (F5)
3. Look at chat

**Expected**: All previous messages still there

### Test 4: Temperature (Creativity)

1. **Sidebar** → Expand "Advanced Options"
2. Set Temperature to **0.2** (low)
3. Type: `"Give me a creative game idea"`
4. Note response
5. Set Temperature to **0.95** (high)
6. Clear conversation
7. Type same question

**Expected**: Low = repetitive, High = more varied

---

## 🐛 Troubleshooting Quick Reference

| If You See | Do This |
|------------|---------|
| "Ollama server is not running" | Go to PowerShell #1, check `ollama serve` is running |
| App stuck loading | Wait 2-5 min, it's downloading models |
| No audio | Check browser speaker icon 🔊 not muted |
| Slow responses | Normal (5-13s). Check GPU: `nvidia-smi` |
| Models not downloading | Check internet, check logs: `logs/agent_*.log` |

**View Logs**:
- In app: Sidebar → "Show Debug Info" → "Recent Logs"
- Or in terminal: `tail logs/agent_*.log`

---

## 🎓 What to Explore Next

### Option A: Test Thoroughly
→ Follow [TESTING.md](TESTING.md) for 10 full checkpoints (25 min)

### Option B: Understand Code
→ Read module docstrings:
- Start: `modules/logger.py`
- Then: `modules/tts.py`
- Advanced: `modules/ollama_client.py`

### Option C: Customize
→ Edit `chat.py` to:
- Change default personality
- Add new voice options
- Tweak UI layout

### Option D: Next Phase
→ When ready for Phase 2 (Knowledge Scoping):
- See [SETUP.md](SETUP.md) Phase 2 section
- Implement document upload + RAG

---

## 📱 3-Window Layout (Recommended)

```
┌──────────────────────────────────────────────┐
│                                              │
│ PowerShell #1           PowerShell #2        │
│ (Ollama serve)          (Downloads models)   │
│                                              │
├──────────────────────────────────────────────┤
│                                              │
│         PowerShell #3 + Browser              │
│      (Streamlit app + chat UI)               │
│                                              │
└──────────────────────────────────────────────┘
```

---

## 🚀 Expected Timeline

```
T+0:00   Install packages
         ↓
T+5:00   Start Ollama
         ↓
T+8:00   Run app
         ↓
T+13:00  "✅ Engines ready!"
         ↓
T+13:30  First message sent
         ↓
T+20:00  Testing features
         ↓
T+25:00  Ready for Phase 2!
```

**Key Milestones**:
- ✅ 5 min: Dependencies installed
- ✅ 8 min: Model downloaded
- ✅ 13 min: App ready
- ✅ 25 min: Full test suite passing

---

## 📞 Need Help?

1. **Check logs**: `logs/agent_*.log`
2. **Read docs**:
   - [README.md](README.md) - Overview
   - [SETUP.md](SETUP.md) - Detailed setup
   - [TESTING.md](TESTING.md) - Full test checklist
3. **Check GPU**: `nvidia-smi` (should show Ollama processes)
4. **Restart services**: Ctrl+C in terminals and restart

---

## 🎯 Success Checklist

- [ ] Python 3.10+ running
- [ ] pip packages installed (no errors)
- [ ] Ollama server responding (`ollama list` works)
- [ ] Model downloaded (qwen2.5:7b shows in `ollama list`)
- [ ] Streamlit opens in browser
- [ ] "✅ Engines ready!" appears
- [ ] Can type and get text response
- [ ] Audio plays after response
- [ ] Personality switching works
- [ ] Voice switching works

**All checked?** → Phase 1 ✅ DONE! Go to [TESTING.md](TESTING.md) for full validation.

---

**Now open `http://localhost:8501` and start chatting! 🎉**
