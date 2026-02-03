# Testing Checklist - Phase 1: Voice Chat with Personality

## Quick Start Test (5 minutes)

```
[ ] 1. Ollama server running: ollama serve
[ ] 2. Virtual env activated: .\venv\Scripts\Activate.ps1
[ ] 3. Install deps: pip install -r requirements.txt
[ ] 4. Start app: streamlit run chat.py
[ ] 5. Wait for "✅ Engines ready!" message
```

---

## Checkpoint Tests

### Test 1: Initialization (Should pass automatically)
**Expected**: App loads without errors

- [ ] No Python errors in terminal
- [ ] Streamlit opens at `http://localhost:8501`
- [ ] Sidebar shows "✅ Engines ready!"
- [ ] Configuration section loads:
  - [ ] Personality dropdown visible
  - [ ] Voice Settings dropdown visible
  - [ ] ASR Model Size dropdown visible
  - [ ] Avatar uploader visible

**Command to debug**:
```bash
tail -100 logs/agent_*.log
```

---

### Test 2: Text Input → LLM → TTS Loop
**Expected**: Full pipeline works end-to-end

**Steps**:
1. Select "Text" input mode
2. Type in chat input: `"Hello, how are you?"`
3. Click Send / Press Enter

**Check**:
- [ ] User message appears in chat with "user" label
- [ ] "🤔 Thinking..." spinner appears
- [ ] LLM response appears (text from qwen2.5:7b)
- [ ] "🔊 Generating voice..." spinner appears
- [ ] Audio player appears with audio controls
- [ ] Can play audio and hear voice output

**If fails**:
```bash
# Check Ollama is running
ollama list

# Check logs
tail -50 logs/agent_*.log
```

---

### Test 3: Personality Switching
**Expected**: Different personality = different response tone

**Steps**:
1. Sidebar → "Agent Personality"
2. Select "Witty Unreal Engine expert from Bengaluru"
3. Clear conversation (sidebar button)
4. Ask: `"What is machine learning?"`
5. Note tone/style of response
6. Switch to "Helpful coding assistant"
7. Ask same question

**Check**:
- [ ] Response 1 sounds witty/informal
- [ ] Response 2 sounds professional/helpful
- [ ] Both responses are different in tone despite same question

**Debug if stuck**:
- [ ] Check "Show Debug Info" → "Session State"
- [ ] Verify "Current personality" matches selection

---

### Test 4: Voice Switching
**Expected**: Different TTS voice = audibly different audio

**Steps**:
1. Sidebar → "Voice Settings"
2. Select voice: "lessac" (current)
3. Type question: `"Say hello in a friendly way"`
4. Note voice characteristics
5. Switch to different voice (e.g., "bryce")
6. Ask similar question

**Check**:
- [ ] Audio sounds noticeably different between voices
- [ ] All available voices work: lessac, bryce, kristin, amy
- [ ] No errors when switching voices

**Available voices to test**:
- [ ] lessac (default, neutral)
- [ ] bryce (smooth)
- [ ] kristin (crisp)
- [ ] amy (warm)

---

### Test 5: Conversation History Persistence
**Expected**: Chat history survives page refresh

**Steps**:
1. Have conversation (3 exchanges minimum)
2. Note messages in chat
3. Press F5 (refresh page)
4. Wait for app to reload

**Check**:
- [ ] All previous messages still visible
- [ ] User messages intact
- [ ] Assistant messages intact
- [ ] Audio players still functional

**Test 5b: Clear Conversation**:
1. Sidebar → "Clear Conversation" button
2. Click it

**Check**:
- [ ] All messages disappear
- [ ] Chat area empty
- [ ] Success message shows
- [ ] New input field ready

---

### Test 6: Advanced Options
**Expected**: Temperature affects response creativity

**Steps**:
1. Sidebar → "Advanced Options" (expand)
2. Set Temperature = 0.2 (low creativity)
3. Clear conversation
4. Ask: `"Give me 3 creative game ideas"`
5. Note responses (should be similar/repetitive)
6. Set Temperature = 0.95 (high creativity)
7. Clear conversation
8. Ask same question

**Check**:
- [ ] Low temp responses are more consistent/repetitive
- [ ] High temp responses are more varied
- [ ] Top-P slider also works (0.0-1.0)

---

### Test 7: Debug Info Panel
**Expected**: Debug panel shows correct system state

**Steps**:
1. Sidebar → "Show Debug Info" button
2. Expand "Session State"

**Check**:
- [ ] Messages count matches actual count
- [ ] Current personality matches sidebar selection
- [ ] Current voice matches selection
- [ ] ASR model matches selection
- [ ] LLM model = "qwen2.5:7b"
- [ ] Temperature value matches slider
- [ ] Top-P value matches slider

**Bonus**: Expand "Recent Logs"
- [ ] Shows last 1000 chars of log file
- [ ] Contains timestamps and log levels

---

### Test 8: Error Handling
**Expected**: Graceful error messages, no crashes

**Test 8a: Ollama Down**:
1. Stop Ollama (Ctrl+C in terminal)
2. Try to send message
3. Should see error: "❌ Error: Ollama server is not running"

**Check**:
- [ ] App doesn't crash
- [ ] Error message is clear
- [ ] Can still interact with UI

**Test 8b: Recovery**:
1. Restart Ollama: `ollama serve`
2. Wait 5 seconds
3. Try to send message again

**Check**:
- [ ] App recovers automatically
- [ ] Message goes through successfully

---

### Test 9: Avatar Upload (Prep for Phase 3)
**Expected**: Avatar image uploads and displays

**Steps**:
1. Sidebar → "Avatar (Optional)"
2. Upload a PNG or JPG image of a character/face
3. Image should display in sidebar

**Check**:
- [ ] File uploader accepts images
- [ ] Image preview shows in sidebar
- [ ] Image dimensions reasonable (100-500px)

**Note**: Avatar video generation not yet implemented (Phase 3)

---

### Test 10: Custom Personality
**Expected**: Can define custom personality and use it

**Steps**:
1. Sidebar → "Agent Personality"
2. Select "Custom"
3. Text area appears with default text
4. Replace with: `"You are a pirate. Speak like a pirate in every response."`
5. Clear conversation
6. Ask: `"What is programming?"`

**Check**:
- [ ] Response uses pirate dialect
- [ ] Custom personality persists across messages (until refresh)
- [ ] Can switch back to presets

---

## Stress Tests (Optional)

### Long Conversation Test
**Steps**:
1. Have 10-15 message exchanges
2. Check memory usage: `nvidia-smi`

**Expected**:
- [ ] No OOM errors
- [ ] VRAM usage stable (~8-10 GB)
- [ ] App remains responsive

### Rapid Fire Test
**Steps**:
1. Send 3 messages quickly without waiting for responses

**Expected**:
- [ ] Queue processes correctly
- [ ] Responses are in order
- [ ] No dropped messages

### Long Input Test
**Steps**:
1. Type very long message (500+ words)
2. Send

**Expected**:
- [ ] LLM processes without error
- [ ] Response is reasonable length
- [ ] TTS handles long text

---

## Performance Benchmarks (RTX 3090)

Track these metrics:

**First Run (one-time)**:
- [ ] ASR model download: _____ seconds
- [ ] TTS model download: _____ seconds
- [ ] Total initialization time: _____ seconds

**Typical Message (once initialized)**:
- [ ] LLM response generation: _____ seconds
- [ ] TTS synthesis: _____ seconds
- [ ] Total end-to-end: _____ seconds

**Example expected times**:
- LLM (qwen2.5:7b): 3-8 seconds
- TTS (Piper): 2-5 seconds
- Total: 5-13 seconds per message

---

## Log File Validation

**File location**: `logs/agent_YYYYMMDD_HHMMSS.log`

**Check for**:
- [ ] File created on test start
- [ ] Timestamps are in order
- [ ] No CRITICAL or ERROR lines (except intentional test errors)
- [ ] INFO lines show expected flow:
  - [ ] "Initializing AI engines..."
  - [ ] "User input received:"
  - [ ] "Sending to Ollama LLM..."
  - [ ] "Response generated and added to history"

**Example valid log**:
```
2025-02-03 14:23:45 - __main__ - INFO - User input received: Hello how are you?
2025-02-03 14:23:45 - __main__ - DEBUG - System prompt: You are Witty Unreal Engine expert...
2025-02-03 14:23:48 - modules.ollama_client - DEBUG - Response received: 725 chars
2025-02-03 14:23:50 - modules.tts - DEBUG - Synthesis complete: 45832 bytes
```

---

## Sign-Off Checklist

When all tests pass, mark these:

- [ ] ✅ Initialization test passed
- [ ] ✅ Text input → LLM → TTS pipeline works
- [ ] ✅ Personality switching works
- [ ] ✅ Voice switching works
- [ ] ✅ Conversation history persists
- [ ] ✅ Advanced options (temp, top-p) work
- [ ] ✅ Debug info panel accurate
- [ ] ✅ Error handling graceful
- [ ] ✅ Avatar upload works
- [ ] ✅ Custom personality works
- [ ] ✅ No OOM errors
- [ ] ✅ Performance acceptable

**Status**: Phase 1 ✅ READY FOR PHASE 2

---

## Next: Phase 2 Setup

Once Phase 1 is fully tested:
1. Install additional dependencies: `pip install sentence-transformers faiss-cpu`
2. Implement RAG (knowledge scoping)
3. Add document upload feature
4. Test knowledge-scoped responses

See `SETUP.md` for detailed Phase 2 instructions (coming soon).
