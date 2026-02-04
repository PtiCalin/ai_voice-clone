# Testing Log

This document serves as a log for all testing activities performed on the AI Voice Clone project.

## Testing Framework

- **Unit Tests**: pytest
- **Integration Tests**: Custom scripts
- **Audio Quality Tests**: PESQ, STOI metrics
- **Performance Tests**: Inference time, memory usage

## Test Categories

### 1. Unit Tests
- Model loading and initialization
- Audio preprocessing functions
- Feature extraction accuracy
- Loss function calculations

### 2. Integration Tests
- End-to-end voice cloning pipeline
- Data loading and batching
- Model training workflow
- Inference and synthesis

### 3. Audio Quality Tests
- Voice similarity assessment
- Audio artifact detection
- Naturalness evaluation

### 4. Performance Tests
- Training time benchmarks
- Inference speed metrics
- Memory consumption analysis

## Automated Testing Checklist

The project should steadily move toward repeatable, automated checks to reduce regression risk. The list below is a suggested baseline for expanding automated coverage.

### Pytest Coverage Targets
- **Unit**: Run fast, deterministic tests for feature extraction, preprocessing, and model configuration defaults.
- **Integration**: Validate end-to-end flows with minimal fixtures (short WAV samples, mocked file I/O).
- **Regression**: Preserve known-good outputs (hashes or metrics) for small reference inputs.

### Suggested Commands
```bash
# Fast local feedback
pytest -q

# Run only integration tests (mark with @pytest.mark.integration)
pytest -m integration

# Generate a coverage report
pytest --cov=ai_voice-clone --cov-report=term-missing
```

### Automation Notes
- Keep integration fixtures small (<5 seconds of audio) to ensure CI runs in minutes.
- Use deterministic seeds for preprocessing and model initialization to prevent flaky tests.
- Prefer mocks for external audio devices or network resources.

## Evaluation Metrics

Automated evaluation should quantify audio quality and speaker similarity across reference and generated samples. Capture results in a consistent report so future model changes are comparable.

### Core Metrics
- **PESQ**: Objective speech quality metric (narrowband/wideband).
- **STOI**: Intelligibility score for synthesized speech.
- **Speaker Similarity**: Cosine similarity between reference and generated embeddings (e.g., ECAPA-TDNN).
- **MOS Prediction (Optional)**: Use MOSNet or similar model for predicted naturalness.

### Evaluation Workflow
1. Collect reference audio and generated samples per speaker.
2. Normalize loudness and resample to consistent sample rate.
3. Compute metrics for each pair and aggregate (mean/median, std dev).
4. Store outputs in a versioned report (CSV/JSON) alongside model metadata.

### Example Evaluation Report Fields
- `model_version`
- `dataset_split`
- `speaker_id`
- `pesq`
- `stoi`
- `speaker_similarity`
- `mos_pred`
- `timestamp`

## Test Log Template

### Test Session: [Date] - [Tester Name]

**Test Category**: [Unit/Integration/Audio Quality/Performance]

**Test Case**: [Brief description]

**Environment**:
- OS: [Windows/Linux/macOS]
- Python Version: [3.x.x]
- GPU/CPU: [Details]
- Dependencies: [Key versions]

**Steps Performed**:
1. [Step 1]
2. [Step 2]
3. ...

**Expected Results**:
- [Expected outcome]

**Actual Results**:
- [Actual outcome]

**Pass/Fail**: [PASS/FAIL]

**Issues Found**:
- [Bug descriptions, error messages]

**Screenshots/Audio Samples**: [Links or attachments]

**Notes**:
- [Additional observations]

---

## Recent Test Sessions

[Add test logs here as they are performed]
