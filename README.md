# MDSS - Multi-Dimensional Dialogue Summarization System

A comprehensive dialogue analysis and summarization system powered by advanced NLP and LLMs that processes conversations through speaker diarization, intent classification, community detection, and hierarchical summarization.

## Overview

MDSS automatically analyzes conversations to identify speakers, categorize dialogue turns by intent, detect conversational modules, and generate coherent multi-level summaries. The system supports both text and audio input, making it versatile for various dialogue analysis applications.

## Key Features

### Core Capabilities
- **Advanced Speaker Diarization**: Automatically identifies 2-6 speakers using:
  - Speaker name extraction from formatted text
  - Semantic embedding clustering with style features
  - Turn-taking pattern refinement
  - Confidence scoring for each speaker assignment

- **Intent Classification**: Categorizes dialogue turns into 6 types:
  - INQUIRY (asking questions)
  - COMPLAINT (expressing problems)
  - REQUEST (asking for action)
  - FEEDBACK (giving opinions)
  - STATEMENT (declarations/explanations)
  - OTHER (uncategorized)

- **Sarcasm Detection**: Identifies sarcastic utterances using sentiment analysis

- **Dialogue Modularization**: Groups related utterances using:
  - Intent graph construction
  - Community detection (Louvain algorithm)
  - Semantic similarity-based clustering

- **Hierarchical Summarization**:
  - Extractive summaries using TextRank
  - Abstractive summaries via Gemini LLM
  - Final polished summary with consistency checking
  - Redundancy removal across summaries

- **Audio Support**: Transcribes audio using OpenAI Whisper (tiny model default)

- **Optimized API Usage**: Ultra-batched processing combining intent classification and module summarization in single API calls

## System Architecture

### Processing Pipeline

1. **Input Processing**
   - Audio transcription (Whisper) or text parsing
   - Sentence tokenization

2. **Speaker Diarization**
   - Name extraction (if formatted as "Speaker: text")
   - Semantic + style feature embedding
   - Agglomerative clustering
   - Turn-taking refinement

3. **Intent Graph Construction**
   - Node creation per utterance with embeddings
   - Edge weighting by semantic similarity
   - Sarcasm detection per node

4. **Module Detection**
   - Community detection on intent graph
   - Minimum module size filtering (default: 2 utterances)

5. **MEGA-BATCH Processing**
   - Combined intent classification + module summarization in 1 API call
   - Extractive summarization per module

6. **Final Assembly**
   - Cross-module summary synthesis
   - Consistency checking
   - Grammar correction and polishing
   - Redundancy removal

7. **Visualization**
   - Intent graph with color-coded nodes by intent type
   - Edge weights showing semantic similarity

## Requirements

### Dependencies
```
Python 3.8+
torch
openai-whisper
sentence-transformers
transformers
sumy
python-louvain
google-generativeai
gradio
spacy (with en_core_web_sm model)
nltk
networkx
numpy
pandas
matplotlib
scikit-learn (optional, falls back to numpy)
```

### API Requirements
- **Gemini API Key**: Required for abstractive summarization and intent classification
  - Get your key from [Google AI Studio](https://aistudio.google.com/apikey)

### Hardware
- GPU recommended for faster processing (CPU supported)
- Minimum 4GB RAM for Whisper + embedding models

## Installation

### Google Colab (Recommended)
The code automatically installs all dependencies when run in Colab:

```python
# Dependencies install automatically on first run
# Add your Gemini API key to Colab Secrets:
# 1. Click the key icon in left sidebar
# 2. Add name: GEMINI_API_KEY
# 3. Add your API key as the value
```

### Local Setup
```bash
# Clone repository
git clone <repository-url>
cd mdss

# Install dependencies
pip install openai-whisper sentence-transformers sumy python-louvain google-generativeai gradio
pip install transformers torch networkx numpy pandas matplotlib scikit-learn nltk

# Download spaCy model
python -m spacy download en_core_web_sm

# Set environment variable
export GEMINI_API_KEY="your_api_key_here"
```

## Usage

### Launching the Interface

Run the script to launch the Gradio web interface:

```python
python mdss.py
```

The interface will start and display:
- Local URL (e.g., `http://127.0.0.1:7860`)
- Public URL (shareable link)

### Interface Tabs

#### 1. Text Input Tab
- Paste dialogue text directly
- Supports formatted text with speaker labels:
  ```
  A (Project Manager): We're behind schedule.
  B (Developer): The API is unstable.
  ```
- Or plain text (auto-tokenizes into utterances)

#### 2. Audio Input Tab
- Record audio using microphone
- Upload audio files (WAV, MP3, etc.)
- Automatically transcribes via Whisper

## Configuration

### Adjustable Parameters

Located in the `MDSSConfig` class:

```python
class MDSSConfig:
    # Model Selection
    WHISPER_MODEL = "tiny"  # Options: tiny, base, small, medium, large
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    SARCASM_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    
    # Processing
    MIN_MODULE_SIZE = 2  # Minimum utterances per module
    EXTRACTIVE_SENTENCES = 3  # Sentences for TextRank summary
    REDUNDANCY_THRESHOLD = 0.85  # Similarity threshold for redundancy
    
    # Speaker Diarization
    AUTO_DETECT_SPEAKERS = True  # Auto-detect speaker count
    MAX_SPEAKERS = 6  # Maximum speakers when auto-detecting
    
    # API Management
    API_DELAY = 7  # Seconds between API calls
    RETRY_ATTEMPTS = 3  # Retry count for failed calls
    RETRY_DELAY = 10  # Seconds between retries
    
    # Batching
    BATCH_SIZE_INTENTS = 999  # Intent classification batch size
    BATCH_SIZE_MODULES = 2  # Module summarization batch size
```

## Output Structure

### Main Result Dictionary

```python
{
    'final_summary': str,  # Coherent summary of entire dialogue
    'module_summaries': {
        module_id: {
            'extractive': str,  # TextRank extractive summary
            'abstractive': str,  # Gemini abstractive summary
            'primary_intent': str  # Dominant intent (INQUIRY, COMPLAINT, etc.)
        },
        ...
    },
    'diarized': [
        {
            'speaker': str,  # "A", "SPEAKER_0", etc.
            'text': str,  # Utterance text
            'confidence': float  # Diarization confidence (0-1)
        },
        ...
    ],
    'consistency': {
        'consistent': bool,  # True if summary is consistent
        'issues': str or None  # Description of inconsistencies
    },
    'num_modules': int,  # Number of detected modules
    'num_speakers': int,  # Number of detected speakers
    'processing_time': float  # Total time in seconds
}
```

### Generated Files

- **`intent_graph.png`**: Visualization showing:
  - Nodes colored by intent type
  - Larger nodes indicate sarcastic utterances
  - Edges weighted by semantic similarity
  - Spring layout for clarity

## Technical Details

### Models Used

- **Whisper (tiny)**: Audio transcription
- **all-MiniLM-L6-v2**: Sentence embeddings (384 dimensions)
- **twitter-roberta-base-sentiment**: Sarcasm detection
- **Gemini 2.5 Flash**: Intent classification and abstractive summarization
- **TextRank**: Extractive summarization
- **Louvain Algorithm**: Community detection for modules

### Algorithms

- **Speaker Diarization**: Agglomerative clustering with ward linkage
- **Intent Graph**: Directed graph with cosine similarity edges
- **Module Detection**: Community detection on undirected graph projection
- **Redundancy Removal**: Cosine similarity filtering (threshold: 0.85)

## Performance

### Typical Processing Times
- 4-speaker, 10-utterance dialogue: ~40-50s
- Includes transcription, diarization, clustering, 3 API calls, and visualization

### Resource Usage
- GPU: Recommended for embeddings and Whisper
- RAM: ~4GB minimum
- Storage: ~2GB for models

## Troubleshooting

### Common Issues

**1. Gemini API Key Error**
```
Solution: Add GEMINI_API_KEY to Colab Secrets or environment variables
```

**2. spaCy Model Not Found**
```bash
python -m spacy download en_core_web_sm
```

**3. Audio Recording Not Working**
```
Solution: Upload audio file instead (microphone may not work in all Colab environments)
```

**4. Rate Limit Errors**
```
Solution: Increase API_DELAY in config (default: 7s)
```

**5. Out of Memory**
```
Solution: Use smaller WHISPER_MODEL (tiny instead of base/small)
```

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/rishitha-1612">
        <img src="https://github.com/rishitha-1612.png" width="100px;" alt="Rishitha Rasineni"/>
        <br />
        <sub><b>Rishitha Rasineni</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/2406-Sowmya">
        <img src="https://github.com/username2.png" width="100px;" alt="Sowmya P R"/>
        <br />
        <sub><b>Sowmya P R</b></sub>
      </a>
    </td>
  </tr>
</table>
