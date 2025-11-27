# MDSS 

Multi-Dimensional Dialogue Summarization System Using LLMs 

## Overview

MDSS is a dialogue analysis and summarization system that processes conversations through speaker diarization, intent classification, and hierarchical summarization. The system automatically identifies speakers, categorizes dialogue turns, and generates coherent summaries.

## Features

- Automatic speaker diarization (2-6 speakers)
- Intent classification (inquiry, complaint, request, feedback, statement)
- Sarcasm detection
- Dialogue clustering using community detection
- Hierarchical summarization (extractive and abstractive)
- Audio transcription support via Whisper
- Optimized API usage (3 calls total)

## Requirements

- Python 3.8+
- Google Colab or local environment
- Gemini API key

## Usage

Running the Interface
When you run the script, a Gradio web interface will automatically launch with a shareable link.
The interface provides two tabs:
1. Text Input Tab: Paste or type dialogue text directly
2. Audio Input Tab: Record audio using your microphone or upload an audio file

Simply select your input method, provide the dialogue content, and click the "Analyze" button. The system will process the input and display:
1. Processing time
2. Final summary
3. Statistics (number of speakers and modules)
4. Module breakdown with intent classifications

## Configuration

Set up your Gemini API key in Google Colab Secrets or environment variables:

```python
# In Colab: Add GEMINI_API_KEY to Secrets
# Local: Set environment variable
export GEMINI_API_KEY="your_key_here"
```

## Output Structure

The system returns a dictionary with:

final_summary: Coherent summary of the entire dialogue
module_summaries: Per-module summaries containing:

extractive: TextRank-based extractive summary
abstractive: Gemini-generated abstractive summary
primary_intent: Dominant intent type for the module

diarized: Speaker-segmented utterances with:

speaker: Speaker identifier (e.g., "A", "SPEAKER_0")
text: Utterance text
confidence: Diarization confidence score

consistency: Consistency check results with:

consistent: Boolean indicating if summary is consistent
issues: Description of any inconsistencies found

num_modules: Number of detected dialogue modules
num_speakers: Number of detected speakers
processing_time: Total processing time in seconds

Additionally, an intent graph visualization is saved as intent_graph.png showing the conversation flow and intent relationships.
## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/rishitha-1612">
        <img src="https://github.com/username1.png" width="100px;" alt="Contributor 1"/>
        <br />
        <sub><b>Rishitha Rasineni</b></sub>
      </a>
      <br />
    </td>
    <td align="center">
      <a href="https://github.com/username2">
        <img src="https://github.com/username2.png" width="100px;" alt="Contributor 2"/>
        <br />
        <sub><b>Sowmya P R</b></sub>
      </a>
      <br />
    </td>
    <td
