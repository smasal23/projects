# API Usage Guide

## Gemini API

### Setup

1. Get API key from Google AI Studio
2. Add to `.env`:
   GEMINI_API_KEY=your_key_here

## Usage

- Used for:
  - Summarization
  - Stakeholder analysis
  - Risk detection

## Notes

- Keep temperature low (0.3) for consistency
- Limit input size to avoid token overflow
- Handle API failures gracefully

---

## Groq API (Backup)

- Used as fallback LLM
- Faster response times
- Limited context window compared to Gemini
