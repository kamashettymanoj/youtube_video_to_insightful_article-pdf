# YouTube to Article & PDF Generator

A Python project that takes a YouTube video URL, extracts the transcript,
summarizes it into a structured article using Google Gemini, and exports it as a PDF.

**Stack:** Python · Streamlit · Gemini API · ReportLab · youtube-transcript-api

---

## Features

- Paste any YouTube URL with captions
- Auto-fetches transcript (manual captions → auto-generated → any language)
- Sends transcript to Gemini and gets a structured article back
- Article sections: Title, TL;DR, Introduction, Key Points, Conclusion
- Keyword extraction using the RAKE algorithm
- Export article as PDF or plain TXT
- Settings for tone, length, and output language

---

## Project Structure

```
youtube_to_article/
├── app.py            ← Streamlit UI (start here)
├── utils.py          ← Core functions: transcript, Gemini, keywords, PDF
├── requirements.txt  ← Python packages needed
└── README.md         ← This file
```

---

## Setup & Installation

### Step 1 — Make sure Python 3.10+ is installed

```bash
python --version
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Get a Gemini API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **Create API Key**
4. Copy the key (starts with `AIza…`)

> The Gemini API has a free tier — no credit card needed for basic use.

---

## Running the App

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## How to Use

1. Paste your Gemini API key in the sidebar
2. Choose your writing tone, article length, and language
3. Paste a YouTube video URL in the input box
4. Click **Generate**
5. Read the article and download as PDF or TXT

---

## Troubleshooting

| Error | What to do |
|---|---|
| `TranscriptsDisabled` | Video has no captions. Try a different video. |
| `VideoUnavailable` | Video is private or deleted. |
| `403 Permission denied` | Check your Gemini API key. |
| `429 Rate limit` | Wait a few seconds and try again. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again. |

---

## Notes

- The transcript is trimmed to 15,000 characters before sending to Gemini to avoid token limit errors.
- Videos must have captions (manual or auto-generated) to work.
- Gemini API is free up to a reasonable limit for student/personal use.
