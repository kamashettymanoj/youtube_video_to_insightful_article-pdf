# utils.py
# Helper functions for the YouTube to Article & PDF Generator project
# Handles: URL parsing, transcript fetching, text cleaning,
#          article generation (Gemini), keyword extraction, PDF building

import re
import io
import json
import textwrap
import unicodedata
from typing import Optional
from datetime import datetime

# YouTube transcript library
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)

# For calling the Gemini REST API
import requests

# Keyword extraction
from rake_nltk import Rake
import nltk

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
    ListFlowable,
    ListItem,
    KeepTogether,
)

# Download required NLTK data on first run
for resource in ["punkt", "stopwords", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


# =============================================================================
# 1. EXTRACT VIDEO ID FROM URL
# =============================================================================

def extract_video_id(url: str) -> Optional[str]:
    """
    Pull the 11-character video ID out of any standard YouTube URL.

    Works with:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID

    Returns the video ID string, or None if no match is found.
    """
    pattern = r"(?:v=|youtu\.be/|embed/|shorts/|watch\?v=)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None


# =============================================================================
# 2. FETCH TRANSCRIPT
# =============================================================================

# Languages to try first when looking for captions
PREFERRED_LANGUAGES = ["en", "en-US", "en-GB", "en-AU", "en-IN"]


def _segments_to_text(segments) -> str:
    """
    Convert transcript segments to a plain string.
    Handles both old dict-style and new object-style segments
    returned by different versions of youtube-transcript-api.
    """
    parts = []
    for seg in segments:
        if isinstance(seg, dict):
            # Old API style: {"text": "...", "start": 0.0, "duration": 1.5}
            parts.append(seg.get("text", ""))
        elif hasattr(seg, "text"):
            # New API style: FetchedTranscriptSnippet object with .text attribute
            parts.append(seg.text)
        else:
            parts.append(str(seg))
    return " ".join(parts)


def fetch_transcript(video_id: str):
    """
    Download the transcript/captions for a YouTube video.

    Strategy:
      1. Quick path — call get_transcript() directly for English
      2. If that fails, list all transcripts and try manual → auto → any language

    Returns a tuple: (transcript_text, language_code, error_message)
    If successful, error_message will be None.
    """

    # --- Quick path: try get_transcript() directly (fastest, most reliable) ---
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id, languages=PREFERRED_LANGUAGES)
        text = _segments_to_text(segments)
        cleaned = clean_transcript(text)
        if len(cleaned.strip()) >= 50:
            return cleaned, "en", None
    except Exception:
        pass  # fall through to the full lookup below

    # --- Full path: list all transcripts and pick the best one ----------------
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        transcript = None

        # 1. Manual English captions
        try:
            transcript = transcript_list.find_manually_created_transcript(PREFERRED_LANGUAGES)
        except NoTranscriptFound:
            pass

        # 2. Auto-generated English captions
        if transcript is None:
            try:
                transcript = transcript_list.find_generated_transcript(PREFERRED_LANGUAGES)
            except NoTranscriptFound:
                pass

        # 3. Any language available
        if transcript is None:
            available = list(transcript_list)
            if not available:
                return "", "unknown", (
                    "No captions found for this video. "
                    "Try a video that has subtitles or auto-generated captions enabled."
                )
            transcript = available[0]

        raw_data = transcript.fetch()
        language  = transcript.language_code

        text    = _segments_to_text(raw_data)
        cleaned = clean_transcript(text)

        if len(cleaned.strip()) < 50:
            return "", language, "Transcript is too short to process (less than 50 characters)."

        return cleaned, language, None

    except TranscriptsDisabled:
        return "", "unknown", (
            "Captions are disabled for this video by the uploader."
        )
    except VideoUnavailable:
        return "", "unknown", (
            "Video is unavailable — it may be private, deleted, or age-restricted."
        )
    except CouldNotRetrieveTranscript as e:
        return "", "unknown", f"Could not retrieve transcript: {e}"
    except Exception as e:
        return "", "unknown", (
            f"Transcript error ({type(e).__name__}): {e}\n\n"
            "Make sure the video is public and has captions enabled."
        )


# =============================================================================
# 3. CLEAN TRANSCRIPT TEXT
# =============================================================================

# Patterns to strip out from raw transcripts
NOISE_PATTERNS = [
    r"\[.*?\]",   # e.g. [Music], [Applause]
    r"\(.*?\)",   # e.g. (inaudible)
    r"<[^>]+>",   # stray HTML tags
    r"\buh+\b",   # filler: uh, uhh
    r"\bum+\b",   # filler: um, umm
    r"\bhmm+\b",  # filler: hmm
]

def clean_transcript(text: str) -> str:
    """
    Clean up a raw YouTube transcript by removing noise and normalising whitespace.
    """
    # Fix odd Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove known noise patterns
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Collapse multiple spaces and newlines into single spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =============================================================================
# 4. GENERATE ARTICLE USING GOOGLE GEMINI
# =============================================================================

# Gemini REST endpoint (gemini-pro model via generateContent)
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

# Max characters to send so we stay within Gemini's token limits
MAX_TRANSCRIPT_CHARS = 15000

def generate_article_with_gemini(
    transcript: str,
    api_key: str,
    tone: str = "Professional",
    target_words: int = 600,
    language: str = "English",
    include_tldr: bool = True,
) -> tuple:
    """
    Send the transcript to Google Gemini and get back a structured article.

    Returns:
        (article_dict, error_message)

    article_dict contains these keys:
        title, tldr, introduction, key_points (list), conclusion, body_text
    """

    # Trim the transcript if it is very long
    truncated = transcript[:MAX_TRANSCRIPT_CHARS]
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        truncated += "\n\n[Transcript truncated…]"

    # Tone instructions
    tone_guide = {
        "Professional":      "Write in a clear, authoritative, professional style.",
        "Casual & Friendly": "Write in a warm, conversational, easy-to-read style.",
        "Academic":          "Write in a formal, scholarly style with precise language.",
        "Journalistic":      "Write in a crisp, news-style with punchy, factual sentences.",
    }.get(tone, "Write in a professional style.")

    tldr_instruction = (
        'Include a "tldr" field: one sentence (25 words max) summarising the whole article.'
        if include_tldr
        else 'Set the "tldr" field to an empty string "".'
    )

    prompt = textwrap.dedent(f"""
        You are an expert content writer. Convert the YouTube transcript below into a
        well-structured article written in {language}.

        {tone_guide}
        Target length: around {target_words} words for the full article.

        Respond ONLY with valid JSON — no markdown fences, no extra text.
        Use this exact structure:

        {{
          "title":        "<Engaging article title, max 12 words>",
          "tldr":         "<One-sentence summary>",
          "introduction": "<2-3 paragraph introduction>",
          "key_points":   ["<Point 1>", "<Point 2>", "...(up to 8 points)"],
          "conclusion":   "<1-2 paragraph conclusion>",
          "body_text":    "<Full article text concatenated: intro + key points + conclusion>"
        }}

        Important:
        - Rewrite everything in your own words. Do not copy the transcript.
        - Every key_points item must be a complete sentence.
        - Make the title specific and engaging.
        {tldr_instruction}

        Transcript:
        {truncated}
    """).strip()

    params  = {"key": api_key}
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature":     0.7,
            "maxOutputTokens": 2048,
        },
    }

    try:
        response = requests.post(
            GEMINI_API_URL,
            params=params,
            json=payload,
            timeout=90,
        )

        # Handle common HTTP errors
        if response.status_code == 400:
            return {}, "Bad request — check that your API key is correct and active."
        if response.status_code == 403:
            return {}, "Permission denied — your Gemini API key may not have access to this model."
        if response.status_code == 429:
            return {}, "Rate limit reached. Please wait a moment and try again."
        if response.status_code != 200:
            return {}, f"API returned HTTP {response.status_code}: {response.text[:300]}"

        data = response.json()

        # Extract the text from Gemini's response structure
        try:
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return {}, "Could not parse the Gemini response structure."

        if not raw_text.strip():
            return {}, "Gemini returned an empty response."

        # Remove any accidental markdown code fences
        raw_text = re.sub(r"```(?:json)?", "", raw_text).strip().strip("`").strip()

        # Parse as JSON
        try:
            article = json.loads(raw_text)
        except json.JSONDecodeError:
            # Try to recover by finding the JSON object inside the response
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                article = json.loads(match.group())
            else:
                return {}, f"Response was not valid JSON.\n\nReceived: {raw_text[:400]}"

        # Fill in any missing fields with safe defaults
        defaults = {
            "title":        "Video Summary",
            "tldr":         "",
            "introduction": "",
            "key_points":   [],
            "conclusion":   "",
            "body_text":    "",
        }
        for key, default in defaults.items():
            if key not in article:
                article[key] = default

        # Build body_text if it came back empty
        if not article.get("body_text"):
            article["body_text"] = (
                article["introduction"] + " " +
                " ".join(article["key_points"]) + " " +
                article["conclusion"]
            )

        return article, None

    except requests.exceptions.Timeout:
        return {}, "Request timed out (90 seconds). The transcript may be too long."
    except requests.exceptions.ConnectionError:
        return {}, "Network error. Please check your internet connection."
    except Exception as e:
        return {}, f"Unexpected error: {type(e).__name__}: {e}"


# =============================================================================
# 5. KEYWORD EXTRACTION
# =============================================================================

def extract_keywords(text: str, top_n: int = 12) -> list:
    """
    Extract the most relevant keywords from a block of text using RAKE.

    RAKE (Rapid Automatic Keyword Extraction) works by scoring words
    based on how often they appear together. It doesn't need a trained model.

    Returns a list of keyword strings (title-cased).
    """
    try:
        sample = text[:8000]  # only need a sample for good results

        r = Rake(min_length=1, max_length=3)
        r.extract_keywords_from_text(sample)

        phrases = r.get_ranked_phrases()[:top_n]

        results = []
        for phrase in phrases:
            phrase = phrase.strip()
            # Skip very short or numeric-only phrases
            if len(phrase) > 2 and not phrase.isnumeric():
                results.append(phrase.title())

        return results[:top_n]

    except Exception:
        return []


# =============================================================================
# 6. BUILD PDF
# =============================================================================

# Colour scheme
COLOR_PURPLE      = colors.HexColor("#667eea")
COLOR_PURPLE_DARK = colors.HexColor("#4f46e5")
COLOR_PURPLE_LITE = colors.HexColor("#ede9fe")
COLOR_BODY_TEXT   = colors.HexColor("#374151")
COLOR_GREY        = colors.HexColor("#9ca3af")
COLOR_DARK        = colors.HexColor("#111827")

def build_pdf(
    article_data: dict,
    keywords: list,
    video_url: str,
    language: str = "English",
) -> tuple:
    """
    Generate a formatted A4 PDF from the article data.

    Returns:
        (pdf_bytes, error_message)
    On success, error_message is None.
    """
    try:
        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=2.5 * cm,
            rightMargin=2.5 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2.5 * cm,
            title=article_data.get("title", "Article"),
            author="YouTube to Article Generator",
        )

        base_styles = getSampleStyleSheet()

        # Define custom paragraph styles
        style_title = ParagraphStyle(
            "MyTitle",
            parent=base_styles["Normal"],
            fontSize=24,
            leading=30,
            textColor=COLOR_DARK,
            fontName="Helvetica-Bold",
            spaceAfter=6,
        )

        style_tldr = ParagraphStyle(
            "MyTLDR",
            parent=base_styles["Normal"],
            fontSize=10,
            leading=16,
            textColor=COLOR_PURPLE_DARK,
            backColor=COLOR_PURPLE_LITE,
            borderPad=8,
            fontName="Helvetica-Oblique",
            spaceAfter=12,
        )

        style_heading = ParagraphStyle(
            "MyHeading",
            parent=base_styles["Normal"],
            fontSize=13,
            leading=18,
            textColor=COLOR_PURPLE,
            fontName="Helvetica-Bold",
            spaceBefore=14,
            spaceAfter=4,
        )

        style_body = ParagraphStyle(
            "MyBody",
            parent=base_styles["Normal"],
            fontSize=10.5,
            leading=17,
            textColor=COLOR_BODY_TEXT,
            fontName="Helvetica",
            spaceBefore=4,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
        )

        style_bullet = ParagraphStyle(
            "MyBullet",
            parent=base_styles["Normal"],
            fontSize=10.5,
            leading=17,
            textColor=COLOR_BODY_TEXT,
            fontName="Helvetica",
            spaceBefore=2,
            spaceAfter=2,
            leftIndent=6,
        )

        style_meta = ParagraphStyle(
            "MyMeta",
            parent=base_styles["Normal"],
            fontSize=8.5,
            leading=13,
            textColor=COLOR_GREY,
            fontName="Helvetica",
        )

        # Build the story (list of flowables)
        story = []

        # Top coloured rule
        story.append(HRFlowable(width="100%", thickness=4, color=COLOR_PURPLE, spaceAfter=12))

        # Title
        story.append(Paragraph(_safe(article_data.get("title", "Article")), style_title))

        # Metadata line
        date_str = datetime.now().strftime("%B %d, %Y")
        meta_line = f"Generated: {date_str}  ·  Language: {language}"
        if video_url:
            short_url = video_url[:60] + ("…" if len(video_url) > 60 else "")
            meta_line += f"  ·  Source: {short_url}"
        story.append(Paragraph(meta_line, style_meta))
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_GREY, spaceAfter=8))

        # TL;DR
        tldr = article_data.get("tldr", "").strip()
        if tldr:
            story.append(KeepTogether([
                Paragraph("TL;DR", style_heading),
                Paragraph(_safe(tldr), style_tldr),
            ]))

        # Introduction
        intro = article_data.get("introduction", "").strip()
        if intro:
            story.append(Paragraph("Introduction", style_heading))
            story.append(HRFlowable(width="35%", thickness=1.5, color=COLOR_PURPLE, spaceAfter=6, hAlign="LEFT"))
            for para in intro.split("\n\n"):
                para = para.strip()
                if para:
                    story.append(Paragraph(_safe(para), style_body))

        # Key Points
        key_points = article_data.get("key_points", [])
        if key_points:
            story.append(Paragraph("Key Points", style_heading))
            story.append(HRFlowable(width="35%", thickness=1.5, color=COLOR_PURPLE, spaceAfter=6, hAlign="LEFT"))

            items = []
            for pt in key_points:
                pt = pt.strip()
                if pt:
                    items.append(ListItem(
                        Paragraph(_safe(pt), style_bullet),
                        bulletColor=COLOR_PURPLE,
                        value="bullet",
                        leftIndent=18,
                        bulletFontSize=10,
                    ))
            if items:
                story.append(ListFlowable(items, bulletType="bullet", start="•"))

        # Conclusion
        conclusion = article_data.get("conclusion", "").strip()
        if conclusion:
            story.append(Paragraph("Conclusion", style_heading))
            story.append(HRFlowable(width="35%", thickness=1.5, color=COLOR_PURPLE, spaceAfter=6, hAlign="LEFT"))
            for para in conclusion.split("\n\n"):
                para = para.strip()
                if para:
                    story.append(Paragraph(_safe(para), style_body))

        # Keywords footer
        if keywords:
            story.append(Spacer(1, 12))
            story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_GREY, spaceAfter=6))
            story.append(Paragraph("Keywords", style_meta))
            story.append(Paragraph("  ·  ".join(keywords), style_meta))

        # Bottom rule + footer text
        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=2, color=COLOR_PURPLE, spaceAfter=4))
        story.append(Paragraph(
            "Generated by YouTube to Article &amp; PDF Generator  |  Powered by Gemini",
            style_meta,
        ))

        doc.build(story)

        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes, None

    except Exception as e:
        return b"", f"{type(e).__name__}: {e}"


def _safe(text: str) -> str:
    """
    Escape XML special characters so ReportLab's Paragraph parser doesn't break.
    ReportLab uses an XML-like renderer internally.
    """
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


# =============================================================================
# 7. ESTIMATE READ TIME
# =============================================================================

def estimate_read_time(text: str) -> int:
    """
    Estimate how long it takes to read a piece of text.
    Assumes an average reading speed of 200 words per minute.
    Returns at least 1 minute.
    """
    word_count = len(text.split())
    return max(1, round(word_count / 200))
