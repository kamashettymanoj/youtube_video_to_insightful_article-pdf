# app.py
# YouTube to Article & PDF Generator
# A mini project that converts YouTube video transcripts into structured articles
# Run: streamlit run app.py

import streamlit as st
from utils import (
    extract_video_id,
    fetch_transcript,
    generate_article_with_gemini,
    extract_keywords,
    build_pdf,
    estimate_read_time,
)

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YT → Article & PDF Generator",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for styling ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
}
.hero h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
.hero p  { font-size: 1.05rem; opacity: 0.9; margin-top: 0.5rem; }

.kw-pill {
    display: inline-block;
    background: #ede9fe;
    color: #5b21b6;
    border-radius: 20px;
    padding: 3px 12px;
    margin: 3px 4px 3px 0;
    font-size: 0.8rem;
    font-weight: 600;
}

.step-badge {
    display: inline-block;
    background: #667eea;
    color: white;
    border-radius: 50%;
    width: 28px; height: 28px;
    line-height: 28px;
    text-align: center;
    font-weight: 700;
    margin-right: 8px;
}

.metric-box {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-box .val { font-size: 1.6rem; font-weight: 700; color: #667eea; }
.metric-box .lbl { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

# ── Hero banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎥 YouTube → 📰 Article & 📄 PDF</h1>
  <p>Paste any YouTube URL · Get a structured article · Download as PDF</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar settings ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    gemini_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get your free key at https://aistudio.google.com/app/apikey",
    )

    st.markdown("---")
    st.markdown("### 📐 Article Style")

    article_tone = st.selectbox(
        "Writing Tone",
        ["Professional", "Casual & Friendly", "Academic", "Journalistic"],
        index=0,
    )

    article_length = st.selectbox(
        "Article Length",
        ["Concise (~300 words)", "Standard (~600 words)", "Detailed (~1000 words)"],
        index=1,
    )

    include_keywords = st.toggle("🏷️ Extract Keywords", value=True)
    include_tldr     = st.toggle("⚡ Add TL;DR", value=True)

    st.markdown("---")
    st.markdown("### 🌐 Language")
    target_lang = st.selectbox(
        "Output Language",
        ["English", "Hindi", "Spanish", "French", "German", "Arabic"],
        index=0,
    )

    st.markdown("---")
    st.markdown("<small>Built with Streamlit & Gemini API</small>", unsafe_allow_html=True)

# ── URL input + Generate button ────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])

with col_input:
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )

with col_btn:
    process_btn = st.button("🚀 Generate", use_container_width=True, type="primary")

# ── How it works (shown on first load) ────────────────────────────────────────
if not process_btn and not youtube_url:
    st.markdown("---")
    st.markdown("### 🗺️ How It Works")
    steps = [
        ("Paste URL",    "Enter any YouTube video link"),
        ("Transcript",   "We pull the video captions automatically"),
        ("Summarize",    "Gemini rewrites it as a structured article"),
        ("Download PDF", "Save your article as a formatted PDF"),
    ]
    cols = st.columns(4)
    for i, (title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(
                f'<div style="text-align:center">'
                f'<span class="step-badge">{i+1}</span>'
                f'<b>{title}</b><br>'
                f'<small style="color:#6b7280">{desc}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── Main processing pipeline ───────────────────────────────────────────────────
if process_btn:

    # Basic input validation
    if not youtube_url.strip():
        st.error("⚠️ Please enter a YouTube URL.")
        st.stop()

    if not gemini_key.strip():
        st.error("⚠️ Please enter your Gemini API Key in the sidebar.")
        st.stop()

    # Map dropdown selection to a target word count
    length_map = {
        "Concise (~300 words)":   300,
        "Standard (~600 words)":  600,
        "Detailed (~1000 words)": 1000,
    }
    target_words = length_map[article_length]

    with st.status("Processing your video…", expanded=True) as status:

        # Step 1 — Validate the URL
        status.update(label="🔍 Checking YouTube URL…")
        video_id = extract_video_id(youtube_url.strip())

        if not video_id:
            status.update(label="❌ Invalid URL", state="error")
            st.error(
                "Could not read a video ID from this URL.\n\n"
                "**Supported formats:**\n"
                "- `https://www.youtube.com/watch?v=VIDEO_ID`\n"
                "- `https://youtu.be/VIDEO_ID`\n"
                "- `https://www.youtube.com/shorts/VIDEO_ID`"
            )
            st.stop()

        st.write(f"✅ Video ID: `{video_id}`")

        # Step 2 — Fetch transcript
        status.update(label="📝 Fetching transcript…")
        transcript_text, transcript_lang, error_msg = fetch_transcript(video_id)

        if error_msg:
            status.update(label="❌ Transcript error", state="error")
            st.error(f"**Could not get transcript:** {error_msg}")
            st.info(
                "**Common reasons:**\n"
                "- Video has no captions enabled\n"
                "- Video is private or age-restricted\n"
                "- Auto-captions were disabled by the uploader"
            )
            st.stop()

        word_count_raw = len(transcript_text.split())
        st.write(f"✅ Transcript ready — {word_count_raw:,} words | Language: `{transcript_lang}`")

        # Step 3 — Generate article using Gemini
        status.update(label="✍️ Generating article with Gemini…")
        article_data, gen_error = generate_article_with_gemini(
            transcript=transcript_text,
            api_key=gemini_key,
            tone=article_tone,
            target_words=target_words,
            language=target_lang,
            include_tldr=include_tldr,
        )

        if gen_error:
            status.update(label="❌ Generation failed", state="error")
            st.error(f"**Gemini API error:** {gen_error}")
            st.stop()

        st.write("✅ Article generated")

        # Step 4 — Keyword extraction (optional)
        keywords = []
        if include_keywords:
            status.update(label="🏷️ Extracting keywords…")
            keywords = extract_keywords(transcript_text, top_n=12)
            st.write(f"✅ Found {len(keywords)} keywords")

        # Step 5 — Build PDF
        status.update(label="📄 Creating PDF…")
        pdf_bytes, pdf_error = build_pdf(
            article_data=article_data,
            keywords=keywords,
            video_url=youtube_url.strip(),
            language=target_lang,
        )

        if pdf_error:
            status.update(label="⚠️ PDF failed (article still shown)", state="error")
            st.warning(f"Could not build PDF: {pdf_error}")
        else:
            st.write("✅ PDF ready")

        status.update(label="🎉 Done!", state="complete", expanded=False)

    # ── Results ────────────────────────────────────────────────────────────────
    st.markdown("---")

    # Stats
    read_time     = estimate_read_time(article_data.get("body_text", ""))
    article_words = len(article_data.get("body_text", "").split())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-box"><div class="val">{word_count_raw:,}</div>'
            f'<div class="lbl">Transcript Words</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-box"><div class="val">{article_words:,}</div>'
            f'<div class="lbl">Article Words</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-box"><div class="val">{read_time} min</div>'
            f'<div class="lbl">Read Time</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-box"><div class="val">{len(keywords)}</div>'
            f'<div class="lbl">Keywords</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Article view + download panel
    col_article, col_actions = st.columns([3, 1])

    with col_article:
        st.markdown("### 📰 Generated Article")

        # Keyword pills
        if keywords:
            kw_html = "".join(f'<span class="kw-pill">{kw}</span>' for kw in keywords)
            st.markdown(kw_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # Article sections
        st.markdown(f"# {article_data.get('title', 'Article')}")

        if article_data.get("tldr"):
            st.info(f"⚡ **TL;DR:** {article_data['tldr']}")

        st.markdown(f"**Introduction**\n\n{article_data.get('introduction', '')}")
        st.markdown("---")

        key_points = article_data.get("key_points", [])
        if key_points:
            st.markdown("**Key Points**")
            for pt in key_points:
                st.markdown(f"- {pt}")

        st.markdown("---")
        st.markdown(f"**Conclusion**\n\n{article_data.get('conclusion', '')}")

    with col_actions:
        st.markdown("### ⬇️ Download")

        # Safe filename from article title
        safe_title = "".join(
            c for c in article_data.get("title", "article")
            if c.isalnum() or c in " _-"
        ).strip().replace(" ", "_")[:50] or "article"

        # PDF
        if pdf_bytes and not pdf_error:
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes,
                file_name=f"{safe_title}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )

        # Plain text
        plain_text = (
            f"{article_data.get('title', '')}\n\n"
            f"TL;DR: {article_data.get('tldr', '')}\n\n"
            "INTRODUCTION\n"
            f"{article_data.get('introduction', '')}\n\n"
            "KEY POINTS\n" +
            "\n".join(f"• {p}" for p in article_data.get("key_points", [])) +
            f"\n\nCONCLUSION\n{article_data.get('conclusion', '')}\n\n"
            f"Keywords: {', '.join(keywords)}"
        )

        st.download_button(
            label="📝 Download TXT",
            data=plain_text,
            file_name=f"{safe_title}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("**📹 Source**")
        st.markdown(f"[Open on YouTube]({youtube_url})")
        st.markdown(f"**Caption language:** `{transcript_lang}`")
        st.markdown(f"**Tone:** `{article_tone}`")

    # Raw transcript viewer
    with st.expander("📜 View Raw Transcript"):
        preview = transcript_text[:5000]
        if len(transcript_text) > 5000:
            preview += "\n\n... [truncated — showing first 5000 characters]"
        st.text_area("Transcript", preview, height=250, label_visibility="collapsed")
