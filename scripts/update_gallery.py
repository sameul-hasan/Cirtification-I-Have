import hashlib
import json
import mimetypes
import os
import re
import time
from datetime import datetime
from html import escape
from typing import Any, Dict, Optional

import fitz  # PyMuPDF
from google import genai
from google.genai import types

# Keep this list to models that are currently known to work in this project.
MODEL_CANDIDATES = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

LOG_PATH = "logs/llm_output.log"
THUMB_DIR = "generated_thumbs"
HTML_PATH = "index.html"
CACHE_PATH = "logs/cert_cache.json"
CERT_ROOT = "certificates"
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".pdf")
MAX_LOG_LINES = 400


def ensure_dirs() -> None:
    os.makedirs("logs", exist_ok=True)
    os.makedirs(THUMB_DIR, exist_ok=True)


def trim_log_file(max_lines: int = MAX_LOG_LINES) -> None:
    if not os.path.exists(LOG_PATH):
        return

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) <= max_lines:
        return

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines[-max_lines:])


def load_cache() -> Dict[str, Any]:
    ensure_dirs()
    default_cache: Dict[str, Any] = {"generated_at_utc": None, "items": {}}

    if not os.path.exists(CACHE_PATH):
        return default_cache

    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        if not raw:
            return default_cache

        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return default_cache

        items = parsed.get("items", {})
        if not isinstance(items, dict):
            items = {}

        return {
            "generated_at_utc": parsed.get("generated_at_utc"),
            "items": items,
        }
    except Exception:
        return default_cache


def save_cache(cache: Dict[str, Any]) -> None:
    ensure_dirs()
    temp_path = CACHE_PATH + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)

    os.replace(temp_path, CACHE_PATH)


def log_llm_output(
    file_path: str,
    cert_name: str,
    model: str,
    prompt: str,
    raw_output: str,
    final_output: str,
) -> None:
    ensure_dirs()
    record = {
        "time_utc": datetime.utcnow().isoformat() + "Z",
        "file_path": file_path,
        "certificate_name": cert_name,
        "model": model,
        "prompt": prompt,
        "raw_output": raw_output,
        "final_output": final_output,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_description(text: str) -> str:
    if not text:
        return ""

    text = text.strip().replace('"', "").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if "." in text:
        text = text.split(".")[0].strip()

    words = text.split()
    if len(words) > 15:
        text = " ".join(words[:15]).strip()

    return text.strip(" .,:;-")


def is_retryable_error(err_text: str) -> bool:
    s = (err_text or "").lower()
    retry_signals = [
        "503",
        "unavailable",
        "429",
        "rate limit",
        "timeout",
        "deadline exceeded",
        "internal",
    ]
    return any(sig in s for sig in retry_signals)


def is_model_not_found_error(err_text: str) -> bool:
    s = (err_text or "").lower()
    return "404" in s or "not found" in s or "is not found" in s


def normalize_rel_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def file_sha256(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_certificate_name(file_name: str) -> str:
    base = os.path.splitext(os.path.basename(file_name))[0]
    base = re.sub(r"[_-]+", " ", base).strip()
    base = re.sub(r"\s+", " ", base)
    return base.title() if base else "Certificate"


def thumb_path_from_source(source_path: str) -> str:
    no_ext = os.path.splitext(normalize_rel_path(source_path))[0].lower()
    safe = re.sub(r"[^a-z0-9._/-]", "-", no_ext).strip("-")
    safe = safe.replace("/", "__")
    return f"{THUMB_DIR}/{safe}.png"


def render_pdf_assets(
    pdf_path: str,
    thumb_rel: Optional[str] = None,
    write_thumb: bool = True,
) -> bytes:
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(0)
        llm_pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        llm_bytes = llm_pix.tobytes("png")

        if write_thumb and thumb_rel:
            thumb_abs = os.path.join(".", thumb_rel)
            thumb_dir = os.path.dirname(thumb_abs)
            if thumb_dir:
                os.makedirs(thumb_dir, exist_ok=True)

            thumb_pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            thumb_pix.save(thumb_abs)

    return llm_bytes


def call_model_with_retries(
    client: genai.Client,
    model_name: str,
    contents: list,
    config: types.GenerateContentConfig,
    max_retries: int = 3,
):
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            return response, ""
        except Exception as err:
            last_error = str(err)
            if attempt < max_retries and is_retryable_error(last_error):
                time.sleep(2**attempt)
                continue
            return None, last_error

    return None, last_error


def describe_certificate(
    file_path: str,
    cert_name: str,
    model_state: Dict[str, Any],
    image_bytes: Optional[bytes] = None,
    mime_type: Optional[str] = None,
) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        fallback = f"Professional Certification for {cert_name}"
        log_llm_output(file_path, cert_name, "NO_API_KEY", "N/A", "Missing GEMINI_API_KEY", fallback)
        return fallback

    prompt = (
        "Read this certificate image and output exactly one short line (max 15 words). "
        "Include certificate title and issuer if visible. "
        "No markdown. No extra text."
    )

    try:
        if model_state.get("client") is None:
            model_state["client"] = genai.Client(api_key=api_key)

        unavailable_models = model_state.setdefault("unavailable_models", set())
        client: genai.Client = model_state["client"]

        if image_bytes is None:
            with open(file_path, "rb") as f:
                image_bytes = f.read()

        inferred_mime = mime_type or mimetypes.guess_type(file_path)[0] or "image/png"
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=inferred_mime)

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f"{prompt}\nTitle hint: {cert_name}\nFilename hint: {cert_name}"),
                    image_part,
                ],
            )
        ]

        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=60,
        )

        for model_name in MODEL_CANDIDATES:
            if model_name in unavailable_models:
                continue

            response, err = call_model_with_retries(client, model_name, contents, config, max_retries=3)
            if response is not None:
                raw_text = response.text or ""
                final_text = clean_description(raw_text) or f"Professional Certification for {cert_name}"
                log_llm_output(file_path, cert_name, model_name, prompt, raw_text, final_text)
                return final_text

            log_llm_output(file_path, cert_name, model_name, prompt, f"ERROR: {err}", "")
            if is_model_not_found_error(err):
                unavailable_models.add(model_name)

        fallback = f"Professional Certification for {cert_name}"
        log_llm_output(file_path, cert_name, "ALL_MODELS_FAILED", prompt, "All model attempts failed", fallback)
        return fallback

    except Exception as err:
        fallback = f"Professional Certification for {cert_name}"
        log_llm_output(file_path, cert_name, "OUTER_EXCEPTION", prompt, f"ERROR: {str(err)}", fallback)
        return fallback


def update_readme(cards_markdown: list) -> None:
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        return

    if cards_markdown:
        gallery_md = "| | |\n| :---: | :---: |\n"
        for idx in range(0, len(cards_markdown), 2):
            c1 = cards_markdown[idx].strip("|").strip()
            c2 = cards_markdown[idx + 1].strip("|").strip() if idx + 1 < len(cards_markdown) else ""
            gallery_md += f"| {c1} | {c2} |\n"
    else:
        gallery_md = "*No certificates found yet. Upload some to see them here!*\n"

    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()

    start = "<!-- START_SECTION:certificates -->"
    end = "<!-- END_SECTION:certificates -->"
    section = f"{start}\n{gallery_md}\n{end}"

    pattern = re.compile(f"{re.escape(start)}.*?{re.escape(end)}", re.DOTALL)
    if pattern.search(readme):
        updated = re.sub(pattern, section, readme)
    else:
        updated = readme.rstrip() + "\n\n## Certificate Gallery\n\n" + section + "\n"

    if updated != readme:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(updated)


def update_html(cards_html: list, generated_at_utc: str) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Certificate Console</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #05070c;
      --bg-soft: #0d1320;
      --card: #101829;
      --text: #ebf2ff;
      --muted: #9fb0cc;
      --accent: #40f3c2;
      --accent-soft: #7cb3ff;
      --border: #1d2a44;
      --shadow: rgba(8, 16, 34, 0.55);
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      font-family: "Space Grotesk", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 12% -5%, rgba(64, 243, 194, 0.12), transparent 34%),
        radial-gradient(circle at 88% 0%, rgba(124, 179, 255, 0.14), transparent 35%),
        linear-gradient(160deg, var(--bg), var(--bg-soft));
      color: var(--text);
      min-height: 100vh;
      position: relative;
    }}

    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: 0.18;
      background-image:
        linear-gradient(to right, rgba(125, 149, 189, 0.15) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(125, 149, 189, 0.15) 1px, transparent 1px);
      background-size: 36px 36px;
      mask-image: radial-gradient(circle at center, black 45%, transparent 100%);
    }}

    .wrap {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 34px 18px 48px;
      position: relative;
      z-index: 1;
    }}

    .hero {{
      margin-bottom: 26px;
      padding: 18px 18px 16px;
      border: 1px solid rgba(98, 128, 179, 0.32);
      border-radius: 18px;
      background: linear-gradient(135deg, rgba(16, 24, 41, 0.94), rgba(8, 13, 24, 0.9));
      box-shadow: 0 18px 50px -24px var(--shadow);
    }}

    .chip {{
      margin: 0;
      width: fit-content;
      font-family: "JetBrains Mono", monospace;
      font-size: 0.72rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(64, 243, 194, 0.45);
      background: rgba(64, 243, 194, 0.08);
      color: #b7ffe9;
    }}

    h1 {{
      margin: 12px 0 8px;
      font-size: clamp(1.7rem, 2.4vw, 2.45rem);
      letter-spacing: 0.02em;
      line-height: 1.1;
    }}

    p.sub {{
      margin: 0;
      color: var(--muted);
      max-width: 72ch;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 18px;
    }}

    .card {{
      background: linear-gradient(185deg, rgba(18, 29, 49, 0.95), rgba(10, 16, 30, 0.96));
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 20px 45px -28px var(--shadow);
      transform: translateY(0);
      transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    }}

    .card:hover {{
      transform: translateY(-3px);
      border-color: rgba(89, 123, 186, 0.95);
      box-shadow: 0 24px 52px -24px var(--shadow);
    }}

    .card-top {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 9px 12px;
      border-bottom: 1px solid rgba(64, 243, 194, 0.12);
      background: rgba(8, 12, 24, 0.75);
      font-family: "JetBrains Mono", monospace;
      font-size: 0.73rem;
      color: #b5c4e0;
    }}

    .dot {{
      width: 8px;
      height: 8px;
      border-radius: 999px;
      display: inline-block;
    }}

    .dot.red {{ background: #ff6f7d; }}
    .dot.amber {{ background: #ffc56f; }}
    .dot.green {{ background: #73f0a5; }}

    .path {{
      margin-left: 2px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    .thumb {{
      width: 100%;
      height: 188px;
      object-fit: cover;
      display: block;
      background: #0f172a;
    }}

    .content {{
      padding: 13px 13px 15px;
    }}

    .title {{
      margin: 0 0 6px;
      font-weight: 700;
      font-size: 1rem;
      color: #f2f6ff;
    }}

    .desc {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.4;
    }}

    .open-link {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      text-decoration: none;
      font-family: "JetBrains Mono", monospace;
      font-size: 0.78rem;
      color: #cefff1;
      border: 1px solid rgba(64, 243, 194, 0.42);
      background: rgba(64, 243, 194, 0.08);
      border-radius: 999px;
      padding: 6px 10px;
    }}

    .open-link:hover {{
      border-color: rgba(124, 179, 255, 0.65);
      background: rgba(124, 179, 255, 0.13);
      color: #d8e8ff;
    }}

    .foot {{
      margin-top: 24px;
      color: var(--muted);
      font-size: 0.85rem;
      font-family: "JetBrains Mono", monospace;
      opacity: 0.9;
    }}

    @media (max-width: 680px) {{
      .wrap {{
        padding: 20px 12px 34px;
      }}

      .hero {{
        margin-bottom: 18px;
      }}

      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <p class="chip">AUTO GENERATED PORTFOLIO</p>
      <h1>Certificate Console</h1>
      <p class="sub">A coding-inspired gallery generated from repository assets with short AI summaries.</p>
    </header>
    <div class="grid">
      {''.join(cards_html) if cards_html else '<p>No certificates found yet.</p>'}
    </div>
    <p class="foot">Generated at: {escape(generated_at_utc or 'N/A')} | Updated automatically by GitHub Actions.</p>
  </div>
</body>
</html>
"""

    if os.path.exists(HTML_PATH):
        with open(HTML_PATH, "r", encoding="utf-8") as f:
            current = f.read()
        if current == html:
            return

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)


def collect_certificates():
    cache = load_cache()
    cache_items = cache.get("items", {})
    if not isinstance(cache_items, dict):
        cache_items = {}

    new_items: Dict[str, Any] = {}
    model_state: Dict[str, Any] = {"client": None, "unavailable_models": set()}
    cards_md = []
    cards_html = []

    if not os.path.exists(CERT_ROOT):
        return cards_md, cards_html, datetime.utcnow().isoformat() + "Z"

    for root, dirs, files in os.walk(CERT_ROOT):
        dirs[:] = sorted([d for d in dirs if not d.startswith(".")])

        for file_name in sorted(files):
            if not file_name.lower().endswith(VALID_EXTENSIONS):
                continue

            file_path = normalize_rel_path(os.path.join(root, file_name))
            cert_name = build_certificate_name(file_name)
            file_hash = file_sha256(file_path)
            cache_item = cache_items.get(file_path, {})
            if not isinstance(cache_item, dict):
                cache_item = {}

            hash_match = cache_item.get("sha256") == file_hash
            ext = os.path.splitext(file_name)[1].lower()
            llm_bytes: Optional[bytes] = None
            cert_href = file_path

            if ext == ".pdf":
                img_src = cache_item.get("thumb_path") or thumb_path_from_source(file_path)
                img_abs = os.path.join(".", img_src)
                need_thumb = (not hash_match) or (not os.path.exists(img_abs))
                if need_thumb:
                    llm_bytes = render_pdf_assets(file_path, thumb_rel=img_src, write_thumb=True)
                llm_mime = "image/png"
            else:
                img_src = file_path
                llm_mime = mimetypes.guess_type(file_path)[0] or "image/png"

            description = str(cache_item.get("description", "")).strip() if hash_match else ""
            if not description:
                if ext == ".pdf" and llm_bytes is None:
                    llm_bytes = render_pdf_assets(file_path, thumb_rel=None, write_thumb=False)

                description = describe_certificate(
                    file_path,
                    cert_name,
                    model_state=model_state,
                    image_bytes=llm_bytes if ext == ".pdf" else None,
                    mime_type=llm_mime,
                )

            if not description:
                description = f"Professional Certification for {cert_name}"

            new_items[file_path] = {
                "sha256": file_hash,
                "certificate_name": cert_name,
                "description": description,
                "thumb_path": img_src,
                "source_path": file_path,
                "updated_at_utc": datetime.utcnow().isoformat() + "Z",
            }

            md_entry = f"| <a href='{cert_href}'><img src='{img_src}' width='250' alt='{cert_name}'></a>"
            md_entry += f"<br>_{description}_ |"
            cards_md.append(md_entry)

            card_html = f"""
<article class="card">
  <div class="card-top">
    <span class="dot red"></span>
    <span class="dot amber"></span>
    <span class="dot green"></span>
    <span class="path">{escape(file_path)}</span>
  </div>
  <img class="thumb" src="{escape(img_src)}" alt="{escape(cert_name)}" loading="lazy" decoding="async">
  <div class="content">
    <h3 class="title">{escape(cert_name)}</h3>
    <p class="desc">{escape(description)}</p>
    <a class="open-link" href="{escape(cert_href)}" target="_blank" rel="noopener">open certificate &gt;</a>
  </div>
</article>
"""
            cards_html.append(card_html)

    generated_at_utc = datetime.utcnow().isoformat() + "Z"
    save_cache({
        "generated_at_utc": generated_at_utc,
        "items": new_items,
    })

    return cards_md, cards_html, generated_at_utc


def main() -> None:
    ensure_dirs()
    trim_log_file()
    cards_md, cards_html, generated_at_utc = collect_certificates()
    update_readme(cards_md)
    update_html(cards_html, generated_at_utc)
    trim_log_file()


if __name__ == "__main__":
    main()
