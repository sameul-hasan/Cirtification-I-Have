import os
import re
import json
import mimetypes
import base64
from datetime import datetime
from google import genai
from google.genai import types
import fitz  # PyMuPDF

MODEL = "gemini-2.5-flash"
LOG_PATH = "logs/llm_output.log"
THUMB_DIR = "generated_thumbs"

def ensure_dirs():
    os.makedirs("logs", exist_ok=True)
    os.makedirs(THUMB_DIR, exist_ok=True)

def log_llm_output(file_path: str, cert_name: str, prompt: str, raw_output: str, final_output: str):
    ensure_dirs()
    record = {
        "time_utc": datetime.utcnow().isoformat() + "Z",
        "file_path": file_path,
        "certificate_name": cert_name,
        "model": MODEL,
        "prompt": prompt,
        "raw_output": raw_output,
        "final_output": final_output
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def clean_description(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip().replace('"', "").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def pdf_first_page_to_png_bytes(pdf_path: str) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    return pix.tobytes("png")

def make_pdf_thumbnail(pdf_path: str) -> str:
    """
    Generates a thumbnail PNG for README display.
    Returns relative path like generated_thumbs/my-cert.png
    """
    ensure_dirs()
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    safe = re.sub(r"[^a-zA-Z0-9._-]", "-", base).lower()
    out_rel = f"{THUMB_DIR}/{safe}.png"
    out_abs = os.path.join(".", out_rel)

    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    pix.save(out_abs)
    return out_rel.replace("\\", "/")

def describe_certificate(file_path: str, cert_name: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        fallback = f"Professional Certification for {cert_name}"
        log_llm_output(file_path, cert_name, "NO_API_KEY", "", fallback)
        return fallback

    prompt = (
        "Read this certificate and write one clear professional description sentence. "
        "Include certificate title and issuer if visible. "
        "Do not use markdown. Do not truncate."
    )

    try:
        client = genai.Client(api_key=api_key)
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            image_bytes = pdf_first_page_to_png_bytes(file_path)
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        else:
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            mime_type = mimetypes.guess_type(file_path)[0] or "image/png"
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=f"{prompt}\nFilename hint: {cert_name}"),
                        image_part,
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=200,
            ),
        )

        raw_text = response.text or ""
        final_text = clean_description(raw_text) or f"Professional Certification for {cert_name}"
        log_llm_output(file_path, cert_name, prompt, raw_text, final_text)
        return final_text

    except Exception as e:
        fallback = f"Professional Certification for {cert_name}"
        log_llm_output(file_path, cert_name, prompt, f"ERROR: {str(e)}", fallback)
        return fallback

def update_readme():
    repo_path = "."
    readme_path = os.path.join(repo_path, "README.md")
    valid_extensions = (".png", ".jpg", ".jpeg", ".webp", ".pdf")
    cards = []

    for root, _, files in os.walk(repo_path):
        if any(skip in root for skip in ["scripts", ".github", "logs", THUMB_DIR, ".git"]):
            continue

        for file in files:
            if not file.lower().endswith(valid_extensions):
                continue

            file_path = os.path.join(root, file).replace("\\", "/").lstrip("./")
            cert_name = os.path.splitext(file)[0].replace("-", " ").replace("_", " ").title()
            description = describe_certificate(file_path, cert_name)
            ext = os.path.splitext(file)[1].lower()

            if ext == ".pdf":
                thumb_path = make_pdf_thumbnail(file_path)
                md_entry = (
                    f"| <img src='{thumb_path}' width='250' alt='{cert_name} PDF thumbnail'>"
                    f"<br>_{description}_ |"
                )
            else:
                md_entry = (
                    f"| <img src='{file_path}' width='250' alt='{cert_name}'>"
                    f"<br>_{description}_ |"
                )

            cards.append(md_entry)

    if cards:
        gallery_md = "| | |\n| :---: | :---: |\n"
        for i in range(0, len(cards), 2):
            c1 = cards[i].strip("|").strip()
            c2 = cards[i + 1].strip("|").strip() if i + 1 < len(cards) else ""
            gallery_md += f"| {c1} | {c2} |\n"
    else:
        gallery_md = "*No certificates found yet. Upload some to see them here!*\n"

    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()

    start = "<!-- START_SECTION:certificates -->"
    end = "<!-- END_SECTION:certificates -->"
    pattern = re.compile(f"{start}.*?{end}", re.DOTALL)
    section = f"{start}\n{gallery_md}\n{end}"
    updated = re.sub(pattern, section, readme)

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(updated)

if __name__ == "__main__":
    ensure_dirs()
    update_readme()
