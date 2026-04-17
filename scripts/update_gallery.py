import os
import re
import mimetypes
from google import genai
from google.genai import types
import fitz  # PyMuPDF

MODEL = "gemini-2.5-flash"

def pdf_first_page_to_png_bytes(pdf_path: str) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # better OCR quality
    return pix.tobytes("png")

def describe_certificate(file_path: str, cert_name: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return f"Professional Certification for {cert_name}"

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

        prompt = (
            "Read this certificate and return exactly one short professional sentence "
            "(maximum 15 words) including certificate title and issuer. "
            "If issuer is unclear, infer best possible from visible text."
        )

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
                temperature=0.3,
                max_output_tokens=60,
            ),
        )

        text = (response.text or "").strip().replace('"', "")
        return text if text else f"Professional Certification for {cert_name}"

    except Exception:
        return f"Professional Certification for {cert_name}"

def update_readme():
    repo_path = "."
    readme_path = os.path.join(repo_path, "README.md")
    valid_extensions = (".png", ".jpg", ".jpeg", ".webp", ".pdf")
    certificates = []

    for root, dirs, files in os.walk(repo_path):
        if "scripts" in root or ".github" in root:
            continue

        for file in files:
            if file.lower().endswith(valid_extensions):
                file_path = os.path.join(root, file).replace("\\", "/").lstrip("./")
                name_without_ext = os.path.splitext(file)[0].replace("-", " ").replace("_", " ").title()

                description = describe_certificate(file_path, name_without_ext)

                if file.lower().endswith(".pdf"):
                    md_entry = f"| 📄 **[{name_without_ext}]({file_path})**<br>_{description}_ |"
                else:
                    md_entry = f"| <a href='{file_path}'><img src='{file_path}' width='250' alt='{name_without_ext}'></a><br>**[{name_without_ext}]({file_path})**<br>_{description}_ |"

                certificates.append(md_entry)

    if certificates:
        gallery_md = "| | |\n| :---: | :---: |\n"
        for i in range(0, len(certificates), 2):
            c1 = certificates[i].strip("|").strip()
            c2 = certificates[i + 1].strip("|").strip() if i + 1 < len(certificates) else ""
            gallery_md += f"| {c1} | {c2} |\n"
    else:
        gallery_md = "*No certificates found yet. Upload some to see them here!*\n"

    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()

    start_marker = "<!-- START_SECTION:certificates -->"
    end_marker = "<!-- END_SECTION:certificates -->"
    pattern = re.compile(f"{start_marker}.*?{end_marker}", re.DOTALL)
    new_section = f"{start_marker}\n{gallery_md}\n{end_marker}"

    updated_readme = re.sub(pattern, new_section, readme_content)

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(updated_readme)

if __name__ == "__main__":
    update_readme()
