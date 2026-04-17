import os
import re
import base64
try:
    from groq import Groq
    import PyPDF2
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

def get_ai_description(file_path):
    if not HAS_DEPS:
        print("Missing dependencies (groq or PyPDF2).")
        return None
        
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("No GROQ_API_KEY found in environment variables.")
        return None
        
    client = Groq(api_key=api_key)
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.pdf':
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages[:1]: # Read first page only
                    text += page.extract_text() + "\n"
            
            print(f"Sending PDF text to Groq for {file_path}...")
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{
                    "role": "user",
                    "content": f"Analyze this certificate text and provide a short, professional 1-sentence description (under 15 words) stating the certification name and issuer. Text: {text[:1500]}"
                }],
                temperature=0.5,
                max_completion_tokens=50,
            )
            return completion.choices[0].message.content.strip().replace('"', '')
            
        elif ext in ['.png', '.jpg', '.jpeg']:
            with open(file_path, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            print(f"Sending Image to Groq Vision for {file_path}...")
            completion = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look at this certificate and provide a short, professional 1-sentence description (under 15 words) stating the certification name and issuer."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }],
                temperature=0.5,
                max_completion_tokens=50,
            )
            return completion.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        print(f"Groq API Error for {file_path}: {e}")
        return None

def update_readme():
    repo_path = '.'
    readme_path = os.path.join(repo_path, 'README.md')
    valid_extensions = ('.png', '.jpg', '.jpeg', '.pdf')
    certificates = []
    
    for root, dirs, files in os.walk(repo_path):
        if 'scripts' in root or '.github' in root:
            continue
            
        for file in files:
            if file.lower().endswith(valid_extensions):
                file_path = os.path.join(root, file).replace('\\', '/').lstrip('./')
                name_without_ext = os.path.splitext(file)[0].replace('-', ' ').replace('_', ' ').title()
                
                print(f"\nProcessing: {file_path}")
                description = get_ai_description(file_path)
                if not description:
                    description = "Professional Certification" # Fallback if AI fails
                
                if file.lower().endswith('.pdf'):
                    md_entry = f"| 📄 **[{name_without_ext}]({file_path})**<br>_{description}_ |"
                else:
                    md_entry = f"| <a href='{file_path}'><img src='{file_path}' width='250' alt='{name_without_ext}'></a><br>**[{name_without_ext}]({file_path})**<br>_{description}_ |"
                
                certificates.append(md_entry)

    gallery_md = ""
    if certificates:
        gallery_md = "| | |\n| :---: | :---: |\n"
        for i in range(0, len(certificates), 2):
            col1 = certificates[i] if i < len(certificates) else "| |"
            c1_content = col1.strip('|').strip()
            c2_content = certificates[i+1].strip('|').strip() if i+1 < len(certificates) else ""
            gallery_md += f"| {c1_content} | {c2_content} |\n"
    else:
        gallery_md = "*No certificates found yet. Upload some to see them here!*\n"

    with open(readme_path, 'r', encoding='utf-8') as f:
        readme_content = f.read()

    start_marker = "<!-- START_SECTION:certificates -->"
    end_marker = "<!-- END_SECTION:certificates -->"
    pattern = re.compile(f"{start_marker}.*?{end_marker}", re.DOTALL)
    new_section = f"{start_marker}\n{gallery_md}\n{end_marker}"
    
    updated_readme = re.sub(pattern, new_section, readme_content)

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(updated_readme)

if __name__ == "__main__":
    update_readme()
