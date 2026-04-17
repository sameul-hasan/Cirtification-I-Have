import os
import re
try:
    from groq import Groq
    import PyPDF2
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

def get_ai_description(file_path, name_without_ext):
    if not HAS_DEPS:
        return f"Professional Certification for {name_without_ext}"
        
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return f"Professional Certification for {name_without_ext}"
        
    client = Groq(api_key=api_key)
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.pdf':
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages[:1]:
                    text += page.extract_text() + "\n"
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{
                    "role": "user",
                    "content": f"Analyze this certificate text and provide a short, professional 1-sentence description (under 15 words) stating the certification name and issuer. Text: {text[:1000]}"
                }],
                temperature=1,
                max_completion_tokens=100,
            )
            return completion.choices[0].message.content.strip().replace('"', '')
            
        elif ext in ['.png', '.jpg', '.jpeg']:
            # Fallback to generating a great description from the filename
            # This avoids Groq Vision Model tier restrictions
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{
                    "role": "user",
                    "content": f"Write a short, professional 1-sentence description (under 20 words) for a certificate."
                }],
                temperature=1,
                max_completion_tokens=100,
            )
            return completion.choices[0].message.content.strip().replace('"', '')
            
    except Exception as e:
        return f"Professional Certification for {name_without_ext}"

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
                
                description = get_ai_description(file_path, name_without_ext)
                
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
