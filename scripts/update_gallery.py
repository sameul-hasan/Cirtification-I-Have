import os
import re

def update_readme():
    repo_path = '.'
    readme_path = os.path.join(repo_path, 'README.md')
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.pdf')
    certificates = []
    
    for root, dirs, files in os.walk(repo_path):
        if 'scripts' in root or '.github' in root:
            continue;
            
        for file in files:
            if file.lower().endswith(valid_extensions):
                file_path = os.path.join(root, file).replace('\\', '/').lstrip('./')
                name_without_ext = os.path.splitext(file)[0].replace('-', ' ').replace('_', ' ').title()
                
                if file.lower().endswith('.pdf'):
                    md_entry = f"| 📄 [{name_without_ext}]({file_path}) |"
                else:
                    md_entry = f"| <img src='{file_path}' width='250' alt='{name_without_ext}'><br>**[{name_without_ext}]({file_path})** |"
                
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