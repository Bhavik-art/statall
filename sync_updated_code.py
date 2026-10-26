"""
Sync actual updated code from experiments/q*.py and their outputs/figures into index.html and all_experiments_code.py
"""
import json
import re
import html
import os

with open('actual_experiments_output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('index.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

section_pattern = re.compile(r'(<section id="(q\d+)">)(.*?)(</section>)', re.DOTALL)

def update_section(match):
    start_tag = match.group(1)
    qid = match.group(2)
    body = match.group(3)
    end_tag = match.group(4)
    
    if qid not in data:
        return match.group(0)
    
    info = data[qid]
    code = info['code']
    output = info['output']
    figures = info['figures']
    
    # 1. Update <pre class="code"><code>...</code></pre>
    escaped_code = html.escape(code)
    code_pattern = re.compile(r'<pre class="code"><code>.*?</code></pre>', re.DOTALL)
    new_code_block = f'<pre class="code"><code>{escaped_code}</code></pre>'
    if code_pattern.search(body):
        body = code_pattern.sub(new_code_block, body)
    else:
        print(f"Warning: No code block found in {qid}")
    
    # 2. Update <pre class="out">...</pre>
    if output:
        out_text = html.escape(output)
    else:
        out_text = 'Plot generated and displayed below.'
    
    new_out_block = f'<pre class="out">{out_text}</pre>'
    out_pattern = re.compile(r'<pre class="out">.*?</pre>', re.DOTALL)
    if out_pattern.search(body):
        body = out_pattern.sub(new_out_block, body)
    else:
        # insert after bar
        bar_pattern = re.compile(r'(<div class="bar"><span>Output</span></div>)')
        if bar_pattern.search(body):
            body = bar_pattern.sub(r'\1\n        ' + new_out_block, body)
    
    # 3. Update or remove <div class="figs">
    figs_pattern = re.compile(r'<div class="figs">.*?</div>', re.DOTALL)
    if figures:
        num = qid[1:]
        img_tags = []
        for idx, fig_b64 in enumerate(figures):
            alt = f"Plot for Q{num}" if len(figures) == 1 else f"Plot {idx+1} for Q{num}"
            img_tags.append(f'<img alt="{alt}"\n            src="data:image/png;base64,{fig_b64}">')
        new_figs_block = '<div class="figs">' + '\n          '.join(img_tags) + '\n        </div>'
        
        if figs_pattern.search(body):
            body = figs_pattern.sub(new_figs_block, body)
        else:
            # insert after <pre class="out">...</pre>
            body = re.sub(
                r'(<pre class="out">.*?</pre>)',
                r'\1\n        ' + new_figs_block,
                body,
                flags=re.DOTALL
            )
    else:
        # No figures in updated experiment
        if figs_pattern.search(body):
            body = figs_pattern.sub('', body)
            
    return start_tag + body + end_tag

new_html_content = section_pattern.sub(update_section, html_content)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(new_html_content)

print(f"Updated index.html: size {len(new_html_content)} bytes")

# Now update all_experiments_code.py
titles = {
    'q1': 'Q1. Iris: Mean, Median, Standard Deviation and Box Plot',
    'q2': 'Q2. Diabetes: BMI Distribution and Histogram',
    'q3': 'Q3. Iris: Pearson Correlation and Heatmap',
    'q4': 'Q4. Euclidean Distance Before and After Standardization',
    'q5': 'Q5. One-Sample t-Test',
    'q6': 'Q6. Bootstrap 95% Confidence Interval for the Mean',
    'q7': 'Q7. Diabetes: Simple Linear Regression',
    'q8': 'Q8. Diabetes: Linear Regression vs Ridge Regression',
    'q9': 'Q9. Iris: Logistic Regression Classifier',
    'q10': 'Q10. Iris: Decision Tree Classifier',
    'q11': 'Q11. Iris: K-Means Clustering and Silhouette Score',
    'q12': 'Q12. Iris: Principal Component Analysis (PCA)',
    'q13': 'Q13. Breast Cancer: 5-Fold Cross-Validation',
    'q14': 'Q14. Breast Cancer: Decision Tree Feature Importance',
    'q15': 'Q15. Breast Cancer: Random Forest Permutation Importance'
}

all_code = [
    "# ==============================================================================",
    "# Statistics for Machine Learning and Data Science: Practical Exam",
    "# Complete Python Code for All 15 Experiments",
    "# ==============================================================================\n"
]

for i in range(1, 16):
    qid = f"q{i}"
    title = titles.get(qid, f"Question {i}")
    code = data[qid]['code']
    all_code.append("# ==============================================================================")
    all_code.append(f"# {title}")
    all_code.append("# ==============================================================================\n")
    all_code.append(code + "\n\n")

all_code_str = "\n".join(all_code)
with open('all_experiments_code.py', 'w', encoding='utf-8') as f:
    f.write(all_code_str)

print(f"Updated all_experiments_code.py: size {len(all_code_str)} bytes")
