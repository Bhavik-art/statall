import sys
import io
import os
import re
import base64
import json
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results = {}

for i in range(1, 19):
    qid = f"q{i}"
    filepath = f"experiments/q{i}.py"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Replace plt.show()
    code_to_exec = re.sub(r'plt\.show\s*\(\s*\)', 'pass  # plt.show() replaced', code)
    
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    plt.close('all')
    
    try:
        exec_globals = {'__builtins__': __builtins__}
        exec(code_to_exec, exec_globals)
        out = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        figs_b64 = []
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            figs_b64.append(base64.b64encode(buf.read()).decode('utf-8'))
        plt.close('all')
        
        results[qid] = {
            "code": code.strip(),
            "output": out.strip(),
            "figures": figs_b64,
            "error": None
        }
        print(f"{qid}: SUCCESS, out_len={len(out.strip())}, figs={len(figs_b64)}")
    except Exception as e:
        out = sys.stdout.getvalue()
        sys.stdout = old_stdout
        plt.close('all')
        err = traceback.format_exc()
        results[qid] = {
            "code": code.strip(),
            "output": out.strip(),
            "figures": [],
            "error": err
        }
        print(f"{qid}: ERROR: {err}")

with open('actual_experiments_output.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print("\nDone running all 15 experiments.")
