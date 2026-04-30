import json

with open('d:/Stroke-Prediction-Model/notebooks/ml.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        changed = False
        for line in cell['source']:
            if 'y_final = (y_prob_xgb >= 0.50).astype(int)' in line:
                line = line.replace('0.50', '0.40')
                changed = True
            elif 'y_pred_final = (y_prob_final >= 0.50).astype(int)' in line:
                line = line.replace('0.50', '0.40')
                changed = True
            elif 'print(f"\\n  Threshold  : 0.50")' in line:
                line = line.replace('0.50', '0.40')
                changed = True
            elif 'y_final = (y_prob_l >= 0.50).astype(int)' in line:
                line = line.replace('0.50', '0.40')
                changed = True
            new_source.append(line)
        if changed:
            cell['source'] = new_source

with open('d:/Stroke-Prediction-Model/notebooks/ml.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
