import os

file_path = 'd:/Stroke-Prediction-Model/notebooks/ml_code.py'
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if 'y_final = (y_prob_xgb >= 0.50).astype(int)' in line:
            line = line.replace('0.50', '0.40')
        elif 'y_pred_final = (y_prob_final >= 0.50).astype(int)' in line:
            line = line.replace('0.50', '0.40')
        elif 'print(f"\\n  Threshold  : 0.50")' in line:
            line = line.replace('0.50', '0.40')
        elif 'y_final = (y_prob_l >= 0.50).astype(int)' in line:
            line = line.replace('0.50', '0.40')
        new_lines.append(line)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
