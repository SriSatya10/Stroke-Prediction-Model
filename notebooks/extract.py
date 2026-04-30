import json
with open("ml.ipynb", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("ml_code.py", "w", encoding="utf-8") as f:
    for cell in data["cells"]:
        if cell["cell_type"] == "code":
            f.write("".join(cell["source"]) + "\n\n")
