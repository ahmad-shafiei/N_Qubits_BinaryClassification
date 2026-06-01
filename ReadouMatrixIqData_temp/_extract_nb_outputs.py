import json

path = "readout_matrix.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    outs = cell.get("outputs", [])
    if not outs:
        continue
    texts = []
    has_img = False
    for o in outs:
        if o.get("output_type") == "stream":
            texts.append("".join(o.get("text", [])))
        if o.get("output_type") == "display_data" and "image/png" in o.get("data", {}):
            has_img = True
    if texts or has_img:
        print(f"--- Cell {i} exec={cell.get('execution_count')} img={has_img} ---")
        for t in texts:
            print(t[:3000])
        if has_img and not texts:
            print("[matplotlib figure embedded]")
