"""Build .ipynb notebooks from the percent-format sources in notebooks/src/.

Cells start with a line '# %%' (code) or '# %% [markdown]' (markdown).
In markdown cells, each line's leading '# ' is stripped.
Usage: python tools/build_notebooks.py
"""
import pathlib, re, nbformat

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "notebooks" / "src", ROOT / "notebooks"

def parse(text):
    cells, kind, buf = [], None, []
    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "md":
            body = "\n".join(re.sub(r"^# ?", "", l) for l in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(body))
        elif body.strip():
            cells.append(nbformat.v4.new_code_cell(body))
    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("md" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells

for src in sorted(SRC.glob("*.py")):
    nb = nbformat.v4.new_notebook()
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": []}}
    nb.cells = parse(src.read_text())
    out = OUT / (src.stem + ".ipynb")
    nbformat.write(nb, out)
    print("wrote", out.relative_to(ROOT))
