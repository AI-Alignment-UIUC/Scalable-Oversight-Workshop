"""Build presentation/slides.html (standalone, keyboard-driven) and presentation/speaker-notes.md
from the deck sources in presentation/deck/ (exported from the workshop's Claude Slides deck).

Usage: python tools/build_slides.py
Keys in slides.html: ← → (or space) to move, N to toggle speaker notes, F for full screen.
"""
import html, json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
DECK = ROOT / "presentation" / "deck"

index = json.loads((DECK / "deck.json").read_text())
css = (DECK / "deck.css").read_text()
fonts = sorted({f["href"] for f in index["faces"].values() if "href" in f})
sections = {v["start"]: v["description"] for v in index.get("sections", {}).values()}

slides, notes_md = [], [f"# {index['title']}: speaker notes\n"]
for n, sid in enumerate(index["order"], 1):
    raw = (DECK / "slides" / f"{sid}.html").read_text()
    m = re.search(r"<aside>(.*?)</aside>", raw, flags=re.S)
    note = m.group(1).strip() if m else ""
    title = re.search(r"<h[12][^>]*>(.*?)</h[12]>", raw, flags=re.S)
    title = re.sub("<[^>]+>", "", title.group(1)) if title else sid
    if sid in sections:
        notes_md.append(f"\n## {sections[sid]}\n")
    notes_md.append(f"\n**Slide {n}. {html.unescape(title)}**\n\n{html.unescape(note)}\n")
    raw = re.sub(r"<aside>.*?</aside>", "", raw, flags=re.S)
    slides.append(raw.replace("<section ", f'<section data-note="{html.escape(note)}" ', 1))

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(index['title'])}</title>
{''.join(f'<link rel="stylesheet" href="{h}">' for h in fonts)}
<style>
html, body {{ margin: 0; height: 100%; background: #0c1a30; overflow: hidden; }}
* {{ box-sizing: content-box; }}
h1, h2, h3, p, ul, ol {{ margin: 0; }}
#stage {{ position: absolute; left: 50%; top: 50%; width: 1920px; height: 1080px; transform-origin: center; }}
#stage > section {{ position: absolute; inset: 0; width: 1664px; height: 824px; display: none !important; overflow: hidden; }}
#stage > section.on {{ display: flex !important; }}
x-shape {{ display: block; }}
x-shape[kind="arrow-right"] {{ clip-path: polygon(0 30%, 60% 30%, 60% 0, 100% 50%, 60% 100%, 60% 70%, 0 70%); }}
th {{ text-align: left; }} td, th {{ padding: 0.35em 0.6em; border-bottom: 1px solid rgba(0,0,0,.12); vertical-align: top; }}
#hud {{ position: fixed; right: 16px; bottom: 12px; font: 14px system-ui, sans-serif; color: #bfd0e6; opacity: .8; }}
#notes {{ position: fixed; left: 0; right: 0; bottom: 0; max-height: 30vh; overflow: auto; background: rgba(12,26,48,.95);
  color: #f5f2ec; font: 17px/1.5 system-ui, sans-serif; padding: 14px 20px; display: none; }}
@media print {{ #hud, #notes {{ display: none !important; }} }}
{css}
</style></head>
<body><div id="stage">
{''.join(slides)}
</div>
<div id="notes"></div><div id="hud"></div>
<script>
const S = [...document.querySelectorAll('#stage > section')]; let i = 0, showNotes = false;
const hud = document.getElementById('hud'), notes = document.getElementById('notes'), stage = document.getElementById('stage');
function fit() {{ const k = Math.min(innerWidth / 1920, innerHeight / 1080); stage.style.transform = `translate(-50%, -50%) scale(${{k}})`; }}
function go(n) {{
  i = Math.max(0, Math.min(S.length - 1, n));
  S.forEach((s, j) => s.classList.toggle('on', j === i));
  hud.textContent = `${{i + 1}} / ${{S.length}}  ·  ← → to move  ·  N notes  ·  F full screen`;
  notes.textContent = S[i].dataset.note || '(no notes)'; notes.style.display = showNotes ? 'block' : 'none';
  history.replaceState(null, '', '#' + (i + 1));
}}
addEventListener('keydown', e => {{
  if (['ArrowRight', 'PageDown', ' '].includes(e.key)) go(i + 1);
  else if (['ArrowLeft', 'PageUp'].includes(e.key)) go(i - 1);
  else if (e.key === 'Home') go(0); else if (e.key === 'End') go(S.length - 1);
  else if (e.key.toLowerCase() === 'n') {{ showNotes = !showNotes; go(i); }}
  else if (e.key.toLowerCase() === 'f') document.documentElement.requestFullscreen?.();
}});
addEventListener('click', e => go(e.clientX > innerWidth / 3 ? i + 1 : i - 1));
addEventListener('resize', fit); fit(); go((parseInt(location.hash.slice(1)) || 1) - 1);
</script></body></html>
"""
(ROOT / "presentation" / "slides.html").write_text(page)
(ROOT / "presentation" / "speaker-notes.md").write_text("".join(notes_md))
print(f"wrote presentation/slides.html ({len(slides)} slides) and presentation/speaker-notes.md")
