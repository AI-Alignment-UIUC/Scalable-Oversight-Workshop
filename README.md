# Scalable Oversight Workshop

**AI Alignment @ Illinois** · SCCS 4403 · 6–8 pm · We will have snacks!

A hands-on introduction to scalable oversight techniques such as **iterated amplification**, **recursive reward modeling**, and **AI safety via debate**.

We will be learning through demos, papers, and illustrative walkthroughs, using Colab notebooks for interactive demos. Our goal is to share recent knowledge to inspire you to get started on your own scalable oversight project.

This workshop builds on the Scalable Oversight series in the [Singularity Reading Club](https://github.com/AI-Alignment-UIUC/Singularity-Reading-Club) (see its [`scalable oversight/`](https://github.com/AI-Alignment-UIUC/Singularity-Reading-Club/tree/main/scalable%20oversight) folder), and on the Iliad Intensive's session on [Scalable Oversight and Debate](https://iliad-intensive.org/safety/debate/).

## Reading (optional, before the workshop)

If you are interested to read them beforehand, the main papers of relevance are:

1. Christiano, Shlegeris & Amodei (2018), [*Supervising strong learners by amplifying weak experts*](https://arxiv.org/abs/1810.08575). Iterated amplification.
2. Leike et al. (2018), [*Scalable agent alignment via reward modeling: a research direction*](https://arxiv.org/abs/1811.07871). Recursive reward modeling.
3. Irving, Christiano & Amodei (2018), [*AI Safety via Debate*](https://arxiv.org/abs/1805.00899).
4. Hubinger (2020), [*An overview of 11 proposals for building safe advanced AI*](https://www.lesswrong.com/posts/fRsjBseRuvRhMPPE5/an-overview-of-11-proposals-for-building-safe-advanced-ai) (LessWrong). Puts all three in one comparison.

Going further: Barnes, [*Debate update: Obfuscated arguments problem*](https://www.alignmentforum.org/posts/PJLABqQ962hZEqhdB/debate-update-obfuscated-arguments-problem) (2020), and the Iliad [debate notes (PDF)](https://iliad-intensive.org/downloads/debate/debate-nosol.pdf).

## Notebooks

New to the topic? Start with notebook 0. Every notebook runs on a free Colab CPU in about a minute, with no API keys and nothing to install beyond what Colab already has (numpy, scikit-learn, matplotlib).

| # | Notebook | What you do | Paper |
|---|---|---|---|
| 0 | **[Start here](notebooks/00_start_here.ipynb)** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-Alignment-UIUC/Scalable-Oversight-Workshop/blob/main/notebooks/00_start_here.ipynb) | A gentle walkthrough of all three ideas with one running example: checking the total of a 1,024-line ledger when you can only add two numbers at a time. Predict-then-run questions, a debate you can judge yourself, no ML. **Interactive:** judge a debate yourself by clicking which half to dispute; play the crooked bookkeeper against two kinds of audit. | All three |
| 1 | [Iterated amplification](notebooks/01_iterated_amplification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-Alignment-UIUC/Scalable-Oversight-Workshop/blob/main/notebooks/01_iterated_amplification.ipynb) | Train a network to compute σᵏ(x) when the "human" can only look up one step and split questions in two. Then give the human random vs. systematic blind spots and watch which ones compound. **Interactive:** step through how the amplified team answers any question; set the overseer's mistakes and decomposition, then retrain. | Christiano et al. §§1–4 |
| 2 | [Debate as path-halving](notebooks/02_debate_reachability.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-Alignment-UIUC/Scalable-Oversight-Workshop/blob/main/notebooks/02_debate_reachability.ipynb) | Two debaters argue whether a maze cell is reachable; a judge who checks only one step decides. See why a weak opponent lets lies through, and a toy version of obfuscated arguments. **Interactive:** play Bob against a possibly lying Alice on the maze; rerun debates with your own claim length and opponent. | Irving et al. §2; Iliad §11 |
| 3 | [The pixel game](notebooks/03_debate_pixels.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-Alignment-UIUC/Scalable-Oversight-Workshop/blob/main/notebooks/03_debate_pixels.ipynb) | A judge that sees 4 pixels of a digit goes from ~30% to ~70% accuracy when an honest and a lying debater pick which pixels to reveal. **Interactive:** judge pixel debates yourself and keep score against the network judge; explore any image, lie and pixel count. | Irving et al. §3.1 |
| 4 | [Recursive reward modeling](notebooks/04_recursive_reward_modeling.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-Alignment-UIUC/Scalable-Oversight-Workshop/blob/main/notebooks/04_recursive_reward_modeling.ipynb) | Learn a reward model from snippet comparisons, build evaluators for longer documents recursively, then optimize against them and watch the optimizer break every seam nobody checks. **Interactive:** hand-write a document to fool the evaluator; change the feedback budget, user noise and seam checks, then re-optimize. | Leike et al. §3.2 |

Each notebook has a 🎛 **Explore** section with sliders, buttons and small games (built with ipywidgets, which Colab supports out of the box), and ends with exercises, from quick tweaks to starting points for a project. Run all cells first (Runtime → Run all), then scroll to the controls.

> The Colab badges work once this repository is public and the notebooks are on `main`.

## Workshop materials

- **Slides** (22, with speaker notes): [`presentation/slides.pdf`](presentation/slides.pdf) to read or project; [`presentation/slides.html`](presentation/slides.html) to present from a browser (download and open it; ← → to move, N for notes, F for full screen); [`presentation/speaker-notes.md`](presentation/speaker-notes.md). The editable sources are in [`presentation/deck/`](presentation/deck/); rebuild the HTML and notes with `python tools/build_slides.py`.
- [`presentation/outline.md`](presentation/outline.md): timed agenda and slide-by-slide outline, with quotes and section numbers.
- [`demo-plan.md`](demo-plan.md): what to run live, what to say, what numbers to expect, and fallbacks.
- [`projects.md`](projects.md): ideas for starting your own scalable oversight project.

## Repository layout

```
notebooks/
  00_start_here.ipynb               ← open these in Colab
  01_iterated_amplification.ipynb
  02_debate_reachability.ipynb
  03_debate_pixels.ipynb
  04_recursive_reward_modeling.ipynb
  src/                              ← the same notebooks as plain Python (edit these)
presentation/                     ← slides.pdf, slides.html, speaker-notes.md, outline.md, deck/ sources
demo-plan.md
projects.md
tools/build_notebooks.py            ← rebuilds notebooks/*.ipynb from notebooks/src/
tools/build_slides.py               ← rebuilds presentation/slides.html and speaker-notes.md from presentation/deck/
```

To change a notebook, edit its source in `notebooks/src/` (cells are separated by `# %%` lines) and run `python tools/build_notebooks.py` (needs `pip install nbformat`). This keeps diffs readable.

To run locally instead of Colab: `pip install numpy scikit-learn matplotlib ipywidgets jupyter`, then `jupyter notebook notebooks/`.
