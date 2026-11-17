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

One notebook per paper. Each is a small toy version of the paper's own setup, built to show the paper's main idea and result, with the paper's numbers alongside for reference. Every notebook runs on a free Colab CPU in under a minute, with no API keys and nothing to install beyond what Colab already has (numpy, scikit-learn, matplotlib, ipywidgets).

| # | Notebook | What you do | Paper |
|---|---|---|---|
| 1 | [Iterated amplification](notebooks/01_iterated_amplification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-Alignment-UIUC/Scalable-Oversight-Workshop/blob/main/notebooks/01_iterated_amplification.ipynb) | The paper's first task, permutation powering, at small size. A "human" H who can only look up one step of σ and split a question in two trains a network to compute σᵏ(x), using the paper's decomposition and 85% curriculum. Compare with supervised learning on true answers, then give H random vs. systematic mistakes and watch which ones spread. **Interactive:** step through how the amplified team answers any question; set the overseer's mistakes and decomposition, then retrain. | Christiano et al. §§1–4, App. C |
| 2 | [Debate: the pixel game](notebooks/02_debate.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-Alignment-UIUC/Scalable-Oversight-Workshop/blob/main/notebooks/02_debate.ipynb) | The paper's MNIST experiment on small 8×8 digits, with its rules: a judge trained on 6 random nonzero pixels, debaters who reveal one pixel per turn, precommit. The judge alone is right about 56% of the time; debate with precommit lifts it to about 90%, and moving second helps, as in the paper's Table 2. **Interactive:** judge debates yourself and keep score against the network judge; explore any image, lie, pixel count and rule. | Irving et al. §§2–3.1 |
| 3 | [Recursive reward modeling](notebooks/03_recursive_reward_modeling.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-Alignment-UIUC/Scalable-Oversight-Workshop/blob/main/notebooks/03_recursive_reward_modeling.ipynb) | Learn a reward model from snippet comparisons, build assisted evaluators for longer documents, then train an agent against the frozen evaluator: its score rises while true quality falls, the shape of the paper's Fig. 4. The paper's fix, feedback while the agent trains (§5.1), closes the gap. **Interactive:** hand-write a document to fool the evaluator; change the feedback budget, user noise and seam checks, then re-optimize. | Leike et al. §§3–5 |

Each notebook has a 🎛 **Explore** section with sliders, buttons and small games (built with ipywidgets, which Colab supports out of the box), and ends with exercises, from quick tweaks to starting points for a project. Run all cells first (Runtime → Run all), then scroll to the controls.

> The Colab badges work once this repository is public and the notebooks are on `main`.

## Workshop materials

- **Slides** (with speaker notes): [`presentation/slides.pdf`](presentation/slides.pdf) to read or project; [`presentation/slides.html`](presentation/slides.html) to present from a browser (download and open it; ← → to move, N for notes, F for full screen); [`presentation/speaker-notes.md`](presentation/speaker-notes.md). The editable sources are in [`presentation/deck/`](presentation/deck/); rebuild the HTML and notes with `python tools/build_slides.py`.
- [`presentation/outline.md`](presentation/outline.md): timed agenda and slide-by-slide outline, with quotes and section numbers.
- [`demo-plan.md`](demo-plan.md): what to run live, what to say, what numbers to expect, and fallbacks.
- [`projects.md`](projects.md): ideas for starting your own scalable oversight project.

## Repository layout

```
notebooks/
  01_iterated_amplification.ipynb   ← open these in Colab
  02_debate.ipynb
  03_recursive_reward_modeling.ipynb
  src/                              ← the same notebooks as plain Python (edit these)
presentation/                     ← slides.pdf, slides.html, speaker-notes.md, outline.md, deck/ sources
demo-plan.md
projects.md
tools/build_notebooks.py            ← rebuilds notebooks/*.ipynb from notebooks/src/
tools/build_slides.py               ← rebuilds presentation/slides.html and speaker-notes.md from presentation/deck/
```

To change a notebook, edit its source in `notebooks/src/` (cells are separated by `# %%` lines) and run `python tools/build_notebooks.py` (needs `pip install nbformat`). This keeps diffs readable.

To run locally instead of Colab: `pip install numpy scikit-learn matplotlib ipywidgets jupyter`, then `jupyter notebook notebooks/`.
