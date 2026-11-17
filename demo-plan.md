# Demo plan

What to run live on the projector, what to point at, and what numbers to expect. The expected numbers come from running each notebook top to bottom on CPU with the fixed random seeds in the code. Colab should match closely (scikit-learn versions can shift the debate numbers by a few points).

There is one notebook per paper. Each is a small toy version of the paper's own setup, so every demo can point back to a section, figure or table of the paper.

## Before the event

- [ ] Open all three Colab links from the README and run each top to bottom once (under a minute each). This warms the runtime and catches any version issue.
- [ ] Put a QR code to the repo on the first slide.
- [ ] Have each notebook open in its own tab, already run, so a slow Colab never stalls the talk. Re-run cells live only where noted.
- [ ] Ask attendees to bring a laptop and a Google account. Pair people up if someone has neither.
- [ ] Print or slide the two "find the mistake" proofs for the warm-up (Iliad notes §4.1 and §4.3).

## Demo 1: Iterated amplification

**Notebook:** `01_iterated_amplification.ipynb` · **Paper:** Christiano et al. §§2–4, Table 3, App. C · **Live time:** 6 minutes · runs in about 20 seconds

1. Section 1: print σ (a shuffle of 0–15) and one ground-truth example, σ²⁹(3). Say: "X is a small network. It starts knowing nothing."
2. Section 2: run the transcript cell. H uses the paper's decomposition (Table 3): σ²⁹(x) = σ(σ¹⁴(σ¹⁴(x))), so it asks X two questions and looks up one step itself. The untrained X gets it wrong; that's expected.
3. Section 3: **run live.** Expect the difficulty to climb from 1 to 5 in four rounds (the paper's 85% rule, which needs no ground truth), and every depth to reach 100% by about round 15.
4. Section 4, the heatmap: "the frontier of questions X can answer moves outward." Yellow fills in from depth 1 up.
5. Section 5, the paper's Fig. 2 comparison: supervised learning on true answers reaches 95% in about 6 rounds, amplification in about 9. That is the paper's "modest slowdown". The difference is that amplification never saw a correct answer to a hard question.
6. Section 6 (our extension, not in the paper): random overseer mistakes (10% of lookups) almost vanish. A blind spot on 2 of the 16 inputs ends up wrong on about 90% of the deepest questions, matching the dashed "paths that pass a blind spot" curve.

**Say:** "Distillation averages away random noise. It faithfully copies consistent bias, to every question built on top of it."

## Demo 2: Debate, the pixel game

**Notebook:** `02_debate.ipynb` · **Paper:** Irving et al. §3.1, Table 2 · **Live time:** 6 minutes · **Run before the talk** (about 40 seconds)

1. Section 1: the judge, trained on 6 random nonzero pixels, is right about 56% of the time (paper's MNIST judge: 59.4%).
2. Section 3: show the two debate pictures. Light blue is hidden from the judge. Green boxes are honest, red the liar's.
3. Section 4, the main result. Expect about:

   | | no precommit | precommit |
   |---|---|---|
   | honest moves first | 52% (paper 67.4%) | 86% (paper 87.4%) |
   | liar moves first | 82% (paper 81.5%) | 93% (paper 90.4%) |
   | mean | 67% (paper 74.4%) | 90% (paper 88.9%) |

   Point out the paper's three patterns: debate beats the judge alone, precommit helps the honest side, and moving second is an advantage.
4. Section 5a: ask the room to judge a few debates with the buttons and compare with the network's score.

**Say:** "Debate doesn't make the judge smarter. The debaters do the work of finding the evidence. And the test is only as strong as the liar."

## Demo 3: Recursive reward modeling

**Notebook:** `03_recursive_reward_modeling.ipynb` · **Paper:** Leike et al. §§3.1–3.2, 4.3, 5.1, Fig. 4 · **Live time:** 5 minutes · runs in about 5 seconds, safe to run live

1. Section 1: the learned pair-reward heatmap. Blue above the diagonal: the reward model recovered "next ≥ previous" from 1,500 noisy comparisons (§3.1).
2. Section 2: on random documents, the naive and careful assisted evaluators both correlate with true quality (about 0.6 and 0.75). "Looks fine, right?"
3. Section 3, the paper's Fig. 4 in miniature: against the frozen naive evaluator, the green score keeps rising while true quality (black) peaks around 57 and falls to 49. The document repeats `2777`, so nearly every seam between chunks (`7 → 2`) is broken.
4. Section 4, the paper's fix (§5.1, online feedback): the user spot-checks random windows of the agent's current output while it trains. True quality stays near the top (62 of 63).

**Say:** "An optimizer finds exactly the parts of the outcome nobody evaluates. Every level of recursion adds seams. Looking at what the agent actually does, while it trains, is how you find them."

## Interactive moments

Every notebook has a 🎛 **Explore** section. These work well on the projector, and as a first thing for people to try themselves:

| Notebook | Control | Use it to show |
|---|---|---|
| 1 | **Ask the amplified team** (sliders) | The decomposition tree for any σᵏ(x), with ✓/✗ at every node. |
| 1 | **Run your own experiment** (button) | Random mistakes still learn; more blind spots spread further; "peel one" needs many more rounds. A few seconds per run. |
| 2 | **You are the judge** (buttons) | The room votes A or B; compare the room's score with the network judge's. |
| 2 | **Explore one debate** (sliders, checkboxes) | Turn precommit off, or let the liar move second, and watch the verdict change. |
| 3 | **Write a document** (text box) | The prefilled optimizer document scores top under the naive evaluator but has 14 broken seams. |

## Hands-on suggestions

Point people at one exercise per notebook so the room doesn't splinter:

| Notebook | Suggested exercise | Why |
|---|---|---|
| 1 | Ex. 2: catch the blind spot with a second decomposition | Directly about designing better overseers |
| 2 | Ex. 1: give the liar a two-move lookahead | Tests the claim that lying is harder than refuting |
| 3 | Ex. 2: cut the comparisons to 100 and look for exploited reward-model errors | Reward hacking in a few lines |

## Fallbacks

- **Colab is slow or down:** every notebook also runs locally with `pip install numpy scikit-learn matplotlib ipywidgets jupyter`. Keep pre-run copies open.
- **Short on time:** skip Demo 1 section 6 and Demo 3 section 2; the main plots carry the point.
- **No projector:** the outline's slides are enough; describe the plots and let people run the notebooks.
