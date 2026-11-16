# Demo plan

What to run live on the projector, what to point at, and what numbers to expect. The expected numbers below come from running each notebook top to bottom on CPU with the fixed random seeds in the code; Colab should match closely (scikit-learn versions can shift the pixel-game numbers by a few points).

## Before the event

- [ ] Open all four Colab links from the README and run each top to bottom once (about 1 minute each). This warms the runtime and catches any version issue.
- [ ] Put a QR code to the repo on the first slide.
- [ ] Have each notebook open in its own tab, already run, so a slow Colab never stalls the talk. Re-run cells live only where noted.
- [ ] Ask attendees to bring a laptop and a Google account. Pair people up if someone has neither.
- [ ] Print or slide the two "find the mistake" proofs for the warm-up (Iliad §4.1 and §4.3).

## Demo 1: Iterated amplification

**Notebook:** `01_iterated_amplification.ipynb` · **Live time:** 6 minutes

1. Section 1: print σ and one ground-truth example. Say: "X is a small network. It starts knowing nothing."
2. Section 2: run the transcript cell. Point out H only does one lookup or two delegated questions, never more. The untrained X gives a wrong answer; that's expected.
3. Section 3: **run live.** The log prints each round. Expect `k_cap` to double each round (2, 4, 8, 16, 32 by round 5) and true accuracy to go from about 0.2 to 1.00 by round 8.
4. Show the heatmap: "the frontier of questions X can answer moves outward." Yellow fills in from the bottom.
5. Section 4: supervised baseline learns faster per round, but used about 61,000 ground-truth labels, each of which secretly costs up to 32 lookups. Amplification used about 11,000 single-step lookups.
6. Section 5 (the punchline): random overseer mistakes (10%) almost vanish; a systematic blind spot on 2 of 16 inputs (12% of lookups) ends up wrong on half of all hard questions, matching the dashed "paths that touch a blind spot" curve exactly.

**Say:** "Distillation averages away random noise. It faithfully copies consistent bias, to every question built on top of it."

## Demo 2: Debate as path-halving

**Notebook:** `02_debate_reachability.ipynb` · **Live time:** 5 minutes

1. Section 2: run both debates and show the transcripts. In the true claim (distance 64, L = 64) Alice wins; in the false claim (distance 65) the judge catches an invalid single step.
2. Show the two maze pictures. Left: honest midpoints. Right: Bob zooms in on the bad step, drawn in red.
3. Section 3: expect judge alone ≈ 54%, debate with strong Bob 100% (true and false claims), debate with random Bob ≈ 50% (all true claims right, about 1% of lies caught, i.e. 1/L).
4. Section 4: the budget plot. Lies win about 85% of the time when Bob can search 25 cells, and almost never once he can search 400.

**Say:** "Debate doesn't make the judge smarter. It moves the work to the opponent. Training has to make sure the opponent actually does it."

## Demo 3: The pixel game

**Notebook:** `03_debate_pixels.ipynb` · **Live time:** 5 minutes · **Run before the talk** (the cell in section 4 takes about 30 seconds)

1. Section 1: judge alone with 4 random pixels ≈ 31% (chance is 10%).
2. Section 2: debate with 4 pixels, honest beats every possible lie ≈ 70%.
3. Section 3: show the debate pictures. Light blue is hidden from the judge. Point out that the honest player often reveals a *blank* pixel: "there's no ink here" is evidence too.
4. Section 4: 2 / 4 / 6 pixels gives judge alone ≈ 20 / 29 / 40% and debate ≈ 61 / 70 / 82%. Compare with the paper's 59.4% → 88.9% at 6 pixels on MNIST. A liar who reveals random pixels loses every time, which tells you nothing; the test is only as strong as the opposition.

## Demo 4: Recursive reward modeling

**Notebook:** `04_recursive_reward_modeling.ipynb` · **Live time:** 4 minutes · runs in about 5 seconds, safe to run live

1. Section 1: the learned pair-reward heatmap. Blue above the diagonal: RM0 recovered "next ≥ previous" from 1,500 noisy comparisons.
2. Section 2: on random documents, naive and careful evaluators both correlate with true quality (≈ 0.6 vs ≈ 0.75). "Looks fine, right?"
3. Section 3: the two-panel plot. Both evaluator scores rise to the top. True quality: careful goes to 63/63; naive peaks around 57 and then *falls* to about 49 while its evaluator score keeps rising.
4. Show the printed documents: naive repeats `2777` so that nearly every seam (`7 → 2`) is broken. Careful is almost all 7s, which is perfect by the stated metric and useless, a bonus point about underspecified rewards.

**Say:** "An optimizer finds exactly the parts of the outcome nobody evaluates. Every level of recursion adds seams."

## Hands-on suggestions

Point people at one exercise per notebook so the room doesn't splinter:

| Notebook | Suggested exercise | Why |
|---|---|---|
| 1 | Ex. 1: catch the blind spot with a second decomposition | Directly about designing better overseers |
| 2 | Ex. 3: write a smarter liar against a budget-limited Bob | Fun, adversarial, short |
| 3 | Ex. 2: remove precommit | The paper reports both settings; compare |
| 4 | Ex. 2: cut the comparisons to 100 and look for exploited RM errors | Reward hacking in a few lines |

## Fallbacks

- **Colab is slow or down:** every notebook also runs locally with `pip install numpy scikit-learn matplotlib jupyter`. Keep pre-run copies open.
- **Short on time:** skip Demo 2 section 4 and Demo 4 section 2; the main plots carry the point.
- **No projector audio/visual:** the outline's slides are enough; describe the plots and let people run the notebooks.
