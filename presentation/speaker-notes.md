# Scalable Oversight Workshop: speaker notes

## The problem: overseers who can’t check the work, and today’s plan.

**Slide 1. Scalable Oversight Workshop**

Welcome. While people settle in: put the repo link or QR code up, ask everyone to open Notebook 0 (Start here) in Colab and run all. It takes seconds and needs no setup. Today is two thirds hands-on: short talks, then notebooks. Builds on the SRC Scalable Oversight series.

**Slide 2. Human feedback works while humans can check the work**

RLHF and its relatives depend on a person being able to tell good output from bad. As systems take on bigger and harder tasks, that stops being true. Ask the room for an example from their own work where they could not check an AI’s answer.

**Slide 3. Can a weak overseer supervise a strong system, if the system helps?**

Three answers, all from 2018, all built on the same idea: use AI to help humans oversee AI. We’ll build a toy version of each today. Notebook 0 runs all three on one example, a 1,024-line ledger.

**Slide 4. Find the mistake**

Show each proof in full (Iliad notes §4.1 and §4.3). Give pairs 5 minutes. Then ask: how long did it take? Who checked every line? The punchline: once someone points at the flawed step, checking it is easy. That gap between finding and checking is the intuition behind debate. Horses: the step from n=1 to n=2 fails because the two groups don’t overlap. ln 2: rearranging a conditionally convergent series changes its sum.

**Slide 5. Short talks, then you run the notebooks**

Roughly a third talk, two thirds hands-on. If we run late, Notebook 4 becomes a live demo only; the closing stays.

## Iterated amplification: split questions, distill the team, watch errors.

**Slide 6. Split the hard question**

H is the human, X is the current model. The amplified system is H plus several copies of X working together. Our running example in Notebook 1 is the paper’s first task, permutation powering: “what is sigma to the k of x?” H can only look up one step of sigma, or split k into two halves and ask X for each.

**Slide 7. Distill the team into X, then amplify again**

Training loop from §2.2, simplified: the paper also learns a model H′ of the human so H isn’t queried every time. There is no external objective: the objective is implicit in how H coordinates the copies of X (§1.1, quote marked for checking against the PDF). Notebook 0 shows HCH on a ledger: 1,023 overseers each doing one addition.

**Slide 8. Five algorithmic tasks, a hard-coded “human”**

The §4.3 quote comes from a text extraction; check it against the PDF before relying on the wording. The paper also reports supervised learning needed tens of millions of examples to learn these algorithms. Point out that every one of these tasks has a clean recursive structure: the open question is whether real tasks do.

**Slide 9. The frontier of questions X can answer moves outward**

Run section 3 live. The log shows k_cap doubling each round (2, 4, 8, 16, 32) and true accuracy climbing from about 0.2 to 1.00 by round 8. Then show the heatmap: yellow fills in from small k upward. The curriculum uses agreement between X and Amplify(X), which needs no ground truth. Supervised learning gets there in fewer rounds but each label for k=32 secretly costs 32 lookups.

**Slide 10. Random slips wash out. Blind spots spread.**

Section 5 of Notebook 1. The systematic-error curve matches the fraction of question paths that ever touch a blind spot exactly; the permutation’s cycle structure caps it at one half. Exercise 1 asks people to design an H that catches the blind spot, using the paper’s “?” trick when two decompositions disagree.

## Debate: refuting a lie is easier than telling one, if the opponent is strong.

**Slide 11. Two agents argue. A judge decides.**

Hubinger’s one-line version (proposal 9): train a model, Alice, to win debates against a copy of itself, Bob, in front of a human judge. The question for today: when does winning the debate line up with telling the truth?

**Slide 12. “In the debate game, it is harder to lie than to refute a lie.”**

This is the whole proposal in one sentence. If it holds, honesty is the winning strategy, even when the judge couldn’t have found the answer. Remind people of the warm-up: once someone pointed at the bad step, checking it was easy.

**Slide 13. Halve the claim to one step**

This is the reachability proof from the Iliad notes, §11: a debate verifies “t is reachable from s in at most L steps” by recursively halving the path. The judge does log2(L) rounds of bookkeeping and one adjacency check. Orange marks the half Bob disputes. Optional aside: cross-examination, letting a debater question frozen copies of the opponent, pushes the theoretical power from PSPACE to NEXP (Iliad §§12–13).

**Slide 14. The maze debate: a judge who checks one step**

Run section 2 to show one honest and one dishonest debate, then the two maze pictures: Bob zooms in on the bad step, drawn in red. Section 3 gives these numbers (400 claims, L = 64). With a random Bob a lie survives unless Bob picks the false half every round, probability 1/L. Section 4: with a compute budget of 25 cells, about 85% of lies win; at 400 cells almost none do. That’s the exploration worry from the Iliad list of criticisms.

**Slide 15. A judge that sees four pixels**

Our version uses scikit-learn’s 8×8 digits and greedy debaters, so numbers differ from the paper’s MNIST and tree search. More pixels: 2 / 4 / 6 pixels gives judge alone about 20 / 29 / 40% and debate about 61 / 70 / 82%. A liar who reveals random pixels always loses, which tests nothing: debate is only as informative as the opposition is strong. Caveat: here debaters can only reveal true pixels, so they cannot lie about evidence.

**Slide 16. Flaws that are hard to find**

The Iliad notes list more criticisms (collusion, prover incentives) and newer protocols such as prover-estimator debate. Check that list on the page before presenting it in detail. Hands-on now (25 min): Notebook 2 section 4 or Notebook 3 exercises 2–3.

## Recursive reward modeling: assisted evaluation and the seams nobody checks.

**Slide 17. Assistants help you evaluate**

Reward modeling: learn a reward function from the user’s feedback, then train an agent against it. The goal (§2): how can we create agents that behave in accordance with the user’s intentions? RRM is Hubinger’s proposal 8. Difference from amplification: it trains agents with reward rather than training a model to imitate a human.

**Slide 18. The optimizer finds the seams nobody checks**

Section 3 of Notebook 4, runs in seconds. Both evaluator scores climb to the top. True quality: careful reaches 63/63; naive peaks around 57 and then falls to about 49. The careful document is almost all 7s: perfect by the stated metric and useless, a bonus point about underspecified rewards. Notebook 0 has the same idea as a cooked ledger hidden in the carry-forward between pages.

**Slide 19. “An important open question is whether errors accumulate: do the mistakes of the more narrow agent Aₖ₋₁ lead to larger mistakes in the training of agent Aₖ?”**

Tie the two halves of the evening together. Same question as SRC criterion C2c: state the per-level error map and classify it as contracting, thresholded, compounding or cascading. Hands-on (10 min): Notebook 4 exercise 2, cut the comparisons to 100 and look for exploited reward-model errors.

## The map of proposals, open problems, and starting your own project.

**Slide 20. Three bets, three failure modes**

Adapted from the SRC Discussion 2 comparison table. Ask: which one would you trust to review a 10,000-line code change? There’s no right answer; get two or three views.

**Slide 21. Six places to start**

Full descriptions in projects.md in the repo. Each notebook’s exercise list is also a starting point. Tell people to post in the club Discord if they start one; we’re happy to help scope it.

**Slide 22. Is debate research more capabilities than alignment?**

The closing question from the Iliad debate session (§21). Thank everyone, point to the repo, the readings in the README (Christiano et al., Leike et al., Irving et al., Hubinger’s 11 proposals), and the SRC reading club for follow-up discussions.
