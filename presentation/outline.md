# Presentation outline

Two hours, SCCS 4403, 6–8 pm. Roughly a third talk, two thirds hands-on. Each part is a short talk followed by a notebook that people run themselves while presenters walk around.

Quotes give section numbers. Quotes marked † came from a text extraction of the paper or page (math stripped) and should be checked against the PDF before they go on a slide. The rest were cross-checked for the SRC [Discussion 2 primer](https://github.com/AI-Alignment-UIUC/Singularity-Reading-Club/blob/main/scalable%20oversight/discussion-2-primer.md).

## Agenda

| Time | Part | Format |
|---|---|---|
| 6:00–6:10 | Arrival, snacks, open Colab (QR code to this repo on screen) | |
| 6:10–6:25 | **0. The problem**, plus "find the mistake" warm-up | talk + pairs |
| 6:25–6:55 | **1. Iterated amplification** → Notebook 1 | 10 min talk, 20 min hands-on |
| 6:55–7:35 | **2. Debate**: theory → Notebook 2, then the pixel game → Notebook 3 | 15 min talk, 25 min hands-on |
| 7:35–7:50 | **3. Recursive reward modeling** → Notebook 4 | 5 min talk, 10 min hands-on |
| 7:50–8:00 | **4. The map, the open problems, and your project** | talk + discussion |

If running late, cut Notebook 4 to a live demo (it runs in seconds) and keep the closing.

---

## Part 0: The problem (6:10–6:25)

**Slide 1. Title.** Scalable Oversight Workshop. AI Alignment @ Illinois. Link/QR to the repo.

**Slide 2. Why oversight stops scaling.** We train models with demonstrations or with rewards from human judgment. Christiano et al.: "there are harder tasks for which we can't compute demonstrations or rewards even with human assistance" (abstract). Examples: reviewing a 10,000-line pull request, checking a long proof, judging a policy proposal.

**Slide 3. The question of this workshop.** Can a weak overseer (a human) reliably supervise a strong system, if the overseer gets help from the system itself? Three answers from 2018: amplification, recursive reward modeling, debate.

**Slide 4. Warm-up: find the mistake (5 min, in pairs).** From the Iliad session's opening game, "Wrong but convincing proofs"† (§2). Put up two:
- "All horses are the same color" (induction on the number of horses; Iliad §4.1).
- "ln(2) = 0" (rearranging a conditionally convergent series; Iliad §4.3).
Ask: how long did it take to find the flaw? Did you check every line, or did you know where to look? That gap (checking is easier than finding, if someone points you to the right line) is the whole intuition for debate.

**Slide 5. Recap of the SRC series.** One line each on SRC Discussion 1 (self-improving systems, recursion, self-similarity) and Discussion 2 (RRM vs. debate vs. STEM AI). Today: build the toy versions.

## Part 1: Iterated amplification (6:25–6:55)

**Slide 6. The idea.** "We propose a new method, Iterated Amplification, for a human expert H to train an ML agent X."† (§1.1) The amplified system is H "and several copies of X working together to solve a problem"† (§1.1).

**Slide 7. The operator.** "Amplify^H(X) answers a question Q by having H identify a sequence of useful subquestions, using X to compute a subanswer to each subquestion" (§1.1). Diagram: Q → H → {Q1, Q2} → X, X → H → A.

**Slide 8. The loop.** (§2.2) 1) sample questions, 2) run Amplify^H(X), 3) train X to imitate it, 4) repeat. Amplify(X) is "modestly smarter than X working on its own, and X chases this moving target"† (§2.3). No external objective: "the objective is implicit in the way that the human coordinates the copies of X"† (§1.1).

**Slide 9. HCH.** The idealized limit: a tree of humans, each able to consult more such trees. Factored cognition is the bet that hard reasoning splits into small checkable pieces.

**Slide 10. What the paper actually tested.** Five algorithmic tasks (§4.1): permutation powering, sequential assignments, wildcard search, shortest path, union find. "Rather than having a human perform the decomposition, we provide a hard-coded algorithm H which decomposes each task"† (§4.1). Result: it solves them "with at worst a modest slowdown" (§4.2) and "Supervised learning required tens of millions of examples in order to learn these algorithms."† (§4.2)

**Slide 11. Notebook 1 live.** See [demo plan](../demo-plan.md#demo-1-iterated-amplification). Show the frontier heatmap moving outward.

**Slide 12. The honest limitation.** "These experiments don't shed any light on whether humans can decompose interesting real world tasks"† (§4.3). Plus our notebook's lesson: random overseer mistakes wash out in distillation; systematic ones compound.

*Hands-on (20 min):* Notebook 1, sections 5–6. Suggested exercise: 1 (catch the blind spot) or 3 (worse decomposition).

## Part 2: Debate (6:55–7:35)

**Slide 13. The game.** "A question q ∈ Q is shown to both agents. The two agents state their answers a₀, a₁ ∈ A. The two agents take turns making statements s₀, s₁, . . . , s_{n−1} ∈ S. The judge sees the debate (q, a, s) and decides which agent wins." (Irving et al. §2)

**Slide 14. The bet.** "In the debate game, it is harder to lie than to refute a lie." (§2) The Iliad notes put the judge's job as: "the judge only needs to evaluate which side makes the stronger argument"† (§7).

**Slide 15. Why it could scale: complexity theory.** "For any problem L ∈ PSPACE, we can choose a polynomial time judge so that optimal play in the debate game with polynomially many steps solves L." (§2.2) Intuition from Iliad §11: "a debate can verify reachability by recursively halving a path"†. Diagram: claim "u → v in ≤ L steps", Alice names a midpoint, Bob picks the half to dispute, repeat until one step.

**Slide 16. Notebook 2 live.** The maze debate. A judge who checks one step is 100% accurate with a strong Bob, about chance with a random Bob.

**Slide 17. Beyond PSPACE (one slide, optional).** Cross-examination: a debater may question frozen copies of the opponent; "cross-examination increases the theoretical power of debate from"† PSPACE to NEXP (Iliad §§12–13). Mention only; point to the Iliad notes.

**Slide 18. The pixel game.** Judge sees a few pixels of an MNIST digit; debaters reveal pixels. "With 6 pixels, debate with precommit turns a 59.4% accurate judge into a 88.9% accurate debate." (§3.1)

**Slide 19. Notebook 3 live.** Our 8×8 version: judge alone ≈ 31% with 4 pixels, debate ≈ 70%. Show one debate picture: the honest player often reveals blank pixels.

**Slide 20. What can go wrong.** The authors' own worry: "Will one of the agents be able to sneak in a statement that appears logical but contains a hidden flaw without the other agent pointing out the flaw?" (§5.2). Barnes's obfuscated arguments: "a dishonest debater can often create arguments that have a fatal error, but where it is very hard to locate the error." Iliad §16 lists more criticisms (exploration, collusion, judge bias); verify that list against the page before presenting it.

*Hands-on (25 min):* Notebook 2 section 4 (bounded Bob) or Notebook 3 exercises 2–3.

## Part 3: Recursive reward modeling (7:35–7:50)

**Slide 21. Reward modeling.** The goal: "How can we create agents that behave in accordance with the user's intentions?" (Leike et al. §2). Learn a reward model from feedback, train the agent against it.

**Slide 22. Recursion.** "In step k we use the agent A_{k-1} to assist the user in evaluating outcomes when training agent A_k." (§3.2) The novel example: "extracting a summary of the plotline, checking spelling and grammar, summarizing character development."

**Slide 23. Notebook 4 live.** Optimize a 64-token document against a naive assisted evaluator: the evaluator score climbs while true quality falls, and the optimizer breaks every seam between chunks. A careful user who checks the seams fixes it.

**Slide 24. The open question.** "An important open question is whether errors accumulate: do the mistakes of the more narrow agent A_{k-1} lead to larger mistakes in the training of agent A_k?" (§3.2). Tie back to Notebook 1: which errors compound depends on whether they are random or systematic.

## Part 4: The map and your project (7:50–8:00)

**Slide 25. Where these sit among the 11 proposals.** Hubinger's list: RRM is proposal 8, debate is 9, and amplification appears in 2, 3, 4, 10 and 11. Use the SRC Discussion 2 table:

| | Where the human sits | What it bets on | Main failure mode |
|---|---|---|---|
| Amplification | Decomposes questions, delegates pieces | Hard reasoning factors into small checkable steps | Systematic overseer errors spread up the tree |
| RRM | Evaluator, helped by earlier agents | Evaluation is easier than doing the task | Optimizers exploit what nobody evaluates |
| Debate | Judge between two adversaries | Refuting a lie is easier than telling one | Weak opponents; obfuscated arguments |

**Slide 26. Start your own project.** Three or four picks from [`projects.md`](../projects.md). Point to each notebook's exercise list as a starting point.

**Slide 27. Discussion question to close on.** From the end of the Iliad session (§21): is debate research more capabilities than alignment? Take a show of hands, then a couple of answers.

---

## Sources

- Christiano, Shlegeris & Amodei (2018), [arXiv:1810.08575](https://arxiv.org/abs/1810.08575)
- Irving, Christiano & Amodei (2018), [arXiv:1805.00899](https://arxiv.org/abs/1805.00899)
- Leike et al. (2018), [arXiv:1811.07871](https://arxiv.org/abs/1811.07871)
- Hubinger (2020), [LessWrong](https://www.lesswrong.com/posts/fRsjBseRuvRhMPPE5/an-overview-of-11-proposals-for-building-safe-advanced-ai)
- Barnes (2020), [Alignment Forum](https://www.alignmentforum.org/posts/PJLABqQ962hZEqhdB/debate-update-obfuscated-arguments-problem)
- Wäldchen, Iliad Intensive, [Scalable Oversight and Debate](https://iliad-intensive.org/safety/debate/) ([PDF](https://iliad-intensive.org/downloads/debate/debate-nosol.pdf), CC BY 4.0)
