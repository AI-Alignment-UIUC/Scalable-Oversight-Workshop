# Presentation outline

Two hours, SCCS 4403, 6–8 pm. Roughly a third talk, two thirds hands-on. There are three notebooks, one per paper, each a small illustrative toy version of that paper's own experiment. Each part follows the same pattern: the paper's method, what the paper found, then the setup of our demo. People then run the notebook themselves while presenters walk around.

The finished deck (25 slides) is in [`slides.pdf`](slides.pdf) and [`slides.html`](slides.html), with full speaker notes in [`speaker-notes.md`](speaker-notes.md). The deck's source files are in [`deck/`](deck/).

Every quote below was checked against the paper or page it cites, with section numbers.

## Agenda

| Time | Part | Notebook | Format |
|---|---|---|---|
| 6:00–6:10 | Arrival, snacks, open Colab (repo link on screen) | 1 · Amplification | |
| 6:10–6:25 | **The problem**, plus "find the mistake" warm-up | none | talk + pairs |
| 6:25–6:55 | **1. Iterated amplification** | 1 · Amplification | 10 min talk, 20 min hands-on |
| 6:55–7:30 | **2. Debate** | 2 · The pixel game | 15 min talk, 20 min hands-on |
| 7:30–7:50 | **3. Recursive reward modeling** | 3 · Reward modeling | 8 min talk, 12 min hands-on |
| 7:50–8:00 | **The map, open problems, and your project** | [`projects.md`](../projects.md) | talk + discussion |

If running late, make Notebook 3 a live demo only (it runs in seconds) and keep the closing.

---

## The problem (6:10–6:25)

**Slide 1. Scalable Oversight Workshop.** Repo link on screen; ask everyone to open Notebook 1 and run all.

**Slide 2. Human feedback works while humans can check the work.** Christiano et al.: "…there are harder tasks for which we can't compute demonstrations or rewards even with human assistance." (§1) Examples: too big (a 10,000-line pull request), too hard (a long proof or a novel experiment), too slow (a policy whose effects take years).

**Slide 3. Can a weak overseer supervise a strong system, if the system helps?** Three answers from 2018: iterated amplification, debate, recursive reward modeling.

**Slide 4. Find the mistake (5 min, in pairs).** Adapted from the Iliad debate session (§4): "All horses are the same color" (§4.1) and "ln(2) = 0" (§4.3). Did you check every line, or did someone point you to the right one? That gap between finding a flaw and checking it is the intuition behind debate.

**Slide 5. Tonight.** The agenda table above.

## 1. Iterated amplification (6:25–6:55)

Christiano, Shlegeris & Amodei (2018), *Supervising strong learners by amplifying weak experts*.

**Slide 6. Method: split the hard question.** "Amplifyᴴ(X) answers a question Q by having H identify a sequence of useful subquestions, using X to compute a subanswer to each subquestion" (§1.1). Diagram: the paper's permutation-powering decomposition (Table 3), with σ²⁹(x) split as 14 + 14 + 1.

**Slide 7. Method: distill the team into X, then amplify again.** Three steps (§2.2): "record every decision made by H during the process"; "We train a model H′ to predict the decisions made by H in each of these transcripts"; "X is trained by supervised learning on these (Q, A) pairs." No outside objective: "the objective is implicit in the way that the human coordinates the copies of X" (§1.1). X "chases this moving target" (§2.3).

**Slide 8. The experiments.** Five algorithmic tasks (§4.1): permutation powering, sequential assignments, wildcard search, shortest path, union find. "Rather than having a human perform the decomposition, we provide a hard-coded algorithm H which decomposes each task" (§4.1). Each task grows from size 8 to 64 (Table 3, App. C.2); the baseline is supervised learning from the ground truth (Fig. 2).

**Slide 9. What the paper found.** "We have shown that Iterated Amplification can successfully solve algorithmically complex tasks where there is no external reward function and the objective is implicit in a learned decomposition." (§6) Speed: "with at worst a modest slowdown" (§4.2). Data: "Supervised learning required tens of millions of examples"; amplification "required tens of thousands" (§4.2). The limit: "These experiments don't shed any light on whether humans can decompose interesting real world tasks" (§4.3).

**Slide 10. Demo setup: Notebook 1.** Paper vs. toy: σ on 64 elements, k up to 64 → 16 elements, k up to 31; same Table 3 decomposition; a Transformer with a random σ as input → a small network with one fixed σ; same 85% curriculum rule (App. C.2), raising k instead of the size; same supervised baseline (Fig. 2). Beyond the paper: an H with random slips or blind spots. Expected results are in the speaker notes and the [demo plan](../demo-plan.md).

*Hands-on (20 min):* Notebook 1 exercise 2 (catch the blind spot with the paper's "?" answer, App. C.1) or exercise 3 (a worse decomposition).

## 2. Debate (6:55–7:30)

Irving, Christiano & Amodei (2018), *AI safety via debate*.

**Slide 11. Method: two agents argue, a judge decides.** "A question q ∈ Q is shown to both agents. The two agents state their answers a₀, a₁ ∈ A (which may be the same). The two agents take turns making statements s₀, s₁, . . . , sₙ₋₁ ∈ S. The judge sees the debate (q, a, s) and decides which agent wins." (§2)

**Slide 12. The claim.** "In the debate game, it is harder to lie than to refute a lie." (§2)

**Slide 13. Theory result: halve the claim to one step.** "In an analogy to complexity theory, debate with optimal play can answer any question in PSPACE given polynomial time judges (direct judging answers only NP questions)." (Abstract; formal statement in §2.2.) Diagram from the Iliad notes' proof sketch (§11): "a debate can verify reachability by recursively halving a path".

**Slide 14. What the paper found.** MNIST, Table 2 means: 6 pixels, judge alone 59.4%, debate 74.4%, with precommit 88.9%; 4 pixels, 48.2%, 59.7%, 85.2%. "For both 4 and 6 pixels precommit is crucial" and "Both versions give an advantage to the second player to move" (§3.1). The limit: "we have demonstrated only a basic experiment for MNIST images" (§8).

**Slide 15. The paper's own worries.** The ten subsection titles of §5, "Reasons to worry", verbatim, and the §5.2 question: "Will one of the agents be able to sneak in a statement that appears logical but contains a hidden flaw without the other agent pointing out the flaw?" Barnes's obfuscated arguments and the Iliad §16 list are in the speaker notes.

**Slide 16. Demo setup: Notebook 2.** Paper vs. toy: MNIST → 8×8 digits; judge trained in advance on random nonzero pixels (§3.1), same; one nonzero pixel per turn, 6 in all, same; Monte Carlo tree search with 10k rollouts per move → greedy debaters; precommit with the liar winning if any of 9 lies wins, same, plus the version without precommit. Then the room judges debates.

*Hands-on (20 min):* Notebook 2 exercise 1 (plan ahead) or exercise 2 (a lazy liar).

## 3. Recursive reward modeling (7:30–7:50)

Leike, Krueger, Everitt, Martic, Maini & Legg (2018), *Scalable agent alignment via reward modeling: a research direction*.

**Slide 17. Method: learn the reward from the user, then optimize it.** Diagram: user → reward model → agent. "The user trains a reward model to learn their intentions by providing feedback." "Both processes happen concurrently, thus we are training the agent with the user in the loop." (§3.1) Assumption 1: "We can learn user intentions to a sufficiently high accuracy." Assumption 2: "For many tasks we want to solve, evaluation of outcomes is easier than producing the correct behavior." (§1)

**Slide 18. Method: earlier agents help the user evaluate.** "In step k we use the agent Aₖ₋₁ to assist the user in evaluating outcomes when training agent Aₖ." (§3.2) The novel example (§3.2) and the Fig. 2 caption.

**Slide 19. What the paper concludes: a research agenda.** The five challenges of §4 (amount of feedback, feedback distribution, reward hacking, unacceptable outcomes, reward-result gap) and the ten approaches of §5, by their subsection titles. No experiments of its own: the direction is "'shovel-ready' for empirical research today" (§8).

**Slide 20. What the paper found: a frozen reward model gets exploited.** Fig. 4 (from Ibarz et al. 2018): "Over time the agent learns to exploit the reward model: the perceived performance (according to the reward model) increases, while the actual performance (according to the game score) plummets." Reward hacking (§4.3): "an effect that lets the agent get more reward than intended by exploiting loopholes in the process determining the reward". The fix (§5.1): "If we provide the agent with reward feedback online, we get a tighter feedback loop between the user's feedback and the agent's behavior."

**Slide 21. The open question.** "An important open question is whether errors accumulate: do the mistakes of the more narrow agent Aₖ₋₁ lead to larger mistakes in the training of agent Aₖ?" (§3.2)

**Slide 22. Demo setup: Notebook 3.** Paper vs. toy: the user's intentions → "flow" in a 64-digit document; feedback → 1,500 noisy comparisons of 4-digit snippets; a learned reward model → a learned score per adjacent pair; Aₖ₋₁ assists evaluation → helpers score chunks of 4, then 16; an agent trained against a frozen reward model (Fig. 4) → an optimizer rewriting the document; online feedback (§5.1) → random spot checks while it trains.

*Hands-on (15 min):* Notebook 3 exercise 2 (fewer comparisons).

## The map and your project (7:50–8:00)

**Slide 23. Three bets, three failure modes.** Adapted from the SRC Discussion 2 table. In Hubinger's 11 proposals, RRM is 8, debate is 9, and amplification appears in 2, 3, 4, 10 and 11.

| | The human… | Bets that… | Fails when… |
|---|---|---|---|
| Amplification | splits questions, delegates pieces | hard reasoning factors into small checkable steps | systematic overseer errors spread up the tree |
| Debate | judges between two adversaries | refuting a lie is easier than telling one | opponents are weak, or flaws are hidden |
| RRM | evaluates, helped by earlier agents | evaluating is easier than doing | optimizers exploit what nobody evaluates |

**Slide 24. Six places to start.** Picks from [`projects.md`](../projects.md); each notebook's exercise list is also a starting point.

**Slide 25. Is debate research more capabilities than alignment?** The closing question from the Iliad session (§21). Show of hands, then two or three answers.

---

## Sources

- Christiano, Shlegeris & Amodei (2018), [Supervising strong learners by amplifying weak experts](https://arxiv.org/abs/1810.08575)
- Irving, Christiano & Amodei (2018), [AI safety via debate](https://arxiv.org/abs/1805.00899)
- Leike et al. (2018), [Scalable agent alignment via reward modeling](https://arxiv.org/abs/1811.07871)
- Ibarz et al. (2018), [Reward learning from human preferences and demonstrations in Atari](https://arxiv.org/abs/1811.06521) (source of Leike et al. Fig. 4)
- Hubinger (2020), [An overview of 11 proposals for building safe advanced AI](https://www.lesswrong.com/posts/fRsjBseRuvRhMPPE5/an-overview-of-11-proposals-for-building-safe-advanced-ai)
- Barnes (2020), [Debate update: obfuscated arguments problem](https://www.alignmentforum.org/posts/PJLABqQ962hZEqhdB/debate-update-obfuscated-arguments-problem)
- Wäldchen, Iliad Intensive, [Scalable Oversight and Debate](https://iliad-intensive.org/safety/debate/) ([PDF](https://iliad-intensive.org/downloads/debate/debate-nosol.pdf), CC BY 4.0)
