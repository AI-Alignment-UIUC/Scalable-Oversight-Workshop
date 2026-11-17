# Presentation outline

Two hours, SCCS 4403, 6–8 pm. Roughly a third talk, two thirds hands-on. There are three notebooks, one per paper, each a small illustrative toy version of that paper's own experiment. Each part is a short talk followed by a notebook that people run themselves while presenters walk around.

The finished deck (21 slides) is in [`slides.pdf`](slides.pdf) and [`slides.html`](slides.html), with full speaker notes in [`speaker-notes.md`](speaker-notes.md). The deck's source files are in [`deck/`](deck/).

Every quote below was checked against the paper or page it cites, with section numbers.

## Agenda

| Time | Part | Notebook | Format |
|---|---|---|---|
| 6:00–6:10 | Arrival, snacks, open Colab (repo link on screen) | 1 · Amplification | |
| 6:10–6:25 | **The problem**, plus "find the mistake" warm-up | none | talk + pairs |
| 6:25–6:55 | **1. Iterated amplification** | 1 · Amplification | 10 min talk, 20 min hands-on |
| 6:55–7:30 | **2. Debate** | 2 · The pixel game | 15 min talk, 20 min hands-on |
| 7:30–7:50 | **3. Recursive reward modeling** | 3 · Reward modeling | 5 min talk, 15 min hands-on |
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

**Slide 6. Split the hard question.** "Amplifyᴴ(X) answers a question Q by having H identify a sequence of useful subquestions, using X to compute a subanswer to each subquestion" (§1.1). Diagram: the paper's permutation-powering decomposition (Table 3), with σ²⁹(x) split as 14 + 14 + 1.

**Slide 7. Distill the team into X, then amplify again.** Amplify, distill, repeat (§2.2). No outside objective: "the objective is implicit in the way that the human coordinates the copies of X" (§1.1). HCH is from Christiano's blog posts, not this paper; mention it only in the notes.

**Slide 8. Five algorithmic tasks, a hard-coded "human".** Permutation powering, sequential assignments, wildcard search, shortest path, union find (§4.1). Amplification solved them "with at worst a modest slowdown" (§4.2). The honest limit: the experiments "don't shed any light on whether humans can decompose interesting real world tasks" (§4.3).

**Slide 9. Notebook 1 live.** Permutation powering on 16 elements, k up to 31, with the paper's decomposition, 85% curriculum rule (App. C.2) and supervised baseline (Fig. 2). X answers all 496 questions by round 15, reaches 95% at round 9 against round 6 for supervised learning, and amplification never uses a true answer to a hard question. See the [demo plan](../demo-plan.md).

**Slide 10. Random slips wash out. Blind spots spread.** Beyond the paper (its H never errs): an H that misreads 10% of lookups at random leaves near-zero error, while an H that always misreads 2 of 16 inputs leaves about 90% error on the deepest questions. Random error is contracting, systematic error is compounding (SRC criterion C2c).

*Hands-on (20 min):* Notebook 1 exercise 2 (catch the blind spot with the paper's "?" answer, App. C.1) or exercise 3 (a worse decomposition).

## 2. Debate (6:55–7:30)

**Slide 11. Two agents argue. A judge decides.** "A question q ∈ Q is shown to both agents. The two agents state their answers a₀, a₁ ∈ A (which may be the same). The two agents take turns making statements s₀, s₁, . . . , sₙ₋₁ ∈ S. The judge sees the debate (q, a, s) and decides which agent wins." (Irving et al. §2)

**Slide 12. The bet.** "In the debate game, it is harder to lie than to refute a lie." (§2)

**Slide 13. Halve the claim to one step.** "For any problem L ∈ PSPACE, we can choose a polynomial time judge so that optimal play in the debate game with polynomially many steps solves L." (§2.2) The proof sketch from the Iliad notes (§11): "a debate can verify reachability by recursively halving a path". No notebook; it is the theory behind the pixel game. Optional aside: cross-examination takes debate from PSPACE to NEXP (Iliad §§12–13).

**Slide 14. Notebook 2 live: a judge that sees six pixels.** "With 6 pixels, debate with precommit turns a 59.4% accurate judge into a 88.9% accurate debate." (§3.1) Our toy follows the paper's rules on 8×8 digits with greedy debaters: the judge alone is 56%, debate with precommit is 90%, and, as in Table 2, moving second helps (86% → 93%). Then the room judges a few debates with the buttons.

**Slide 15. Flaws that are hard to find.** "Will one of the agents be able to sneak in a statement that appears logical but contains a hidden flaw without the other agent pointing out the flaw?" (§5.2) Barnes on obfuscated arguments: "a dishonest debater can often create arguments that have a fatal error, but where it is very hard to locate the error." Exploration and judge bias from the Iliad list of criticisms (§16).

*Hands-on (20 min):* Notebook 2 exercise 1 (plan ahead) or exercise 2 (a lazy liar).

## 3. Recursive reward modeling (7:30–7:50)

**Slide 16. Assistants help you evaluate.** "In step k we use the agent Aₖ₋₁ to assist the user in evaluating outcomes when training agent Aₖ." (Leike et al. §3.2) The novel example: "the user is assisted by an agent that provides auxiliary input: extracting a summary of the plotline, checking spelling and grammar, summarizing character development, assessing the flow of the prose, and so on." (§3.2) The goal (§1): "How can we create agents that behave in accordance with the user's intentions?"

**Slide 17. Notebook 3 live: a frozen evaluator gets gamed.** The paper runs no experiments of its own. Its Fig. 4 (from Ibarz et al. 2018) shows a frozen reward model being exploited: "the perceived performance (according to the reward model) increases, while the actual performance (according to the game score) plummets." In our toy an agent writes a 64-token document scored in 4-token chunks: true quality peaks around 57 of 63, then falls to 49 while the score keeps rising. The §5.1 fix, "reward feedback online", has the user spot-check a few random windows every 250 steps, and true quality stays at 62.

**Slide 18. The open question.** "An important open question is whether errors accumulate: do the mistakes of the more narrow agent Aₖ₋₁ lead to larger mistakes in the training of agent Aₖ?" (§3.2) Notebook 1's answer: it depends on whether the errors are random or systematic. Notebook 3's: it depends on what nobody evaluates.

*Hands-on (15 min):* Notebook 3 exercise 2 (fewer comparisons).

## The map and your project (7:50–8:00)

**Slide 19. Three bets, three failure modes.** Adapted from the SRC Discussion 2 table. In Hubinger's 11 proposals, RRM is 8, debate is 9, and amplification appears in 2, 3, 4, 10 and 11.

| | The human… | Bets that… | Fails when… |
|---|---|---|---|
| Amplification | splits questions, delegates pieces | hard reasoning factors into small checkable steps | systematic overseer errors spread up the tree |
| Debate | judges between two adversaries | refuting a lie is easier than telling one | opponents are weak, or flaws are hidden |
| RRM | evaluates, helped by earlier agents | evaluating is easier than doing | optimizers exploit what nobody evaluates |

**Slide 20. Six places to start.** Picks from [`projects.md`](../projects.md); each notebook's exercise list is also a starting point.

**Slide 21. Is debate research more capabilities than alignment?** The closing question from the Iliad session (§21). Show of hands, then two or three answers.

---

## Sources

- Christiano, Shlegeris & Amodei (2018), [Supervising strong learners by amplifying weak experts](https://arxiv.org/abs/1810.08575)
- Irving, Christiano & Amodei (2018), [AI safety via debate](https://arxiv.org/abs/1805.00899)
- Leike et al. (2018), [Scalable agent alignment via reward modeling](https://arxiv.org/abs/1811.07871)
- Ibarz et al. (2018), [Reward learning from human preferences and demonstrations in Atari](https://arxiv.org/abs/1811.06521) (source of Leike et al. Fig. 4)
- Hubinger (2020), [An overview of 11 proposals for building safe advanced AI](https://www.lesswrong.com/posts/fRsjBseRuvRhMPPE5/an-overview-of-11-proposals-for-building-safe-advanced-ai)
- Barnes (2020), [Debate update: obfuscated arguments problem](https://www.alignmentforum.org/posts/PJLABqQ962hZEqhdB/debate-update-obfuscated-arguments-problem)
- Wäldchen, Iliad Intensive, [Scalable Oversight and Debate](https://iliad-intensive.org/safety/debate/) ([PDF](https://iliad-intensive.org/downloads/debate/debate-nosol.pdf), CC BY 4.0)
