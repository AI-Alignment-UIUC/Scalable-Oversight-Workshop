# Scalable Oversight Workshop: speaker notes

## The problem: overseers who can’t check the work, and today’s plan.

**Slide 1. Scalable Oversight Workshop**

Welcome. While people settle in: put the repo link or QR code up, ask everyone to open Notebook 1 (Iterated amplification) in Colab and run all. It takes about 20 seconds and needs no setup. There are three notebooks, one per paper, each a small toy version of the paper’s own experiment. Today is two thirds hands-on: short talks, then notebooks. Builds on the SRC Scalable Oversight series.

**Slide 2. Human feedback works while humans can check the work**

RLHF and its relatives depend on a person being able to tell good output from bad. As systems take on bigger and harder tasks, that stops being true. Ask the room for an example from their own work where they could not check an AI’s answer.

**Slide 3. Can a weak overseer supervise a strong system, if the system helps?**

Three answers, all from 2018, all built on the same idea: use AI to help humans oversee AI. We’ll build a toy version of each today: one notebook per paper, each a small version of that paper’s own experiment.

**Slide 4. Find the mistake**

Show each proof in full (Iliad notes §4.1 and §4.3). Give pairs 5 minutes. Then ask: how long did it take? Who checked every line? The punchline: once someone points at the flawed step, checking it is easy. That gap between finding and checking is the intuition behind debate. Horses: the step from n=1 to n=2 fails because the two groups don’t overlap. ln 2: rearranging a conditionally convergent series changes its sum.

**Slide 5. Short talks, then you run the notebooks**

Roughly a third talk, two thirds hands-on. If we run late, Notebook 3 becomes a live demo only (it runs in seconds); the closing stays.

## Iterated amplification: split questions, distill the team, watch errors.

**Slide 6. Split the hard question**

H is the human, X is the current model. The amplified system is H plus several copies of X working together. Notebook 1 is a small version of the paper’s first task, permutation powering: “what is sigma to the k of x?” H uses the paper’s decomposition (Table 3): sigma^(2m)(x) = sigma^m(sigma^m(x)), and for odd k one more step of sigma, which H looks up itself. H never solves a hard question.

**Slide 7. Distill the team into X, then amplify again**

Training loop from §2.2, simplified: the paper also learns a model H′ of the human so H isn’t queried every time. Amplify(X) “is modestly smarter than X working on its own, and X chases this moving target” (§2.3). In the limit the system approximates a huge tree of copies of H. Christiano’s name for that ideal, HCH (“humans consulting HCH”), comes from his blog posts, not this paper.

**Slide 8. Five algorithmic tasks, a hard-coded “human”**

Quotes checked against the paper. It also reports “Supervised learning required tens of millions of examples in order to learn these algorithms” (§4.2), while H itself was queried only 6,000 to 24,000 times per task (Table 2), because a learned model H′ stood in for H the rest of the time. Point out that every one of these tasks has a clean recursive structure: the open question is whether real tasks do.

**Slide 9. The frontier of questions X can answer moves outward**

Run section 3 live (about 20 seconds). The difficulty climbs from 1 to 5 in four rounds, using the paper’s 85% rule, which compares X with its own training targets and needs no ground truth. Every depth reaches 100% by about round 15. Then the heatmap: yellow fills in from depth 1 upward. Section 5 is the paper’s Fig. 2 comparison: amplification is a few rounds slower than supervised learning, the paper’s “modest slowdown”, but never sees a correct answer to a hard question.

**Slide 10. Random slips wash out. Blind spots spread.**

Section 6 of Notebook 1. This is our extension: the paper’s H never makes mistakes. The systematic-error curve follows the share of questions whose chain of shuffles passes through a blind spot. Exercise 2 asks people to design an H that catches the blind spot, using the paper’s “?” answer (App. C.1) when two decompositions disagree.

## Debate: refuting a lie is easier than telling one, if the opponent is strong.

**Slide 11. Two agents argue. A judge decides.**

Hubinger’s one-line version (proposal 9): train a model, Alice, to win debates against a copy of itself, Bob, in front of a human judge. The question for today: when does winning the debate line up with telling the truth?

**Slide 12. “In the debate game, it is harder to lie than to refute a lie.”**

This is the whole proposal in one sentence. If it holds, honesty is the winning strategy, even when the judge couldn’t have found the answer. Remind people of the warm-up: once someone pointed at the bad step, checking it was easy.

**Slide 13. Halve the claim to one step**

The proof sketch from the Iliad notes, §11 (“Theorem: Debate=PSPACE”): “a debate can verify reachability by recursively halving a path”. Alice claims a path of at most 2^t steps between two configurations of the machine and names a midpoint; Bob picks the half to challenge; at t = 0 the judge checks directly that the two configurations are equal or one step apart. Orange marks the half Bob disputes. No notebook for this one: it is the theory behind the pixel game. Optional aside: cross-examination, letting a debater question frozen copies of the opponent, pushes the theoretical power from PSPACE to NEXP (Iliad §§12–13).

**Slide 14. A judge that sees six pixels**

Notebook 2 follows the paper’s rules: the judge is trained only on random nonzero pixels and never sees a debate; each move reveals one nonzero pixel; with precommit the liar wins an image if any of the 9 lies wins. Differences: 8×8 digits instead of MNIST, greedy debaters instead of Monte Carlo tree search. Expected table (paper in brackets): no precommit 52% honest first [67.4], 82% liar first [81.5], mean 67% [74.4]; precommit 86% [87.4], 93% [90.4], mean 90% [88.9]. Then let the room judge a few debates with the buttons in section 5a. Caveat: debaters can only reveal true pixels, so they cannot lie about evidence.

**Slide 15. Flaws that are hard to find**

The Iliad notes’ list of criticisms (§16): obfuscated arguments, a judge persuaded to run code with harmful effects, no exploration guarantees, collusion between debaters, and biased judges. §18 covers a newer protocol, prover-estimator debate. Hands-on now (20 min): Notebook 2 exercises 1–2.

## Recursive reward modeling: assisted evaluation, and a frozen evaluator gets gamed.

**Slide 16. Assistants help you evaluate**

Reward modeling: learn a reward function from the user’s feedback, then train an agent against it. The goal (§1): how can we create agents that behave in accordance with the user’s intentions? RRM is Hubinger’s proposal 8. Difference from amplification: it trains agents with reward rather than training a model to imitate a human.

**Slide 17. A frozen evaluator gets gamed**

Sections 3 and 4 of Notebook 3, run in seconds. The paper runs no experiments of its own; its Fig. 4, reproduced from Ibarz et al. (2018), shows a frozen reward model on Atari: “Over time the agent learns to exploit the reward model: the perceived performance (according to the reward model) increases, while the actual performance (according to the game score) plummets.” Our toy has the same shape: true quality peaks around 57, then falls to 49 while the green score keeps rising. The fix in §5.1: “If we provide the agent with reward feedback online, we get a tighter feedback loop between the user’s feedback and the agent’s behavior.” Here the user reads a few random 4-token windows of the agent’s current document every 250 steps, and true quality stays at 62.

**Slide 18. “An important open question is whether errors accumulate: do the mistakes of the more narrow agent Aₖ₋₁ lead to larger mistakes in the training of agent Aₖ?”**

Tie the two halves of the evening together. Same question as SRC criterion C2c: state the per-level error map and classify it as contracting, thresholded, compounding or cascading. Hands-on (15 min): Notebook 3 exercise 2, cut the comparisons to 100 and look for exploited reward-model errors.

## The map of proposals, open problems, and starting your own project.

**Slide 19. Three bets, three failure modes**

Adapted from the SRC Discussion 2 comparison table. Ask: which one would you trust to review a 10,000-line code change? There’s no right answer; get two or three views.

**Slide 20. Six places to start**

Full descriptions in projects.md in the repo. Each notebook’s exercise list is also a starting point. Tell people to post in the club Discord if they start one; we’re happy to help scope it.

**Slide 21. Is debate research more capabilities than alignment?**

The closing question from the Iliad debate session (§21). Thank everyone, point to the repo, the readings in the README (Christiano et al., Leike et al., Irving et al., Hubinger’s 11 proposals), and the SRC reading club for follow-up discussions.
