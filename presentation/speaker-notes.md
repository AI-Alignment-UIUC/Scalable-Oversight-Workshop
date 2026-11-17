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

## Iterated amplification: the method, what the paper found, and the Notebook 1 setup.

**Slide 6. Split the hard question**

H is the human, X is the current model. The amplified system is H plus several copies of X working together. Notebook 1 is a small version of the paper’s first task, permutation powering: “what is sigma to the k of x?” H uses the paper’s decomposition (Table 3): sigma^(2k)(x) = sigma^k(sigma^k(x)) and sigma^(2k+1)(x) = sigma(sigma^k(sigma^k(x))), where the last sigma is a primitive question, “What is σ(x)?”, that H answers itself. H never solves a hard question. The abstract’s one-line summary: amplification “progressively builds up a training signal for difficult problems by combining solutions to easier subproblems”.

**Slide 7. Distill the team into X, then amplify again**

Why H′ exists (§2.1): “In order to reduce the burden on the human expert H, we train a ‘human predictor’ H′”. Because H′ only learns how to split questions and combine answers, the paper expects “to train it with much less data”. In the limit the system approximates a huge tree of copies of H. Christiano’s name for that ideal, HCH (“humans consulting HCH”), comes from his blog posts, not this paper. Our notebook skips H′: H answers directly.

**Slide 8. Five algorithmic tasks, a hard-coded “human”**

The tasks are chosen because the right decomposition is known, so the paper can test the training method without a person in the loop. Permutation powering, the one our notebook uses: “Given a permutation σ: {1, …, 64} → {1, …, 64}, compute σ^k(x) for k up to 64” (§4.1). X is a Transformer: “We implement X as an encoder-decoder architecture with self-attention, closely following the Transformer architecture” (§2.5). Point out that every one of these tasks has a clean recursive structure; the open question is whether real tasks do.

**Slide 9. Almost as fast, with no ground truth**

Full §4.2 sentence on data: “Supervised learning required tens of millions of examples in order to learn these algorithms”, while “Iterated Amplification required tens of thousands of examples in order to learn the much simpler decompositions”. Those tens of thousands are queries to H: Table 2 gives 6,000 to 24,000 per task. The cost: “training with amplification requires about twice as much computation per question” (§4.2). The §4.3 limit continues: “nor on whether it would be feasible to learn messy real world decompositions.” The conclusion’s hope (§6): “This offers hope for applying ML in domains where we cannot compute a suitable objective, even with human help, as long as humans are able to decompose a task into simpler pieces.”

**Slide 10. Permutation powering, small enough to train in seconds**

Run section 3 live (about 20 seconds). What to expect: the difficulty climbs from 1 to 5 in four rounds under the 85% rule, which compares X with its own training targets and needs no ground truth. X answers all 496 questions by round 15. Section 5 is the Fig. 2 comparison: amplification reaches 95% at round 9, supervised learning on true answers at round 6, the paper’s “modest slowdown”. Amplification never sees a true answer to a hard question. Section 6 is our extension, since the paper’s H never errs: an H that misreads 10% of lookups at random leaves near-zero error, while an H that always misreads 2 of the 16 inputs leaves about 90% error on the deepest questions. Random error washes out in distillation; consistent bias is copied into every answer built on it. Another simplification: the paper also trains a model H′ of the human so H isn’t queried every time; our H answers directly. Hands-on (20 min): exercise 2 (catch the blind spot with the paper’s “?” answer, App. C.1) or exercise 3 (a worse decomposition).

## Debate: the game, the theory, the MNIST results, the paper’s worries, and the Notebook 2 setup.

**Slide 11. Two agents argue. A judge decides.**

Hubinger’s one-line version (proposal 9): train a model, Alice, to win debates against a copy of itself, Bob, in front of a human judge. The question for today: when does winning the debate line up with telling the truth?

**Slide 12. “In the debate game, it is harder to lie than to refute a lie.”**

This is the whole proposal in one sentence. If it holds, honesty is the winning strategy, even when the judge couldn’t have found the answer. Remind people of the warm-up: once someone pointed at the bad step, checking it was easy.

**Slide 13. Halve the claim to one step**

The formal statement (§2.2): “For any problem L ∈ PSPACE, we can choose a polynomial time judge so that optimal play in the debate game with polynomially many steps solves L.” The point of the NP contrast: a judge checking an answer directly can only verify what fits in one certificate; debate lets the judge verify much more by checking one step. The proof sketch from the Iliad notes, §11 (“Theorem: Debate=PSPACE”): “a debate can verify reachability by recursively halving a path”. Alice claims a path of at most 2^t steps between two configurations of the machine and names a midpoint; Bob picks the half to challenge; at t = 0 the judge checks directly that the two configurations are equal or one step apart. Orange marks the half Bob disputes. No notebook for this one: it is the theory behind the pixel game. Optional aside: cross-examination, letting a debater question frozen copies of the opponent, pushes the theoretical power from PSPACE to NEXP (Iliad §§12–13).

**Slide 14. Debate lifts a weak judge**

The setup, from the Fig. 2 caption: “The debaters state their claimed label up front, then reveal one nonzero pixel per turn to the judge up to a total of 4 or 6.” How the experiment works (§3.1): “the judge is trained to classify MNIST from 6 (resp. 4) nonzero pixels, with the pixels chosen at random at training time.” The debaters use “pure Monte Carlo Tree Search” with “10k rollouts per move”. Precommit: “The two players precommit to their claimed label up front”; the paper plays “9 different games for the same image” and “the liar wins if any lie wins”. Without precommit, “the judge uses the maximum logit over all 10 classes” and “any incorrect guess by the judge is a win for the liar”. Table 2 in full, honest first / liar first: 6 px, no precommit 67.4 / 81.5, precommit 87.4 / 90.4; 4 px, no precommit 51.0 / 68.4, precommit 83.8 / 86.7. Table 2’s caption: “the honest player wins more often than a random judge, showing that honesty has an advantage”. The abstract puts it as “boosting the classifier’s accuracy from 59.4% to 88.9% given 6 pixels and from 48.2% to 85.2% given 4 pixels”. The §8 limit in full: “debate is proposal only for the natural language case, and we have demonstrated only a basic experiment for MNIST images.” Also note (§3.1) that debaters “cannot lie about these pixels, but a dishonest player could choose them adversarially to fool the judge.”

**Slide 15. Ten reasons to worry (§5)**

These are the subsection titles of §5, “Reasons to worry”, verbatim. Later work sharpened the hidden-flaw worry. Barnes (2020) on obfuscated arguments: “a dishonest debater can often create arguments that have a fatal error, but where it is very hard to locate the error.” The Iliad notes’ list of criticisms (§16) adds a judge persuaded to run code with harmful effects, no exploration guarantees, collusion between debaters, and biased judges; §18 covers a newer protocol, prover-estimator debate.

**Slide 16. The pixel game, on 8×8 digits**

Run section 4 live (about 40 seconds). What to expect, with the paper’s Table 2 in brackets: the judge alone is right 56% of the time [59.4]. Without precommit, honest first 52% [67.4], liar first 82% [81.5], mean 67% [74.4]. With precommit, 86% [87.4], 93% [90.4], mean 90% [88.9]. As in the paper, precommit helps the honest side and moving second is an advantage. Then let the room judge a few debates with the buttons in section 5a. Caveat, as in the paper: debaters can only reveal true pixels, so they cannot lie about evidence. Greedy debaters are a weak test; exercise 1 adds lookahead. Hands-on (20 min): Notebook 2 exercises 1–2.

## Recursive reward modeling: the method, the paper’s agenda and its one result, and the Notebook 3 setup.

**Slide 17. Learn the reward from the user, then optimize it**

Leike, Krueger, Everitt, Martic, Maini and Legg (2018), Scalable agent alignment via reward modeling: a research direction. The question it starts from (§1): “How can we create agents that behave in accordance with the user’s intentions?” The abstract defines reward modeling as “learning a reward function from interaction with the user and optimizing the learned reward function”. Fig. 1’s caption: “a reward model is trained with user feedback; this reward model provides rewards to an agent trained with RL by interacting with the environment.” The intuition behind Assumption 2: “it is easier for the user to evaluate an outcome in the environment than it would be to teach behavior directly.” RRM is Hubinger’s proposal 8. Difference from amplification: it trains agents with reward rather than training a model to imitate a human.

**Slide 18. Earlier agents help the user evaluate**

The definition (§3.2): “we call recursive reward modeling: leveraging agents trained with reward modeling on simpler tasks” to help evaluate harder ones. Each assistant was itself trained with reward modeling, one level down. This is where Assumption 2 does the work: the user never has to produce a novel, only judge one with help.

**Slide 19. A research agenda: five challenges, ten approaches**

Section and subsection titles are verbatim. The §5 approaches “should be thought of as directions to explore”. Section 6, Establishing trust, adds five avenues: design choices, testing, interpretability, formal verification and theoretical guarantees. The outlook (§8): “agent alignment research is still in its early stages, but we believe that there is substantial reason for optimism”.

**Slide 20. A frozen reward model gets exploited**

Fig. 4’s setup: “The fully trained reward model from the best seed is frozen and used to train an new agent from scratch.” (“an new” is the paper’s typo.) The §5.1 diagnosis: “Preliminary experiments show failure modes when the reward model is not trained online, i.e. in parallel with the agent”. “In these cases the agent learns to exploit reward models that are frozen.” “Because there is no additional user feedback, loopholes in the reward model that the agent discovers cannot be corrected.” This is the one result in the paper, and Notebook 3 reproduces its shape.

**Slide 21. “An important open question is whether errors accumulate: do the mistakes of the more narrow agent Aₖ₋₁ lead to larger mistakes in the training of agent Aₖ?”**

The paper leaves this open. Our two notebooks give toy answers: in Notebook 1 it depends on whether the overseer’s errors are random or systematic; in Notebook 3 it depends on what nobody evaluates. Tie the two halves of the evening together. Same question as SRC criterion C2c: state the per-level error map and classify it as contracting, thresholded, compounding or cascading.

**Slide 22. A document too long for the user to read**

Sections 3 and 4 of Notebook 3 run in seconds. The paper runs no experiments of its own; its Fig. 4, reproduced from Ibarz et al. (2018), shows the failure we reproduce: “The fully trained reward model from the best seed is frozen and used to train an new agent from scratch.” What to expect: against the frozen evaluator, the score keeps rising while true quality peaks around 57 of 63 and falls to 49. The optimizer repeats the 4-digit snippet the reward model likes best (2777 2777 2777 …), so every chunk is perfect and nearly every seam between chunks is broken, because no helper looks at the seams. With online feedback (8 random 4-digit windows of the agent’s current document, re-drawn every 250 steps) true quality stays at 62. Hands-on (about 12 min): exercise 2, cut the comparisons to 100 and look for exploited reward-model errors.

## The map of proposals, open problems, and starting your own project.

**Slide 23. Three bets, three failure modes**

Adapted from the SRC Discussion 2 comparison table. Ask: which one would you trust to review a 10,000-line code change? There’s no right answer; get two or three views.

**Slide 24. Six places to start**

Full descriptions in projects.md in the repo. Each notebook’s exercise list is also a starting point. Tell people to post in the club Discord if they start one; we’re happy to help scope it.

**Slide 25. Is debate research more capabilities than alignment?**

The closing question from the Iliad debate session (§21). Thank everyone, point to the repo, the readings in the README (Christiano et al., Leike et al., Irving et al., Hubinger’s 11 proposals), and the SRC reading club for follow-up discussions.
