# Starting your own scalable oversight project

Ideas sized for a few weeks, roughly in order of how much setup they need. Each builds on a notebook in this repo.

1. **Catch systematic overseer errors in amplification** (Notebook 1). Give H a blind spot and design decompositions or consistency checks that detect it. Measure how much extra overseer work each check costs. Question: is there a decomposition whose error map is contracting even for correlated errors?

2. **Bounded debaters and obfuscated arguments** (Notebook 2). Build problems where every split of a false claim looks equally plausible, so a compute-limited opponent can't find the flaw. Barnes's 2020 post on the [obfuscated arguments problem](https://www.alignmentforum.org/posts/PJLABqQ962hZEqhdB/debate-update-obfuscated-arguments-problem) is the starting point.

3. **Train the debaters** (Notebook 3). Replace greedy pixel choice with debaters trained by self-play, and check whether training finds honest strategies or judge exploits. The paper's precommit vs. no-precommit comparison (§3.1) is a good first target.

4. **Distilled recursive reward models** (Notebook 4). Train a new reward model at each level from the assisted evaluator's comparisons, as real RRM would, and measure whether blind spots get inherited, amplified or corrected across levels.

5. **Language-model debate on hidden passages.** Two small open models debate a reading-comprehension question about a passage the judge model can't see. Measure judge accuracy with and without debate, and with one debater weaker than the other. (Needs a GPU or an API; Colab's free GPU runs small models.)

6. **Factored cognition with an LLM.** Have a model answer multi-step math or logic questions only by decomposing them into subquestions answered by fresh copies of itself (a shallow HCH). Compare with answering directly, and look at where decompositions fail.

If you start one of these, tell us in the club Discord. We are happy to help scope it.
