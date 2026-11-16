# %% [markdown]
# # Notebook 2: Debate as recursive path-halving
#
# **Scalable Oversight Workshop · AI Alignment @ Illinois**
#
# Papers and notes:
# - Irving, Christiano & Amodei (2018), [*AI Safety via Debate*](https://arxiv.org/abs/1805.00899)
# - Iliad Intensive, [*Scalable Oversight and Debate*](https://iliad-intensive.org/safety/debate/) (Stephan Wäldchen), §§7–11
#
# The debate game (Irving et al., §2):
#
# > "A question q ∈ Q is shown to both agents. The two agents state their answers a₀, a₁ ∈ A. The two agents take turns making statements s₀, s₁, . . . , s_{n−1} ∈ S. The judge sees the debate (q, a, s) and decides which agent wins."
#
# The bet that makes it useful (§2): **"In the debate game, it is harder to lie than to refute a lie."** And the theory result (§2.2): "For any problem L ∈ PSPACE, we can choose a polynomial time judge so that optimal play in the debate game with polynomially many steps solves L."
#
# The Iliad notes prove that result with one picture: "a debate can verify reachability by recursively halving a path" (§11). This notebook plays that debate on a maze. A weak judge, who only ever checks **one step** of the maze, correctly decides whether a far-away cell is reachable, as long as the debaters are strong.
#
# Runs in seconds on CPU. Only numpy and matplotlib are needed.

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

rng = np.random.default_rng(1)
SIZE = 32                                     # a 32 x 32 maze
open_cell = rng.random((SIZE, SIZE)) > 0.33   # True = open, False = wall

def neighbours(c):
    r, k = c
    for dr, dk in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, kk = r + dr, k + dk
        if 0 <= rr < SIZE and 0 <= kk < SIZE and open_cell[rr, kk]:
            yield (rr, kk)

def one_step(u, v):
    """The ONLY thing the judge can do: check that v equals u or is next to it."""
    return u == v or v in set(neighbours(u))

BFS_WORK = 0
_cache = {}
def bfs(src):
    """Shortest-path tree from src. This is what the (strong) debaters can afford and the judge cannot."""
    global BFS_WORK
    if src in _cache:
        return _cache[src]
    dist, parent, q = {src: 0}, {src: None}, deque([src])
    while q:
        u = q.popleft(); BFS_WORK += 1
        for w in neighbours(u):
            if w not in dist:
                dist[w], parent[w] = dist[u] + 1, u
                q.append(w)
    _cache[src] = (dist, parent)
    return dist, parent

def dist(u, v):
    return bfs(u)[0].get(v, np.inf)

def path(u, v):
    _, parent = bfs(u)
    p = [v]
    while p[-1] != u:
        p.append(parent[p[-1]])
    return p[::-1]

open_cells = [(int(r), int(k)) for r, k in np.argwhere(open_cell)]
print(f"{len(open_cells)} open cells")

# %% [markdown]
# ## 1. The protocol
#
# The claim on the table is **"cell v can be reached from cell u in at most L steps"**, with L a power of two. Alice argues it is true; Bob argues it is false.
#
# Each round:
# 1. Alice names a midpoint c, implicitly claiming "u → c in ≤ L/2 steps" **and** "c → v in ≤ L/2 steps".
# 2. Bob picks the half he thinks is false. The debate continues on that half only.
# 3. When L = 1, the judge checks a single step with `one_step(u, v)`.
#
# If the original claim is false, at least one half is false every round, so a Bob who always points at a false half drives Alice down to a one-step claim that is false, and the judge sees it. If the claim is true, an honest Alice can always pick a midpoint with both halves true. The judge does log₂ L rounds of bookkeeping and one adjacency check, however big the maze is.

# %%
def honest_midpoint(u, v, L):
    p = path(u, v)                            # a true path with len(p) - 1 <= L
    return p[min(len(p) - 1, L // 2)]

def lying_midpoint(u, v, L):
    """Alice defends a FALSE claim. Best trick: make one half true, so Bob has to find the other."""
    reach_u = [c for c, d in bfs(u)[0].items() if d <= L // 2]
    if rng.random() < 0.5 and reach_u:        # first half true; get as close to v as possible
        return min(reach_u, key=lambda c: (dist(c, v), rng.random()))
    reach_v = [c for c in open_cells if dist(c, v) <= L // 2]
    if reach_v:                               # second half true; get as close to u as possible
        return min(reach_v, key=lambda c: (dist(u, c), rng.random()))
    return v

def alice(u, v, L):
    return honest_midpoint(u, v, L) if dist(u, v) <= L else lying_midpoint(u, v, L)

def bob_optimal(u, c, v, L):
    """Bob can run BFS too, so he checks both halves and attacks a false one."""
    return 0 if dist(u, c) > L // 2 else 1

def bob_random(u, c, v, L):
    return int(rng.integers(2))

def debate(u, v, L, bob=bob_optimal, verbose=False):
    transcript = []
    while L > 1:
        c = alice(u, v, L)
        side = bob(u, c, v, L)
        transcript.append((u, c, v, L, side))
        if verbose:
            print(f"L={L:3d}  Alice: '{u} -> {c} -> {v}'   Bob disputes the {'first' if side == 0 else 'second'} half")
        u, v = (u, c) if side == 0 else (c, v)
        L //= 2
    verdict = one_step(u, v)
    if verbose:
        print(f"Judge checks one step {u} -> {v}: {'valid, Alice wins' if verdict else 'INVALID, Bob wins'}")
    return verdict, transcript

# %% [markdown]
# ## 2. One honest debate and one dishonest debate

# %%
L = 64
s = open_cells[0]
d_from_s = bfs(s)[0]
t_true = max((c for c, d in d_from_s.items() if d <= L), key=lambda c: d_from_s[c])
t_false = min((c for c, d in d_from_s.items() if d > L), key=lambda c: d_from_s[c])

print(f"TRUE claim: {t_true} reachable from {s} in <= {L} steps (real distance {dist(s, t_true)})")
_, tr_true = debate(s, t_true, L, verbose=True)
print(f"\nFALSE claim: {t_false} reachable from {s} in <= {L} steps (real distance {dist(s, t_false)})")
_, tr_false = debate(s, t_false, L, verbose=True)

# %%
def draw(transcript, title, ax):
    ax.imshow(~open_cell, cmap="Greys", vmin=0, vmax=1.6)
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(transcript)))
    for (u, c, v, L, side), col in zip(transcript, colors):
        ax.plot([u[1], c[1], v[1]], [u[0], c[0], v[0]], "o-", color=col, lw=2, ms=5, label=f"L={L}")
    u, c, v, L, side = transcript[-1]
    a, b = (u, c) if side == 0 else (c, v)
    ax.plot([a[1], b[1]], [a[0], b[0]], "-", color="red", lw=4)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=7, loc="lower right")

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
draw(tr_true, "Honest Alice: every midpoint holds up", axes[0])
draw(tr_false, "Lying Alice: Bob zooms in on the bad step (red)", axes[1])
plt.show()

# %% [markdown]
# ## 3. How good is the judge?
#
# We run many random claims, half true and half false, and compare three settings:
#
# - **Judge alone**: the judge can only check one step, so for long claims the best it can do is guess (we let it say "true" if u and v are within one step, otherwise flip a coin).
# - **Debate with a strong Bob** who checks both halves.
# - **Debate with a random Bob** who never checks anything.

# %%
def sample_claim(L, want_true):
    while True:
        u, v = open_cells[rng.integers(len(open_cells))], open_cells[rng.integers(len(open_cells))]
        d = dist(u, v)
        if want_true and d <= L and d > 1: return u, v
        if not want_true and L < d < np.inf: return u, v

def judge_alone(u, v, L):
    return True if one_step(u, v) else bool(rng.integers(2))

results = {"judge alone": [], "debate, strong Bob": [], "debate, random Bob": []}
for i in range(400):
    truth = i % 2 == 0
    u, v = sample_claim(L, truth)
    results["judge alone"].append(judge_alone(u, v, L) == truth)
    results["debate, strong Bob"].append(debate(u, v, L, bob_optimal)[0] == truth)
    results["debate, random Bob"].append(debate(u, v, L, bob_random)[0] == truth)

truths = np.arange(400) % 2 == 0
for k, r in results.items():
    r = np.array(r)
    print(f"{k:22s} accuracy = {r.mean():.1%}   (true claims: {r[truths].mean():.0%}, false claims: {r[~truths].mean():.0%})")
print(f"\nJudge work per debate: {int(np.log2(L))} rounds + 1 adjacency check.")
print(f"Debaters' work: {BFS_WORK:,} BFS node expansions so far across all games.")

# %% [markdown]
# With a strong Bob the judge is perfect. With a random Bob, a lie survives unless Bob happens to pick the false half every single round, which happens with probability (1/2)^log₂L = 1/L. So:
#
# > **Debate does not make the judge smart. It moves the hard work onto the opponent, and it only works if the opponent actually finds the flaw.**
#
# This is the "exploration" worry from the Iliad notes' list of criticisms (§16): in training, a debater that has not learned to find flaws gives the liar a free pass.
#
# ## 4. Computationally bounded debaters (a taste of obfuscated arguments)
#
# Real debaters are not all-powerful. Let Bob have a compute budget: checking a half means searching every cell within L/2 steps of its start, and if that costs more than his budget he has to guess. As the maze gets bigger relative to Bob's budget, how often does the lie win?
#
# This is a toy version of Beth Barnes's **obfuscated arguments problem** ([2020](https://www.alignmentforum.org/posts/PJLABqQ962hZEqhdB/debate-update-obfuscated-arguments-problem)): "a dishonest debater can often create arguments that have a fatal error, but where it is very hard to locate the error."

# %%
def make_bob_budget(budget):
    def bob(u, c, v, L):
        # Checking "u -> c in <= L/2 steps" means searching every cell within L/2 of u.
        cost = sum(1 for d in bfs(u)[0].values() if d <= L // 2)
        if cost <= budget:
            return 0 if dist(u, c) > L // 2 else 1
        return int(rng.integers(2))
    return bob

false_claims = [sample_claim(L, False) for _ in range(150)]
budgets = [25, 50, 100, 200, 400, 800, len(open_cells)]
liar_wins = []
for B in budgets:
    bob = make_bob_budget(B)
    liar_wins.append(np.mean([debate(u, v, L, bob)[0] for u, v in false_claims]))

plt.figure(figsize=(6, 3.5))
plt.semilogx(budgets, liar_wins, "o-")
plt.xlabel("Bob's compute budget (cells he can search)"); plt.ylabel("fraction of lies that win")
plt.title("Lies win when refuting them costs more than Bob has")
plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
plt.show()

# %% [markdown]
# A weak Bob has to guess during the early rounds, where checking a half means searching a big region. One wrong guess sends the debate into a half that is actually true, and from then on Alice wins honestly. So lies mostly win until Bob can afford to check the top-level claims. In this maze the checks do get cheap as the debate zooms in; obfuscated arguments are the case where *no* split makes the flaw cheap to find: every half looks equally plausible and checking any of them costs as much as the whole problem.
#
# ## 5. Exercises
#
# 1. **Iliad Exercise 11.x.** Count the rounds and the length of each message in this protocol, as a function of the maze size and L. Why is that polynomial in the input size even though the number of paths is exponential?
# 2. **Weaken the judge.** Let `one_step` be wrong 10% of the time. What happens to accuracy with a strong Bob? Is debate robust to a noisy judge at the leaves?
# 3. **Smarter liar.** Our liar picks a midpoint that makes one half true. Can you write a liar that beats `make_bob_budget(100)` more often, e.g. by choosing midpoints whose false half is far from both endpoints, so Bob's search is expensive?
# 4. **Cross-examination.** The Iliad notes (§12) say "cross-examination increases the theoretical power of debate" by letting a debater question a frozen copy of the opponent. In this game, what would Bob gain if he could ask a past copy of Alice "what was your midpoint for (u, c)?" before choosing?
# 5. **Warm-up from the Iliad session (§4):** "Find the mistake in the presented proof" (all horses are the same color; ln 2 = 0). Which of those flawed proofs would a step-by-step debate catch fastest, and why?
