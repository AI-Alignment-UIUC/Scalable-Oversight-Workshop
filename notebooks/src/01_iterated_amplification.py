# %% [markdown]
# # Notebook 1: Iterated amplification
#
# **Scalable Oversight Workshop · AI Alignment @ Illinois**
#
# Paper: Christiano, Shlegeris & Amodei (2018), [*Supervising strong learners by amplifying weak experts*](https://arxiv.org/abs/1810.08575), §§1–4 and Appendix C.
#
# > "We propose a new method, Iterated Amplification, for a human expert H to train an ML agent X." (§1.1)
#
# The amplified system, Amplify^H(X), consists "of H and several copies of X working together to solve a problem" (§1.1). X is trained to imitate that team. Then the team gets stronger, because X got stronger: Amplify^H(X) "is modestly smarter than X working on its own, and X chases this moving target" (§2.3).
#
# The paper tests this on five algorithmic tasks (§4.1). This notebook is a small toy version of the first one, **permutation powering**:
#
# > "Given a permutation σ: {1, …, 64} → {1, …, 64}, compute σ^k(x) for k up to 64." (§4.1)
#
# The paper doesn't use a human. "Rather than having a human perform the decomposition, we provide a hard-coded algorithm H which decomposes each task" (§4.1). H can look up σ(x) for a single x, and it can split a question into two easier ones. It never solves a hard question itself, and **ground truth is never used for training**, only to measure how X is doing.
#
# **In plain words.** σ is a fixed shuffle of the numbers 0–15. σ(3) is where 3 goes after one shuffle; σ^k(3) is where it ends up after k shuffles. "Where does 3 end up after 29 shuffles?" is hard if all you can do is look at one shuffle at a time. That is the overseer's situation.
#
# ### What we keep from the paper, and what we simplify
#
# | | Paper | This toy |
# |---|---|---|
# | Task | σ on 64 elements, k up to 64 | σ on 16 elements, k up to 31 |
# | H's decomposition | σ^{2k}(x) = σ^k(σ^k(x)) and σ^{2k+1}(x) = σ(σ^k(σ^k(x))) (Table 3) | same |
# | Curriculum | raise the difficulty whenever X gets 85% of its training targets right (App. C.2); the difficulty is the size of σ, from 8 to 64 | same rule, but the difficulty is how large k can be |
# | Baseline | supervised learning on ground truth (Fig. 2) | same |
# | What X sees | a fresh random σ every time, so X must learn the *algorithm* | one fixed σ, so X only memorizes its powers |
# | X | a transformer | a small numpy network |
#
# The fixed σ is the big simplification: it lets a tiny network learn in seconds. What carries over is the training signal, which is the part that matters for oversight.
#
# > **What you'll see**
# > 1. A network that starts out knowing nothing learns to answer 31-shuffle questions, though its teacher only ever looked at one shuffle at a time (section 4).
# > 2. It gets there a little more slowly than supervised learning on true answers, as in the paper's Fig. 2 (section 5).
# > 3. Beyond the paper: when H makes mistakes, random ones wash out but consistent ones spread (section 6).
# > 4. Controls to run your own experiments (section 7).
#
# Everything runs on a CPU in under a minute. Use **Runtime → Run all**. Needs numpy, matplotlib and ipywidgets (all in Colab).

# %%
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

N = 16          # sigma permutes {0, ..., 15}
BITS = 5        # k is written with 5 bits, so 1 <= k <= 31

sigma = rng.permutation(N)

def truth(x, k):
    """Ground truth sigma^k(x), vectorized. Used ONLY to measure X, never to train it."""
    out, k = np.array(x, copy=True), np.array(k, copy=True)
    while (k > 0).any():
        m = k > 0
        out[m] = sigma[out[m]]; k[m] -= 1
    return out

def depth(k):
    """Recursion depth of a question = the number of bits in k (1 for k = 1, ..., 5 for k = 16..31)."""
    return np.floor(np.log2(np.maximum(k, 1))).astype(int) + 1

print("sigma =", sigma)
print("sigma^29(3) =", truth(3, 29), "  (a depth", depth(np.array(29)), "question)")

# %% [markdown]
# ## 1. The learner X
#
# X is a small neural network. Its input is the question (x, k): x as a one-hot vector and k in binary, as in the paper (App. C.1: "For permutation powering, we represent the exponent k in binary"). Its output is a probability distribution over the 16 possible answers.
#
# X is the "strong learner": it answers any question in one fast step, but it only knows what it has been trained on. At the start it knows nothing.

# %%
class MLP:
    """Classifier with two hidden layers, trained with Adam."""
    def __init__(self, n_in, n_hidden, n_out, seed=0, lr=3e-3):
        r = np.random.default_rng(seed)
        self.p = {"W1": r.normal(0, 1 / np.sqrt(n_in), (n_in, n_hidden)), "b1": np.zeros(n_hidden),
                  "W2": r.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, n_hidden)), "b2": np.zeros(n_hidden),
                  "W3": r.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, n_out)), "b3": np.zeros(n_out)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t, self.lr = 0, lr

    def forward(self, X):
        h1 = np.maximum(0, X @ self.p["W1"] + self.p["b1"])
        h2 = np.maximum(0, h1 @ self.p["W2"] + self.p["b2"])
        z = h2 @ self.p["W3"] + self.p["b3"]
        z = z - z.max(1, keepdims=True)
        prob = np.exp(z); prob /= prob.sum(1, keepdims=True)
        return h1, h2, prob

    def train_step(self, X, y):
        h1, h2, prob = self.forward(X)
        g = prob.copy(); g[np.arange(len(y)), y] -= 1; g /= len(y)
        grads = {"W3": h2.T @ g, "b3": g.sum(0)}
        d2 = (g @ self.p["W3"].T) * (h2 > 0); grads.update(W2=h1.T @ d2, b2=d2.sum(0))
        d1 = (d2 @ self.p["W2"].T) * (h1 > 0); grads.update(W1=X.T @ d1, b1=d1.sum(0))
        self.t += 1
        for k in self.p:
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * grads[k] ** 2
            mh, vh = self.m[k] / (1 - 0.9 ** self.t), self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)

def encode(x, k):
    x, k = np.atleast_1d(x), np.atleast_1d(k)
    out = np.zeros((len(x), N + BITS))
    out[np.arange(len(x)), x] = 1
    out[:, N:] = (k[:, None] >> np.arange(BITS)) & 1        # k in binary
    return out

class Agent:
    """X: answers 'sigma^k(x)?' in one forward pass."""
    def __init__(self, seed=0):
        self.net = MLP(N + BITS, 256, N, seed=seed)
    def __call__(self, x, k):
        return self.net.forward(encode(x, k))[2].argmax(1)

ALL_X, ALL_K = [a.ravel() for a in np.meshgrid(np.arange(N), np.arange(1, 2 ** BITS), indexing="ij")]
TRUE = truth(ALL_X, ALL_K)           # all 16 x 31 = 496 questions, for measuring only

def accuracy_by_depth(agent):
    correct = agent(ALL_X, ALL_K) == TRUE
    d = depth(ALL_K)
    return np.array([correct[d == i].mean() for i in range(1, BITS + 1)])

print("Untrained X, accuracy at depths 1..5:", accuracy_by_depth(Agent()).round(2))

# %% [markdown]
# ## 2. The weak expert H and the Amplify operator
#
# > "Amplify^H(X) answers a question Q by having H identify a sequence of useful subquestions, using X to compute a subanswer to each subquestion" (§1.1)
#
# H follows the paper's decomposition for this task (Table 3):
#
# - σ^{2m}(x) = σ^m(σ^m(x)): ask X for y = σ^m(x), then ask X for σ^m(y).
# - σ^{2m+1}(x) = σ(σ^m(σ^m(x))): the same, then H looks up one more step of σ itself.
# - σ^1(x): H looks it up.
#
# H never does more than two delegated questions and one lookup, however large k is. That is the point: H stays weak. We count every time H is asked, and every lookup of σ it makes.

# %%
COUNT = {"H": 0, "lookups": 0}

def lookup(x):
    """H reads sigma(x) for one x: the only access to the task anyone gets during training."""
    COUNT["lookups"] += len(np.atleast_1d(x))
    return sigma[x]

def amplify(X, x, k):
    """Amplify^H(X): H splits 'sigma^k(x)?' into subquestions, and copies of X answer them."""
    x, k = np.atleast_1d(x), np.atleast_1d(k)
    COUNT["H"] += len(x)
    ans = np.empty_like(x)
    base = k == 1
    ans[base] = lookup(x[base])                        # H answers the base case itself
    xs, ks = x[~base], k[~base]
    m = ks // 2
    z = X(X(xs, m), m)                                 # two subquestions, each answered by X
    odd = ks % 2 == 1
    z[odd] = lookup(z[odd])                            # odd k: one more step of sigma, by H
    ans[~base] = z
    return ans

# A transcript in the paper's format (Q, Q1, A1, Q2, A2, A) (§2.2), with an untrained X:
X0 = Agent()
x, k = 3, 29
m = k // 2
a1 = int(X0(x, m)[0]); a2 = int(X0(a1, m)[0])
print(f"Q : sigma^{k}({x})?")
print(f"Q1: sigma^{m}({x})?    A1 (from X): {a1}")
print(f"Q2: sigma^{m}({a1})?   A2 (from X): {a2}")
print(f"H : k is odd, so look up sigma({a2}) = {sigma[a2]}")
print(f"A : {amplify(X0, x, k)[0]}   (truth: {truth(x, k)})")

# %% [markdown]
# The untrained X gives nonsense subanswers, so the team's answer is wrong. That is expected. What matters is what happens when X is trained to imitate the team, again and again.
#
# ## 3. The training loop
#
# The paper's loop (§2.2), without the learned model H′ of the human:
#
# 1. Sample questions Q.
# 2. Run Amplify^H(X) on them to get answers A.
# 3. Train X by supervised learning on the (Q, A) pairs.
# 4. Repeat.
#
# Two details come from the paper's appendices:
#
# - **Question distribution** (App. C.4): the paper picks the leading bit of k at random, so every depth gets about the same share of questions.
# - **Curriculum** (App. C.2): "Whenever X achieves an accuracy of at least 85% at predicting the (Q,A) pairs in its dataset, we increment the difficulty." Our difficulty is the largest depth allowed. This check needs **no ground truth**: it compares X with its own training targets.

# %%
def sample_questions(n, difficulty):
    """Questions whose depth is at most `difficulty`, half of them at the current difficulty."""
    lead = np.where(rng.random(n) < 0.5, difficulty, rng.integers(1, difficulty + 1, n))   # leading bit of k
    low = 2 ** (lead - 1)
    return rng.integers(0, N, n), low + (rng.random(n) * low).astype(int)

def iterated_amplification(rounds=25, amplify_fn=amplify, seed=0, threshold=0.85, verbose=True,
                           n=2048, steps=60, batch=256, keep_rounds=3):
    COUNT.update(H=0, lookups=0)
    X = Agent(seed)
    difficulty, data, log = 1, [], []
    for r in range(rounds):
        xs, ks = sample_questions(n, difficulty)
        data.append((xs, ks, amplify_fn(X, xs, ks)))      # 1-2. the amplified team's answers
        data = data[-keep_rounds:]                         # keep the last few rounds of (Q, A) pairs
        dx, dk, dy = (np.concatenate(c) for c in zip(*data))
        enc = encode(dx, dk)
        for _ in range(steps):                             # 3. distill: train X to imitate the team
            i = rng.integers(0, len(dy), batch)
            X.net.train_step(enc[i], dy[i])
        fit = (X(xs, ks) == data[-1][2]).mean()           # X vs. its own targets: no ground truth
        if fit >= threshold and difficulty < BITS:
            difficulty += 1                                # the curriculum (App. C.2)
        log.append(dict(round=r + 1, difficulty=difficulty, fit=fit, acc_x=accuracy_by_depth(X),
                        H=COUNT["H"], lookups=COUNT["lookups"]))
        if verbose and (r < 8 or r % 5 == 4):
            print(f"round {r+1:2d}  difficulty {difficulty}  X fits its targets {fit:4.0%}  "
                  f"true accuracy by depth {np.round(log[-1]['acc_x'], 2)}")
    return X, log

X_amp, log_amp = iterated_amplification()

# %% [markdown]
# ## 4. The frontier moves outward
#
# The heatmap shows X's true accuracy at each depth after each round. Depth 1 means k = 1; depth 5 means k between 16 and 31.

# %%
SHARE = np.array([(depth(ALL_K) == i).mean() for i in range(1, BITS + 1)])   # share of questions at each depth
acc = np.array([l["acc_x"] for l in log_amp])
plt.figure(figsize=(9, 3.2))
plt.imshow(acc.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1,
           extent=[0.5, len(acc) + 0.5, 0.5, BITS + 0.5])
plt.colorbar(label="accuracy vs. ground truth")
plt.xlabel("round"); plt.ylabel("depth (bits in k)")
plt.title("What X can answer, round by round")
plt.show()
print(f"Final accuracy over all {len(TRUE)} questions: {np.mean(X_amp(ALL_X, ALL_K) == TRUE):.0%}")

# %% [markdown]
# This is the dynamic the paper describes in §2.3. At first X answers randomly, and only the questions H can answer alone (depth 1) come out right. X learns those. Then H can answer depth-2 questions by splitting them, X learns those, and so on. By the end X answers 31-shuffle questions, although H never looked at more than one shuffle at a time.
#
# ## 5. Compare with supervised learning (the idea of the paper's Fig. 2)
#
# Fig. 2 compares "Iterated Amplification (orange) and supervised learning from the ground truth data (blue)". The paper's result is that amplification solves its tasks "with at worst a modest slowdown" (§4.2). We train the same network on the same questions, with true answers as targets.

# %%
def supervised(rounds=25, seed=1, n=2048, steps=60, batch=256, keep_rounds=3):
    X = Agent(seed); difficulty, data, curve = 1, [], []
    for r in range(rounds):
        xs, ks = sample_questions(n, difficulty)
        data.append((xs, ks, truth(xs, ks)))              # ground-truth labels
        data = data[-keep_rounds:]
        dx, dk, dy = (np.concatenate(c) for c in zip(*data))
        enc = encode(dx, dk)
        for _ in range(steps):
            i = rng.integers(0, len(dy), batch)
            X.net.train_step(enc[i], dy[i])
        if (X(xs, ks) == data[-1][2]).mean() >= 0.85 and difficulty < BITS:
            difficulty += 1
        curve.append(np.mean(X(ALL_X, ALL_K) == TRUE))
    return curve

sup_curve = supervised()
amp_curve = [(l["acc_x"] * SHARE).sum() for l in log_amp]
plt.figure(figsize=(6, 3.5))
plt.plot(range(1, len(amp_curve) + 1), amp_curve, color="tab:orange", label="iterated amplification")
plt.plot(range(1, len(sup_curve) + 1), sup_curve, color="tab:blue", label="supervised on ground truth")
plt.xlabel("round"); plt.ylabel("accuracy on all questions"); plt.ylim(0, 1.02)
plt.legend(); plt.title("Amplification vs. supervised learning"); plt.show()

first = lambda curve: next((r + 1 for r, a in enumerate(curve) if a >= 0.95), None)
print(f"Rounds to reach 95%: supervised {first(sup_curve)}, amplification {first(amp_curve)}")

# %% [markdown]
# Amplification gets there a few rounds later, which is the paper's "modest slowdown". The real difference is what each method needed from outside:
#
# - **Supervised learning** needed a correct answer to every hard question it trained on. Computing σ^29(x) takes 29 lookups, so each label hides a lot of work by someone who could already solve the task.
# - **Amplification** never saw a correct answer to a hard question. It needed only H: one split and at most one lookup per question.
#
# In the paper's words: "Supervised learning required tens of millions of examples in order to learn these algorithms" (§4.2). That is the scalable oversight point. If nobody can produce correct answers to the hard questions, supervised learning is not available, and amplification still is.
#
# ## 6. Beyond the paper: what if H makes mistakes?
#
# The paper's H is a perfect algorithm. A human overseer is not. This section is **our extension**, not something the paper tests. It asks the question Leike et al. raise about recursive reward modeling: "do the mistakes of the more narrow agent A_{k-1} lead to larger mistakes in the training of agent A_k?" (Leike et al. 2018, §3.2).
#
# We try two kinds of mistakes in H's lookups:
#
# - **Random**: 10% of lookups return a random number.
# - **Systematic**: for 2 of the 16 inputs, H always reads the same wrong value (a blind spot).

# %%
def make_amplify(noise=0.0, blind_spots=0, split="halve"):
    blind = {int(sigma[i]): int((sigma[sigma[i]] + 1) % N) for i in range(blind_spots)}
    def bad_lookup(x):
        COUNT["lookups"] += len(x)
        y = sigma[x].copy()
        wrong = rng.random(len(x)) < noise
        y[wrong] = rng.integers(0, N, wrong.sum())
        for b, w in blind.items():
            y[x == b] = w
        return y
    def amp(X, x, k):
        x, k = np.atleast_1d(x), np.atleast_1d(k)
        COUNT["H"] += len(x)
        ans = np.empty_like(x); base = k == 1
        ans[base] = bad_lookup(x[base])
        xs, ks = x[~base], k[~base]
        if split == "halve":
            m = ks // 2
            z = X(X(xs, m), m); odd = ks % 2 == 1
            z[odd] = bad_lookup(z[odd])
        else:                                             # "peel one": sigma^k(x) = sigma(sigma^(k-1)(x))
            z = bad_lookup(X(xs, ks - 1))
        ans[~base] = z
        return ans
    return amp, set(blind)

amp_random, _ = make_amplify(noise=0.10)
amp_blind, BLIND = make_amplify(blind_spots=2)
X_random, log_random = iterated_amplification(amplify_fn=amp_random, threshold=0.85 * 0.9, verbose=False)
X_blind, log_blind = iterated_amplification(amplify_fn=amp_blind, verbose=False)

def touches_blind_spot(x, k):
    """Does the true path x, sigma(x), ..., sigma^(k-1)(x) pass through a blind spot?"""
    for _ in range(k):
        if x in BLIND: return True
        x = sigma[x]
    return False
predicted = [np.mean([touches_blind_spot(x, k) for x, k in zip(ALL_X, ALL_K) if depth(k) == d])
             for d in range(1, BITS + 1)]

ds = np.arange(1, BITS + 1)
plt.figure(figsize=(6, 3.5))
plt.plot(ds, 1 - log_amp[-1]["acc_x"], "o-", label="H never wrong")
plt.plot(ds, 1 - log_random[-1]["acc_x"], "o-", label="random mistakes (10% of lookups)")
plt.plot(ds, 1 - log_blind[-1]["acc_x"], "o-", label="blind spot (2 of 16 inputs)")
plt.plot(ds, predicted, "k--", label="share of questions whose path\npasses a blind spot")
plt.xticks(ds); plt.xlabel("depth"); plt.ylabel("error of the final X"); plt.ylim(-0.02, 1.02); plt.legend(fontsize=8); plt.show()

# %% [markdown]
# Random mistakes almost vanish. X sees each question many times, and training with cross-entropy learns the *most common* answer, so independent mistakes average out before they get passed up a level.
#
# A blind spot does not average out. It is wrong the same way every time, so every question whose chain of shuffles passes through it inherits the mistake, and the error grows with depth along the dashed line. H is wrong on 2 of 16 inputs, but X ends up wrong on far more of the deepest questions.
#
# **Takeaway:** distillation removes random noise and faithfully copies consistent bias, to every question built on top of it.

# %% [markdown]
# ## 7. 🎛 Explore
#
# ### 7a. Ask the amplified team a question
#
# Pick x and k. You'll see H's decomposition with the trained X's answer at every node, checked against the truth. (This uses `X_amp` from section 3.)

# %%
import ipywidgets as widgets

def show_tree(x=3, k=29, levels=3):
    def walk(x, k, level):
        guess, true = int(X_amp(x, k)[0]), int(truth(x, k))
        mark = "✓" if guess == true else f"✗ (truth {true})"
        pad = "    " * level
        if k == 1:
            print(f"{pad}σ^1({x}) = {guess} {mark}   ← H looks this up"); return
        m = k // 2
        y = int(X_amp(x, m)[0])
        extra = ", then H looks up σ of the result" if k % 2 else ""
        print(f"{pad}σ^{k}({x}) = {guess} {mark}   H asks σ^{m}({x}) and σ^{m}({y}){extra}")
        if level + 1 < levels:
            walk(x, m, level + 1); walk(y, m, level + 1)
    walk(x, k, 0)

widgets.interact(show_tree,
                 x=widgets.IntSlider(value=3, min=0, max=N - 1, description="x"),
                 k=widgets.IntSlider(value=29, min=1, max=2 ** BITS - 1, description="k shuffles"),
                 levels=widgets.IntSlider(value=3, min=1, max=5, description="show levels"));

# %% [markdown]
# ### 7b. Run your own experiment
#
# Set up the overseer, then click **Run Interact** (a few seconds). Things to try:
#
# - **Random mistakes**: how high can the rate go before X stops learning?
# - **Blind spots**: does the damage grow with the number of blind spots, or with depth?
# - **Decomposition**: "peel one" uses σ^k(x) = σ(σ^(k−1)(x)) instead of halving. How many rounds does it need now?

# %%
def experiment(mistakes="random", rate=0.10, blind_spots=2, split="halve", rounds=25):
    amp, _ = make_amplify(noise=rate if mistakes == "random" else 0.0,
                          blind_spots=blind_spots if mistakes == "blind spots" else 0, split=split)
    thr = 0.85 * (1 - (rate if mistakes == "random" else 0))   # so the curriculum can still advance
    _, log = iterated_amplification(rounds=rounds, amplify_fn=amp, verbose=False, threshold=thr)
    a = np.array([l["acc_x"] for l in log])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    axes[0].imshow(a.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1,
                   extent=[0.5, len(a) + 0.5, 0.5, BITS + 0.5])
    axes[0].set_xlabel("round"); axes[0].set_ylabel("depth"); axes[0].set_title("accuracy by round (yellow = right)")
    axes[1].plot(ds, 1 - a[-1], "o-"); axes[1].set_ylim(-0.02, 1.02); axes[1].set_xticks(ds)
    axes[1].set_xlabel("depth"); axes[1].set_ylabel("error"); axes[1].set_title("error of the final X")
    plt.show()
    print(f"Final accuracy over all questions: {(a[-1] * SHARE).sum():.0%}.  H was asked {log[-1]['H']:,} times.")

widgets.interact_manual(experiment,
                        mistakes=widgets.ToggleButtons(options=["none", "random", "blind spots"], value="random"),
                        rate=widgets.FloatSlider(value=0.10, min=0, max=0.6, step=0.05, description="random rate"),
                        blind_spots=widgets.IntSlider(value=2, min=0, max=8, description="# blind spots"),
                        split=widgets.Dropdown(options=["halve", "peel one"], description="H splits by"),
                        rounds=widgets.IntSlider(value=25, min=5, max=60, step=5, description="rounds"));

# %% [markdown]
# ## 8. Exercises
#
# 1. **Another task from the paper.** Implement *sequential assignments* (§4.1): "Given a function f:{1,…,8}^2 → {1,…,8} and a sequence of 64 assignments of the form x:=3 or x:=f(y,z), evaluate a particular variable." Table 3's decomposition: look up the definition of x; if x := f(y,z), evaluate y, evaluate z, and look up f(y, z). What plays the role of the depth?
# 2. **Catch the blind spot.** The paper adds a special answer "?" that H gives when the recursive answers are inconsistent or inconclusive (App. C.1). Write an `amplify` that computes σ^k(x) along two different splits, returns "?" when they disagree, and see whether that contains the blind spot.
# 3. **A worse decomposition.** Use `split="peel one"`. How many rounds does it take now, and why does the depth of the decomposition matter so much?
# 4. **No curriculum.** In `iterated_amplification`, start with `difficulty = BITS` instead of 1. What goes wrong, and why does the paper need the curriculum?
# 5. **Discussion.** The paper says plainly: "These experiments don't shed any light on whether humans can decompose interesting real world tasks" (§4.3). Pick a real task (grading an essay, reviewing a pull request). Could you write an H for it that only ever does a small amount of work? Where would its *systematic* blind spots be?
