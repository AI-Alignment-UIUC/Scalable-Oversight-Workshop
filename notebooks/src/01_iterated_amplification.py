# %% [markdown]
# # Notebook 1: Iterated Amplification on a toy task
#
# **Scalable Oversight Workshop · AI Alignment @ Illinois**
#
# Paper: Christiano, Shlegeris & Amodei (2018), [*Supervising strong learners by amplifying weak experts*](https://arxiv.org/abs/1810.08575).
#
# > "We propose a new method, Iterated Amplification, for a human expert H to train an ML agent X." (§1.1)
#
# The amplified system is H "and several copies of X working together to solve a problem" (§1.1). X is trained to imitate that team, then the team gets stronger because X got stronger, and so on:
#
# > "Amplify(X) is modestly smarter than X working on its own, and X chases this moving target" (§2.3, lightly reformatted)
#
# We rebuild the paper's first task, **permutation powering** (§4.1): given a permutation σ and a number k, answer "what is σᵏ(x)?". The trick is that our "human" H never solves a hard question. H can only:
#
# 1. look up σ(x) directly (one step, the base case), and
# 2. split "σᵏ(x)?" into two easier questions, "y = σᵃ(x)?" then "σᵇ(y)?" with a + b = k, and ask X to answer them.
#
# Like the paper, our H is a hand-coded algorithm: "Rather than having a human perform the decomposition, we provide a hard-coded algorithm H which decomposes each task" (§4.1). Ground truth is used **only to measure** how well X does, never to train it.
#
# Everything runs on CPU in about two minutes. Only numpy and matplotlib are needed.

# %%
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

N = 16        # permutation of {0, ..., N-1}
K_MAX = 32    # we want X to answer sigma^k(x) for every k up to K_MAX

sigma = rng.permutation(N)

def truth(x, k):
    """Ground truth sigma^k(x). Used ONLY for evaluation."""
    x = np.array(x)
    for _ in range(int(k)):
        x = sigma[x]
    return x

print("sigma =", sigma)
print("sigma^5(3) =", truth(3, 5))

# %% [markdown]
# ## 1. The learner X
#
# X is a small neural network (one hidden layer, written in numpy so there is nothing to install). Its input is the question (x, k) as two one-hot vectors, and its output is a probability distribution over the answer.
#
# X is the "strong learner": fast, but it only knows what it has been trained on. At the start it knows nothing.

# %%
class MLP:
    """Tiny classifier: one-hot(x) + one-hot(k) -> softmax over N answers. Trained with Adam."""
    def __init__(self, n_in, n_hidden, n_out, seed=0, lr=3e-3):
        r = np.random.default_rng(seed)
        self.p = {"W1": r.normal(0, 1 / np.sqrt(n_in), (n_in, n_hidden)), "b1": np.zeros(n_hidden),
                  "W2": r.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, n_out)), "b2": np.zeros(n_out)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t, self.lr = 0, lr

    def forward(self, X):
        h = np.maximum(0, X @ self.p["W1"] + self.p["b1"])
        z = h @ self.p["W2"] + self.p["b2"]
        z = z - z.max(1, keepdims=True)
        prob = np.exp(z); prob /= prob.sum(1, keepdims=True)
        return h, prob

    def train_step(self, X, y):
        h, prob = self.forward(X)
        g = prob.copy(); g[np.arange(len(y)), y] -= 1; g /= len(y)
        grads = {"W2": h.T @ g, "b2": g.sum(0)}
        gh = (g @ self.p["W2"].T) * (h > 0)
        grads.update({"W1": X.T @ gh, "b1": gh.sum(0)})
        self.t += 1
        for k in self.p:
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * grads[k] ** 2
            mh, vh = self.m[k] / (1 - 0.9 ** self.t), self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)

def encode(x, k):
    x, k = np.atleast_1d(x), np.atleast_1d(k)
    out = np.zeros((len(x), N + K_MAX))
    out[np.arange(len(x)), x] = 1
    out[np.arange(len(x)), N + k - 1] = 1
    return out

class Agent:
    """X: answers the question 'sigma^k(x)?' in one fast forward pass."""
    def __init__(self, seed=0):
        self.net = MLP(N + K_MAX, 256, N, seed=seed)
    def __call__(self, x, k):
        return self.net.forward(encode(x, k))[1].argmax(1)
    def fit(self, x, k, y, steps=60, batch=256):
        X = encode(x, k)
        for _ in range(steps):
            idx = rng.integers(0, len(y), batch)
            self.net.train_step(X[idx], y[idx])

ALL_X, ALL_K = np.meshgrid(np.arange(N), np.arange(1, K_MAX + 1), indexing="ij")
ALL_X, ALL_K = ALL_X.ravel(), ALL_K.ravel()
TRUE = np.array([truth(x, k) for x, k in zip(ALL_X, ALL_K)])

def accuracy_by_k(agent):
    correct = agent(ALL_X, ALL_K) == TRUE
    return np.array([correct[ALL_K == k].mean() for k in range(1, K_MAX + 1)])

X0 = Agent()
print("Untrained X, accuracy for k = 1..8:", accuracy_by_k(X0)[:8].round(2))

# %% [markdown]
# ## 2. The weak expert H and the Amplify operator
#
# > "Amplify(X) answers a question Q by having H identify a sequence of useful subquestions, using X to compute a subanswer to each subquestion" (§1.1)
#
# Our H follows the paper's decomposition for this task, which halves the exponent (App. C.3): to answer σᵏ(x) it asks X for y = σ^⌊k/2⌋(x), then asks X for σ^⌈k/2⌉(y). When k = 1, H reads σ(x) straight from the context. Those base-case lookups are the only "oracle calls" in the whole method, and we count them.
#
# Notice that H never does more than one lookup or two delegated questions, however large k is. That is the point: H stays weak.

# %%
ORACLE_CALLS = 0

def lookup(x, noise=0.0):
    """H reads sigma(x) from the context. With probability `noise` H misreads it."""
    global ORACLE_CALLS
    x = np.atleast_1d(x)
    ORACLE_CALLS += len(x)
    y = sigma[x].copy()
    wrong = rng.random(len(x)) < noise
    y[wrong] = rng.integers(0, N, wrong.sum())
    return y

def amplify(X, x, k, noise=0.0):
    """Amplify^H(X): H decomposes 'sigma^k(x)?' and delegates the pieces to copies of X."""
    x, k = np.atleast_1d(x), np.atleast_1d(k)
    ans = np.empty_like(x)
    base = k == 1
    ans[base] = lookup(x[base], noise)                    # H answers the easy case itself
    a = k[~base] // 2; b = k[~base] - a
    y = X(x[~base], a)                                    # subquestion 1, answered by X
    ans[~base] = X(y, b)                                  # subquestion 2, answered by X
    return ans

# A tiny transcript, in the paper's format tau = (Q, Q1, A1, Q2, A2, A) (§2.2), for an untrained X:
x, k = 3, 6
a, b = k // 2, k - k // 2
y = X0(x, a)[0]
print(f"Q : sigma^{k}({x})?")
print(f"Q1: sigma^{a}({x})?   A1 (from X): {y}")
print(f"Q2: sigma^{b}({y})?   A2 (from X): {X0(y, b)[0]}")
print(f"A : {amplify(X0, x, k)[0]}   (truth: {truth(x, k)})")

# %% [markdown]
# ## 3. Iterate: distill Amplify(X) into X, repeat
#
# The training loop (§2.2, simplified by dropping the learned model H′ of the human):
#
# 1. Sample questions Q.
# 2. Run Amplify^H(X) on them to get answers A.
# 3. Train X by supervised learning on the (Q, A) pairs.
#
# We also use a curriculum like the paper's (App. C.2: difficulty rises "when accuracy reaches 85%"), with one twist worth noticing: we measure progress by how often **X agrees with Amplify(X)**, which needs no ground truth at all.

# %%
def iterated_amplification(rounds=30, noise=0.0, amplify_fn=amplify, seed=0, verbose=True):
    global ORACLE_CALLS
    ORACLE_CALLS = 0
    X = Agent(seed)
    k_cap, history, calls = 1, [], []
    for r in range(rounds):
        xs = rng.integers(0, N, 2048)
        ks = rng.integers(1, k_cap + 1, 2048)
        targets = amplify_fn(X, xs, ks, noise)            # the amplified overseer's answers
        X.fit(xs, ks, targets)                            # distillation
        agree = (X(xs, ks) == amplify_fn(X, xs, ks, noise)).mean()
        if agree > 0.85 and k_cap < K_MAX:
            k_cap = min(2 * k_cap, K_MAX)
        history.append(accuracy_by_k(X)); calls.append(ORACLE_CALLS)
        if verbose and (r < 8 or r % 5 == 4):
            print(f"round {r+1:2d}  k_cap={k_cap:2d}  agreement={agree:.2f}  "
                  f"true accuracy (all k)={history[-1].mean():.2f}  oracle calls so far={ORACLE_CALLS}")
    return X, np.array(history), calls

X_amp, hist, calls = iterated_amplification()

# %%
plt.figure(figsize=(9, 3.5))
plt.imshow(hist.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1,
           extent=[0.5, len(hist) + 0.5, 0.5, K_MAX + 0.5])
plt.colorbar(label="accuracy vs. ground truth")
plt.xlabel("amplification round"); plt.ylabel("k (question difficulty)")
plt.title("The frontier of questions X can answer moves outward")
plt.show()

# %% [markdown]
# **What to look for.** Early rounds only get k = 1 and 2 right. Each round, X learns to answer slightly harder questions, because Amplify(X) could answer them using the previous X. Questions with k = 32 are answered correctly even though H only ever looked up a single step of σ.
#
# > "X starts random; H answers simple questions alone; X copies those; the frontier moves outward." (our paraphrase of §2.3)

# %% [markdown]
# ## 4. Compare with supervised learning on ground truth
#
# The paper reports that amplification solves its tasks "with at worst a modest slowdown" (§4.2) compared to training on true labels. Here the cost that matters is the number of times anyone consults the true σ. Supervised learning needs a true label for every training question. Amplification only consults σ for k = 1.

# %%
X_sup = Agent(seed=1)
sup_acc, sup_labels = [], []
for r in range(len(hist)):
    xs = rng.integers(0, N, 2048); ks = rng.integers(1, K_MAX + 1, 2048)
    X_sup.fit(xs, ks, np.array([truth(x, k) for x, k in zip(xs, ks)]))
    sup_acc.append(accuracy_by_k(X_sup).mean()); sup_labels.append(2048 * (r + 1))

plt.figure(figsize=(6, 3.5))
plt.plot(hist.mean(1), label="iterated amplification (H + X)")
plt.plot(sup_acc, label="supervised on ground truth")
plt.xlabel("round"); plt.ylabel("mean accuracy over k = 1..32"); plt.legend(); plt.show()
print(f"Ground-truth labels used by supervised learning: {sup_labels[-1]:,}")
print(f"Base-case lookups used by amplification:        {calls[-1]:,}")
print("Note: each supervised label for k=32 secretly costs 32 lookups of sigma.")

# %% [markdown]
# ## 5. When the overseer makes mistakes: do errors compound?
#
# Real overseers make mistakes. σ³²(x) is built from 32 lookups stacked through several levels of decomposition, so if H misreads σ(x) now and then, does the error grow with k?
#
# This is the question Leike et al. ask of recursive reward modeling: "do the mistakes of the more narrow agent A_{k-1} lead to larger mistakes in the training of agent A_k?" (§3.2). In SRC's criterion v1.1 language (C2c), we want to classify the per-level error map as **contracting, thresholded, compounding or cascading**.
#
# ### 5a. Random mistakes
#
# First, H misreads σ(x) **at random**: 10% of lookups return a random number. The dashed line is what you would expect if every one of the k lookups had to be right.

# %%
X_noisy, hist_noisy, _ = iterated_amplification(noise=0.10, verbose=False)
ks = np.arange(1, K_MAX + 1)
plt.figure(figsize=(6, 3.5))
plt.plot(ks, 1 - hist[-1], "o-", label="H never wrong")
plt.plot(ks, 1 - hist_noisy[-1], "o-", label="H wrong on 10% of lookups, at random")
plt.plot(ks, 1 - 0.9 ** ks, "k--", label="1 - 0.9^k (every lookup must be right)")
plt.xlabel("k"); plt.ylabel("error of final X"); plt.legend(); plt.show()

# %% [markdown]
# Surprise: the error is (close to) zero. X sees each question many times, and a model trained with cross-entropy learns the *most common* answer, so independent mistakes average out before they are passed up a level. Distillation is a denoiser. For random errors the map is **contracting**.
#
# ### 5b. Systematic mistakes
#
# Now H has a blind spot: for two particular inputs it *always* misreads σ(x), the same wrong way every time. Real overseers have biases like this.

# %%
BLIND = [int(sigma[0]), int(sigma[5])]       # two inputs H always gets wrong
WRONG = {x: int((sigma[x] + 1) % N) for x in BLIND}

def lookup_biased(x, noise=0.0):
    global ORACLE_CALLS
    x = np.atleast_1d(x)
    ORACLE_CALLS += len(x)
    y = sigma[x].copy()
    for b, w in WRONG.items():
        y[x == b] = w
    return y

def amplify_biased(X, x, k, noise=0.0):
    x, k = np.atleast_1d(x), np.atleast_1d(k)
    ans = np.empty_like(x)
    base = k == 1
    ans[base] = lookup_biased(x[base])
    a = k[~base] // 2; b = k[~base] - a
    ans[~base] = X(X(x[~base], a), b)
    return ans

X_bias, hist_bias, _ = iterated_amplification(amplify_fn=amplify_biased, verbose=False)

# Prediction: a question goes wrong if the true path x, sigma(x), ..., sigma^(k-1)(x) visits a blind spot.
def hits_blind_spot(x, k):
    for _ in range(k):
        if x in BLIND:
            return True
        x = sigma[x]
    return False
predicted = [np.mean([hits_blind_spot(x, k) for x in range(N)]) for k in ks]

plt.figure(figsize=(6, 3.5))
plt.plot(ks, 1 - hist_noisy[-1], "o-", label="random mistakes (10%)")
plt.plot(ks, 1 - hist_bias[-1], "o-", label="systematic mistakes (2 of 16 inputs)")
plt.plot(ks, predicted, "k--", label="fraction of paths that touch a blind spot")
plt.xlabel("k"); plt.ylabel("error of final X"); plt.legend(); plt.show()
print(f"H's base-case error rate: {len(BLIND)/N:.0%} of inputs.  Error of X at k=1: {1-hist_bias[-1][0]:.0%}, "
      f"at k=32: {1-hist_bias[-1][-1]:.0%}")

# %% [markdown]
# H is wrong on only 12.5% of single lookups, but X ends up wrong on about half of the hard questions, because every long chain of reasoning that passes through a blind spot inherits the mistake. Averaging cannot fix an error that is the same every time. For systematic errors the map is **compounding**: per-level error grows with depth until it saturates at the share of questions whose path ever reaches a blind spot (here σ's cycle structure caps that at one half).
#
# **Takeaway for the talk:** amplification is only as good as the overseer's *correlated* errors. Random sloppiness is cheap; a consistent blind spot is spread to every question that depends on it.
#
# ## 6. Exercises
#
# 1. **Catch the blind spot.** H knows σ is a permutation (every output appears exactly once). Write an `amplify` that uses this to detect, or route around, a misread. Hint: the paper's H returns a special "?" symbol "when recursive results are inconsistent" (App. C); try computing σᵏ(x) along two different decompositions and compare.
# 2. **Thresholds.** In 5a, raise `noise` to 0.2, then higher. Two things break: the curriculum stalls (X can never agree with a noisy Amplify more than about 1 − noise of the time, so `k_cap` stops growing), and eventually averaging itself stops working. Fix the first, then find the noise level where the second happens. Why does it depend on N (a *thresholded* regime)?
# 3. **A worse decomposition.** Change `amplify` to split k into 1 + (k − 1) instead of halving. How many rounds does it take to reach k = 32 now? How does the oracle-call count change?
# 4. **Another task.** Adapt the code to one of the paper's other tasks (§4.1), e.g. *union find*: "find the root of x's tree" in a rooted forest. What is the base case, and how should H decompose?
# 5. **Discussion.** The paper is honest that "These experiments don't shed any light on whether humans can decompose interesting real world tasks" (§4.3). Pick a real task (grading an essay, reviewing a pull request). Could you write an H for it that only ever does a small amount of work? Where would its *systematic* blind spots be?
