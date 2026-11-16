# %% [markdown]
# # Notebook 3: Debate with a sparse judge (the "pixel game")
#
# **Scalable Oversight Workshop · AI Alignment @ Illinois**
#
# Paper: Irving, Christiano & Amodei (2018), [*AI Safety via Debate*](https://arxiv.org/abs/1805.00899), §3.1.
#
# The paper's first experiment gives the judge a severe handicap: it sees only a few pixels of an image. Two debaters, who see the whole image, take turns revealing pixels to the judge. One argues for the true label, the other for a false one. The paper reports (§3.1): "With 6 pixels, debate with precommit turns a 59.4% accurate judge into a 88.9% accurate debate."
#
# We rebuild a small version with scikit-learn's built-in 8×8 handwritten digits (no download, CPU only, about a minute). Our numbers will differ from the paper's (smaller images, simpler judge, greedy debaters instead of Monte Carlo tree search), but the shape of the result is the point.
#
# **Why it matters for oversight.** The judge stands in for a human who cannot inspect everything a strong model sees. The debaters stand in for strong models. If honest debaters reliably win, the weak judge can supervise questions it could never answer alone.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import warnings; warnings.filterwarnings("ignore")   # sklearn convergence notices

rng = np.random.default_rng(0)
digits = load_digits()
images = digits.images / 16.0                 # 1797 images, 8 x 8, values in [0, 1]
labels = digits.target
X_tr, X_te, y_tr, y_te = train_test_split(images.reshape(len(images), -1), labels, test_size=0.3, random_state=0)

K = 4                                         # pixels the judge gets to see

fig, axes = plt.subplots(1, 8, figsize=(10, 1.6))
for ax, img, lab in zip(axes, images, labels):
    ax.imshow(img, cmap="gray_r"); ax.set_title(lab); ax.axis("off")
plt.show()

# %% [markdown]
# ## 1. Train the sparse judge
#
# The judge sees a masked image: the values of the revealed pixels, plus a mask saying which pixels were revealed (so "revealed and blank" differs from "hidden"). Like the paper, we train it on **random** sets of K revealed pixels, so it knows nothing about debate.

# %%
def masked_input(x, mask):
    """x: (n, 64) pixel values, mask: (n, 64) booleans -> judge input (n, 128)."""
    return np.concatenate([x * mask, mask.astype(float)], axis=1)

def random_masks(n, k):
    m = np.zeros((n, 64), bool)
    for i in range(n):
        m[i, rng.choice(64, k, replace=False)] = True
    return m

REPEATS = 30
Xj = np.repeat(X_tr, REPEATS, axis=0); yj = np.repeat(y_tr, REPEATS)
judge = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=40, random_state=0)
judge.fit(masked_input(Xj, random_masks(len(Xj), K)), yj)

baseline = judge.score(masked_input(X_te, random_masks(len(X_te), K)), y_te)
print(f"Judge alone, {K} random pixels: {baseline:.1%} accurate (chance = 10%)")

# %% [markdown]
# ## 2. The debate
#
# For one test image with true label `y`:
#
# 1. **Precommit.** The honest debater claims `y`; the liar claims some other label `l` (paper §3.1 calls this "debate with precommit").
# 2. **Reveal.** The two debaters alternate, honest first, until K pixels are revealed. Each picks the pixel that, once revealed, most increases the judge's belief in its own label relative to the opponent's (a greedy one-step search; the paper used Monte Carlo tree search).
# 3. **Verdict.** The judge looks at the K revealed pixels and picks whichever of the two claimed labels it finds more likely.
#
# We say the honest debater wins an image only if it beats **every** possible lie, which is the liar's best case.

# %%
def judge_probs(x, mask):
    return judge.predict_proba(masked_input(x[None].repeat(len(mask), 0), mask))

def best_pixel(x, mask, mine, theirs):
    candidates = np.flatnonzero(~mask)
    trial = np.repeat(mask[None], len(candidates), 0)
    trial[np.arange(len(candidates)), candidates] = True
    p = judge_probs(x, trial)
    score = np.log(p[:, mine] + 1e-9) - np.log(p[:, theirs] + 1e-9)
    return candidates[score.argmax()]

def debate(x, honest, lie, k=K, liar_strategy="greedy"):
    mask = np.zeros(64, bool); order = []
    for turn in range(k):
        if turn % 2 == 0:
            px = best_pixel(x, mask, honest, lie)
        elif liar_strategy == "greedy":
            px = best_pixel(x, mask, lie, honest)
        else:                                          # a liar who reveals pixels at random
            px = rng.choice(np.flatnonzero(~mask))
        mask[px] = True; order.append(px)
    p = judge_probs(x, mask[None])[0]
    return p[honest] > p[lie], mask, order

def honest_wins_all(x, y, k=K):
    return all(debate(x, y, lie, k)[0] for lie in range(10) if lie != y)

N_TEST = 200
wins = [honest_wins_all(X_te[i], y_te[i]) for i in range(N_TEST)]
print(f"Judge alone ({K} random pixels):          {baseline:.1%}")
print(f"Debate ({K} pixels), honest beats every lie: {np.mean(wins):.1%}")

# %% [markdown]
# ## 3. Look at a debate

# %%
def show_debate(i, lie):
    x, y = X_te[i], y_te[i]
    won, mask, order = debate(x, y, lie)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(x.reshape(8, 8), cmap="gray_r"); axes[0].set_title(f"what debaters see (true: {y})")
    shown = np.full(64, np.nan); shown[mask] = x[mask]
    cmap = plt.cm.gray_r.copy(); cmap.set_bad("#cfe3f5")      # light blue = hidden from the judge
    axes[1].imshow(np.ma.masked_invalid(shown.reshape(8, 8)), cmap=cmap, vmin=0, vmax=1)
    for t, px in enumerate(order):
        r, c = divmod(px, 8)
        axes[1].add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, lw=3,
                                        color="tab:green" if t % 2 == 0 else "tab:red"))
        axes[1].text(c, r, str(t + 1), ha="center", va="center", color="tab:orange", fontsize=12, weight="bold")
    axes[1].set_title(f"judge sees: honest '{y}' vs liar '{lie}'\n{'honest wins' if won else 'LIAR WINS'}")
    for a in axes: a.set_xticks([]); a.set_yticks([])
    plt.show()

show_debate(0, lie=(y_te[0] + 1) % 10)
show_debate(1, lie=(y_te[1] + 3) % 10)

# %% [markdown]
# Light blue is hidden from the judge. Green boxes are pixels the honest debater chose, red boxes the liar's; numbers give the order. Notice how often the honest player reveals a *blank* pixel: "there is no ink here" can be the most damning evidence against a lie.
#
# ## 4. More pixels, and a lazy liar
#
# Two quick comparisons:
# - Does a longer debate help? (We reuse the same judge, so more pixels also means more information.)
# - What if the liar does not try, and reveals random pixels? Debate is only as informative as the opposition is strong: a weak liar makes the honest debater look good without testing anything.

# %%
for k in (2, 4, 6):
    judge_alone = judge.score(masked_input(X_te[:N_TEST], random_masks(N_TEST, k)), y_te[:N_TEST])
    deb = np.mean([honest_wins_all(X_te[i], y_te[i], k) for i in range(N_TEST)])
    print(f"{k} pixels: judge alone {judge_alone:5.1%}   debate {deb:5.1%}")

lazy = np.mean([all(debate(X_te[i], y_te[i], l, K, "random")[0] for l in range(10) if l != y_te[i])
                for i in range(N_TEST)])
print(f"\n{K} pixels, liar reveals at random: honest beats every lie {lazy:.1%}")

# %% [markdown]
# ## 5. What this does and does not show
#
# - The judge was trained on random pixels, not on debates, yet a debate between strong players makes it far more accurate. The debaters, not the judge, did the work of finding the informative evidence.
# - It is a toy: our debaters can only reveal true pixels, so they **cannot lie about the evidence**, only choose it. The paper flags this limit itself, and asks whether in richer settings an agent can "sneak in a statement that appears logical but contains a hidden flaw without the other agent pointing out the flaw?" (§5.2).
# - Later work moved this to language models judging debates about reading-comprehension passages the judge cannot see (see the reading list in the README).
#
# ## 6. Exercises
#
# 1. **Who moves first?** Let the liar reveal first. Does accuracy change? Why might the last word matter?
# 2. **Without precommit.** Let the liar pick its lie *after* seeing the honest player's first pixel. How much does that help the liar? (The paper reports both settings.)
# 3. **Lying about evidence.** Allow the liar, once per game, to report a pixel value that is wrong by 0.5, and let the honest player spend a turn to "challenge" it (the judge then sees the true value). Is honesty still the winning strategy?
# 4. **A better searcher.** Replace the greedy `best_pixel` with a two-step lookahead for the honest player only. Does a smarter honest debater close the gap?
# 5. **Discussion.** The judge here is a fixed classifier. Human judges get tired, have biases, and can be persuaded by style. Which of those would break this game first?
