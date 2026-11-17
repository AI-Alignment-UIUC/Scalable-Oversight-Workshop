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
# > **What you'll see**
# > 1. A "judge" network that sees only 4 pixels of a handwritten digit guesses right about 30% of the time (section 1).
# > 2. When an honest and a lying debater choose which pixels to show it, it picks the true digit about 70% of the time (section 2).
# > 3. Pictures of debates, and what happens with more pixels or a lazy liar (sections 3–4).
# > 4. **You judge a debate yourself**, and controls to explore (section 5).
#
# Use **Runtime → Run all**. The longest cell takes about 30 seconds. Needs numpy, matplotlib, scikit-learn and ipywidgets (all in Colab).
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
# ## 5. 🎛 Explore
#
# ### 5a. You are the judge
#
# Two debaters, **A** and **B**, each claim a different digit. Between them they reveal 4 pixels (light blue is hidden). One of them is honest. Which digit is it? Click your answer, then see the full image and what our judge network decided.

# %%
import ipywidgets as widgets
from IPython.display import display, HTML

class HumanJudge:
    def __init__(self, k=K):
        self.k, self.score, self.played = k, 0, 0
        self.out = widgets.Output()
        self.ba = widgets.Button(description="A is honest", button_style="info")
        self.bb = widgets.Button(description="B is honest", button_style="info")
        self.bn = widgets.Button(description="Next debate", button_style="warning")
        self.ba.on_click(lambda _: self.answer("A")); self.bb.on_click(lambda _: self.answer("B"))
        self.bn.on_click(lambda _: self.new())
        display(widgets.VBox([widgets.HBox([self.ba, self.bb, self.bn]), self.out]))
        self.new()

    def new(self):
        self.i = int(rng.integers(len(X_te)))
        x, y = X_te[self.i], int(y_te[self.i])
        # the liar picks the lie that does best against the honest debater
        best = None
        for lie in range(10):
            if lie == y: continue
            won, mask, order = debate(x, y, lie, self.k)
            margin = judge_probs(x, mask[None])[0]
            gap = margin[y] - margin[lie]
            if best is None or gap < best[0]:
                best = (gap, lie, mask, order)
        _, self.lie, self.mask, self.order = best
        self.honest_is = "A" if rng.random() < 0.5 else "B"
        self.claims = {self.honest_is: y, ("B" if self.honest_is == "A" else "A"): self.lie}
        self.answered = False
        self.draw(reveal=False)

    def draw(self, reveal, note=""):
        x = X_te[self.i]
        self.out.clear_output(wait=True)
        with self.out:
            fig, axes = plt.subplots(1, 2 if reveal else 1, figsize=(7 if reveal else 3.5, 3.5))
            axes = np.atleast_1d(axes)
            shown = np.full(64, np.nan); shown[self.mask] = x[self.mask]
            cmap = plt.cm.gray_r.copy(); cmap.set_bad("#cfe3f5")
            axes[0].imshow(np.ma.masked_invalid(shown.reshape(8, 8)), cmap=cmap, vmin=0, vmax=1)
            for t, px in enumerate(self.order):
                r, c = divmod(px, 8)
                who = "A" if (t % 2 == 0) == (self.honest_is == "A") else "B"
                axes[0].add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, lw=3,
                                                color="tab:purple" if who == "A" else "tab:orange"))
                axes[0].text(c, r, who, ha="center", va="center", color="tab:red", fontsize=11, weight="bold")
            axes[0].set_title(f"A says {self.claims['A']},  B says {self.claims['B']}")
            if reveal:
                axes[1].imshow(x.reshape(8, 8), cmap="gray_r"); axes[1].set_title(f"the real digit: {y_te[self.i]}")
            for a in axes: a.set_xticks([]); a.set_yticks([])
            plt.show()
            display(HTML(note or "<p>Boxes show who revealed each pixel. Which debater is honest?</p>"))

    def answer(self, pick):
        if self.answered: return
        self.answered, self.played = True, self.played + 1
        right = pick == self.honest_is
        self.score += right
        p = judge_probs(X_te[self.i], self.mask[None])[0]
        y = int(y_te[self.i])
        net = "the honest debater" if p[y] > p[self.lie] else "the liar"
        self.draw(True, f"<p>{'✓ Right' if right else '✗ Fooled'}: {self.honest_is} was honest. "
                        f"Our judge network sided with {net}. Your score: {self.score}/{self.played}.</p>")

human_judge = HumanJudge()

# %% [markdown]
# ### 5b. Explore one debate
#
# Pick a test image, the lie the liar argues for, how many pixels get revealed, and how hard the liar tries. The bar chart shows how the judge's belief ends up split across all ten digits.

# %%
def explore(image=0, lie=1, pixels=4, liar="greedy"):
    x, y = X_te[image], int(y_te[image])
    if lie == y:
        print(f"Image {image} really is a {y}; pick a different lie."); return
    won, mask, order = debate(x, y, lie, pixels, liar)
    p = judge_probs(x, mask[None])[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    axes[0].imshow(x.reshape(8, 8), cmap="gray_r"); axes[0].set_title(f"true digit {y}")
    shown = np.full(64, np.nan); shown[mask] = x[mask]
    cmap = plt.cm.gray_r.copy(); cmap.set_bad("#cfe3f5")
    axes[1].imshow(np.ma.masked_invalid(shown.reshape(8, 8)), cmap=cmap, vmin=0, vmax=1)
    for t, px in enumerate(order):
        r, c = divmod(px, 8)
        axes[1].add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, lw=3,
                                        color="tab:green" if t % 2 == 0 else "tab:red"))
    axes[1].set_title("judge sees (green honest, red liar)")
    for a in axes[:2]: a.set_xticks([]); a.set_yticks([])
    colors = ["tab:green" if d == y else "tab:red" if d == lie else "#bbbbbb" for d in range(10)]
    axes[2].bar(range(10), p, color=colors); axes[2].set_xticks(range(10))
    axes[2].set_title(f"judge's belief: {'honest wins' if won else 'LIAR WINS'}")
    plt.show()

widgets.interact(explore,
                 image=widgets.IntSlider(value=0, min=0, max=len(X_te) - 1, description="image", continuous_update=False),
                 lie=widgets.Dropdown(options=list(range(10)), value=1, description="liar claims"),
                 pixels=widgets.IntSlider(value=4, min=2, max=10, description="pixels", continuous_update=False),
                 liar=widgets.ToggleButtons(options=["greedy", "random"], description="liar plays"));

# %% [markdown]
# ## 6. What this does and does not show
#
# - The judge was trained on random pixels, not on debates, yet a debate between strong players makes it far more accurate. The debaters, not the judge, did the work of finding the informative evidence.
# - It is a toy: our debaters can only reveal true pixels, so they **cannot lie about the evidence**, only choose it. The paper flags this limit itself, and asks whether in richer settings an agent can "sneak in a statement that appears logical but contains a hidden flaw without the other agent pointing out the flaw?" (§5.2).
# - Later work moved this to language models judging debates about reading-comprehension passages the judge cannot see (see the reading list in the README).
#
# ## 7. Exercises
#
# 1. **Who moves first?** Let the liar reveal first. Does accuracy change? Why might the last word matter?
# 2. **Without precommit.** Let the liar pick its lie *after* seeing the honest player's first pixel. How much does that help the liar? (The paper reports both settings.)
# 3. **Lying about evidence.** Allow the liar, once per game, to report a pixel value that is wrong by 0.5, and let the honest player spend a turn to "challenge" it (the judge then sees the true value). Is honesty still the winning strategy?
# 4. **A better searcher.** Replace the greedy `best_pixel` with a two-step lookahead for the honest player only. Does a smarter honest debater close the gap?
# 5. **Discussion.** The judge here is a fixed classifier. Human judges get tired, have biases, and can be persuaded by style. Which of those would break this game first?
