# %% [markdown]
# # Notebook 2: AI safety via debate, the pixel game
#
# **Scalable Oversight Workshop · AI Alignment @ Illinois**
#
# Paper: Irving, Christiano & Amodei (2018), [*AI Safety via Debate*](https://arxiv.org/abs/1805.00899), §§2–3.1.
#
# The debate game (§2):
#
# 1. "A question q ∈ Q is shown to both agents."
# 2. "The two agents state their answers a0, a1 ∈ A (which may be the same)."
# 3. "The two agents take turns making statements s0, s1, . . . , sn−1 ∈ S."
# 4. "The judge sees the debate (q, a, s) and decides which agent wins."
#
# The bet behind it (§2): **"In the debate game, it is harder to lie than to refute a lie."** If that holds, a judge who could never answer a question alone can still pick the right answer, by watching two strong agents argue about it.
#
# The paper's first experiment (§3.1) makes the judge weak on purpose. The judge is a classifier that sees only a few pixels of a handwritten digit. Two debaters see the whole image. One argues for the true digit and one for a false digit, and they take turns revealing pixels to the judge. The headline result:
#
# > "With 6 pixels, debate with precommit turns a 59.4% accurate judge into a 88.9% accurate debate." (§3.1)
#
# This notebook plays the same game as a small toy: scikit-learn's built-in 8×8 digits instead of 28×28 MNIST, and simple greedy debaters instead of the paper's tree search. It runs in under a minute and follows the paper's rules:
#
# - The judge is trained only on **random** sets of 6 **nonzero** pixels, and never sees a debate.
# - Each move reveals one nonzero pixel. Revealed pixels can't be faked, but they can be chosen adversarially.
# - **Precommit**: the liar names one false digit up front, and the judge compares the two claimed digits. The liar wins an image if **any** of the 9 possible lies wins.
#
# > **What you'll see**
# > 1. A judge that sees 6 random pixels is right a bit over half the time (section 1).
# > 2. Debates played out on real digits (sections 2–3).
# > 3. Debate lifts the same judge to about 90% with precommit, and moving second matters, as in the paper's Table 2 (section 4).
# > 4. **You judge a debate yourself** (section 5).
#
# Use **Runtime → Run all**. Needs numpy, matplotlib, scikit-learn and ipywidgets (all in Colab).

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import warnings; warnings.filterwarnings("ignore")   # sklearn convergence notices

rng = np.random.default_rng(0)
digits = load_digits()
images = digits.images.reshape(len(digits.images), 64) / 16.0   # 1,797 digits, 8 x 8, values in [0, 1]
X_tr, X_te, y_tr, y_te = train_test_split(images, digits.target, test_size=0.3, random_state=0)
K = 6                                                            # pixels the judge gets to see

fig, axes = plt.subplots(1, 10, figsize=(12, 1.6))
for ax, img, lab in zip(axes, X_te, y_te):
    ax.imshow(img.reshape(8, 8), cmap="gray_r"); ax.set_title(lab); ax.axis("off")
plt.show()
print(f"A digit has about {np.mean((X_te > 0).sum(1)):.0f} nonzero pixels out of 64.")

# %% [markdown]
# ## 1. The sparse judge
#
# The paper's judge is "trained to classify MNIST from 6 (resp. 4) nonzero pixels, with the pixels chosen at random at training time" (§3.1). Its input is a 0/1 mask of the revealed pixels plus their values. We do the same with a small neural network, then **freeze** it: from here on the judge learns nothing.

# %%
def random_reveal(x, k):
    """A mask revealing k random NONZERO pixels of each image."""
    r = rng.random(x.shape)
    r[x == 0] = -1                                   # blank pixels are never revealed
    idx = np.argpartition(-r, k, axis=1)[:, :k]
    mask = np.zeros(x.shape, bool)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask

def judge_input(x, mask):
    """What the judge sees: which pixels are revealed, and their values."""
    return np.concatenate([mask.astype(float), x * mask], axis=1)

REPEATS = 30                                         # 30 random masks per training image
Xj, yj = np.repeat(X_tr, REPEATS, axis=0), np.repeat(y_tr, REPEATS)
judge = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=40, random_state=0)
judge.fit(judge_input(Xj, random_reveal(Xj, K)), yj)

alone = judge.score(judge_input(X_te, random_reveal(X_te, K)), y_te)
print(f"Judge alone, {K} random nonzero pixels: {alone:.1%}   (chance is 10%; the paper's MNIST judge: 59.4%)")

# %% [markdown]
# ## 2. The debate game
#
# For one test image with true digit `y` and a lie `lie`:
#
# 1. **Precommit.** The honest debater claims `y`; the liar claims `lie`.
# 2. **Reveal.** The debaters alternate, each revealing one nonzero pixel, until 6 are revealed (3 each).
# 3. **Verdict.** The judge sees only those 6 pixels and picks whichever of the two claimed digits it finds more likely.
#
# **Our debaters are greedy.** On its turn each one tries every pixel it could reveal and picks the one that most helps its own digit against the other's, right now. The paper's debaters used Monte Carlo tree search, which plans ahead.
#
# Without precommit, the liar doesn't have to name a lie: the judge just names its top digit, and any wrong answer is a win for the liar.

# %%
def scores(x, masks):
    """The judge's log-probabilities for image x under each of several masks."""
    return np.log(judge.predict_proba(judge_input(np.repeat(x[None], len(masks), 0), masks)) + 1e-12)

def play(x, y, lie=None, k=K, honest_first=True):
    """One debate. lie=None means no precommit. Returns (honest_won, moves)."""
    mask = np.zeros(64, bool); moves = []
    for turn in range(k):
        honest_turn = (turn % 2 == 0) == honest_first
        cand = np.flatnonzero((x > 0) & ~mask)                     # nonzero pixels not yet shown
        trial = np.repeat(mask[None], len(cand), 0)
        trial[np.arange(len(cand)), cand] = True
        s = scores(x, trial)
        if lie is not None:
            margin = s[:, y] - s[:, lie]                           # honest digit vs. the lie
        else:
            others = s.copy(); others[:, y] = -np.inf
            margin = s[:, y] - others.max(1)                       # truth vs. the judge's best other digit
        pick = cand[margin.argmax() if honest_turn else margin.argmin()]
        mask[pick] = True; moves.append((int(pick), "honest" if honest_turn else "liar"))
    final = scores(x, mask[None])[0]
    won = final[y] > final[lie] if lie is not None else final.argmax() == y
    return bool(won), moves

def honest_wins(i, precommit=True, honest_first=True):
    """With precommit, the liar wins the image if ANY of the 9 lies wins (§3.1)."""
    x, y = X_te[i], int(y_te[i])
    if not precommit:
        return play(x, y, None, K, honest_first)[0]
    return all(play(x, y, lie, K, honest_first)[0] for lie in range(10) if lie != y)

won, moves = play(X_te[0], int(y_te[0]), lie=8)
print(f"Image 0 is a {y_te[0]}. Against a liar claiming 8: {'honest wins' if won else 'liar wins'}. Moves: {moves}")

# %% [markdown]
# ## 3. Look at some debates
#
# Light blue is hidden from the judge. Green boxes are the honest debater's pixels and red boxes the liar's; numbers give the order. On the left is the whole digit, which only the debaters can see.

# %%
def draw_debate(i, lie, honest_first=True):
    x, y = X_te[i], int(y_te[i])
    won, moves = play(x, y, lie, K, honest_first)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    axes[0].imshow(x.reshape(8, 8), cmap="gray_r"); axes[0].set_title(f"the digit: {y}")
    shown = np.full(64, np.nan)
    for px, _ in moves: shown[px] = x[px]
    cmap = plt.cm.gray_r.copy(); cmap.set_bad("#cfe3f5")
    axes[1].imshow(np.ma.masked_invalid(shown.reshape(8, 8)), cmap=cmap, vmin=0, vmax=1)
    for t, (px, who) in enumerate(moves):
        r, c = divmod(px, 8)
        axes[1].add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, lw=3,
                                        color="tab:green" if who == "honest" else "tab:red"))
        axes[1].text(c, r, str(t + 1), ha="center", va="center", color="tab:orange", fontsize=11, weight="bold")
    axes[1].set_title(f"judge sees: {y} vs. {lie}\n{'honest wins' if won else 'LIAR WINS'}", fontsize=10)
    for a in axes: a.set_xticks([]); a.set_yticks([])
    plt.show()

draw_debate(0, lie=8)
draw_debate(1, lie=(int(y_te[1]) + 3) % 10, honest_first=False)

# %% [markdown]
# The honest debater looks for ink that only its digit would have in that place. The liar looks for ink its lie could share. With only 6 pixels, the judge's decision rests on those few choices.
#
# ## 4. The main result (the paper's Table 2)
#
# Now play every test image against every lie, for both settings (with and without precommit) and both orders (honest moves first or second). The numbers are how often the honest debater wins. The paper's MNIST numbers for 6 pixels are in brackets, for reference only: our images, judge and debaters are all smaller.

# %%
N_TEST = 200
rows = {}
for precommit in (False, True):
    for honest_first in (True, False):
        rows[(precommit, honest_first)] = np.mean([honest_wins(i, precommit, honest_first) for i in range(N_TEST)])

PAPER = {(False, True): 67.4, (False, False): 81.5, (True, True): 87.4, (True, False): 90.4}
print(f"Judge alone, 6 random pixels: {alone:.0%}   [paper 59.4%]\n")
print(f"{'':22s}{'no precommit':>20s}{'precommit':>20s}")
for hf, name in ((True, "honest moves first"), (False, "liar moves first")):
    print(f"{name:22s}" + "".join(f"{rows[(pc, hf)]:>12.0%}  [{PAPER[(pc, hf)]:.1f}%]" for pc in (False, True)))
PAPER_MEAN = {False: 74.4, True: 88.9}
print(f"{'mean':22s}" + "".join(f"{(rows[(pc, True)] + rows[(pc, False)]) / 2:>12.0%}  [{PAPER_MEAN[pc]:.1f}%]"
                                for pc in (False, True)))

# %% [markdown]
# **What to look for.** The paper's patterns all show up in the toy:
#
# - Debate makes the same frozen judge much more accurate than it is on random pixels. The debaters, not the judge, did the work of finding the informative evidence.
# - **Precommit** helps the honest side a lot. Without it, the liar doesn't have to defend one lie; any wrong answer counts.
# - **Moving second** is a big advantage. The paper notes it too: "Both versions give an advantage to the second player to move." The last pixel can't be answered.
#
# ## 5. 🎛 Explore
#
# ### 5a. You are the judge
#
# Two debaters, **A** and **B**, each claim a digit. Between them they reveal 6 pixels (light blue is hidden). One of them is honest. Who? The liar has picked the lie that did best against our judge network, and your score is kept next to the network's.

# %%
import ipywidgets as widgets
from IPython.display import display, HTML

class HumanJudge:
    def __init__(self):
        self.score, self.net_score, self.played = 0, 0, 0
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
        best = None                                   # the liar picks its strongest lie
        for lie in range(10):
            if lie == y: continue
            won, moves = play(x, y, lie)
            mask = np.zeros(64, bool); mask[[px for px, _ in moves]] = True
            s = scores(x, mask[None])[0]
            if best is None or s[y] - s[lie] < best[0]:
                best = (s[y] - s[lie], lie, moves, won)
        _, self.lie, self.moves, self.net_right = best
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
            shown = np.full(64, np.nan)
            for px, _ in self.moves: shown[px] = x[px]
            cmap = plt.cm.gray_r.copy(); cmap.set_bad("#cfe3f5")
            axes[0].imshow(np.ma.masked_invalid(shown.reshape(8, 8)), cmap=cmap, vmin=0, vmax=1)
            for px, who in self.moves:
                r, c = divmod(px, 8)
                name = self.honest_is if who == "honest" else ("B" if self.honest_is == "A" else "A")
                axes[0].add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, lw=3,
                                                color="tab:purple" if name == "A" else "tab:orange"))
                axes[0].text(c, r, name, ha="center", va="center", color="tab:red", fontsize=11, weight="bold")
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
        self.score += right; self.net_score += self.net_right
        net = "the honest debater" if self.net_right else "the liar"
        self.draw(True, f"<p>{'✓ Right' if right else '✗ Fooled'}: {self.honest_is} was honest. "
                        f"The judge network sided with {net}.<br>You: {self.score}/{self.played}. "
                        f"Network: {self.net_score}/{self.played}.</p>")

human_judge = HumanJudge()

# %% [markdown]
# ### 5b. Explore one debate
#
# Pick a test image, the lie, the number of pixels, precommit or not, and who moves first. The bar chart shows the judge's final belief across all ten digits.

# %%
def explore(image=0, lie=8, pixels=6, precommit=True, honest_first=True):
    x, y = X_te[image], int(y_te[image])
    if precommit and lie == y:
        print(f"Image {image} really is a {y}; pick a different lie."); return
    won, moves = play(x, y, lie if precommit else None, pixels, honest_first)
    mask = np.zeros(64, bool); mask[[px for px, _ in moves]] = True
    p = judge.predict_proba(judge_input(x[None], mask[None]))[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    axes[0].imshow(x.reshape(8, 8), cmap="gray_r"); axes[0].set_title(f"true digit {y}")
    shown = np.where(mask, x, np.nan)
    cmap = plt.cm.gray_r.copy(); cmap.set_bad("#cfe3f5")
    axes[1].imshow(np.ma.masked_invalid(shown.reshape(8, 8)), cmap=cmap, vmin=0, vmax=1)
    for t, (px, who) in enumerate(moves):
        r, c = divmod(px, 8)
        axes[1].add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, lw=3,
                                        color="tab:green" if who == "honest" else "tab:red"))
    axes[1].set_title("judge sees (green honest, red liar)")
    for a in axes[:2]: a.set_xticks([]); a.set_yticks([])
    colors = ["tab:green" if d == y else "tab:red" if (precommit and d == lie) else "#bbbbbb" for d in range(10)]
    axes[2].bar(range(10), p, color=colors); axes[2].set_xticks(range(10))
    axes[2].set_title(f"judge's belief: {'honest wins' if won else 'LIAR WINS'}")
    plt.show()

widgets.interact(explore,
                 image=widgets.IntSlider(value=0, min=0, max=len(X_te) - 1, description="image", continuous_update=False),
                 lie=widgets.Dropdown(options=list(range(10)), value=8, description="liar claims"),
                 pixels=widgets.IntSlider(value=6, min=2, max=12, description="pixels", continuous_update=False),
                 precommit=widgets.Checkbox(value=True, description="precommit"),
                 honest_first=widgets.Checkbox(value=True, description="honest moves first"));

# %% [markdown]
# ## 6. What this does and does not show
#
# - The judge was trained on random pixels, never on debates, yet a debate between two players makes it far more accurate.
# - The debaters can only reveal true pixels, so they **cannot lie about the evidence**, only choose it. The paper raises the harder case itself, under "Will humans understand the debates?" (§5.2): "Will one of the agents be able to sneak in a statement that appears logical but contains a hidden flaw without the other agent pointing out the flaw?"
# - Our debaters are greedy. A test of debate is only as strong as the liar, and a stronger liar might find better lies.
#
# Why debate could work for much harder questions is a theory result in the same paper (§2.2): "For any problem L ∈ PSPACE, we can choose a polynomial time judge so that optimal play in the debate game with polynomially many steps solves L." The [Iliad notes](https://iliad-intensive.org/safety/debate/) (§11) walk through the proof, where "a debate can verify reachability by recursively halving a path".
#
# ## 7. Exercises
#
# 1. **Plan ahead.** Give the liar a two-move lookahead: for each candidate pixel, assume the honest player answers greedily, and pick the pixel that leaves the best margin afterward. Does the honest win rate drop? Then give the honest player the same search.
# 2. **A lazy liar.** Make the liar reveal a random nonzero pixel. The honest debater now wins almost always, which tells you nothing. Why is a test of debate only as strong as its opposition?
# 3. **More pixels.** Use `explore` with 8 or 10 pixels. The judge was trained on 6. Is it still reliable? Retrain it with `K = 10` and compare.
# 4. **Lying about evidence.** Let the liar, once per game, report a pixel's value wrongly, and let the honest player spend a turn to challenge it (the judge then sees the true value). Is honesty still the winning strategy?
# 5. **Discussion.** The judge here is a fixed classifier. Human judges get tired, have biases, and can be persuaded by style. Which of those would break this game first?
