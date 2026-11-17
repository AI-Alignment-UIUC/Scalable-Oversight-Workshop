# %% [markdown]
# # Notebook 3: Recursive reward modeling
#
# **Scalable Oversight Workshop · AI Alignment @ Illinois**
#
# Paper: Leike, Krueger, Everitt, Martic, Maini & Legg (2018), [*Scalable agent alignment via reward modeling: a research direction*](https://arxiv.org/abs/1811.07871), §§1–5.
#
# The question the paper starts from (§1): "How can we create agents that behave in accordance with the user's intentions?"
#
# Its answer has two parts:
#
# - **Reward modeling** (§3.1): "The user trains a *reward model* to learn their intentions by providing feedback." An agent is then trained with RL against that reward model.
# - **Recursion** (§3.2): when outcomes get too hard for the user to evaluate, earlier agents help. "In step k we use the agent A_{k-1} to assist the user in evaluating outcomes when training agent A_k." Their example is a novel, where "the user is assisted by an agent that provides auxiliary input: extracting a summary of the plotline, checking spelling and grammar, summarizing character development, assessing the flow of the prose, and so on."
#
# It all rests on an assumption (§1): "For many tasks we want to solve, evaluation of outcomes is easier than producing the correct behavior."
#
# This paper is a research agenda and runs no experiments of its own. Its one data figure, Fig. 4 (reproduced from Ibarz et al. 2018), shows the main way this goes wrong: "Over time the agent learns to exploit the reward model: the perceived performance (according to the reward model) increases, while the actual performance (according to the game score) plummets." This notebook builds a small toy of the whole pipeline and shows that same failure, and the fix the paper proposes.
#
# **Our toy.** An agent writes a "document": a sequence of 64 tokens, each a digit 0–9. The user's true preference is *flow*: a document is better the more adjacent pairs are in non-decreasing order. The catch is that the user can only directly compare **snippets of 4 tokens**. To evaluate a 64-token document, the user needs help.
#
# We will (1) learn a reward model from the user's snippet comparisons, (2) build evaluators for longer documents recursively, and (3) optimize documents against those evaluators and watch what the optimizer exploits.
#
# **In plain words.** Think of the agent as a writer, and of "flow" as a stand-in for "the essay reads well from sentence to sentence." You can only read four sentences at a time, so you rely on helpers who each read a chunk and report a score. What does a writer who is optimizing for your helpers' scores end up writing?
#
# > **What you'll see**
# > 1. A reward model learns your preference from 1,500 snippet comparisons (section 1).
# > 2. Assisted evaluators for long documents look fine on random documents (section 2).
# > 3. An agent optimized against the frozen evaluator raises its score while the real quality *falls*, the shape of the paper's Fig. 4 (section 3).
# > 4. The paper's fix, feedback from the user while the agent trains, closes the gap (section 4).
# > 5. **Write a document yourself** to fool the evaluator, and run your own experiments (section 5).
#
# Runs in under a minute on CPU. Use **Runtime → Run all**. Needs numpy, scikit-learn, matplotlib and ipywidgets (all in Colab).

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
V, SNIP, DOC = 10, 4, 64

def true_quality(seq):
    """What the user actually wants: the number of adjacent pairs in order. Hidden from the agent."""
    seq = np.asarray(seq)
    return int(np.sum(seq[..., 1:] >= seq[..., :-1]))

def user_prefers(a, b, temperature=1.0):
    """The user compares two SHORT snippets, with Bradley-Terry noise (they sometimes misjudge)."""
    assert len(a) <= SNIP and len(b) <= SNIP, "the user can only read short snippets"
    d = (true_quality(a) - true_quality(b)) / temperature
    return rng.random() < 1 / (1 + np.exp(-d))

print(true_quality([1, 2, 2, 7]), true_quality([7, 2, 2, 1]))

# %% [markdown]
# ## 1. Level 0: a reward model from snippet comparisons
#
# Following the reward-modeling recipe (§3.1), we show the user pairs of random snippets and record which one they prefer, the kind of feedback the paper calls trajectory preferences. Then we fit a reward model r(snippet) to those choices with logistic regression (a Bradley-Terry model).
#
# Our reward model uses one feature per possible *adjacent pair of tokens* (100 features), summed over the snippet. It does not know which pairs are "in order"; it has to learn that from the user's choices.

# %%
def pair_features(seq):
    seq = np.asarray(seq)
    f = np.zeros(V * V)
    np.add.at(f, seq[:-1] * V + seq[1:], 1)
    return f

N_COMPARISONS = 1500
diffs, ys = [], []
for _ in range(N_COMPARISONS):
    a, b = rng.integers(0, V, SNIP), rng.integers(0, V, SNIP)
    diffs.append(pair_features(a) - pair_features(b)); ys.append(user_prefers(a, b))
rm0 = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(np.array(diffs), ys)
pair_reward = rm0.coef_[0].reshape(V, V)

def RM0(snippet):
    """Learned reward for a snippet: sum of learned pair rewards."""
    snippet = np.asarray(snippet)
    return float(pair_reward[snippet[:-1], snippet[1:]].sum())

plt.figure(figsize=(4, 3.5))
plt.imshow(pair_reward, cmap="RdBu", vmin=-np.abs(pair_reward).max(), vmax=np.abs(pair_reward).max())
plt.xlabel("next token"); plt.ylabel("token"); plt.colorbar(label="learned reward")
plt.title("RM0: learned pair rewards")
plt.show()

# %% [markdown]
# The learned reward is blue (good) above the diagonal and red below: the model recovered the user's preference from 1,500 noisy comparisons. Notice it is not perfect; some cells are noisy, and pairs the user rarely saw are poorly estimated.
#
# ## 2. Levels 1 and 2: assisted evaluation
#
# The user still cannot read a 16-token document. But they can ask the level-0 assistant to score each 4-token chunk, and add the scores up. That gives an evaluator for 16 tokens. A 16-token evaluator, in turn, lets the user evaluate 64 tokens in four chunks.
#
# There are two ways the user might combine the chunk scores:
#
# - **Naive:** add up the assistant's chunk scores and stop.
# - **Careful:** also look *personally* at the one pair of tokens at each seam between chunks. That is one tiny comparison per seam, cheap enough for the user.
#
# This is the RRM structure: each level's evaluator is the user plus the previous level's assistant, the toy version of the plot summarizer and the spell checker.

# %%
def make_evaluator(level, careful):
    """Evaluator for documents of length SNIP * 4**level."""
    if level == 0:
        return RM0
    sub = make_evaluator(level - 1, careful)
    size = SNIP * 4 ** (level - 1)
    seam_value = pair_reward.max()            # what the user gives a seam pair they judge to be in order
    def evaluate(doc):
        doc = np.asarray(doc)
        chunks = [doc[i:i + size] for i in range(0, len(doc), size)]
        score = sum(sub(c) for c in chunks)                       # the assistant's reports
        if careful:                                               # the user checks each seam directly
            for i in range(size, len(doc), size):
                score += seam_value if doc[i] >= doc[i - 1] else -seam_value
        return score
    return evaluate

evaluators = {f"level {l}, {'careful' if c else 'naive'}": (l, make_evaluator(l, c))
              for l in (1, 2) for c in (False, True)}

# How well does each evaluator track the truth on random documents of its size?
print(f"{'evaluator':22s} {'doc length':>10s} {'correlation with true quality':>32s}")
for name, (l, ev) in evaluators.items():
    docs = rng.integers(0, V, (500, SNIP * 4 ** l))
    r = np.corrcoef([ev(d) for d in docs], [true_quality(d) for d in docs])[0, 1]
    print(f"{name:22s} {SNIP * 4 ** l:10d} {r:32.2f}")

# %% [markdown]
# On random documents both evaluators look decent, the naive one a bit worse because it ignores the seams. If we stopped here we might conclude the naive evaluator is good enough.
#
# ## 3. Train an agent against the frozen evaluator (the paper's Fig. 4)
#
# Fig. 4's setup: "The fully trained reward model from the best seed is frozen and used to train an new agent from scratch." We do the same with our level-2 evaluators. The agent's job is to produce a 64-token document that the evaluator rates highly. Our "agent" is a simple optimizer (hill climbing: rewrite a short run of tokens, keep the change if the evaluator's score does not drop). Like Fig. 4, we plot the evaluator's score in green and the **true** quality in black.

# %%
def optimize(evaluate, steps=15000, seed=0):
    r = np.random.default_rng(seed)
    doc = r.integers(0, V, DOC)
    score = evaluate(doc)
    trace = []
    for t in range(steps):
        new = doc.copy()
        i = r.integers(DOC); j = min(DOC, i + r.integers(1, 9))
        new[i:j] = r.integers(V)                  # rewrite a short run of tokens
        s = evaluate(new)
        if s >= score:
            doc, score = new, s
        if t % 250 == 0:
            trace.append((score, true_quality(doc)))
    return doc, np.array(trace)

docs, traces = {}, {}
for name in ("level 2, naive", "level 2, careful"):
    docs[name], traces[name] = optimize(evaluators[name][1])

def fig4(ax, trace, title):
    """Plot like the paper's Fig. 4: evaluator's score (green) and true quality (black)."""
    steps = np.arange(len(trace)) * 250
    ax.plot(steps, trace[:, 1], color="black", label="true quality")
    ax.set_ylabel("true quality (max 63)"); ax.set_ylim(25, 64); ax.set_xlabel("optimization step")
    ax2 = ax.twinx()
    ax2.plot(steps, trace[:, 0], color="tab:green", label="evaluator's score")
    ax2.set_ylabel("evaluator's score", color="tab:green")
    ax.set_title(title)
    ax.legend(handles=ax.get_lines() + ax2.get_lines(), loc="lower right", fontsize=8)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
fig4(axes[0], traces["level 2, naive"], "frozen naive evaluator")
fig4(axes[1], traces["level 2, careful"], "frozen careful evaluator")
plt.tight_layout(); plt.show()

for name, d in docs.items():
    print(f"{name:18s} true quality {true_quality(d)}/63")
    print("   ", " ".join("".join(map(str, d[i:i + SNIP])) for i in range(0, DOC, SNIP)))

# %% [markdown]
# **What happened.** On the left is the paper's Fig. 4 in miniature. The evaluator's score keeps climbing, while the true quality rises at first, peaks, and then falls. The optimizer found the 4-token snippet RM0 likes best and repeated it, so every chunk is perfect and **nearly every one of the 15 seams is broken** (the printout is split into 4-token chunks so you can see it). Nothing ever looked at the seams.
#
# This is what the paper calls reward hacking (§4.3), "an effect that lets the agent get more reward than intended by exploiting loopholes in the process determining the reward". Here the loophole is in the assistance: the decomposition of the evaluation left gaps, and the optimizer found them. Each level of recursion adds new seams, which is why the paper leaves this open (§3.2): "An important open question is whether errors accumulate: do the mistakes of the more narrow agent A_{k-1} lead to larger mistakes in the training of agent A_k?"
#
# On the right, a user who knew where the seams were checked each one directly, and the gap closed. But that needed the user to know in advance where the loophole would be.

# %% [markdown]
# ## 4. The paper's fix: feedback while the agent trains
#
# The paper's diagnosis of Fig. 4 (§5.1): "Preliminary experiments show failure modes when the reward model is not trained online, i.e. in parallel with the agent". With a frozen model, "loopholes in the reward model that the agent discovers cannot be corrected." Its proposal: "If we provide the agent with reward feedback online, we get a tighter feedback loop between the user's feedback and the agent's behavior."
#
# In our toy, online feedback means the user keeps looking at what the agent is actually producing. Every 250 steps the user reads a few 4-token windows of the agent's **current** document, at random places (with the level-0 assistant), and those scores count toward the reward. The user doesn't need to know where the loophole is. They just look at the agent's real output.

# %%
def optimize_online(steps=15000, seed=0, every=250, n_windows=8):
    r = np.random.default_rng(seed)
    doc = r.integers(0, V, DOC)
    windows = r.integers(0, DOC - SNIP + 1, n_windows)
    naive = evaluators["level 2, naive"][1]
    evaluate = lambda d: naive(d) + sum(RM0(d[o:o + SNIP]) for o in windows)
    trace = []
    for t in range(steps):
        if t % every == 0:                        # new feedback on the agent's current output
            windows = r.integers(0, DOC - SNIP + 1, n_windows)
            score = evaluate(doc)
        new = doc.copy()
        i = r.integers(DOC); j = min(DOC, i + r.integers(1, 9))
        new[i:j] = r.integers(V)
        s = evaluate(new)
        if s >= score:
            doc, score = new, s
        if t % 250 == 0:
            trace.append((naive(doc), true_quality(doc)))
    return doc, np.array(trace)

docs["online feedback"], traces["online feedback"] = optimize_online()
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
fig4(axes[0], traces["level 2, naive"], "frozen naive evaluator")
fig4(axes[1], traces["online feedback"], "naive evaluator + online spot checks")
plt.tight_layout(); plt.show()
d = docs["online feedback"]
print(f"online feedback   true quality {true_quality(d)}/63")
print("   ", " ".join("".join(map(str, d[i:i + SNIP])) for i in range(0, DOC, SNIP)))

# %% [markdown]
# With online spot checks the true quality climbs to near the maximum. (The green curve on the right is still the naive evaluator's score, for comparison.) The broken seams are gone, because any seam might be the one the user reads next.
#
# ## 5. 🎛 Explore
#
# ### 5a. Write a document that fools the evaluator
#
# Type a document of exactly **64 digits** (spaces are ignored). You'll see its true quality and what the naive and careful evaluators think of it. Can you write something the naive evaluator rates near the top while its true quality is low? The optimizer's answer from section 3 is pre-filled.

# %%
import ipywidgets as widgets

naive2, careful2 = evaluators["level 2, naive"][1], evaluators["level 2, careful"][1]
best_naive = max(naive2(d) for d in docs.values())
best_careful = max(careful2(d) for d in docs.values())

def score_doc(text=" ".join("".join(map(str, docs["level 2, naive"][i:i + SNIP])) for i in range(0, DOC, SNIP))):
    digits = [int(c) for c in text if c.isdigit()]
    if len(digits) != DOC:
        print(f"Need exactly {DOC} digits; you have {len(digits)}."); return
    d = np.array(digits)
    bad = [i for i in range(1, DOC) if d[i] < d[i - 1]]
    seams = [i for i in bad if i % SNIP == 0]
    print(f"True quality:       {true_quality(d)}/63   ({len(bad)} pairs out of order, {len(seams)} of them at chunk seams)")
    print(f"Naive evaluator:    {naive2(d):6.1f}   (best the optimizer found: {best_naive:.1f})")
    print(f"Careful evaluator:  {careful2(d):6.1f}   (best the optimizer found: {best_careful:.1f})")

widgets.interact(score_doc, text=widgets.Textarea(
    value=" ".join("".join(map(str, docs["level 2, naive"][i:i + SNIP])) for i in range(0, DOC, SNIP)),
    description="document", layout=widgets.Layout(width="95%", height="60px"), continuous_update=False));

# %% [markdown]
# ### 5b. Run your own experiment
#
# Change how much feedback the user gives, how noisy it is, and how carefully the seams get checked. Then click **Run Interact** (takes a few seconds). Things to try:
#
# - Cut the comparisons to 100. Does the optimizer now exploit mistakes in RM0 itself, even against the careful evaluator?
# - Make the seam checks wrong 20% of the time. Is a noisy check still worth doing?
# - Make the user very noisy (temperature 4). How much worse does RM0 get?

# %%
def rrm_experiment(comparisons=1500, temperature=1.0, seam_check_error=0.0, steps=15000, seed=0):
    global pair_reward
    r = np.random.default_rng(seed)
    diffs, ys = [], []
    for _ in range(comparisons):
        a, b = r.integers(0, V, SNIP), r.integers(0, V, SNIP)
        diffs.append(pair_features(a) - pair_features(b)); ys.append(user_prefers(a, b, temperature))
    saved = pair_reward
    pair_reward = LogisticRegression(fit_intercept=False, max_iter=2000).fit(np.array(diffs), ys).coef_[0].reshape(V, V)

    def careful_noisy(doc):                       # level-2 careful evaluator whose seam checks can be wrong
        sub = make_evaluator(1, False)
        score = sum(sub(doc[i:i + 16]) for i in range(0, DOC, 16))
        for i in range(SNIP, DOC, SNIP):
            ok = doc[i] >= doc[i - 1]
            if r.random() < seam_check_error:
                ok = not ok
            score += pair_reward.max() if ok else -pair_reward.max()
        return score

    runs = {"naive": make_evaluator(2, False), "careful": careful_noisy}
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    axes[0].imshow(pair_reward, cmap="RdBu", vmin=-np.abs(pair_reward).max(), vmax=np.abs(pair_reward).max())
    axes[0].set_title(f"RM0 from {comparisons} comparisons"); axes[0].set_xlabel("next token"); axes[0].set_ylabel("token")
    for name, ev in runs.items():
        d, tr = optimize(ev, steps=steps, seed=seed)
        axes[1].plot(tr[:, 0] / np.abs(tr[:, 0]).max(), label=name)
        axes[2].plot(tr[:, 1], label=name)
        print(f"{name:8s} true quality {true_quality(d)}/63:  " + " ".join("".join(map(str, d[i:i + SNIP])) for i in range(0, DOC, SNIP)))
    axes[1].set_title("evaluator score (normalized)"); axes[2].set_title("TRUE quality (max 63)")
    for a in axes[1:]: a.set_xlabel("step (x250)"); a.legend()
    plt.show()
    pair_reward = saved

widgets.interact_manual(rrm_experiment,
    comparisons=widgets.IntSlider(value=1500, min=50, max=5000, step=50, description="comparisons"),
    temperature=widgets.FloatSlider(value=1.0, min=0.25, max=4.0, step=0.25, description="user noise"),
    seam_check_error=widgets.FloatSlider(value=0.0, min=0, max=0.5, step=0.05, description="seam error"),
    steps=widgets.IntSlider(value=15000, min=2000, max=30000, step=1000, description="opt. steps"),
    seed=widgets.IntSlider(value=0, min=0, max=20, description="seed"));

# %% [markdown]
# ## 6. Exercises
#
# 1. **Count the seams.** At level L the naive evaluator misses how many pairs? Write the formula and check it against the documents above. What happens at level 3 (256 tokens)?
# 2. **Fewer comparisons.** Cut `N_COMPARISONS` to 100. Re-run the careful optimization. Does the optimizer now exploit errors in RM0 itself (look for pairs in the final document that are out of order but that RM0 rates highly)?
# 3. **A careless user.** Make the careful user's seam check wrong 20% of the time. Is checking a noisy seam still better than not checking?
# 4. **Distillation.** Real RRM trains a new reward model at each level instead of composing the old one. Fit a level-1 reward model on 16-token documents using comparisons from the *assisted* evaluator. Does it inherit the seam blind spot?
# 5. **Discussion.** In the novel example, what are the "seams"? Think of something a plot summary, a spell checker and a character summary would all miss.
