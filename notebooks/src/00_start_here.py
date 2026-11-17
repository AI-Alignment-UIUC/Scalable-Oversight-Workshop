# %% [markdown]
# # Start here: scalable oversight in one example
#
# **Scalable Oversight Workshop · AI Alignment @ Illinois**
#
# This notebook is a gentle walkthrough. It uses **one running example** to explain the three ideas of the workshop, iterated amplification, debate, and recursive reward modeling, with very little code and no machine learning. The four notebooks after it go deeper into each one.
#
# How to use it: read a section, **make the prediction it asks for** before you run the next cell, then check. Answers to the questions are hidden under ▶ arrows.
#
# Everything runs instantly. Run all cells top to bottom (Runtime → Run all) or one at a time with Shift+Enter.

# %% [markdown]
# ## 0. The problem
#
# We train AI systems with human feedback: people look at what the system did and say whether it was good. That works while people can check the work. It stops working when the work gets too big or too hard to check.
#
# Christiano, Shlegeris & Amodei put it this way: "there are harder tasks for which we can't compute demonstrations or rewards even with human assistance" ([2018](https://arxiv.org/abs/1810.08575), abstract).
#
# **Scalable oversight** asks: can a *weak* overseer reliably supervise a *strong* system, if the overseer gets help, often from AI systems themselves?
#
# ### The running example: a ledger
#
# A company's ledger has 1,024 transactions. Someone (perhaps an AI) reports the total. You want to know whether that total is right.
#
# You are the overseer. You are careful but slow: **in one step you can read one transaction, or add two numbers.** Checking the whole ledger yourself takes 1,023 additions. Pretend that's far too many.

# %%
import random
random.seed(0)

N = 1024
ledger = [random.randint(-500, 1000) for _ in range(N)]
true_total = sum(ledger)

print("First ten transactions:", ledger[:10])
print("Number of transactions:", N)
# We print the true total only so you can check the methods below. The overseer never sees it.
print("True total (hidden from the overseer):", true_total)

# %% [markdown]
# ## 1. Amplification: split the question, delegate the pieces
#
# The first idea: the overseer doesn't answer the hard question. They **break it into easier sub-questions** and hand each to an assistant, then combine the answers.
#
# > "Amplify(X) answers a question Q by having H identify a sequence of useful subquestions, using X to compute a subanswer to each subquestion" (Christiano et al., §1.1)
#
# For the ledger: "What is the total of transactions 0–1023?" becomes "total of 0–511?" plus "total of 512–1023?". Each of those splits again, down to single transactions. Every person in the tree does **one** addition. This idealized tree of overseers each consulting more overseers is called **HCH** ("Humans Consulting HCH").

# %%
def hch(lo, hi, depth=0, log=None):
    """Total of ledger[lo:hi], answered by a tree of overseers who each do one step."""
    if hi - lo == 1:
        return ledger[lo]                                  # base case: read one transaction
    mid = (lo + hi) // 2
    left = hch(lo, mid, depth + 1, log)                    # delegate sub-question 1
    right = hch(mid, hi, depth + 1, log)                   # delegate sub-question 2
    if log is not None and depth < 2:
        log.append(f"{'   ' * depth}total[{lo}:{hi}] = {left} + {right} = {left + right}")
    return left + right                                    # one addition

log = []
answer = hch(0, N, log=log)
print("\n".join(sorted(log, key=lambda s: (len(s) - len(s.lstrip()), s))))
print("\nHCH answer:", answer, "  correct?", answer == true_total)

# %% [markdown]
# **Where does learning come in?** A tree of 1,023 people is too slow to use. Iterated amplification trains a fast model X to *imitate* the tree (that's called **distillation**), then uses copies of the faster X as the assistants in a new, stronger tree, and repeats. Notebook 1 does exactly this with a neural network.
#
# ### Predict before you run
#
# Real overseers make mistakes. Suppose each overseer in the tree adds wrongly 1 time in 200. **What fraction of the time will the final total be right?** Write down a guess, then run the cell.

# %%
def hch_noisy(lo, hi, p_error):
    if hi - lo == 1:
        return ledger[lo]
    mid = (lo + hi) // 2
    s = hch_noisy(lo, mid, p_error) + hch_noisy(mid, hi, p_error)
    if random.random() < p_error:
        s += random.randint(1, 9)                          # a small slip (always upward, so slips never cancel)
    return s

trials = 300
right = sum(hch_noisy(0, N, 1 / 200) == true_total for _ in range(trials))
print(f"Final total correct in {right}/{trials} trials ({right / trials:.0%})")
print(f"Prediction if every one of the 1023 additions must be right: {(1 - 1/200) ** 1023:.1%}")

# %% [markdown]
# <details><summary>▶ What's going on?</summary>
#
# Every one of the 1,023 additions feeds into the final answer, so the chance that all are right is (1 − 1/200)¹⁰²³ ≈ 0.6%. Small per-step errors **compound** across a big tree. Notebook 1 shows a subtler version: when a model learns from many noisy examples, *random* slips average out, but a *systematic* blind spot gets copied into every answer that depends on it.
# </details>
#
# ### 🎛 Try it: how reliable must each overseer be?
#
# Move the slider. How small does the per-overseer error rate need to be before the whole tree is usually right? (The trials rerun when you let go of the slider.)

# %%
import ipywidgets as widgets
from IPython.display import display, HTML

def try_tree(errors_per_10000=50, ledger_size=1024):
    p = errors_per_10000 / 10_000
    global ledger, true_total
    full = ledger
    ledger = full[:ledger_size]; true_total = sum(ledger)
    ok = sum(hch_noisy(0, ledger_size, p) == true_total for _ in range(200))
    print(f"Each overseer slips {p:.2%} of the time, tree of {ledger_size - 1} additions:")
    print(f"  final total right in {ok}/200 trials ({ok / 200:.0%});  theory (1-p)^{ledger_size - 1} = {(1 - p) ** (ledger_size - 1):.0%}")
    ledger = full; true_total = sum(ledger)

widgets.interact(try_tree,
                 errors_per_10000=widgets.IntSlider(value=50, min=0, max=200, step=5, description="slips /10k", continuous_update=False),
                 ledger_size=widgets.SelectionSlider(options=[16, 64, 256, 1024], value=1024, description="ledger size", continuous_update=False));

# %% [markdown]
# This motivates the second idea: what if you didn't have to trust every node of the tree, only the one place where two experts disagree?

# %% [markdown]
# ## 2. Debate: let two experts argue, check one step
#
# Two strong debaters both know the whole ledger. **Alice** claims one total, **Bob** claims another. Exactly one of them is telling the truth (we don't know which). You, the judge, will do only a handful of steps.
#
# > "In the debate game, it is harder to lie than to refute a lie." (Irving, Christiano & Amodei, [2018](https://arxiv.org/abs/1805.00899), §2)
#
# **The protocol** (recursive halving, as in the Iliad notes on debate, §11):
#
# 1. Each debater splits their claimed total into two half-totals that must add up to it. *You check that addition: one step.*
# 2. If both debaters agree on a half, it's settled. They must disagree on at least one half (their totals differ). The debate continues on a half where they disagree.
# 3. Repeat until the dispute is about a **single transaction**. You read it: one step. Whoever got it right wins.
#
# The trick: a liar's half-totals must add up to the false total, so at least one half-total is also false. The lie can be pushed around but never gets smaller, and at the bottom it lands on a single transaction you can read.

# %%
def honest(lo, hi):
    return sum(ledger[lo:hi])

def make_liar(claimed_total):
    """A liar who claims `claimed_total` for the whole ledger, and keeps every sub-claim
    as close to the truth as possible while adding up correctly."""
    claims = {(0, N): claimed_total}
    def liar(lo, hi):
        if (lo, hi) in claims:
            return claims[(lo, hi)]
        return honest(lo, hi)
    def split(lo, hi):
        mid = (lo + hi) // 2
        total = liar(lo, hi)
        # Put the whole lie into one randomly chosen half; the other half is honest.
        if random.random() < 0.5:
            claims[(lo, mid)], claims[(mid, hi)] = total - honest(mid, hi), honest(mid, hi)
        else:
            claims[(lo, mid)], claims[(mid, hi)] = honest(lo, mid), total - honest(lo, mid)
        return claims[(lo, mid)], claims[(mid, hi)]
    return liar, split

def debate(alice_is_honest=True, lie_size=37, judge=None, verbose=True):
    """Returns True if the honest debater wins. `judge(…)` can be you (see below)."""
    lie = true_total + lie_size
    liar, liar_split = make_liar(lie)
    claim = {"Alice": true_total if alice_is_honest else lie,
             "Bob": lie if alice_is_honest else true_total}
    who_lies = "Bob" if alice_is_honest else "Alice"
    steps = 0
    lo, hi = 0, N
    if verbose:
        print(f"Alice claims the total is {claim['Alice']}; Bob claims {claim['Bob']}.")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        h = (honest(lo, mid), honest(mid, hi))
        l = liar_split(lo, hi)
        halves = {"Alice": h if who_lies == "Bob" else l, "Bob": l if who_lies == "Bob" else h}
        steps += 2                                         # judge checks both additions
        disputed = [i for i in (0, 1) if halves["Alice"][i] != halves["Bob"][i]]
        side = disputed[0] if judge is None else judge(lo, mid, hi, halves, disputed)
        if verbose:
            name = f"[{lo}:{mid}]" if side == 0 else f"[{mid}:{hi}]"
            print(f"  [{lo}:{hi}]  Alice: {halves['Alice'][0]} + {halves['Alice'][1]}   "
                  f"Bob: {halves['Bob'][0]} + {halves['Bob'][1]}   → dispute {name}")
        lo, hi = (lo, mid) if side == 0 else (mid, hi)
    steps += 1                                             # read one transaction
    alice_leaf = halves["Alice"][side]; bob_leaf = halves["Bob"][side]
    winner = "Alice" if alice_leaf == ledger[lo] else "Bob"
    if verbose:
        print(f"Judge reads transaction {lo}: {ledger[lo]}.  Alice said {alice_leaf}, Bob said {bob_leaf}.  "
              f"{winner} wins.  Judge work: {steps} steps (vs {N - 1} to add everything).")
    return winner != who_lies

debate(alice_is_honest=False)

# %% [markdown]
# Ten rounds, about twenty additions and one lookup, and the judge caught the lie. Try it the other way round, and with a huge lie:

# %%
debate(alice_is_honest=True, lie_size=-123_456)

# %% [markdown]
# ### 🎛 Be the judge yourself
#
# Run the cell below. You'll get a fresh debate: one debater is lying, and **you don't know which**. Each round, click the half you want to dispute. When the dispute is down to one transaction, you read it and the liar is revealed (or not).
#
# Question while you play: **does it matter which half you pick?** Try once picking where the debaters disagree, and once picking where they agree.

# %%
class JudgeGame:
    def __init__(self):
        self.out = widgets.Output()
        self.left = widgets.Button(description="Dispute LEFT half", button_style="info")
        self.right = widgets.Button(description="Dispute RIGHT half", button_style="info")
        self.new = widgets.Button(description="New debate", button_style="warning")
        self.left.on_click(lambda _: self.pick(0)); self.right.on_click(lambda _: self.pick(1))
        self.new.on_click(lambda _: self.reset())
        display(widgets.VBox([widgets.HBox([self.left, self.right, self.new]), self.out]))
        self.reset()

    def reset(self):
        self.alice_honest = random.random() < 0.5
        lie = true_total + random.choice([-1, 1]) * random.randint(5, 400)
        self.liar_claim, self.liar_split = make_liar(lie)
        self.claims = {"Alice": true_total if self.alice_honest else lie,
                       "Bob": lie if self.alice_honest else true_total}
        self.lo, self.hi, self.steps, self.history = 0, N, 0, []
        self.show()

    def show(self):
        self.out.clear_output()
        with self.out:
            if self.hi - self.lo == 1:
                return self.finish()
            mid = (self.lo + self.hi) // 2
            h = (honest(self.lo, mid), honest(mid, self.hi))
            l = self.liar_split(self.lo, self.hi)
            self.cur = {"Alice": h if self.alice_honest else l, "Bob": l if self.alice_honest else h}
            bar = "".join("█" if self.lo <= i * 16 < self.hi else "·" for i in range(N // 16))
            A, B = self.cur["Alice"], self.cur["Bob"]
            def cell(a, b):
                color = "#b5371a" if a != b else "#46536a"
                return f"<td style='padding:4px 14px;color:{color}'><b>{a}</b></td>"
            display(HTML(f"""
            <p><b>Round {len(self.history) + 1}.</b> Disputed range: transactions {self.lo}–{self.hi - 1}
            &nbsp; <code>{bar}</code></p>
            <table style='font-size:15px'>
              <tr><th></th><th style='padding:4px 14px'>left half [{self.lo}:{mid}]</th>
                  <th style='padding:4px 14px'>right half [{mid}:{self.hi}]</th><th style='padding:4px 14px'>adds up to</th></tr>
              <tr><td><b>Alice</b></td>{cell(A[0], B[0])}{cell(A[1], B[1])}<td style='padding:4px 14px'>{A[0] + A[1]}</td></tr>
              <tr><td><b>Bob</b></td>{cell(B[0], A[0])}{cell(B[1], A[1])}<td style='padding:4px 14px'>{B[0] + B[1]}</td></tr>
            </table>
            <p style='color:#46536a'>Numbers in orange are where they disagree. You've done {self.steps} steps of work so far.</p>"""))
            self.steps += 2

    def pick(self, side):
        if self.hi - self.lo == 1:
            return
        mid = (self.lo + self.hi) // 2
        self.history.append(side)
        self.picked = {k: v[side] for k, v in self.cur.items()}
        self.lo, self.hi = (self.lo, mid) if side == 0 else (mid, self.hi)
        self.show()

    def finish(self):
        self.steps += 1
        a, b = self.picked["Alice"], self.picked["Bob"]
        truth = ledger[self.lo]
        liar = "Bob" if self.alice_honest else "Alice"
        if a == b:
            verdict = (f"Both debaters say transaction {self.lo} is {a}, and it is {truth}. Nobody is caught: "
                       f"you followed a half they agreed on, and the lie went with the half you dropped. "
                       f"<b>{liar} was the liar and got away with it.</b>")
        else:
            caught = "Alice" if a != truth else "Bob"
            verdict = (f"You read transaction {self.lo}: it is <b>{truth}</b>. Alice said {a}, Bob said {b}. "
                       f"<b>{caught} was lying, and you caught them.</b>")
        display(HTML(f"<p>{verdict}</p><p style='color:#46536a'>Total work: {self.steps} steps "
                     f"(adding everything yourself: {N - 1}). Click <i>New debate</i> to play again.</p>"))

game = JudgeGame()

# %% [markdown]
# <details><summary>▶ Does it matter which half you pick?</summary>
#
# Yes. If you pick a half where the debaters **agree**, the dispute is over a number both accept, and the liar's lie is hidden in the half you threw away. In the real protocol the *debaters* point to the half they disagree on, and a strong honest debater always points at the liar's false half. That's the point of debate: the opponent, not the judge, does the work of finding the flaw. If the opponent is weak or lazy, the lie survives. Notebook 2 measures this on a maze.
# </details>
#
# **Why this matters.** The judge's work grew with the *number of rounds* (log₂ 1024 = 10), not with the size of the ledger. The paper's theory result says this generalizes far: "For any problem L ∈ PSPACE, we can choose a polynomial time judge so that optimal play in the debate game with polynomially many steps solves L" (Irving et al., §2.2).
#
# **What could go wrong.** Here every claim eventually bottoms out in a transaction you can read. In real arguments, a liar might make a claim that is wrong but *very hard to locate* the error in. Beth Barnes calls this the obfuscated arguments problem: "a dishonest debater can often create arguments that have a fatal error, but where it is very hard to locate the error" ([2020](https://www.alignmentforum.org/posts/PJLABqQ962hZEqhdB/debate-update-obfuscated-arguments-problem)).

# %% [markdown]
# ## 3. Recursive reward modeling: evaluate with assistants
#
# The third idea flips the question. Instead of asking for an answer, you **evaluate** work an AI did, with help from assistants trained in earlier rounds.
#
# > "In step k we use the agent A_{k-1} to assist the user in evaluating outcomes when training agent A_k." (Leike et al., [2018](https://arxiv.org/abs/1811.07871), §3.2)
#
# For the ledger: an AI bookkeeper writes the books as **32 pages of 32 transactions**. Each page ends with a page total and a running *carried-forward* balance. You audit it with help: an assistant checks each page (does the page total match its transactions?) and you only look at the assistants' reports.

# %%
PAGE = 32

def write_books(cook=False):
    """The bookkeeper's report: per page, (transactions, page_total, carried_forward)."""
    books, running = [], 0
    for p in range(N // PAGE):
        rows = ledger[p * PAGE:(p + 1) * PAGE]
        page_total = sum(rows)
        running += page_total
        if cook and p == 20:
            running += 5_000                       # quietly inflate the carried-forward balance
        books.append((rows, page_total, running))
    return books

def assistant_checks_page(rows, page_total):
    """A level-1 assistant: verifies that one page adds up. (Trusted, for this example.)"""
    return sum(rows) == page_total

def audit(books, check_carry=False):
    ok = all(assistant_checks_page(rows, t) for rows, t, _ in books)
    if check_carry:                                # the user also checks each seam: one addition per page
        prev = 0
        for rows, t, carried in books:
            ok &= (prev + t == carried)
            prev = carried
    return ok

honest_books, cooked_books = write_books(), write_books(cook=True)
print("Reported total, honest books:", honest_books[-1][2], "  (true:", true_total, ")")
print("Reported total, cooked books:", cooked_books[-1][2])

# %% [markdown]
# ### Predict before you run
#
# The cooked books add 5,000 to the carried-forward balance on page 20, but every page's transactions and page total are correct. **Will the assisted audit catch it?**

# %%
print("Audit with page assistants only:      honest passes:", audit(honest_books),
      "  cooked passes:", audit(cooked_books))
print("Audit + you check each carry-forward: honest passes:", audit(honest_books, True),
      "  cooked passes:", audit(cooked_books, True))

# %% [markdown]
# <details><summary>▶ Why did the first audit miss it?</summary>
#
# Every assistant did its job perfectly. The fraud sat in the **seam between pages**, the one thing no assistant was asked to check. An AI that is being trained to pass the audit will find exactly those unchecked places, not out of malice but because they are where reward is cheapest to get. Checking the seams cost you only 32 more additions.
#
# Leike et al. raise the general worry themselves (§3.2): "An important open question is whether errors accumulate: do the mistakes of the more narrow agent A_{k-1} lead to larger mistakes in the training of agent A_k?" Notebook 4 shows an optimizer discovering the seams on its own.
# </details>

# %% [markdown]
# ### 🎛 Try it: be the crooked bookkeeper
#
# Pick *how* to cook the books and *which* checks the auditor runs. Which tricks does each audit catch? Can you find a trick that passes even the careful audit?
#
# - **carry-forward**: inflate the running balance carried from one page to the next (a seam).
# - **one transaction**: change a transaction inside a page but keep the page total.
# - **page total**: change a page total (and carry that through honestly).
# - **transaction + page total**: change a transaction *and* fix up the page total to match.

# %%
def cook_books(trick, page, amount):
    books, running = [], 0
    for p in range(N // PAGE):
        rows = list(ledger[p * PAGE:(p + 1) * PAGE])
        page_total = sum(rows)
        if p == page and trick == "one transaction":
            rows[3] += amount                              # the page no longer adds up
        if p == page and trick == "page total":
            page_total += amount
        if p == page and trick == "transaction + page total":
            rows[3] += amount; page_total = sum(rows)
        running += page_total
        if p == page and trick == "carry-forward":
            running += amount
        books.append((rows, page_total, running))
    return books

def try_audit(trick="carry-forward", page=20, amount=5000, check_seams=False):
    books = cook_books(trick, page, amount)
    passes = audit(books, check_carry=check_seams)
    who = "page assistants" + (" + your seam checks" if check_seams else " only")
    print(f"Reported total: {books[-1][2]}   (true total {true_total}, off by {books[-1][2] - true_total:+})")
    print(f"Audit with {who}: {'PASSES ✗ (fraud missed)' if passes else 'FAILS ✓ (fraud caught)'}")

widgets.interact(try_audit,
                 trick=widgets.Dropdown(options=["carry-forward", "one transaction", "page total", "transaction + page total"], description="trick"),
                 page=widgets.IntSlider(value=20, min=0, max=N // PAGE - 1, description="on page"),
                 amount=widgets.IntSlider(value=5000, min=100, max=20000, step=100, description="amount"),
                 check_seams=widgets.Checkbox(value=False, description="also check every carry-forward"));

# %% [markdown]
# <details><summary>▶ Which trick beats every audit, and what does that teach?</summary>
#
# "transaction + page total" passes both audits: every page adds up and every seam carries forward, because the bookkeeper changed the *evidence itself*. Our auditors trust the raw transactions; nobody compares them with the receipts. That is the next seam. Every layer of checking leaves some input it takes on trust, and an optimizer under pressure will find it. (The debate notebooks have the same caveat: there, debaters can only reveal true evidence.)
# </details>

# %% [markdown]
# ## 4. The three ideas side by side
#
# | | Amplification | Debate | Recursive reward modeling |
# |---|---|---|---|
# | **The overseer…** | splits questions and combines answers | judges one disputed step | evaluates outcomes with help |
# | **In the ledger** | a tree of one-addition overseers | two debaters halve the dispute | page assistants + checking seams |
# | **Overseer's work** | 1 step per node, but the tree is huge until distilled | about 2 × log₂ N steps | one check per page plus seams |
# | **Bet** | hard reasoning factors into small steps | refuting a lie is easier than telling one | evaluating is easier than doing |
# | **Failure mode** | errors compound across the tree | weak opponents, obfuscated arguments | optimizers exploit what nobody checks |
#
# All three appear in Evan Hubinger's [overview of 11 proposals for building safe advanced AI](https://www.lesswrong.com/posts/fRsjBseRuvRhMPPE5/an-overview-of-11-proposals-for-building-safe-advanced-ai): recursive reward modeling is proposal 8, debate is proposal 9, and amplification appears in several others.
#
# ## 5. Check yourself
#
# 1. In the debate, the judge did about 21 steps for 1,024 transactions. About how many would it need for a ledger of one million transactions?
# 2. Why does a liar's lie never get "smaller" as the debate goes down the tree?
# 3. In the audit, name another "seam" a cooked ledger could hide in.
# 4. Which of the three methods would you trust most for reviewing a 10,000-line code change, and why?
#
# <details><summary>▶ Answers</summary>
#
# 1. log₂(1,000,000) ≈ 20 rounds, so about 41 steps. The judge's work grows with the logarithm of the problem size.
# 2. The liar's two half-totals must add up to their claimed total (the judge checks), so if the total is off by 37, at least one half is off by a nonzero amount, and that half is what gets disputed next.
# 3. For example: the opening balance, transactions split across two pages, currency conversions, or the order of pages.
# 4. There's no single right answer; that's the discussion for today.
# </details>
#
# ## Where to go next
#
# - **Notebook 1:** iterated amplification with a real learner (and systematic vs. random overseer errors)
# - **Notebook 2:** debate on a maze, and what happens when the opponent is weak
# - **Notebook 3:** the pixel game from the debate paper
# - **Notebook 4:** recursive reward modeling, where an optimizer finds the seams by itself
