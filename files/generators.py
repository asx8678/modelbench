"""
Procedural problem generators for the reasoning benchmark.

Design goals
------------
* Contamination-proof: every problem is generated on the fly, so it cannot be
  in any model's training set.
* Gold answers are correct *by construction* (computed while the problem is built).
* Three independent random streams per problem:
    - structure rng : fixes the underlying logic (numbers, order, rule) and the gold.
    - surface  rng : changes only cosmetic surface (names, items, clause order).
    - distractor rng: builds an irrelevant "NoOp" clause.
  Splitting these lets us hold the computation fixed while varying surface,
  which is exactly the invariance / robustness probe from GSM-Symbolic.

Each generator has the signature:
    gen(difficulty: int, structure_seed: int, surface_seed: int, distractor: bool)
        -> (prompt: str, gold: str, answer_type: str, choices: list|None)

answer_type is "int" or "choice".
"""

import hashlib
import itertools
import random
import re
from dataclasses import dataclass, asdict
from typing import Optional, List

# ---------------------------------------------------------------- word pools
NAMES = ["Maria", "Tomás", "Aisha", "Kenji", "Lena", "Omar", "Priya", "Diego",
         "Nadia", "Sven", "Yuki", "Rosa", "Ivan", "Mei", "Pablo", "Hana",
         "Olek", "Zara", "Bruno", "Anika"]
ITEMS = ["apples", "pencils", "marbles", "coins", "stickers", "books",
         "cookies", "stamps", "shells", "buttons", "candles", "ribbons"]
CONTAINERS = ["the red box", "the blue box", "the green box", "the wooden crate",
              "the metal tin", "the paper bag", "the glass jar", "the basket"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "teal", "grey"]

# adjective tuples: (comparative, antonym-comparative, superlative-high, superlative-low)
REL = [
    ("taller", "shorter", "tallest", "shortest"),
    ("older", "younger", "oldest", "youngest"),
    ("heavier", "lighter", "heaviest", "lightest"),
    ("faster", "slower", "fastest", "slowest"),
    ("richer", "poorer", "richest", "poorest"),
]


def _rng(*parts) -> random.Random:
    """Deterministic Random seeded by a hash of the parts."""
    seed = int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest(), 16) % (2 ** 32)
    return random.Random(seed)


def _drop_redundant(clues, still_unique):
    """Return an irredundant subset: drop every clue whose removal still leaves the
    solution unique. One forward pass suffices — removing clues only makes the
    survivors more essential — so the result has no redundant giveaways and every
    remaining clue is load-bearing. Keeps order, so it stays deterministic."""
    kept = list(clues)
    i = 0
    while i < len(kept):
        trial = kept[:i] + kept[i + 1:]
        if still_unique(trial):
            kept = trial
        else:
            i += 1
    return kept


# ----------------------------------------------------------- 1. arithmetic
def gen_arithmetic(difficulty, structure_seed, surface_seed, distractor):
    """Multi-hop quantitative word problem. Difficulty = number of operations."""
    rs = _rng("arith-struct", difficulty, structure_seed)
    ru = _rng("arith-surf", structure_seed, surface_seed)
    rd = _rng("arith-distract", structure_seed)

    name = ru.choice(NAMES)
    item = ru.choice(ITEMS)
    current = rs.randint(3, 20)
    clauses = [f"{name} starts with {current} {item}."]

    for _ in range(difficulty):
        op = rs.choice(["add", "sub", "mul", "div"])
        if op == "sub" and current <= 1:
            op = "add"
        divisors = [f for f in (2, 3, 4) if current % f == 0 and current // f >= 1]
        if op == "div" and not divisors:
            op = "add"
        if op == "add":
            k = rs.randint(2, 15); current += k
            # Verb is purely cosmetic, so draw it from the SURFACE rng (E8): this
            # makes surface variants vary the phrasing, not just names/items, so
            # the invariance probe tests lexical robustness too. The gold is set by
            # rs (magnitudes/ops) and stays fixed across a grp's surface variants.
            verb = ru.choice(["buys", "finds", "is given", "picks up"])
            clauses.append(f"{name} {verb} {k} more {item}.")
        elif op == "sub":
            k = rs.randint(1, current - 1); current -= k
            verb = ru.choice(["gives away", "loses", "uses", "drops"])
            clauses.append(f"{name} {verb} {k} {item}.")
        elif op == "mul":
            f = rs.choice([2, 3]); current *= f
            clauses.append(f"{name} {'doubles' if f == 2 else 'triples'} the number of {item} they have.")
        else:  # div: exact by construction, so the gold stays an integer
            f = rs.choice(divisors); current //= f
            clauses.append(f"{name} divides the {item} into {f} equal groups and keeps one group.")

    if distractor:
        # Relevance-true NoOp (bench-ukn / E1): the SAME subject acts on a
        # DIFFERENT item. A subject-name filter no longer separates the clause --
        # the model (and the item-aware verifier) must bind each operation to the
        # queried item, not just the queried name.
        oitem = rd.choice([it for it in ITEMS if it != item])
        verb = rd.choice(["buys", "finds", "is given", "picks up"])
        ox = rd.randint(2, 20)
        clauses.insert(1, f"{name} {verb} {ox} {oitem}.")

    prompt = " ".join(clauses) + f" How many {item} does {name} have now?"
    return prompt, str(current), "int", None


# -------------------------------------------------------- 2. state tracking
def gen_state(difficulty, structure_seed, surface_seed, distractor):
    """Track item counts across containers through a sequence of updates."""
    rs = _rng("state-struct", difficulty, structure_seed)
    ru = _rng("state-surf", structure_seed, surface_seed)
    rd = _rng("state-distract", structure_seed)

    item = ru.choice(ITEMS)
    names = ru.sample(CONTAINERS, 3)            # cosmetic slot names
    state = [rs.randint(5, 20) for _ in range(3)]  # structural amounts, by slot
    clauses = [f"{names[i].capitalize()} has {state[i]} {item}." for i in range(3)]

    for _ in range(difficulty):
        op = rs.choice(["add", "remove", "move"])
        i = rs.randrange(3)
        if op == "add" or (op == "remove" and state[i] <= 1):
            k = rs.randint(2, 12); state[i] += k
            clauses.append(f"{k} {item} are added to {names[i]}.")
        elif op == "remove":
            k = rs.randint(1, state[i]); state[i] -= k
            clauses.append(f"{k} {item} are removed from {names[i]}.")
        else:  # move
            j = rs.choice([x for x in range(3) if x != i])
            if state[i] <= 0:
                k = rs.randint(2, 12); state[i] += k
                clauses.append(f"{k} {item} are added to {names[i]}.")
            else:
                k = rs.randint(1, state[i]); state[i] -= k; state[j] += k
                clauses.append(f"{k} {item} are moved from {names[i]} to {names[j]}.")

    qi = rs.randrange(3)
    if distractor:
        # Relevance-true NoOp (E1): a real-looking UPDATE to a TRACKED container,
        # but about a DIFFERENT item. A container-name filter no longer separates
        # it; the item-aware verifier binds updates to the queried item, so the
        # different-item clause is correctly inert.
        oitem = rd.choice([it for it in ITEMS if it != item])
        oc = names[rd.randrange(3)]
        ox = rd.randint(2, 12)
        clauses.insert(3, f"{ox} {oitem} are added to {oc}.")

    prompt = " ".join(clauses) + f" How many {item} are in {names[qi]} now?"
    return prompt, str(state[qi]), "int", None


# ----------------------------------------------------- 2c. retroactive edit
# A state-tracking problem where a late "Actually..." clause retroactively
# changes an earlier value. The gold is the queried container's final amount
# after replaying the edit.
def gen_retroactive_edit(difficulty, structure_seed, surface_seed, distractor):
    rs = _rng("re-struct", difficulty, structure_seed)
    ru = _rng("re-surf", structure_seed, surface_seed)

    item = ru.choice(ITEMS)
    names = ru.sample(CONTAINERS, 3)
    initial = [rs.randint(5, 20) for _ in range(3)]
    state = list(initial)
    clauses = [f"{names[i].capitalize()} has {state[i]} {item}." for i in range(3)]

    # Structural updates.
    for _ in range(difficulty):
        op = rs.choice(["add", "remove", "move"])
        i = rs.randrange(3)
        if op == "add" or (op == "remove" and state[i] <= 1):
            k = rs.randint(2, 12); state[i] += k
            clauses.append(f"{k} {item} are added to {names[i]}.")
        elif op == "remove":
            k = rs.randint(1, state[i]); state[i] -= k
            clauses.append(f"{k} {item} are removed from {names[i]}.")
        else:
            j = rs.choice([x for x in range(3) if x != i])
            if state[i] <= 0:
                k = rs.randint(2, 12); state[i] += k
                clauses.append(f"{k} {item} are added to {names[i]}.")
            else:
                k = rs.randint(1, state[i]); state[i] -= k; state[j] += k
                clauses.append(f"{k} {item} are moved from {names[i]} to {names[j]}.")

    # Retroactive edit: a late clause says one container held a multiple
    # of its originally stated amount. Vary the factor (not just ×2/×3)
    # to break the pattern-matchable trigger, and target a container
    # that participates in a LATER structural op so the edit must
    # propagate to the final state.
    edit_i = rs.randrange(3)
    factor = rs.choice([2, 3, 4, 5])
    edit_clause = f"Actually, {names[edit_i].lower()} held {factor} times as many {item} as originally stated."
    # Ensure downstream dependence: if the last structural op does not
    # touch edit_i, append a no-op add to it so the factor must be
    # carried through a later op.
    clauses.insert(-1, edit_clause)
    last_s = clauses[-1] if clauses[-1] != edit_clause else clauses[-2]
    if not re.search(re.escape(names[edit_i].lower()), last_s, re.I):
        k = rs.randint(1, 3)
        clauses.append(f"{k} {item} are added to {names[edit_i]}.")

    # Query the EDITED container so the retroactive edit is ALWAYS load-bearing:
    # the queried final value carries a coefficient-1 dependence on the scaled
    # initial amount, so a different factor always changes the gold. (Querying an
    # independent container made the "Actually..." pivot a no-op ~2/3 of the time.)
    qi = edit_i
    if distractor:
        # Relevance-true NoOp (E1): a different-item update to a TRACKED container.
        # The item-aware verifier (re.escape(item) below) already ignores it, so
        # the gold stays a true NoOp while a container-only filter would mis-apply it.
        oitem = ru.choice([it for it in ITEMS if it != item])
        oc = names[ru.randrange(3)]
        ox = ru.randint(2, 12)
        clauses.insert(3, f"{ox} {oitem} are added to {oc}.")

    prompt = " ".join(clauses) + f" How many {item} are in {names[qi]} now?"
    # Recompute gold from the INITIAL state, with the edit applied, then replay
    # all recorded operations.
    edited_state = [initial[j] for j in range(3)]
    edited_state[edit_i] *= factor
    name_to_idx = {n.lower(): i for i, n in enumerate(names)}
    for s in clauses:
        if (m := re.match(r"(\d+) " + re.escape(item) + r" are added to (.+?)\.", s)):
            idx = name_to_idx[m.group(2).lower().rstrip()]
            edited_state[idx] += int(m.group(1))
        elif (m := re.match(r"(\d+) " + re.escape(item) + r" are removed from (.+?)\.", s)):
            idx = name_to_idx[m.group(2).lower().rstrip()]
            edited_state[idx] -= int(m.group(1))
        elif (m := re.match(r"(\d+) " + re.escape(item) + r" are moved from (.+?) to (.+?)\.", s)):
            src = name_to_idx[m.group(2).lower().rstrip()]
            dst = name_to_idx[m.group(3).lower().rstrip()]
            k = int(m.group(1))
            edited_state[src] -= k
            edited_state[dst] += k
    return prompt, str(edited_state[qi]), "int", None


# ----------------------------------------------------- 2b. multi-turn inject
# Turn 1 establishes a state-tracking setup. Turn 2 injects a new rule and asks a
# question whose answer depends on the state from turn 1 plus the injected rule.
# This tests whether the model carries state across conversational turns.
def gen_multi_turn_inject(difficulty, structure_seed, surface_seed, distractor):
    rs = _rng("mt-struct", difficulty, structure_seed)
    ru = _rng("mt-surf", structure_seed, surface_seed)

    item = ru.choice(ITEMS)
    names = ru.sample(CONTAINERS, 3)
    state = [rs.randint(5, 20) for _ in range(3)]
    clauses = [f"{names[i].capitalize()} has {state[i]} {item}." for i in range(3)]

    # Structural updates applied before turn 2; gold is computed after these.
    for _ in range(difficulty):
        op = rs.choice(["add", "remove", "move"])
        i = rs.randrange(3)
        if op == "add" or (op == "remove" and state[i] <= 1):
            k = rs.randint(2, 12); state[i] += k
            clauses.append(f"{k} {item} are added to {names[i]}.")
        elif op == "remove":
            k = rs.randint(1, state[i]); state[i] -= k
            clauses.append(f"{k} {item} are removed from {names[i]}.")
        else:
            j = rs.choice([x for x in range(3) if x != i])
            if state[i] <= 0:
                k = rs.randint(2, 12); state[i] += k
                clauses.append(f"{k} {item} are added to {names[i]}.")
            else:
                k = rs.randint(1, state[i]); state[i] -= k; state[j] += k
                clauses.append(f"{k} {item} are moved from {names[i]} to {names[j]}.")

    qi = rs.randrange(3)
    # Turn 1: the state-establishment prompt only.
    turn1 = " ".join(clauses)

    # Turn 2: inject a new local rule and ask the question.
    # The injected rule is a simple multiplier/bonus applied to the queried container.
    inject = rs.choice([("double", 2), ("triple", 3), ("keep", 1)])
    rule_name, mult = inject
    injected_gold = state[qi] * mult
    if rule_name == "keep":
        turn2 = (f"Now apply this rule: use the current count in {names[qi]} exactly as is. "
                 f"How many {item} are in {names[qi]} now?")
    else:
        turn2 = (f"Now apply this rule: {rule_name} the number of {item} in {names[qi]}. "
                 f"How many {item} are in {names[qi]} now?")

    prompt = turn1 + " " + turn2
    turns = [turn1, turn2]
    return prompt, str(injected_gold), "int", None, turns


# -------------------------------------------------------------- 3. ordering
def _ordinal(n):
    """Convert integer to ordinal string (1->'1st', 2->'2nd', etc.)."""
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th'][min(n % 10, 4)]}"

def gen_order(difficulty, structure_seed, surface_seed, distractor):
    """Transitive comparison. Difficulty = number of entities - 2.

    Ships the adjacent chain (the minimal sufficient set) PLUS a subset of
    the non-adjacent pairwise facts so degree-counting shortcuts become
    less effective. The verifier in _verify_order does Kahn's algorithm
    on the full edge set and rejects ambiguous graphs.
    """
    rs = _rng("order-struct", difficulty, structure_seed)
    ru = _rng("order-surf", structure_seed, surface_seed)
    rd = _rng("order-distract", structure_seed)

    m = difficulty + 2
    names = rs.sample(NAMES, m)
    comp, anti, sup_hi, sup_lo = rs.choice(REL)
    order = names[:]                            # order[0] highest ... order[-1] lowest

    # Adjacent chain edges: the minimal sufficient set.
    edges = {order[i]: {order[i + 1]} for i in range(m - 1)}

    # Non-adjacent transitive extras: pick a random subset of size
    # roughly m (so total edges are ~2*(m-1), giving each entity
    # balanced connectivity). Withheld edges = m*(m-1)/2 - 1 - m.
    non_adjacent = [(order[i], order[j])
                    for i in range(m) for j in range(i + 2, m)]
    rs.shuffle(non_adjacent)
    for a, b in non_adjacent[:m]:
        edges.setdefault(a, set()).add(b)

    # Ask a non-extreme rank to prevent degree-counting shortcuts.
    rank_idx = rs.randint(1, m - 2)             # 1..m-2 (never 0 or m-1)
    rank_word = _ordinal(rank_idx + 1)          # 1-indexed ordinal
    gold = order[rank_idx]

    # Emit facts in random order.
    flat = [(a, b) for a, bs in edges.items() for b in bs]
    ru.shuffle(flat)
    clauses = []
    for a, b in flat:
        if ru.random() < 0.5:
            clauses.append(f"{a} is {comp} than {b}.")
        else:
            clauses.append(f"{b} is {anti} than {a}.")

    if distractor:
        # Relevance-true NoOp (E2): an in-grammar comparative on an ORTHOGONAL
        # relation family. Same "X is <comp> than Y" surface as a real clue, but a
        # different adjective axis (e.g. "richer" inside a height ordering), so a
        # "than"-regex no longer separates it. The relation-bound verifier ignores
        # off-axis edges because they don't match the asked superlative, so the
        # clue is genuinely inert -- the model must bind the relation to the query.
        dtup = rd.choice([r for r in REL if r != (comp, anti, sup_hi, sup_lo)])
        a, b = rd.sample(order, 2)
        clauses.insert(0, f"{a} is {dtup[0]} than {b}." if rd.random() < 0.5
                       else f"{a} is {dtup[1]} than {b}.")

    prompt = " ".join(clauses) + f" Who is the {rank_word} {sup_hi}?"
    return prompt, gold, "choice", names



# ------------------------------------------------------------- 4. sequences
def gen_sequence(difficulty, structure_seed, surface_seed, distractor):
    """Find-the-next-term rule induction. Difficulty 1..6 selects rule type."""
    rs = _rng("seq-struct", difficulty, structure_seed)
    lvl = min(max(difficulty, 1), 6)

    if lvl == 1:                                  # arithmetic
        a0, d = rs.randint(1, 9), rs.randint(2, 9)
        terms = [a0 + i * d for i in range(6)]; nxt = a0 + 6 * d
    elif lvl == 2:                                # geometric
        a0, r = rs.randint(1, 5), rs.choice([2, 3])
        terms = [a0 * (r ** i) for i in range(6)]; nxt = a0 * (r ** 6)
    elif lvl == 3:                                # quadratic (constant 2nd diff)
        a0, b, c = rs.randint(0, 5), rs.randint(1, 5), rs.randint(1, 4)
        f = lambda n: a0 + n * b + c * (n * (n - 1) // 2)
        terms = [f(i) for i in range(6)]; nxt = f(6)
    elif lvl == 4:                                # fibonacci-like
        seq = [rs.randint(1, 6), rs.randint(1, 6)]
        for _ in range(6):
            seq.append(seq[-1] + seq[-2])
        terms, nxt = seq[:7], seq[7]
    elif lvl == 5:                                # interleaved two APs
        a0, da, b0, db = rs.randint(1, 6), rs.randint(2, 6), rs.randint(1, 6), rs.randint(2, 6)
        seq = []
        for i in range(8):
            seq.append(a0 + (i // 2) * da if i % 2 == 0 else b0 + (i // 2) * db)
        terms, nxt = seq, a0 + 4 * da
    else:                                         # cubic (constant 3rd diff)
        a, b, c, e = rs.randint(0, 3), rs.randint(1, 4), rs.randint(1, 3), rs.randint(1, 2)
        f = lambda n: (a + b * n + c * (n * (n - 1) // 2)
                       + e * (n * (n - 1) * (n - 2) // 6))
        terms = [f(i) for i in range(7)]; nxt = f(7)

    prompt = "Sequence: " + ", ".join(map(str, terms)) + ", ...  What is the next number?"
    return prompt, str(nxt), "int", None


# ---------------------------------------------------- 5. knights & knaves
# Constraint-satisfaction deduction. Each islander is a knight (always truthful)
# or a knave (always lying); a knight's statement is true, a knave's is false.
# We assign types per *slot* (structure rng), emit statements that are consistent
# with that assignment by construction, then greedily keep adding statements until
# brute force confirms the assignment is the UNIQUE one. Names are attached last
# from the surface rng, so the logic (and the gold) is invariant to renaming.

def _kk_truth(st, t):
    """Truth value of statement `st` under type map `t` (entity -> is_knight)."""
    if st[0] == "ABS":                      # "X says Y is a knight/knave"
        _, _x, y, want_knight = st
        return t[y] if want_knight else (not t[y])
    _, _x, y, z, same = st                  # "X says Y and Z are (same|different)"
    rel = (t[y] == t[z])
    return rel if same else (not rel)


def _kk_consistent(st, t):
    """A statement is satisfied iff (speaker is a knight) == (statement is true)."""
    return t[st[1]] == _kk_truth(st, t)


def _kk_all_solutions(entities, stmts):
    """Every type assignment over `entities` satisfying all statements (brute force)."""
    ents = list(entities)
    out = []
    for bits in range(1 << len(ents)):
        t = {e: bool((bits >> i) & 1) for i, e in enumerate(ents)}
        if all(_kk_consistent(s, t) for s in stmts):
            out.append(t)
    return out


def gen_knights(difficulty, structure_seed, surface_seed, distractor):
    rs = _rng("kk-struct", difficulty, structure_seed)
    ru = _rng("kk-surf", structure_seed, surface_seed)
    n = difficulty + 2
    slots = list(range(n))
    typ = {s: rs.random() < 0.5 for s in slots}        # structure: type per slot

    pool = []                                          # all consistent statements
    for x in slots:
        others = [o for o in slots if o != x]
        for y in others:                               # ABS: links x and y's types
            pool.append(("ABS", x, y, typ[x] == typ[y]))
        for i in range(len(others)):                   # REL: ties x to a relation of
            for j in range(i + 1, len(others)):        # two others (breaks the global
                y, z = others[i], others[j]            # knight<->knave flip symmetry)
                eq = (typ[y] == typ[z])
                pool.append(("REL", x, y, z, typ[x] == eq))
    rs.shuffle(pool)

    remaining = _kk_all_solutions(slots, [])           # greedily shrink to one solution
    chosen, changed = [], True
    while len(remaining) > 1 and changed:
        changed = False
        for st in pool:
            if st in chosen:
                continue
            nr = [t for t in remaining if _kk_consistent(st, t)]
            if len(nr) < len(remaining):
                chosen.append(st); remaining = nr; changed = True
                if len(remaining) == 1:
                    break
    chosen = _drop_redundant(chosen, lambda kept: len(_kk_all_solutions(slots, kept)) == 1)

    names = ru.sample(NAMES, n)                         # surface: attach labels
    ru.shuffle(chosen)
    sents = []
    for st in chosen:
        if st[0] == "ABS":
            sents.append(f"{names[st[1]]} says that {names[st[2]]} is a "
                         f"{'knight' if st[3] else 'knave'}.")
        else:
            sents.append(f"{names[st[1]]} says that {names[st[2]]} and {names[st[3]]} "
                         f"are {'the same type' if st[4] else 'different types'}.")
    # De-collapse the answer space from binary to 2^n (set of all knaves).
    # Gold is the sorted, comma-separated list of knave names; empty if
    # all inhabitants are knights.
    knave_names = sorted(names[s] for s in slots if not typ[s])
    gold = ", ".join(knave_names)
    prompt = ("On an island, every inhabitant is either a knight (who always tells the "
              "truth) or a knave (who always lies). Its inhabitants are "
              + ", ".join(names) + ". They say:\n" + "\n".join(sents)
              + "\nList every knave on this island (comma-separated, e.g. 'Maria, Tom\u00e1s'). "
                "If everyone is a knight, answer 'none'.")
    return prompt, gold, "set", names



# ---------------------------------------------------- 6. logic grid (zebra-lite)
# N people occupy N distinct floors (1..N). Relative-position clues pin a unique
# arrangement; the question asks one person's floor. As above, the arrangement and
# the chosen clue set are decided in slot space (structure rng) and only labelled
# from the surface rng, so renaming people leaves the computation and gold fixed.

def _lg_holds(cl, pos):
    """Whether clue `cl` holds under floor map `pos` (entity -> floor int)."""
    k = cl[0]
    if k == "FLOOR":  return pos[cl[1]] == cl[2]
    if k == "ABOVEK": return pos[cl[1]] - pos[cl[3]] == cl[2]
    if k == "DIR":    return pos[cl[1]] - pos[cl[2]] == 1
    if k == "HI":     return pos[cl[1]] > pos[cl[2]]
    return abs(pos[cl[1]] - pos[cl[2]]) == 1                          # ADJ


def _lg_solutions(entities, n, clues):
    """Floor assignments (1..n, distinct) over `entities` satisfying all clues.
    Stops at two, since uniqueness is all any caller needs."""
    ents = list(entities)
    out = []
    for perm in itertools.permutations(range(1, n + 1)):
        pos = dict(zip(ents, perm))
        if all(_lg_holds(c, pos) for c in clues):
            out.append(pos)
            if len(out) > 1:
                break
    return out


def gen_logic_grid(difficulty, structure_seed, surface_seed, distractor):
    rs = _rng("lg-struct", difficulty, structure_seed)
    ru = _rng("lg-surf", structure_seed, surface_seed)
    n = difficulty + 2
    slots = list(range(n))
    rs.shuffle(slots)
    pos = dict(zip(slots, range(1, n + 1)))
    q = rs.choice(slots)

    rel = []
    for a, b in itertools.combinations(slots, 2):
        da = pos[a] - pos[b]
        if da >= 1:
            rel.append(("HI", a, b))
            rel.append(("DIR", a, b) if da == 1 else ("ABOVEK", a, da, b))
        if pos[a] - pos[b] == 1:
            rel.append(("ADJ", a, b))
    rs.shuffle(rel)
    floorclues = [("FLOOR", a, pos[a]) for a in slots if a != q]
    rs.shuffle(floorclues)
    ordered = rel + floorclues

    remaining = [dict(zip(slots, perm))
                 for perm in itertools.permutations(range(1, n + 1))]
    chosen, changed = [], True
    while len(remaining) > 1 and changed:
        changed = False
        for cl in ordered:
            if cl in chosen:
                continue
            nr = [p for p in remaining if _lg_holds(cl, p)]
            if len(nr) < len(remaining):
                chosen.append(cl); remaining = nr; changed = True
                if len(remaining) == 1:
                    break
    chosen = _drop_redundant(chosen, lambda kept: len(_lg_solutions(slots, n, kept)) == 1)

    names = ru.sample(NAMES, n)
    ru.shuffle(chosen)
    nm = lambda s: names[s]
    sents = []
    for cl in chosen:
        if cl[0] == "FLOOR":    sents.append(f"{nm(cl[1])} lives on floor {cl[2]}.")
        elif cl[0] == "ABOVEK": sents.append(f"{nm(cl[1])} lives {cl[2]} floors above {nm(cl[3])}.")
        elif cl[0] == "DIR":    sents.append(f"{nm(cl[1])} lives directly above {nm(cl[2])}.")
        elif cl[0] == "HI":     sents.append(f"{nm(cl[1])} lives on a higher floor than {nm(cl[2])}.")
        else:                   sents.append(f"{nm(cl[1])} and {nm(cl[2])} live on adjacent floors.")
    gold = str(pos[q])
    prompt = (f"A building has floors numbered 1 (lowest) to {n} (highest). "
              "Exactly one of these people lives on each floor: " + ", ".join(names)
              + ".\n" + "\n".join(sents) + f"\nOn which floor does {nm(q)} live?")
    return prompt, gold, "int", None


# ---------------------------------------------------------- 7. composed
# Chain three genuinely load-bearing hops: a knights_knaves knight-count K
# (deduced, never printed) seeds an arithmetic chain -> R; solving an ordering
# yields C, the position of a named inhabitant; the final gold is R * C, so a
# slip in ANY hop changes the answer. Every hop is embedded so each can be
# verified independently.

def _composed_parse_hops(prompt):
    """Return dict: knights_prompt, arith_prompt, arith_subj, order_block, pivot.

    All three stages are load-bearing. Stage 1 (knights) yields a count K that
    seeds Stage 2 (arithmetic -> R). Stage 3 (ordering) yields C, the position
    of `pivot`; the gold is R * C. The parser extracts each stage from the
    prompt text so the verifier can re-derive every hop independently.
    """
    m = re.search(
        r"Stage 1:\s*(.+?)\n+Stage 2:\s*(.+?)\n+Stage 3:\s*(.+)$",
        prompt, re.S,
    )
    if not m:
        return None
    knights_block = m.group(1).strip()
    stage2 = m.group(2).strip()
    stage3 = m.group(3).strip()
    # Stage 2: drop the trailing "Let R be ..." gloss; keep the arithmetic
    # (the "starts with as many ..." opener plus the update sentences).
    arith_prompt = re.split(r"\n+Let R be\b", stage2)[0].strip()
    sm = re.match(r"(\S[^\n]*?) starts with as many", arith_prompt)
    arith_subj = sm.group(1).strip() if sm else None
    # Stage 3: the named pivot whose 1-indexed position is C.
    pm = re.search(r"position 1, what is (.+?)'s position", stage3)
    pivot = pm.group(1).strip() if pm else None
    return {
        "knights_prompt": knights_block,
        "arith_prompt": arith_prompt,
        "arith_subj": arith_subj,
        "order_block": stage3,
        "pivot": pivot,
    }



def _verify_composed(prompt, gold):
    parsed = _composed_parse_hops(prompt)
    if not parsed:
        return False
    # Hop A: re-derive the knight count K from Stage 1's clues.
    kk_names, kk_stmts = _kk_parse(parsed["knights_prompt"])
    if not kk_names or not kk_stmts:
        return False
    sols = _kk_all_solutions(kk_names, kk_stmts)
    if len(sols) != 1:
        return False
    knight_count = sum(1 for v in sols[0].values() if v)
    # Hop B: replay Stage 2's arithmetic seeded by K -> R.
    arith_result = _verify_arithmetic_raw(parsed["arith_prompt"], knight_count)
    if arith_result is None:
        return False
    # Hop C: re-solve Stage 3's ordering -> C, the pivot's 1-indexed position.
    pivot = parsed.get("pivot")
    order, _expected = _verify_order_raw(parsed.get("order_block", ""), "")
    if not pivot or order is None or pivot not in order:
        return False
    count_c = order.index(pivot) + 1
    try:
        return int(gold) == arith_result * count_c
    except (TypeError, ValueError):
        return False



def _verify_arithmetic_raw(prompt, start):
    """Re-derive the arithmetic result given an explicit starting integer.

    The opener references the start verbally ("starts with as many ... as
    there are knights"), so the integer itself is never read from the text --
    only the subject name is, and the caller supplies the start.
    """
    m = re.search(r"^(\S[^\n]*?) starts with as many\b", prompt, re.M)
    if not m:
        return None
    subj = m.group(1).strip()
    total = start
    sents = _sentences(prompt)
    for s in sents[1:]:
        if not s.startswith(subj + " "):
            continue
        rest = s[len(subj) + 1:]
        if (m := re.match(r"(?:buys|finds|is given|picks up) (\d+) more", rest)):
            total += int(m.group(1))
        elif (m := re.match(r"(?:gives away|loses|uses|drops) (\d+)", rest)):
            total -= int(m.group(1))
        elif rest.startswith("doubles"):
            total *= 2
        elif rest.startswith("triples"):
            total *= 3
        elif (m := re.match(r"divides the .+? into (\d+) equal groups and keeps one group", rest)):
            total //= int(m.group(1))
        else:
            return None
    return total


def _verify_order_raw(prompt, query):
    """Return (ordered_names, gold_name) for an ordering sub-prompt."""
    rel_hi = {r[0] for r in REL}
    rel_lo = {r[1] for r in REL}
    relations = []
    names = set()
    for m in re.finditer(r"(\w[\w']*) is (\w+) than (\w[\w']*)\.", prompt):
        a, rel, b = m.group(1), m.group(2), m.group(3)
        if rel in rel_hi:
            relations.append((a, b))
        elif rel in rel_lo:
            relations.append((b, a))
        else:
            continue
        names.update((a, b))
    if not names:
        return None, None
    # The sub-prompt must have a unique total order consistent with all clues.
    order = None
    for perm in itertools.permutations(names):
        ok = True
        for hi, lo in relations:
            if perm.index(hi) >= perm.index(lo):
                ok = False
                break
        if ok:
            if order is not None:
                return None, None
            order = list(perm)
    if order is None:
        return None, None
    # Parse ordinal rank from query (e.g., "2nd tallest" -> index 1)
    ordinals = {"1st": 0, "2nd": 1, "3rd": 2, "4th": 3, "5th": 4, "6th": 5, "7th": 6, "8th": 7}
    mq = re.search(r"(\w+) (\w+)", query)
    if mq and mq.group(1) in ordinals:
        rank = ordinals[mq.group(1)]
        expected = order[rank] if rank < len(order) else None
    elif query.lower() in ("shortest", "lightest", "youngest", "coldest"):
        expected = order[-1]
    else:
        expected = order[0]
    return order, expected


def gen_composed(difficulty, structure_seed, surface_seed, distractor):
    """Chain three genuinely load-bearing hops: knights -> arithmetic -> ordering.

    Every hop affects the final integer, so none can be skipped:
      * Hop A (knights): the model must DEDUCE the knight count K. K is never
        printed -- Stage 2 refers to it only as "as many ... as there are
        knights". K is forced >= 1 so the arithmetic start (and the final
        product) is non-zero.
      * Hop B (arithmetic): a chain seeded by K produces R.
      * Hop C (ordering): solving the ordering yields C, the 1-indexed position
        of a named inhabitant. The final answer is R * C.
    A one-unit slip in ANY hop changes the gold. There is no modulo wrap to
    mask arithmetic drift, and no stage is a labelled distractor.
    """
    rs = _rng("composed-struct", difficulty, structure_seed)
    ru = _rng("composed-surf", structure_seed, surface_seed)

    # Hop A: knights & knaves (small, fixed difficulty; unique by construction).
    # Re-roll until at least one knight, so the arithmetic start is non-zero.
    kk_diff = 2
    knight_count, kk_block = 0, ""
    for _ in range(64):
        kk_struct = rs.randint(0, 2 ** 16)
        kk_prompt, _kk_gold, _kk_at, _kk_n = gen_knights(kk_diff, kk_struct, 0, False)
        kk_names, kk_stmts = _kk_parse(kk_prompt)
        knight_count = sum(1 for v in _kk_all_solutions(kk_names, kk_stmts)[0].values() if v)
        if knight_count >= 1:
            kk_block = kk_prompt.rsplit("\n", 1)[0]   # clues only (drop the question)
            break

    # Hop B: arithmetic seeded by the UN-printed knight count.
    arith_diff = max(1, difficulty)
    arith_name = ru.choice(NAMES)
    arith_item = ru.choice(ITEMS)
    current = knight_count
    arith_clauses = [f"{arith_name} starts with as many {arith_item} as there "
                     f"are knights on the island."]
    for _ in range(arith_diff):
        op = rs.choice(["add", "sub", "mul", "div"])
        if op == "sub" and current <= 1:
            op = "add"
        divisors = [f for f in (2, 3, 4) if current % f == 0 and current // f >= 1]
        if op == "div" and not divisors:
            op = "add"
        if op == "add":
            k = rs.randint(2, 15); current += k
            verb = rs.choice(["buys", "finds", "is given", "picks up"])
            arith_clauses.append(f"{arith_name} {verb} {k} more {arith_item}.")
        elif op == "sub":
            k = rs.randint(1, current - 1); current -= k
            verb = rs.choice(["gives away", "loses", "uses", "drops"])
            arith_clauses.append(f"{arith_name} {verb} {k} {arith_item}.")
        elif op == "mul":
            f = rs.choice([2, 3]); current *= f
            arith_clauses.append(f"{arith_name} {'doubles' if f == 2 else 'triples'} the number of {arith_item} they have.")
        else:  # div
            f = rs.choice(divisors); current //= f
            arith_clauses.append(f"{arith_name} divides the {arith_item} into {f} equal groups and keeps one group.")
    arith_prompt = " ".join(arith_clauses)
    arith_result = current

    # Hop C: ordering. Solving it yields C = the 1-indexed position (from the
    # top) of a named inhabitant; the final answer is R * C, so a slip in the
    # ordering changes the gold (no modulo wrap to mask drift).
    order_diff = max(1, difficulty)
    order_struct = rs.randint(0, 2 ** 16)
    order_prompt, _order_gold, _order_at, order_names = gen_order(
        order_diff, order_struct, 0, False)
    order_clauses = order_prompt.split(" Who is the ")[0]
    sup_hi = re.search(r"Who is the \w+ (\w+)\?", order_prompt).group(1)
    n_people = len(order_names)              # order_names[0] = top ... [-1] = bottom
    pivot_idx = rs.randint(1, n_people - 1)  # C in 2..n_people (never identity)
    pivot_name = order_names[pivot_idx]
    count_c = pivot_idx + 1
    gold = str(arith_result * count_c)

    prompt = (
        "Stage 1: " + kk_block + "\n"
        "First work out how many knights live on the island; you need that "
        "count for Stage 2.\n\n"
        "Stage 2: " + arith_prompt + "\n"
        f"Let R be how many {arith_item} {arith_name} has at the end of Stage 2.\n\n"
        "Stage 3: " + order_clauses + "\n"
        f"Counting the {sup_hi} as position 1, what is {pivot_name}'s position "
        "in this ordering? Call it C. Your final answer is R multiplied by C. "
        "What is R times C?"
    )
    return prompt, gold, "int", None






# ------------------------------------------------ 8. redefined_ops (counterfactual)
# Arithmetic where the prompt redefines operators (e.g. "⊕ means a+b+3").
# Gold computed by applying the redefined table; verifier replays from
# prompt text. Stronger variants use non-commutative (order-sensitive)
# operators and chain two distinct redefined ops in a single problem.
def gen_redefined_ops(difficulty, structure_seed, surface_seed, distractor):
    rs = _rng("rops-struct", difficulty, structure_seed)
    ru = _rng("rops-surf", structure_seed, surface_seed)
    rd = _rng("rops-distract", structure_seed)

    name = ru.choice(NAMES)
    item = ru.choice(ITEMS)
    # Stronger operator table: commutative (add+bias, mul-bias, add+double)
    # PLUS non-commutative (order-sensitive) operators. Withholding the
    # standard operator meaning forces the model to read the redefined
    # table; a model that applies standard semantics scores at chance.
    op_bank = [
        (lambda sym, bias: (f"{sym} means add the two numbers then add {bias}",
                             lambda a, b: a + b + bias)),
        (lambda sym, bias: (f"{sym} means multiply the two numbers then subtract {bias}",
                             lambda a, b: a * b - bias)),
        (lambda sym, bias: (f"{sym} means double the sum of the two numbers, then add {bias}",
                             lambda a, b: (a + b) * 2 + bias)),
        # Non-commutative: subtraction from the LEFT operand (order-sensitive)
        (lambda sym, bias: (f"{sym} means subtract the right number from the left, then add {bias}",
                             lambda a, b: a - b + bias)),
        (lambda sym, bias: (f"{sym} means take the left number, divide by the right, then add {bias}",
                             lambda a, b: (a // b if b else 0) + bias if b else a + bias)),
        # Modulo (non-commutative with the operand order)
        (lambda sym, bias: (f"{sym} means take the left modulo the right, then add {bias}",
                             lambda a, b: (a % b if b else 0) + bias if b else a + bias)),
    ]
    # Pick one or two distinct operators to chain. With difficulty >= 3
    # we chain two distinct operators (alternating), forcing the model
    # to apply each step independently.
    use_chain = difficulty >= 3 and rs.random() < 0.5
    if use_chain:
        n_ops = 2
        picks = rs.sample(range(len(op_bank)), 2)
        op_syms = rs.sample(["⊕", "⊗", "⊖", "⊘", "⊙"], 2)
        ops = []
        for p, sym in zip(picks, op_syms):
            bias = rs.randint(1, 4)
            text, fn = op_bank[p](sym, bias)
            ops.append((sym, text, fn))
        # Alternate between the two ops across the difficulty steps
        def make_apply(idx, current, k):
            return ops[idx % 2][2](current, k)
        op_descs = [ops[0][1], ops[1][1]]
    else:
        n_ops = 1
        pick = rs.randrange(len(op_bank))
        op_sym = rs.choice(["⊕", "⊗", "⊖", "⊘", "⊙"])
        bias = rs.randint(1, 5)
        op_text, op_fn = op_bank[pick](op_sym, bias)
        ops = [(op_sym, op_text, op_fn)]
        def make_apply(idx, current, k):
            return op_fn(current, k)
        op_descs = [ops[0][1]]
    current = rs.randint(3, 15)
    clauses = [f"{name} starts with {current} {item}.",
               " ".join(f"In this problem, {t}." for t in op_descs)]
    for i in range(difficulty):
        k = rs.randint(2, 10)
        sym = ops[i % len(ops)][0]
        new = make_apply(i, current, k)
        clauses.append(f"{name} {sym} {k} {item}.")
        current = new
    if distractor:
        # Relevance-true NoOp (E1): the SAME subject, a DIFFERENT item, and no
        # operator symbol. Not separable by "different person"; the symbol-keyed
        # verifier still ignores it (no redefined operator to apply).
        oitem = rd.choice([it for it in ITEMS if it != item])
        clauses.insert(2, f"{name} keeps {rd.randint(2, 20)} {oitem} in a drawer.")
    prompt = " ".join(clauses) + f" How many {item} does {name} have now?"
    return prompt, str(current), "int", None



def _verify_redefined_ops(prompt, gold):
    """Re-derive the answer by parsing the redefined operators from the
    prompt text. Supports multiple distinct operator definitions
    (chained ops) and non-commutative definitions.
    """
    # Parse all "X means Y" operator definitions
    op_defs = {}  # sym -> callable
    for m in re.finditer(r"(\S) means (.*?)\.", prompt):
        sym, definition = m.group(1), m.group(2)
        if "add the two numbers then add" in definition:
            bias = int(re.search(r"add (\d+)", definition).group(1))
            op_defs[sym] = lambda a, b, bias=bias: a + b + bias
        elif "multiply the two numbers then subtract" in definition:
            bias = int(re.search(r"subtract (\d+)", definition).group(1))
            op_defs[sym] = lambda a, b, bias=bias: a * b - bias
        elif "double the sum of the two numbers, then add" in definition:
            bias = int(re.search(r"add (\d+)", definition).group(1))
            op_defs[sym] = lambda a, b, bias=bias: (a + b) * 2 + bias
        elif "subtract the right number from the left, then add" in definition:
            bias = int(re.search(r"add (\d+)", definition).group(1))
            op_defs[sym] = lambda a, b, bias=bias: a - b + bias
        elif "take the left number, divide by the right, then add" in definition:
            bias = int(re.search(r"add (\d+)", definition).group(1))
            def _div_op(a, b, bias=bias):
                if not b:
                    return a + bias
                return a // b + bias
            op_defs[sym] = _div_op
        elif "take the left modulo the right, then add" in definition:
            bias = int(re.search(r"add (\d+)", definition).group(1))
            def _mod_op(a, b, bias=bias):
                if not b:
                    return a + bias
                return a % b + bias
            op_defs[sym] = _mod_op
    if not op_defs:
        return False
    # Parse initial value
    mi = re.search(r"starts with (\d+)", prompt)
    if not mi:
        return False
    current = int(mi.group(1))
    sym_pattern = "|".join(re.escape(s) for s in op_defs)
    op_iter = re.finditer(rf"(\S+) ({sym_pattern}) (\d+) (\S+)", prompt)
    for m in op_iter:
        name, sym, k, _item = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        if sym not in op_defs:
            return False
        current = op_defs[sym](current, k)
    return str(current) == str(gold)





# ------------------------------------------------ 10. unsat_csp (premise-flaw)
# Build a KK or logic-grid puzzle, then with controlled probability make it
# ill-posed: drop a clue (→ multiple solutions → UNDETERMINED) or inject a
# contradiction (→ zero solutions → NO_SOLUTION). Otherwise keep unique.
def gen_unsat_csp(difficulty, structure_seed, surface_seed, distractor):
    rs = _rng("unsat-struct", difficulty, structure_seed)
    ru = _rng("unsat-surf", structure_seed, surface_seed)
    n = difficulty + 2
    slots = list(range(n))
    typ = {s: rs.random() < 0.5 for s in slots}

    # Build a unique KK puzzle (same as gen_knights).
    pool = []
    for x in slots:
        others = [o for o in slots if o != x]
        for y in others:
            pool.append(("ABS", x, y, typ[x] == typ[y]))
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                y, z = others[i], others[j]
                eq = (typ[y] == typ[z])
                pool.append(("REL", x, y, z, typ[x] == eq))
    rs.shuffle(pool)

    remaining = _kk_all_solutions(slots, [])
    chosen, changed = [], True
    while len(remaining) > 1 and changed:
        changed = False
        for st in pool:
            if st in chosen:
                continue
            nr = [t for t in remaining if _kk_consistent(st, t)]
            if len(nr) < len(remaining):
                chosen.append(st); remaining = nr; changed = True
                if len(remaining) == 1:
                    break
    chosen = _drop_redundant(chosen, lambda kept: len(_kk_all_solutions(slots, kept)) == 1)

    # Decide the class, aiming for a roughly uniform label prior over
    # {knight, knave, UNDETERMINED, NO_SOLUTION}: ~32% drop-a-clue (most queried
    # at a VARYING slot below -> UNDETERMINED ~25%, the rest at an invariant slot
    # -> determinate), ~24% contradiction (NO_SOLUTION), ~44% unique/
    # over-constrained (knight/knave). `chosen` is a minimal unique set, so
    # dropping any clue reliably yields multiple solutions.
    r = rs.random()
    if r < 0.32 and len(chosen) > 1:
        drop_idx = rs.randrange(len(chosen))
        ill_clues = chosen[:drop_idx] + chosen[drop_idx + 1:]
    elif r < 0.56:
        # Construct a statement that's FALSE under the true types.
        contradiction = None
        for x in slots:
            for y in slots:
                if x == y:
                    continue
                # "x says y is knight" — false when knight says knave or knave says knight
                want_knight = typ[y]
                st = ("ABS", x, y, not want_knight)  # opposite of truth
                if not _kk_consistent(st, typ):
                    contradiction = st
                    break
            if contradiction:
                break
        ill_clues = chosen + [contradiction] if contradiction else chosen
    elif r < 0.85:
        ill_clues = chosen
    else:
       # Over-constrained but still unique: add redundant clues that preserve
       # the unique solution until no more can be added.
       over = list(chosen)
       extra_candidates = [st for st in pool if st not in over]
       rs.shuffle(extra_candidates)
       for st in extra_candidates:
           if not _kk_consistent(st, typ):
               continue
           trial = over + [st]
           if len(_kk_all_solutions(slots, trial)) == 1:
               over.append(st)
       ill_clues = over

    # Compute gold from actual solution count — single source of truth.
    sols = _kk_all_solutions(slots, ill_clues)
    names = ru.sample(NAMES, n)
    # When several solutions remain, query a slot that VARIES across them so the
    # dropped-clue class is genuinely UNDETERMINED instead of collapsing to a
    # determinate knight/knave (which under-represented UNDETERMINED). With >= 2
    # distinct solutions at least one slot must vary.
    if len(sols) >= 2:
        varying = [s for s in slots if len({sol[s] for sol in sols}) > 1]
        invariant = [s for s in slots if s not in varying]
        # Mostly query a VARYING slot (-> UNDETERMINED) so the label prior stays
        # balanced, but ~20% of the time query an INVARIANT slot (-> determinate)
        # to preserve the "locally determinate despite global ambiguity" probe
        # (bench-le7.1/le7.2).
        if varying and (not invariant or rs.random() < 0.8):
            query_slot = rs.choice(varying)
        elif invariant:
            query_slot = rs.choice(invariant)
        else:
            query_slot = rs.randrange(n)
    else:
        query_slot = rs.randrange(n)
    if len(sols) == 0:
        gold = "NO_SOLUTION"
    elif len(sols) == 1:
        gold = "knight" if sols[0][query_slot] else "knave"
    else:
        values = {s[query_slot] for s in sols}
        gold = "UNDETERMINED" if len(values) > 1 else ("knight" if next(iter(values)) else "knave")

    ru.shuffle(ill_clues)
    sents = []
    for st in ill_clues:
        if st[0] == "ABS":
            sents.append(f"{names[st[1]]} says that {names[st[2]]} is a "
                         f"{'knight' if st[3] else 'knave'}.")
        else:
            sents.append(f"{names[st[1]]} says that {names[st[2]]} and {names[st[3]]} "
                         f"are {'the same type' if st[4] else 'different types'}.")
    prompt = ("On an island, every inhabitant is either a knight (who always tells the "
              "truth) or a knave (who always lies). Its inhabitants are "
              + ", ".join(names) + ". They say:\n" + "\n".join(sents)
              + f"\nIs {names[query_slot]} a knight or a knave?")
    return prompt, gold, "choice", ["knight", "knave", "UNDETERMINED", "NO_SOLUTION"]


def _verify_unsat_csp(prompt, gold):
    """Re-derive solution set from prompt text and check gold matches the queried slot."""
    names, stmts = _kk_parse(prompt)
    if not names:
        return False
    sols = _kk_all_solutions(names, stmts)
    mq = re.search(r"Is (\w+) a knight or a knave\?", prompt)
    if not mq:
        return False
    query_name = mq.group(1)
    if len(sols) == 0:
        return gold == "NO_SOLUTION"
    values = {s.get(query_name) for s in sols}
    if len(values) == 1:
        expected = "knight" if next(iter(values)) else "knave"
        return gold == expected
    return gold == "UNDETERMINED"


# ------------------------------------------------ 11. dynamic_pivot (true backtracking)
# Genuine commit-then-backtrack (E3/H3). Turn 1 asks for the count under the literal
# reading -- the model COMMITS the sub-gold WITHOUT yet seeing the pivot. Turn 2 reveals
# that every "moved" operation did not actually happen (the items stayed at the source
# and the destination never received them), forcing the model to revise its committed
# answer. Gold is the revised count; subgold is the committed (literal) count. This is
# the one family that requires genuine sequential multi-turn execution (the runner calls
# the model per turn) -- in a single shot a model would just read the pivot first and
# never commit, which is exactly the E3 weakness of retroactive_edit.
def gen_dynamic_pivot(difficulty, structure_seed, surface_seed, distractor):
    rs = _rng("pivot-struct", difficulty, structure_seed)
    ru = _rng("pivot-surf", structure_seed, surface_seed)
    item = ru.choice(ITEMS)
    names = ru.sample(CONTAINERS, 3)
    literal = [rs.randint(5, 20) for _ in range(3)]   # moves relocate (the literal reading)
    pivot = list(literal)                             # moves never happened (the revision)
    move_delta = [0, 0, 0]                            # literal-only net move effect per container
    clauses = [f"{names[i].capitalize()} has {literal[i]} {item}." for i in range(3)]
    for _ in range(max(2, difficulty)):
        op = rs.choice(["add", "remove", "move", "move"])   # bias toward moves so the pivot bites
        i = rs.randrange(3)
        # A remove must stay valid under BOTH readings so neither count goes negative.
        max_rm = min(literal[i], pivot[i])
        if op == "remove" and max_rm < 1:
            op = "add"
        if op == "move" and literal[i] < 1:
            op = "add"
        if op == "add":
            k = rs.randint(2, 12); literal[i] += k; pivot[i] += k
            clauses.append(f"{k} {item} are added to {names[i]}.")
        elif op == "remove":
            k = rs.randint(1, max_rm); literal[i] -= k; pivot[i] -= k
            clauses.append(f"{k} {item} are removed from {names[i]}.")
        else:  # move i -> j: literal relocates; pivot leaves both untouched
            j = rs.choice([x for x in range(3) if x != i])
            k = rs.randint(1, literal[i])
            literal[i] -= k; literal[j] += k
            move_delta[i] -= k; move_delta[j] += k
            clauses.append(f"{k} {item} are moved from {names[i]} to {names[j]}.")
    # Query a container whose count DIFFERS between the two readings, so the pivot is
    # load-bearing (gold != subgold). If moves happened to cancel everywhere, force one.
    candidates = [i for i in range(3) if move_delta[i] != 0]
    if not candidates:
        i, j = rs.sample(range(3), 2)
        k = rs.randint(1, max(1, literal[i]))
        literal[i] -= k; literal[j] += k; move_delta[j] += k
        clauses.append(f"{k} {item} are moved from {names[i]} to {names[j]}.")
        candidates = [j]
    qi = rs.choice(candidates)
    subgold = str(literal[qi])
    gold = str(pivot[qi])
    turn1 = " ".join(clauses) + f" How many {item} are in {names[qi]} now?"
    turn2 = (f"Now reconsider the whole sequence: every time {item} were described as "
             f"'moved' from one place to another, the move did not actually happen — the "
             f"{item} stayed where they were and the destination never received them. "
             f"With that correction, how many {item} are in {names[qi]} now?")
    prompt = turn1 + " " + turn2
    return prompt, gold, "int", None, [turn1, turn2], subgold


def _replay_state(prompt, item, ignore_moves):
    """Re-derive container counts for `item` from prompt text. With ignore_moves,
    'moved' operations are treated as no-ops (the dynamic_pivot revision)."""
    state = {}
    for s in _sentences(prompt):
        m = re.match(r"(.+?) has (\d+) (\w+)", s)
        if m and " are " not in s and m.group(3) == item:
            state[m.group(1).lower()] = int(m.group(2))
    for s in _sentences(prompt):
        if (m := re.match(r"(\d+) (\w+) are added to (.+?)\.", s)) and m.group(2) == item:
            c = m.group(3).lower(); state[c] = state.get(c, 0) + int(m.group(1))
        elif (m := re.match(r"(\d+) (\w+) are removed from (.+?)\.", s)) and m.group(2) == item:
            c = m.group(3).lower(); state[c] = state.get(c, 0) - int(m.group(1))
        elif (m := re.match(r"(\d+) (\w+) are moved from (.+?) to (.+?)\.", s)) and m.group(2) == item:
            if ignore_moves:
                continue
            k = int(m.group(1))
            state[m.group(3).lower()] = state.get(m.group(3).lower(), 0) - k
            state[m.group(4).lower()] = state.get(m.group(4).lower(), 0) + k
    return state


def _verify_dynamic_pivot(prompt, gold):
    # The primary gold is the REVISED count: moves never happened.
    mq = re.search(r"How many (.+?) are in (.+?) now\?", prompt)
    if not mq:
        return False
    item, qc = mq.group(1), mq.group(2).lower()
    state = _replay_state(prompt, item, ignore_moves=True)
    return str(state.get(qc)) == str(gold)


# ------------------------------------------------ 12. false_lemma (premise-flaw trap)
# A state-tracking total where a plausible-but-FALSE "helpful note" claims the total
# is conserved ("every step only relocates items"). It is false because the sequence
# also adds and/or removes items. A model that trusts the lemma answers the initial
# total; the correct answer must reject the premise and re-sum. (H7 / Phase-2 epistemic
# premise-flaw vector.)
def gen_false_lemma(difficulty, structure_seed, surface_seed, distractor):
    rs = _rng("lemma-struct", difficulty, structure_seed)
    ru = _rng("lemma-surf", structure_seed, surface_seed)
    item = ru.choice(ITEMS)
    names = ru.sample(CONTAINERS, 3)
    state = [rs.randint(5, 20) for _ in range(3)]
    total0 = sum(state)
    clauses = [f"{names[i].capitalize()} has {state[i]} {item}." for i in range(3)]
    net = 0                                        # change to the TOTAL (moves are net-zero)
    for _ in range(max(2, difficulty)):
        op = rs.choice(["add", "remove", "move"])
        i = rs.randrange(3)
        if op == "add" or (op == "remove" and state[i] <= 1):
            k = rs.randint(2, 12); state[i] += k; net += k
            clauses.append(f"{k} {item} are added to {names[i]}.")
        elif op == "remove":
            k = rs.randint(1, state[i]); state[i] -= k; net -= k
            clauses.append(f"{k} {item} are removed from {names[i]}.")
        else:
            j = rs.choice([x for x in range(3) if x != i])
            if state[i] <= 0:
                k = rs.randint(2, 12); state[i] += k; net += k
                clauses.append(f"{k} {item} are added to {names[i]}.")
            else:
                k = rs.randint(1, state[i]); state[i] -= k; state[j] += k
                clauses.append(f"{k} {item} are moved from {names[i]} to {names[j]}.")
    if net == 0:                                   # ensure the false lemma is load-bearing
        i = rs.randrange(3); k = rs.randint(2, 12); state[i] += k; net += k
        clauses.append(f"{k} {item} are added to {names[i]}.")
    total_final = total0 + net
    clauses.append(f"Note: every step above simply relocates {item} between the "
                   f"containers, so the total number of {item} never changes; it "
                   f"stays {total0}.")
    prompt = (" ".join(clauses) +
              f" How many {item} are there in total across all three containers now?")
    return prompt, str(total_final), "int", None


def _verify_false_lemma(prompt, gold):
    mq = re.search(r"How many (\w+) are there in total", prompt)
    if not mq:
        return False
    item = mq.group(1)
    total = 0
    for s in _sentences(prompt):
        m = re.match(r"(.+?) has (\d+) " + re.escape(item) + r"\.", s)
        if m and " are " not in s:
            total += int(m.group(2))
    for s in _sentences(prompt):                   # the 'Note:' lemma is prose, not an op
        if (m := re.match(r"(\d+) " + re.escape(item) + r" are added to ", s)):
            total += int(m.group(1))
        elif (m := re.match(r"(\d+) " + re.escape(item) + r" are removed from ", s)):
            total -= int(m.group(1))
        # moves leave the total unchanged
    return str(total) == str(gold)


# ------------------------------------------------ 13. noise_haystack (high-similarity distractors)
# A real arithmetic problem about `subj`, buried under several COMPLETE,
# structurally-identical arithmetic chains about OTHER people with the SAME item
# (maximal embedding similarity, zero parse overlap once the subject is bound). The
# model must locate the queried subject's chain among near-identical decoys. (H8 /
# Phase-2 semantic-noise vector.)
def gen_noise_haystack(difficulty, structure_seed, surface_seed, distractor):
    rd = _rng("haystack-distract", difficulty, structure_seed, surface_seed)
    core_prompt, gold, _at, _ch = gen_arithmetic(difficulty, structure_seed, surface_seed, False)
    subj = re.search(r"does (.+?) have now\?", core_prompt).group(1)
    item = re.search(r"How many (\w+) does", core_prompt).group(1)
    others = [n for n in NAMES if n != subj]
    rd.shuffle(others)
    decoys = []
    for d in range(3 + difficulty):                # decoys scale with difficulty
        oname = others[d % len(others)]
        cur = rd.randint(3, 20)
        decoys.append(f"{oname} starts with {cur} {item}.")
        for _ in range(2 + rd.randrange(3)):
            verb = rd.choice(["buys", "finds", "is given", "picks up"])
            decoys.append(f"{oname} {verb} {rd.randint(2, 15)} more {item}.")
    rd.shuffle(decoys)                             # scatter so the needle isn't first/last
    prompt = " ".join(decoys) + " " + core_prompt
    return prompt, gold, "int", None


GENERATORS = {
    "arithmetic": gen_arithmetic,
    "state_tracking": gen_state,
    "retroactive_edit": gen_retroactive_edit,
    "multi_turn_inject": gen_multi_turn_inject,
    "ordering": gen_order,
    "sequences": gen_sequence,
    "knights_knaves": gen_knights,
    "logic_grid": gen_logic_grid,
    "composed": gen_composed,
    "redefined_ops": gen_redefined_ops,
    "unsat_csp": gen_unsat_csp,
    "dynamic_pivot": gen_dynamic_pivot,
    "false_lemma": gen_false_lemma,
    "noise_haystack": gen_noise_haystack,
}
SUPPORTS_DISTRACTOR = {"arithmetic", "state_tracking", "ordering", "retroactive_edit",
                       "redefined_ops"}
# The CSP families pick their structure in slot space and only label it from the
# surface rng, so renaming is a true cosmetic perturbation with the gold held fixed.
SUPPORTS_SURFACE = {"arithmetic", "state_tracking", "ordering", "retroactive_edit",
                    "knights_knaves", "logic_grid", "redefined_ops", "unsat_csp",
                    "dynamic_pivot"}

# Families that require GENUINE sequential multi-turn execution: the model must
# commit an answer to turn 1 BEFORE it sees the turn-2 pivot. Items carry a subgold
# (the committed turn-1 answer) graded separately from the final gold.
REQUIRES_SEQUENTIAL_TURNS = {"dynamic_pivot"}

# For most families difficulty == number of reasoning steps and is open-ended.
# Some families select a discrete tier / a brute-forced structure instead, where a
# higher difficulty would either silently reuse the top tier (sequences) or blow up
# the gold-verification search (the CSP families). Cap those rather than emit
# "difficulties" that are not actually harder or not feasible to verify.
#   sequences      : rule-complexity tier 1..6
FAMILY_MAX_DIFF = {"sequences": 6, "knights_knaves": 6, "logic_grid": 5, "composed": 5,
                    "unsat_csp": 6}
# difficulty == number of update operations before the pivot, so it is a depth axis.

# What the `difficulty` integer MEANS per family (bench-lop / E6). For most
# families it is the number of reasoning steps, so the degradation curve is a true
# depth axis. For a few it is NOT commensurable with steps: `sequences` selects a
# rule TIER (d1=AP .. d6=cubic -- a different rule, not more steps) and the CSP
# families scale the problem SIZE n. Reporting uses this so the per-family x-axis
# is labelled honestly and the curves are not read as one shared "more steps" axis.
DIFFICULTY_AXIS = {
    "sequences": "rule tier",
    "knights_knaves": "islanders (n)",
    "logic_grid": "floors (n)",
    "unsat_csp": "islanders (n)",
    "composed": "chain / per-hop difficulty",
}


def difficulty_axis(family: str) -> str:
    """Human-readable name of a family's difficulty axis (default: reasoning steps)."""
    return DIFFICULTY_AXIS.get(family, "reasoning steps")


@dataclass
class Problem:
    item_id: str
    family: str
    difficulty: int
    structure_seed: int
    surface_seed: int
    has_distractor: bool
    probe: str           # base | distractor | surface
    grp: str             # links matched items (base/distractor/surface share a grp)
    answer_type: str
    gold: str
    choices: Optional[List[str]]
    prompt: str
    turns: Optional[List[str]] = None
    subgold: Optional[str] = None        # intermediate (pre-pivot) gold for backtracking families

    def row(self):
        d = asdict(self)
        d["choices"] = "|".join(self.choices) if self.choices else ""
        d["has_distractor"] = int(self.has_distractor)
        d["turns"] = "|".join(self.turns) if self.turns else ""
        d["subgold"] = self.subgold or ""
        return d


def _mk(family, difficulty, structure_seed, surface_seed, distractor, probe, grp):
    result = GENERATORS[family](difficulty, structure_seed, surface_seed, distractor)
    subgold = None
    if len(result) == 6:                 # (prompt, gold, atype, choices, turns, subgold)
        prompt, gold, atype, choices, turns, subgold = result
    elif len(result) == 5:
        prompt, gold, atype, choices, turns = result
    else:
        prompt, gold, atype, choices = result
        turns = None
    iid = hashlib.sha1(f"{family}|{difficulty}|{structure_seed}|{surface_seed}|{distractor}|{probe}".encode()).hexdigest()[:16]
    return Problem(iid, family, difficulty, structure_seed, surface_seed, distractor,
                   probe, grp, atype, gold, choices, prompt, turns, subgold)


# ---------------------------------------------------- independent gold verifiers
# These recompute the answer from the PROMPT TEXT (not the generator's internal
# state), so they are a genuinely independent check: they catch both arithmetic
# slips and prompt/gold desync. build_dataset runs them on every item by default.

def _sentences(prompt):
    return re.split(r"(?<=\.)\s+", prompt)


def _verify_arithmetic(prompt, gold):
    mq = re.search(r"How many (.+?) does (.+?) have now\?", prompt)
    if not mq:
        return False
    item, subj, total = mq.group(1), mq.group(2), None
    for s in _sentences(prompt):
        if not s.startswith(subj + " "):
            continue
        if not re.search(r"\b" + re.escape(item) + r"\b", s):
            continue                                   # different-item NoOp distractor
        rest = s[len(subj) + 1:]
        if (m := re.match(r"starts with (\d+)", rest)):       total = int(m.group(1))
        elif total is None:                                    continue
        elif (m := re.match(r"(?:buys|finds|is given|picks up) (\d+) more", rest)): total += int(m.group(1))
        elif (m := re.match(r"(?:gives away|loses|uses|drops) (\d+)", rest)):       total -= int(m.group(1))
        elif rest.startswith("doubles"):                       total *= 2
        elif rest.startswith("triples"):                       total *= 3
        elif (m := re.match(r"divides the .+? into (\d+) equal groups and keeps one group", rest)): total //= int(m.group(1))
    return total is not None and str(total) == str(gold)


def _verify_state(prompt, gold):
    mq = re.search(r"How many (.+?) are in (.+?) now\?", prompt)
    if not mq:
        return False
    item, qc = mq.group(1), mq.group(2).lower()
    state = {}
    for s in _sentences(prompt):
        m = re.match(r"(.+?) has (\d+) (\w+)", s)
        if m and " are " not in s and m.group(3) == item:
            state[m.group(1).lower()] = int(m.group(2))
    for s in _sentences(prompt):
        # Bind every update to the queried item: a different-item NoOp distractor
        # to a tracked container must not move the count.
        if (m := re.match(r"(\d+) (\w+) are added to (.+?)\.", s)) and m.group(2) == item:
            state[m.group(3).lower()] = state.get(m.group(3).lower(), 0) + int(m.group(1))
        elif (m := re.match(r"(\d+) (\w+) are removed from (.+?)\.", s)) and m.group(2) == item:
            state[m.group(3).lower()] = state.get(m.group(3).lower(), 0) - int(m.group(1))
        elif (m := re.match(r"(\d+) (\w+) are moved from (.+?) to (.+?)\.", s)) and m.group(2) == item:
            k = int(m.group(1))
            state[m.group(3).lower()] = state.get(m.group(3).lower(), 0) - k
            state[m.group(4).lower()] = state.get(m.group(4).lower(), 0) + k
    return str(state.get(qc)) == str(gold)


def _verify_order(prompt, gold):
    """Transitive resolution: topo-order the chain, index the asked rank.

    The relation family is bound to the question's superlative, so a comparative
    on an ORTHOGONAL family (the E2 distractor, e.g. "richer" in a height
    ordering) contributes no edge and is correctly ignored.
    """
    from collections import defaultdict, deque
    # Parse rank query first: "Who is the Nth tallest?" -> resolve the relation
    # family whose superlative-high is the asked word, and only read that family.
    mq = re.search(r"Who is the (\w+) (\w+)\?", prompt)
    if not mq:
        return False
    rank_word, sup = mq.group(1), mq.group(2)
    tup = next((r for r in REL if r[2] == sup), None)
    if tup is None:
        return False
    hi_adj, lo_adj = tup[0], tup[1]
    edges = defaultdict(set)   # edges[a] = {b} means a > b
    names = set()
    for m in re.finditer(r"(\w[\w']*) is (\w+) than (\w[\w']*)\.", prompt):
        a, rel, b = m.group(1), m.group(2), m.group(3)
        if rel == hi_adj:
            edges[a].add(b)
        elif rel == lo_adj:
            edges[b].add(a)
        else:
            continue                                   # off-axis comparative: inert
        names.update((a, b))
    # Convert ordinal to 0-indexed rank
    ordinals = {"1st": 0, "2nd": 1, "3rd": 2, "4th": 3, "5th": 4, "6th": 5, "7th": 6, "8th": 7}
    rank = ordinals.get(rank_word, None)
    if rank is None:
        return False
    # Topo-sort via Kahn's algorithm
    in_degree = {n: 0 for n in names}
    for a in edges:
        for b in edges[a]:
            in_degree[b] = in_degree.get(b, 0) + 1
    queue = deque(n for n in names if in_degree[n] == 0)
    order = []
    while queue:
        if len(queue) > 1:
            # Ambiguous: multiple nodes with in-degree 0
            return False
        node = queue.popleft()
        order.append(node)
        for child in edges.get(node, set()):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    if len(order) != len(names):
        return False  # cycle
    return rank < len(order) and order[rank] == gold


def _verify_sequence(prompt, gold):
    """Independent verifier: predict the next term from the simplest fitting
    rule and REJECT genuinely ambiguous sequences.

    It collects a candidate next-term from every independent rule that fully
    fits the observed terms -- lowest-degree polynomial (finite differences),
    geometric, fibonacci-like, and interleaved two-APs. If more than one
    *distinct* next-term is predicted, the sequence is ambiguous and rejected
    (this is what the README's "unique next term" guarantee requires, and what
    the old hollow `pass` gate never enforced). Shares no rule ladder with the
    generator.
    """
    m = re.search(r"Sequence:\s*(.+?),\s*\.\.\.", prompt)
    if not m:
        return False
    terms = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    if len(terms) < 4:                       # too few terms to pin any rule
        return False

    candidates = set()

    # (1) Lowest-degree polynomial via finite differences. Require the constant
    # level to be confirmed by >= 3 differences: a high-degree fit "confirmed"
    # by only 2 equal differences is a coincidence (e.g. an interleaved sequence
    # whose 6th differences happen to repeat), not a real polynomial. Every
    # generator polynomial (AP/quadratic/cubic) has >= 4 confirming differences.
    diffs = [terms[:]]
    while len(diffs[-1]) > 1:
        prev = diffs[-1]
        diffs.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])
    for deg, d in enumerate(diffs):
        if len(d) >= 3 and len(set(d)) == 1:
            candidates.add(sum(level[-1] for level in diffs[:deg + 1]))
            break

    # (2) Geometric: constant integer ratio.
    if all(t != 0 for t in terms) and terms[1] % terms[0] == 0:
        r0 = terms[1] // terms[0]
        if r0 not in (0, 1) and all(terms[i + 1] == terms[i] * r0
                                    for i in range(len(terms) - 1)):
            candidates.add(terms[-1] * r0)

    # (3) Fibonacci-like: each term is the sum of the two preceding.
    if all(terms[i] == terms[i - 1] + terms[i - 2] for i in range(2, len(terms))):
        candidates.add(terms[-1] + terms[-2])

    # (4) Interleaved two APs: even- and odd-index subsequences are each APs.
    even, odd = terms[0::2], terms[1::2]
    if len(even) >= 2 and len(odd) >= 2:
        de = {even[i + 1] - even[i] for i in range(len(even) - 1)}
        do = {odd[i + 1] - odd[i] for i in range(len(odd) - 1)}
        if len(de) == 1 and len(do) == 1:
            de0, do0 = next(iter(de)), next(iter(do))
            candidates.add(even[-1] + de0 if len(terms) % 2 == 0 else odd[-1] + do0)

    # Unambiguous iff exactly one distinct prediction, and it matches the gold.
    if len(candidates) != 1:
        return False
    return str(candidates.pop()) == str(gold)


def _verify_retroactive_edit(prompt, gold):
    # Replay the state-tracking problem; the "Actually..." clause overrides
    # the initial value of the referenced container.
    state = {}
    edit_m = re.search(r"actually, ([\w ]+?) held (\d+) times as many (\w+) as originally stated", prompt, re.I)
    if not edit_m:
        return False
    edit_name = edit_m.group(1).strip().lower()
    factor = int(edit_m.group(2))
    item = edit_m.group(3)
    for s in _sentences(prompt):
        m = re.match(r"(.+?) has (\d+) " + re.escape(item) + r"\.", s)
        if m and " are " not in s:
            name = m.group(1).lower().strip()
            amount = int(m.group(2))
            if name == edit_name:
                amount *= factor
            state[name] = amount             # an untracked distractor container is never queried
    for s in _sentences(prompt):
        if (m := re.match(r"(\d+) " + re.escape(item) + r" are added to (.+?)\.", s)):
            state[m.group(2).lower().strip()] = state.get(m.group(2).lower().strip(), 0) + int(m.group(1))
        elif (m := re.match(r"(\d+) " + re.escape(item) + r" are removed from (.+?)\.", s)):
            state[m.group(2).lower().strip()] = state.get(m.group(2).lower().strip(), 0) - int(m.group(1))
        elif (m := re.match(r"(\d+) " + re.escape(item) + r" are moved from (.+?) to (.+?)\.", s)):
            k = int(m.group(1))
            state[m.group(2).lower().strip()] = state.get(m.group(2).lower().strip(), 0) - k
            state[m.group(3).lower().strip()] = state.get(m.group(3).lower().strip(), 0) + k
    mq = re.search(r"are in (.+?) now\?", prompt)
    if not mq:
        return False
    return str(state.get(mq.group(1).lower().strip())) == str(gold)
def _kk_parse(prompt):
    """Re-derive (names, statements) from a knights & knaves prompt's text."""
    m = re.search(r"inhabitants are (.+?)\. They say\b", prompt, re.S)
    names = [x.strip() for x in m.group(1).split(",")] if m else []
    stmts = []
    for mm in re.finditer(r"(\w+) says that (\w+) is a (knight|knave)\.", prompt):
        stmts.append(("ABS", mm.group(1), mm.group(2), mm.group(3) == "knight"))
    for mm in re.finditer(r"(\w+) says that (\w+) and (\w+) are "
                          r"(the same type|different types)\.", prompt):
        stmts.append(("REL", mm.group(1), mm.group(2), mm.group(3),
                      mm.group(4) == "the same type"))
    referenced = {s[1] for s in stmts} | {s[2] for s in stmts} \
        | {s[3] for s in stmts if s[0] == "REL"}
    for nm in referenced:                       # defensive: never miss a speaker
        if nm not in names:
            names.append(nm)
    return names, stmts


def _verify_knights(prompt, gold):
    names, stmts = _kk_parse(prompt)
    if not names or not stmts:
        return False
    sols = _kk_all_solutions(names, stmts)
    if len(sols) != 1:
        return False
    # Gold is the sorted, comma-separated list of knave names (2^n answer
    # space). Compare case-insensitively so a model that lowercases a
    # name still passes.
    knave_set = {n.lower() for n, is_knight in sols[0].items() if not is_knight}
    gold_set = {t.strip().lower() for t in gold.split(",") if t.strip()}
    return knave_set == gold_set



def _lg_parse(prompt):
    """Re-derive (names, n, clues, queried_name) from a logic-grid prompt's text."""
    mn = re.search(r"numbered 1 \(lowest\) to (\d+) \(highest\)", prompt)
    mp = re.search(r"lives on each floor: (.+?)\.", prompt, re.S)
    mq = re.search(r"On which floor does (\w+) live\?", prompt)
    if not (mn and mp and mq):
        return None
    n = int(mn.group(1))
    names = [x.strip() for x in mp.group(1).split(",")]
    clues = []
    for m in re.finditer(r"(\w+) lives on floor (\d+)\.", prompt):
        clues.append(("FLOOR", m.group(1), int(m.group(2))))
    for m in re.finditer(r"(\w+) lives (\d+) floors above (\w+)\.", prompt):
        clues.append(("ABOVEK", m.group(1), int(m.group(2)), m.group(3)))
    for m in re.finditer(r"(\w+) lives directly above (\w+)\.", prompt):
        clues.append(("DIR", m.group(1), m.group(2)))
    for m in re.finditer(r"(\w+) lives on a higher floor than (\w+)\.", prompt):
        clues.append(("HI", m.group(1), m.group(2)))
    for m in re.finditer(r"(\w+) and (\w+) live on adjacent floors\.", prompt):
        clues.append(("ADJ", m.group(1), m.group(2)))
    return names, n, clues, mq.group(1)


def _verify_logic_grid(prompt, gold):
    parsed = _lg_parse(prompt)
    if not parsed:
        return False
    names, n, clues, q = parsed
    if len(names) != n or not clues or q not in names:
        return False
    sols = _lg_solutions(names, n, clues)         # also enforces a UNIQUE solution
    return len(sols) == 1 and str(sols[0][q]) == str(gold)


def _verify_multi_turn_inject(prompt, gold):
    state = {}
    for s in _sentences(prompt):
        m = re.match(r"(.+?) has (\d+) ", s)
        if m and " are " not in s:
            state[m.group(1).lower()] = int(m.group(2))
    for s in _sentences(prompt):
        if (m := re.match(r"(\d+) \S+ are added to (.+?)\.", s)):
            state[m.group(2).lower()] = state.get(m.group(2).lower(), 0) + int(m.group(1))
        elif (m := re.match(r"(\d+) \S+ are removed from (.+?)\.", s)):
            state[m.group(2).lower()] = state.get(m.group(2).lower(), 0) - int(m.group(1))
        elif (m := re.match(r"(\d+) \S+ are moved from (.+?) to (.+?)\.", s)):
            k = int(m.group(1))
            state[m.group(2).lower()] = state.get(m.group(2).lower(), 0) - k
            state[m.group(3).lower()] = state.get(m.group(3).lower(), 0) + k
    mq = re.search(r"are in (.+?) now\?", prompt)
    if not mq:
        return False
    key = mq.group(1).lower()
    total = state.get(key)
    if total is None:
        return False
    if re.search(r"double the number of", prompt):
        total *= 2
    elif re.search(r"triple the number of", prompt):
        total *= 3
    return str(total) == str(gold)


_VERIFIERS = {
    "arithmetic": _verify_arithmetic,
    "state_tracking": _verify_state,
    "retroactive_edit": _verify_retroactive_edit,
    "multi_turn_inject": _verify_multi_turn_inject,
    "ordering": _verify_order,
    "sequences": _verify_sequence,
    "knights_knaves": _verify_knights,
    "logic_grid": _verify_logic_grid,
    "composed": _verify_composed,
    "redefined_ops": _verify_redefined_ops,
    "unsat_csp": _verify_unsat_csp,
    "dynamic_pivot": _verify_dynamic_pivot,
    "false_lemma": _verify_false_lemma,
    "noise_haystack": _verify_arithmetic,        # core is arithmetic; decoys are subject-inert
}

def verify_gold(p) -> bool:
    """Independently re-derive the answer from the prompt text and check it == gold.
    Accepts a Problem or a dataset-row dict."""
    fam = p["family"] if isinstance(p, dict) else p.family
    prompt = p["prompt"] if isinstance(p, dict) else p.prompt
    gold = p["gold"] if isinstance(p, dict) else p.gold
    fn = _VERIFIERS.get(fam)
    return fn(prompt, gold) if fn else True

def build_dataset(families, min_diff, max_diff, reps, with_distractor=False,
                  surface_variants=0, verify=True):
    """
    Build a list of Problem objects.

    reps             : how many distinct structures per (family, difficulty).
    with_distractor  : also emit a matched NoOp-distractor version of each base item.
    surface_variants : also emit this many cosmetic variants (same gold) per base item.
    verify           : independently re-verify every gold from the prompt text.
    """
    items, seen = [], set()
    for fam in families:
        if fam not in GENERATORS:
            raise ValueError(f"unknown family: {fam}")
        fam_max = min(max_diff, FAMILY_MAX_DIFF.get(fam, max_diff))
        for diff in range(min_diff, fam_max + 1):
            for r in range(reps):
                grp = f"{fam}-d{diff}-r{r}"
                base = _mk(fam, diff, r, 0, False, "base", grp)
                if base.item_id not in seen:
                    items.append(base); seen.add(base.item_id)
                if with_distractor and fam in SUPPORTS_DISTRACTOR:
                    di = _mk(fam, diff, r, 0, True, "distractor", grp)
                    if di.item_id not in seen:
                        items.append(di); seen.add(di.item_id)
                if surface_variants and fam in SUPPORTS_SURFACE:
                    for s in range(1, surface_variants + 1):
                        v = _mk(fam, diff, r, s, False, "surface", grp)
                        if v.item_id not in seen:
                            items.append(v); seen.add(v.item_id)
    if verify:
        for p in items:
            if not verify_gold(p):
                raise AssertionError(
                    f"gold re-verification failed for {p.family} d{p.difficulty} "
                    f"({p.probe}, id={p.item_id}): gold={p.gold!r}\n  {p.prompt}")
    return items


if __name__ == "__main__":
    # quick smoke test
    ds = build_dataset(list(GENERATORS), 1, 3, 2, with_distractor=True, surface_variants=2)
    print(f"generated {len(ds)} items\n")
    for p in ds[:6]:
        print(f"[{p.family} d{p.difficulty} {p.probe}] gold={p.gold}")
        print("  " + p.prompt + "\n")
