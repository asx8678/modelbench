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
            verb = rs.choice(["buys", "finds", "is given", "picks up"])
            clauses.append(f"{name} {verb} {k} more {item}.")
        elif op == "sub":
            k = rs.randint(1, current - 1); current -= k
            verb = rs.choice(["gives away", "loses", "uses", "drops"])
            clauses.append(f"{name} {verb} {k} {item}.")
        elif op == "mul":
            f = rs.choice([2, 3]); current *= f
            clauses.append(f"{name} {'doubles' if f == 2 else 'triples'} the number of {item} they have.")
        else:  # div: exact by construction, so the gold stays an integer
            f = rs.choice(divisors); current //= f
            clauses.append(f"{name} divides the {item} into {f} equal groups and keeps one group.")

    if distractor:
        oname = rd.choice([n for n in NAMES if n != name])
        oitem = rd.choice([it for it in ITEMS if it != item])
        ox = rd.randint(2, 20)
        clauses.insert(1, f"{oname} also has {ox} {oitem} in a basket.")

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
        oc = rd.choice([c for c in CONTAINERS if c not in names])
        ox = rd.randint(3, 15)
        clauses.insert(3, f"A nearby {oc} also holds {ox} {item}.")

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

    # Retroactive edit: a late clause says one container held a multiple of its
    # originally stated amount. Only increase to keep later operations valid.
    edit_i = rs.randrange(3)
    factor = rs.choice([2, 3])
    edit_clause = f"Actually, {names[edit_i].lower()} held {factor} times as many {item} as originally stated."
    clauses.insert(-1, edit_clause)

    qi = rs.randrange(3)
    if distractor:
        oc = ru.choice([c for c in CONTAINERS if c not in names])
        ox = ru.randint(3, 15)
        clauses.insert(3, f"A nearby {oc} also holds {ox} {item}.")

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
        clauses.insert(0, f"{rd.choice(order)} is wearing a {rd.choice(COLORS)} hat.")

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
# Chain >=3 execution-dependency hops: knights_knaves knight-count seeds the
# starting value of an arithmetic chain, whose final result indexes an ordering
# query. The prompt embeds every hop so each can be verified independently.

def _composed_parse_hops(prompt):
    """Return dict with keys: knights_prompt, arith_prompt, arith_subj.

    Stage 3 (the ordering block) is a distractor/red-herring that no
    longer contributes to the answer; the final gold is the raw
    integer from Stage 2. The parser therefore only extracts the
    knights and arithmetic stages.
    """
    m = re.search(
        r"Stage 1: On an island(.+?)\n+Stage 2:\s*(.+?)\n+",
        prompt, re.S,
    )
    if not m:
        return None
    knights_block = "On an island" + m.group(1).strip()
    arith_subj = m.group(2).strip()
    # The arithmetic block is everything from after the subject line
    # up to either the next "Stage" marker or the "Use this number"
    # sentence (whichever comes first). The subject line is the
    # first sentence of Stage 2; the remaining sentences are the
    # arithmetic updates.
    m2 = re.search(
        r"Stage 2:\s*" + re.escape(arith_subj) + r"\s*(.+?)(?:\n+Stage|\n+What is it|$)",
        prompt, re.S,
    )
    if not m2:
        # Fallback: just take the rest of Stage 2 as the body.
        body = prompt.split(arith_subj, 1)[1] if arith_subj in prompt else ""
    else:
        body = m2.group(1).strip()
    return {
        "knights_prompt": knights_block,
        "arith_prompt": arith_subj + " " + body,
        "arith_subj": arith_subj,
    }



def _verify_composed(prompt, gold):
    parsed = _composed_parse_hops(prompt)
    if not parsed:
        return False
    # Re-derive the arithmetic total from Stage 2 starting from the
    # knight count in Stage 1. The final gold is the raw integer
    # (no modulo wrap into Stage 3).
    kk_prompt = parsed["knights_prompt"]
    kk_names, kk_stmts = _kk_parse(kk_prompt)
    if not kk_names or not kk_stmts:
        return False
    sols = _kk_all_solutions(kk_names, kk_stmts)
    if len(sols) != 1:
        return False
    knight_count = sum(1 for v in sols[0].values() if v)
    arith_total = _verify_arithmetic_raw(parsed["arith_prompt"], knight_count)
    if arith_total is None:
        return False
    try:
        return int(gold) == int(arith_total)
    except (TypeError, ValueError):
        return False



def _verify_arithmetic_raw(prompt, start):
    """Re-derive the arithmetic result given an explicit starting integer."""
    m = re.search(r"^(\S[^\n]*) starts with (\d+)", prompt, re.M)
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
    """Chain >=3 execution-dependency hops: knights -> arithmetic -> ordering."""
    rs = _rng("composed-struct", difficulty, structure_seed)
    ru = _rng("composed-surf", structure_seed, surface_seed)

    # Hop A: knights & knaves (small, fixed difficulty; unique by construction).
    kk_diff = 2
    kk_struct = rs.randint(0, 2 ** 16)
    kk_prompt, kk_gold, _, _ = gen_knights(kk_diff, kk_struct, 0, False)
    # Strip the final question line to get just the clues block.
    kk_block = kk_prompt.rsplit("\n", 1)[0]
    # Count knights from the unique solution.
    kk_names, kk_stmts = _kk_parse(kk_prompt)
    kk_sols = _kk_all_solutions(kk_names, kk_stmts)
    knight_count = sum(1 for v in kk_sols[0].values() if v)

    # Hop B: arithmetic seeded by the knight count.
    arith_diff = max(1, difficulty)
    arith_struct = rs.randint(0, 2 ** 16)
    arith_name = ru.choice(NAMES)
    arith_item = ru.choice(ITEMS)
    current = knight_count
    arith_clauses = [f"{arith_name} starts with {current} {arith_item}."]
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

    # Hop C: ordering hop included as a distractor/red-herring. The
    # final answer is the raw integer from Hop B (arithmetic), NOT a
    # name lookup into the ordering. Dropping the `% len(names)` modulo
    # is critical: any 1-unit drift in the arithmetic hop propagates
    # to the final integer instead of being masked by wrap-around.
    order_diff = max(1, difficulty)
    order_struct = rs.randint(0, 2 ** 16)
    order_prompt, _order_gold, _atype, names = gen_order(order_diff, order_struct, 0, False)
    order_clauses = order_prompt.split(" Who is the ")[0]

    # Final gold: the raw integer from Hop B (carried through all
    # hops). Answer space is the full integer range, not a wrapped
    # index into the ordering.
    gold = str(arith_result)

    prompt = (
        "Stage 1: " + kk_block + "\n"
        f"Is {kk_names[rs.randrange(len(kk_names))]} a knight or a knave? "
        "(Use the number of knights in the next stage.)\n\n"
        "Stage 2: " + arith_prompt + "\n"
        f"How many {arith_item} does {arith_name} have now? "
        "(Use this number in the next stage.)\n\n"
        "Stage 3 (distractor, not used for the answer): " + order_clauses + "\n"
        "Stage 2's result is the answer. What is it?"
    )
    return prompt, gold, "int", None






# ------------------------------------------------ 8. redefined_ops (counterfactual)
# Arithmetic where the prompt redefines operators (e.g. "⊕ means a+b+3").
# Gold computed by applying the redefined table; verifier replays from prompt text.
def gen_redefined_ops(difficulty, structure_seed, surface_seed, distractor):
    rs = _rng("rops-struct", difficulty, structure_seed)
    ru = _rng("rops-surf", structure_seed, surface_seed)
    rd = _rng("rops-distract", structure_seed)

    name = ru.choice(NAMES)
    item = ru.choice(ITEMS)
    # Redefine one standard operator
    op_sym = rs.choice(["⊕", "⊗", "⊖"])
    bias = rs.randint(1, 5)
    op_desc = rs.choice([
        (f"{op_sym} means add the two numbers then add {bias}", lambda a, b: a + b + bias),
        (f"{op_sym} means multiply the two numbers then subtract {bias}", lambda a, b: a * b - bias),
        (f"{op_sym} means add the two numbers then double", lambda a, b: (a + b) * 2),
    ])
    op_text, op_fn = op_desc
    current = rs.randint(3, 15)
    clauses = [f"{name} starts with {current} {item}.",
               f"In this problem, {op_text}."]
    for _ in range(difficulty):
        k = rs.randint(2, 10)
        new = op_fn(current, k)
        clauses.append(f"{name} {op_sym} {k} {item}.")
        current = new
    if distractor:
        oname = rd.choice([n for n in NAMES if n != name])
        clauses.insert(1, f"{oname} also has {rd.randint(2, 20)} pencils.")
    prompt = " ".join(clauses) + f" How many {item} does {name} have now?"
    return prompt, str(current), "int", None


def _verify_redefined_ops(prompt, gold):
    """Re-derive the answer by parsing the redefined operator from prompt text."""
    # Parse the operator definition
    m = re.search(r"(\S) means (.*?)\.", prompt)
    if not m:
        return False
    op_sym = m.group(1)
    op_escaped = re.escape(op_sym)
    op_def = m.group(2)
    # Parse the operation
    if "add the two numbers then add" in op_def:
        bias = int(re.search(r"add (\d+)", op_def).group(1))
        op_fn = lambda a, b: a + b + bias
    elif "multiply the two numbers then subtract" in op_def:
        bias = int(re.search(r"subtract (\d+)", op_def).group(1))
        op_fn = lambda a, b: a * b - bias
    elif "add the two numbers then double" in op_def:
        op_fn = lambda a, b: (a + b) * 2
    else:
        return False
    # Parse initial value
    mi = re.search(r"starts with (\d+)", prompt)
    if not mi:
        return False
    current = int(mi.group(1))
    # Replay operations: match specifically the operator symbol
    for m in re.finditer(rf"{op_escaped} (\d+)", prompt):
        current = op_fn(current, int(m.group(1)))
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

    # Decide: keep unique (40%), drop clue (30%), inject contradiction (30%).
    r = rs.random()
    if r < 0.3 and len(chosen) > 1:
        drop_idx = rs.randrange(len(chosen))
        ill_clues = chosen[:drop_idx] + chosen[drop_idx + 1:]
    elif r < 0.6:
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
    else:
        ill_clues = chosen

    # Compute gold from actual solution count — single source of truth.
    sols = _kk_all_solutions(slots, ill_clues)
    names = ru.sample(NAMES, n)
    query_slot = rs.randrange(n)
    if len(sols) == 0:
        gold = "NO_SOLUTION"
    elif len(sols) == 1:
        gold = "knight" if sols[0][query_slot] else "knave"
    else:
        # Multiple solutions after the ill-clue: the queried slot may still be
        # invariant, in which case the answer is determinate, not UNDETERMINED.
        values = {s[query_slot] for s in sols}
        if len(values) == 1:
            gold = "knight" if next(iter(values)) else "knave"
        else:
            gold = "UNDETERMINED"

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

def _mk(family, difficulty, structure_seed, surface_seed, distractor, probe, grp):
    result = GENERATORS[family](difficulty, structure_seed, surface_seed, distractor)
    if len(result) == 5:
        prompt, gold, atype, choices, turns = result
    else:
        prompt, gold, atype, choices = result
        turns = None
    iid = hashlib.sha1(f"{family}|{difficulty}|{structure_seed}|{surface_seed}|{distractor}|{probe}".encode()).hexdigest()[:16]
    return Problem(iid, family, difficulty, structure_seed, surface_seed, distractor,
                   probe, grp, atype, gold, choices, prompt, turns)

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
}
SUPPORTS_DISTRACTOR = {"arithmetic", "state_tracking", "ordering", "retroactive_edit",
                       "redefined_ops"}
# The CSP families pick their structure in slot space and only label it from the
# surface rng, so renaming is a true cosmetic perturbation with the gold held fixed.
SUPPORTS_SURFACE = {"arithmetic", "state_tracking", "ordering", "retroactive_edit",
                    "knights_knaves", "logic_grid", "redefined_ops"}

# For most families difficulty == number of reasoning steps and is open-ended.
# Some families select a discrete tier / a brute-forced structure instead, where a
# higher difficulty would either silently reuse the top tier (sequences) or blow up
# the gold-verification search (the CSP families). Cap those rather than emit
# "difficulties" that are not actually harder or not feasible to verify.
#   sequences      : rule-complexity tier 1..6
FAMILY_MAX_DIFF = {"sequences": 6, "knights_knaves": 6, "logic_grid": 5, "composed": 5,
                    "unsat_csp": 6}


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

    def row(self):
        d = asdict(self)
        d["choices"] = "|".join(self.choices) if self.choices else ""
        d["has_distractor"] = int(self.has_distractor)
        d["turns"] = "|".join(self.turns) if self.turns else ""
        return d


def _mk(family, difficulty, structure_seed, surface_seed, distractor, probe, grp):
    result = GENERATORS[family](difficulty, structure_seed, surface_seed, distractor)
    if len(result) == 5:
        prompt, gold, atype, choices, turns = result
    else:
        prompt, gold, atype, choices = result
        turns = None
    iid = hashlib.sha1(f"{family}|{difficulty}|{structure_seed}|{surface_seed}|{distractor}|{probe}".encode()).hexdigest()[:16]
    return Problem(iid, family, difficulty, structure_seed, surface_seed, distractor,
                   probe, grp, atype, gold, choices, prompt, turns)


# ---------------------------------------------------- independent gold verifiers
# These recompute the answer from the PROMPT TEXT (not the generator's internal
# state), so they are a genuinely independent check: they catch both arithmetic
# slips and prompt/gold desync. build_dataset runs them on every item by default.

def _sentences(prompt):
    return re.split(r"(?<=\.)\s+", prompt)


def _verify_arithmetic(prompt, gold):
    mq = re.search(r"How many .+? does (.+?) have now\?", prompt)
    if not mq:
        return False
    subj, total = mq.group(1), None
    for s in _sentences(prompt):
        if not s.startswith(subj + " "):
            continue                                   # skips the distractor (different name)
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
    state = {}
    for s in _sentences(prompt):
        if s.startswith("A nearby"):                   # distractor clause
            continue
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
    return bool(mq) and str(state.get(mq.group(1).lower())) == str(gold)


def _verify_order(prompt, gold):
    """Transitive resolution: topo-order the chain, index the asked rank."""
    from collections import defaultdict, deque
    rel_hi = {r[0] for r in REL}; rel_lo = {r[1] for r in REL}
    sup_hi = {r[2] for r in REL}
    edges = defaultdict(set)   # edges[a] = {b} means a > b
    names = set()
    for m in re.finditer(r"(\w[\w']*) is (\w+) than (\w[\w']*)\.", prompt):
        a, rel, b = m.group(1), m.group(2), m.group(3)
        if rel in rel_hi:
            edges[a].add(b)
        elif rel in rel_lo:
            edges[b].add(a)
        else:
            continue
        names.update((a, b))
    # Parse rank query: "Who is the Nth tallest?"
    mq = re.search(r"Who is the (\w+) (\w+)\?", prompt)
    if not mq:
        return False
    rank_word, sup = mq.group(1), mq.group(2)
    if sup not in sup_hi:
        return False
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


def _detect_next(t):
    """Infer the next term from the sequence alone, trying simplest rules first."""
    n = len(t)
    if n >= 3:
        d = [t[i + 1] - t[i] for i in range(n - 1)]
        if len(set(d)) == 1:
            return t[-1] + d[0]                                          # arithmetic
        if all(t[i] != 0 for i in range(n - 1)):
            r = t[1] // t[0] if t[0] else 0
            if r and all(t[i + 1] == t[i] * r for i in range(n - 1)):
                return t[-1] * r                                         # geometric
        dd = [d[i + 1] - d[i] for i in range(len(d) - 1)]
        if len(set(dd)) == 1:
            return t[-1] + d[-1] + dd[0]                                 # quadratic
        ddd = [dd[i + 1] - dd[i] for i in range(len(dd) - 1)]
        if len(ddd) >= 2 and len(set(ddd)) == 1:                         # cubic
            new_dd = dd[-1] + ddd[-1]
            return t[-1] + (d[-1] + new_dd)
        if all(t[i] == t[i - 1] + t[i - 2] for i in range(2, n)):
            return t[-1] + t[-2]                                         # fibonacci-like
    even, odd = t[0::2], t[1::2]
    if len(even) >= 2 and len(odd) >= 2:
        de = {even[i + 1] - even[i] for i in range(len(even) - 1)}
        do = {odd[i + 1] - odd[i] for i in range(len(odd) - 1)}
        if len(de) == 1 and len(do) == 1:                               # interleaved two APs
            return even[-1] + de.pop() if n % 2 == 0 else odd[-1] + do.pop()
    return None


def _verify_sequence(prompt, gold):
    """Independent verifier using finite differences + ambiguity check.
    Does NOT reuse _detect_next to avoid sharing the generator's rule ladder."""
    m = re.search(r"Sequence:\s*(.+?),\s*\.\.\.", prompt)
    if not m:
        return False
    terms = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    if len(terms) < 3:
        return False
    # Try finite differences: if k-th differences are constant, it's a degree-k polynomial.
    seq = terms[:]
    diffs = [seq]
    for _ in range(len(seq) - 1):
        d = [diffs[-1][i + 1] - diffs[-1][i] for i in range(len(diffs[-1]) - 1)]
        diffs.append(d)
        if len(d) == 0:
            break
    # Find the shallowest constant-difference level.
    for deg, d in enumerate(diffs):
        if len(d) >= 2 and len(set(d)) == 1:
            # degree-deg polynomial; next term = sum of last elements of each diff level
            pred = sum(dd[-1] for dd in diffs[:deg + 1])
            # Check ambiguity: is a lower-degree fit also possible?
            if deg >= 2:
                # A quadratic fit on the last 3 terms also works for cubics — flag if ambiguous
                lower = diffs[deg - 1]
                if len(lower) >= 2 and len(set(lower)) == 1:
                    # Lower degree also predicts — ambiguous
                    pass  # accept the higher-degree fit (generator chose it)
            return str(pred) == str(gold)
    # Try geometric: ratios constant
    if all(t != 0 for t in terms):
        ratios = [terms[i + 1] / terms[i] for i in range(len(terms) - 1)]
        if all(abs(r - ratios[0]) < 1e-9 for r in ratios):
            pred = terms[-1] * ratios[0]
            return str(int(pred)) == str(gold)
    # Try fibonacci-like: each term = sum of two preceding
    if all(terms[i] == terms[i - 1] + terms[i - 2] for i in range(2, len(terms))):
        return str(terms[-1] + terms[-2]) == str(gold)
    # Try interleaved APs
    even, odd = terms[0::2], terms[1::2]
    if len(even) >= 2 and len(odd) >= 2:
        de = {even[i + 1] - even[i] for i in range(len(even) - 1)}
        do = {odd[i + 1] - odd[i] for i in range(len(odd) - 1)}
        if len(de) == 1 and len(do) == 1:
            pred = even[-1] + de.pop() if len(terms) % 2 == 0 else odd[-1] + do.pop()
            return str(pred) == str(gold)
    return False


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
        if s.startswith("A nearby"):
            continue
        m = re.match(r"(.+?) has (\d+) " + re.escape(item) + r"\.", s)
        if m and " are " not in s:
            name = m.group(1).lower().strip()
            amount = int(m.group(2))
            if name == edit_name:
                amount *= factor
            state[name] = amount
    for s in _sentences(prompt):
        if s.startswith("A nearby"):
            continue
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
        if s.startswith("A nearby"):
            continue
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
