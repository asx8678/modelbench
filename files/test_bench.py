"""
Test suite for reasoning-bench. Run with:  pytest -q

Covers the things that silently corrupt results if they regress: independent gold
verification, grading robustness, error/duplicate handling, and the metrics math.
No model or network needed — the deterministic mock drives the end-to-end paths.
(report.py is not imported here so the suite needs no matplotlib.)
"""

import dataclasses
import itertools
import json

import re
from pathlib import Path
import pytest

import generators
import grading
import storage
import runner
import metrics
import providers


def _cfg(**kw):
    base = dict(mock=None, base_url="", api_key="", model="m", temperature=0,
                max_tokens=64, n=1, workers=2, timeout=5, retries=0,
                ask_confidence=True, resume=False, context_window=None)
    base.update(kw)
    return base


def _db_with(items):
    con = storage.connect(":memory:")
    storage.save_dataset(con, items)
    return con, storage.load_dataset(con)


# --------------------------------------------------------- generators / gold (#12)
def test_all_generated_golds_verify():
    # build_dataset(verify=True) raises if any gold fails independent re-derivation
    ds = generators.build_dataset(list(generators.GENERATORS), 1, 6, 6,
                                  with_distractor=True, surface_variants=2, verify=True)
    assert ds and all(generators.verify_gold(p) for p in ds)


@pytest.mark.parametrize("family", sorted(generators.SUPPORTS_DISTRACTOR))
def test_distractor_is_noop_without_stock_phrase(family):
    # bench-9bt: the distractor must (a) be a true NoOp -- the gold is identical
    # to the matched base item -- and (b) carry no fixed stock phrase that lets a
    # model (or the verifier) skip it without tracking relevance.
    stock = ("A nearby", "in a basket", "also has")
    changed = False
    for diff in range(1, 7):
        for seed in range(40):
            base = generators._mk(family, diff, seed, 0, False, "base", "g")
            dist = generators._mk(family, diff, seed, 0, True, "distractor", "g")
            assert generators.verify_gold(dist) is True
            assert base.gold == dist.gold                     # NoOp: gold preserved
            if dist.prompt != base.prompt:
                changed = True
                assert not any(s in dist.prompt for s in stock), dist.prompt
    assert changed                                            # the probe actually injects a clause


# agent tokens a shallow "subject/container filter" solver would key on, per family.
_AGENT_POOL = {
    "arithmetic": generators.NAMES,
    "redefined_ops": generators.NAMES,
    "ordering": generators.NAMES,
    "state_tracking": generators.CONTAINERS,
    "retroactive_edit": generators.CONTAINERS,
}


def _injected_clause(base, dist):
    bset = set(re.split(r"(?<=\.)\s+", base.prompt))
    for s in re.split(r"(?<=\.)\s+", dist.prompt):
        if s not in bset:
            return s
    return None


@pytest.mark.parametrize("family", sorted(generators.SUPPORTS_DISTRACTOR))
def test_distractor_not_separable_by_token_filter(family):
    # bench-ukn / E1+E2: the distractor must require RELEVANCE reasoning, not a
    # surface token filter. It must introduce no NEW agent token (so a
    # "different-person/different-container" filter can't separate it) and instead
    # be off-axis: a different item (numeric families) or an orthogonal relation
    # (ordering). The stock-phrase fix (bench-9bt) did not guarantee this.
    pool = _AGENT_POOL[family]
    injected = 0
    for diff in range(1, 7):
        for seed in range(40):
            base = generators._mk(family, diff, seed, 0, False, "base", "g")
            dist = generators._mk(family, diff, seed, 0, True, "distractor", "g")
            if dist.prompt == base.prompt:
                continue
            injected += 1
            inj = _injected_clause(base, dist)
            assert inj is not None
            base_agents = {a for a in pool if a.lower() in base.prompt.lower()}
            inj_agents = {a for a in pool if a.lower() in inj.lower()}
            # no agent token absent from the base => "new person/container" filter is useless
            assert inj_agents <= base_agents, (family, inj)
            if family == "ordering":
                # off-axis comparative: same grammar, adjective not in the asked family
                assert " than " in inj
                qsup = re.search(r"the \w+ (\w+)\?", base.prompt).group(1)
                fam_tup = next(r for r in generators.REL if r[2] == qsup)
                assert not re.search(rf"\b{fam_tup[0]}\b|\b{fam_tup[1]}\b", inj), inj
            else:
                # off-item: the injected clause names an ITEM other than the queried one
                qitem = re.search(r"How many (\w+) (?:does|are)", base.prompt).group(1)
                inj_items = {it for it in generators.ITEMS if re.search(rf"\b{it}\b", inj)}
                assert inj_items and qitem not in inj_items, (family, inj)
    assert injected


@pytest.mark.parametrize("family", ["arithmetic", "state_tracking", "ordering"])
def test_naive_solver_now_fails_on_distractor(family):
    # The probe earns its name only if a relevance-blind heuristic is actually
    # WRONG on it. A solver that ignores the off-axis dimension (item or relation
    # family) and applies every clause about the right agent must mis-grade at
    # least some distractor items -- otherwise the distractor is still inert to
    # shallow matching.
    def naive_answer(p):
        if family == "ordering":
            # relation-blind: treat ANY "X is <adj> than Y" as an ordering edge.
            from collections import defaultdict
            edges = defaultdict(set); names = set()
            hi = {r[0] for r in generators.REL}; lo = {r[1] for r in generators.REL}
            for m in re.finditer(r"(\w+) is (\w+) than (\w+)\.", p.prompt):
                a, rel, b = m.groups()
                if rel in hi: edges[a].add(b); names |= {a, b}
                elif rel in lo: edges[b].add(a); names |= {a, b}
            indeg = {n: 0 for n in names}
            for a in edges:
                for b in edges[a]: indeg[b] = indeg.get(b, 0) + 1
            order, q = [], [n for n in names if indeg[n] == 0]
            while len(q) == 1:
                n = q.pop(); order.append(n)
                for c in edges.get(n, ()):
                    indeg[c] -= 1
                    if indeg[c] == 0: q.append(c)
            mq = re.search(r"the (\w+) \w+\?", p.prompt)
            rk = {"1st":0,"2nd":1,"3rd":2,"4th":3,"5th":4,"6th":5}.get(mq.group(1))
            return order[rk] if rk is not None and rk < len(order) else None
        # numeric: item-blind subject/container filter.
        if family == "arithmetic":
            subj = re.search(r"does (.+?) have now", p.prompt).group(1)
            total = None
            for s in re.split(r"(?<=\.)\s+", p.prompt):
                if not s.startswith(subj + " "): continue
                rest = s[len(subj) + 1:]
                if (m := re.match(r"starts with (\d+)", rest)): total = int(m.group(1))
                elif total is None: continue
                elif (m := re.match(r"(?:buys|finds|is given|picks up) (\d+)", rest)): total += int(m.group(1))
                elif (m := re.match(r"(?:gives away|loses|uses|drops) (\d+)", rest)): total -= int(m.group(1))
                elif rest.startswith("doubles"): total *= 2
                elif rest.startswith("triples"): total *= 3
            return str(total)
        # state_tracking: container filter, item-blind.
        state = {}
        for s in re.split(r"(?<=\.)\s+", p.prompt):
            m = re.match(r"(.+?) has (\d+) ", s)
            if m and " are " not in s: state[m.group(1).lower()] = int(m.group(2))
        for s in re.split(r"(?<=\.)\s+", p.prompt):
            if (m := re.match(r"(\d+) \w+ are added to (.+?)\.", s)):
                state[m.group(2).lower()] = state.get(m.group(2).lower(), 0) + int(m.group(1))
            elif (m := re.match(r"(\d+) \w+ are removed from (.+?)\.", s)):
                state[m.group(2).lower()] = state.get(m.group(2).lower(), 0) - int(m.group(1))
            elif (m := re.match(r"(\d+) \w+ are moved from (.+?) to (.+?)\.", s)):
                k = int(m.group(1))
                state[m.group(2).lower()] = state.get(m.group(2).lower(), 0) - k
                state[m.group(3).lower()] = state.get(m.group(3).lower(), 0) + k
        mq = re.search(r"are in (.+?) now\?", p.prompt)
        return str(state.get(mq.group(1).lower()))

    fooled = 0
    for diff in range(1, 6):
        for seed in range(40):
            dist = generators._mk(family, diff, seed, 0, True, "distractor", "g")
            base = generators._mk(family, diff, seed, 0, False, "base", "g")
            if dist.prompt == base.prompt:
                continue
            if naive_answer(dist) != dist.gold:
                fooled += 1
    assert fooled > 0, f"{family}: relevance-blind solver was never fooled — distractor still inert"


def _redefined_items(mode_key, diff=4, n=80):
    """Generated redefined_ops items whose prompt is in the requested H4 mode."""
    needle = "for its first" if mode_key == "positional" else "to that result and the right number"
    return [p for seed in range(n)
            if needle in (p := generators._mk("redefined_ops", diff, seed, 0, False, "base", "g")).prompt]


def test_redefined_ops_positional_is_load_bearing():
    # H4/bench-81k: the SAME symbol's meaning depends on its POSITION in the chain.
    # A solver that looks the symbol up once (applies phase-1 to every step) must be
    # WRONG on at least some items — otherwise position is not load-bearing and the
    # old fixed-lookup / i%2 shortcut would still win.
    items = _redefined_items("positional")
    assert items, "no positional redefined_ops items generated"
    fooled = 0
    for p in items:
        assert generators.verify_gold(p) is True
        dm = re.search(r"(\S) means: for its first \d+ uses, (.*?); for every later use, .*?\.", p.prompt)
        sym, f1 = dm.group(1), generators._rop_phrase_fn(dm.group(2))
        cur = int(re.search(r"starts with (\d+)", p.prompt).group(1))
        for m in re.finditer(rf"\S+ {re.escape(sym)} (\d+) \S+", p.prompt):
            cur = f1(cur, int(m.group(1)))          # position-blind: always phase-1
        if str(cur) != p.gold:
            fooled += 1
    assert fooled > 0, "position-blind solver was never wrong — position is not load-bearing"


def test_redefined_ops_compositional_requires_base_expansion():
    # H4/bench-81k: one operator is defined IN TERMS OF another (a ⊗ b applies the
    # base op twice). A solver that applies the base op only ONCE for the composed
    # symbol must be wrong on some items — proving the composition is load-bearing.
    items = _redefined_items("compositional")
    assert items, "no compositional redefined_ops items generated"
    fooled = 0
    for p in items:
        assert generators.verify_gold(p) is True
        base_def = re.search(r"(\S) means ([^:.]+?)\.", p.prompt)      # the simple base op
        comp_m = re.search(r"(\S) means: apply (\S) to the two numbers", p.prompt)
        base_sym, comp_sym = comp_m.group(2), comp_m.group(1)
        bf = generators._rop_phrase_fn(base_def.group(2))
        cur = int(re.search(r"starts with (\d+)", p.prompt).group(1))
        for m in re.finditer(rf"\S+ ({re.escape(base_sym)}|{re.escape(comp_sym)}) (\d+) \S+", p.prompt):
            cur = bf(cur, int(m.group(2)))          # WRONG: applies base once, never twice
        if str(cur) != p.gold:
            fooled += 1
    assert fooled > 0, "single-apply solver was never wrong — composition is not load-bearing"


def test_surface_variants_vary_phrasing_hold_gold():
    # bench-1s0 / E8: surface variants must hold the gold fixed (matched-pair flip
    # metric) AND actually vary cosmetic surface beyond names — verb phrasing now
    # comes from the surface rng, so at least some grp shows >1 distinct verb.
    verbs = ("buys", "finds", "is given", "picks up", "gives away", "loses", "uses", "drops")
    saw_lexical_variation = False
    for diff in range(2, 6):
        for seed in range(30):
            grp = [generators._mk("arithmetic", diff, seed, s, False,
                                  "base" if s == 0 else "surface", "g")
                   for s in range(4)]
            golds = {p.gold for p in grp}
            assert len(golds) == 1, (diff, seed, golds)   # gold invariant across the grp
            used = [{v for v in verbs if re.search(rf"\b{v}\b", p.prompt)} for p in grp]
            if len({frozenset(u) for u in used}) > 1:
                saw_lexical_variation = True
    assert saw_lexical_variation, "surface variants never changed verb phrasing"


def test_false_lemma_trap_is_load_bearing():
    # bench-dqw / H7: the "Note: ... total never changes; it stays N" lemma is FALSE
    # (the sequence also adds/removes). Trusting it (answering N) must be WRONG, and
    # the independent verifier must reject the premise and re-sum to the true gold.
    fooled = 0
    for diff in range(1, 7):
        for seed in range(30):
            p = generators._mk("false_lemma", diff, seed, 0, False, "base", "g")
            assert generators.verify_gold(p) is True
            claimed = re.search(r"stays (\d+)", p.prompt).group(1)
            assert claimed != p.gold              # lemma is false: trusting it is wrong
            if claimed != p.gold:
                fooled += 1
    assert fooled > 0


def test_noise_haystack_needle_survives_decoys():
    # bench-dqw / H8: a real arithmetic problem buried under structurally-identical
    # decoy chains about other people (same item). The subject-bound verifier finds
    # the needle; a subject-BLIND running-total solver is polluted by the decoys.
    fooled = 0
    for diff in range(1, 6):
        for seed in range(30):
            p = generators._mk("noise_haystack", diff, seed, 0, False, "base", "g")
            assert generators.verify_gold(p) is True
            # subject-blind solver: sum every 'starts with' and 'more' it sees
            naive = 0
            for s in re.split(r"(?<=\.)\s+", p.prompt):
                if (m := re.search(r"starts with (\d+)", s)):
                    naive += int(m.group(1))
                elif (m := re.search(r"(\d+) more", s)):
                    naive += int(m.group(1))
            if str(naive) != p.gold:
                fooled += 1
    assert fooled > 0, "decoys never polluted a relevance-blind solver"


def test_dynamic_pivot_structure_and_subgold():
    # bench-7fo / E3-H3: two turns, a committed subgold, and a load-bearing pivot
    # (gold != subgold). The subgold must re-derive as the LITERAL reading (moves
    # relocate); the gold as the REVISED reading (moves never happened).
    import re as _re
    for diff in range(1, 7):
        for seed in range(25):
            p = generators._mk("dynamic_pivot", diff, seed, 0, False, "base", "g")
            assert p.turns and len(p.turns) == 2
            assert p.subgold is not None and p.gold != p.subgold      # pivot is load-bearing
            assert generators.verify_gold(p) is True                  # gold = revised reading
            mq = _re.search(r"How many (.+?) are in (.+?) now", p.prompt)
            item, qc = mq.group(1), mq.group(2).lower()
            literal = generators._replay_state(p.prompt, item, ignore_moves=False)
            assert str(literal.get(qc)) == p.subgold                  # subgold = literal reading


def test_dynamic_pivot_backtracking_metric_end_to_end():
    # The genuine multi-turn runner must grade the committed turn-1 reply (subgold)
    # separately from the revised final (gold), and metrics must expose both.
    items = generators.build_dataset(["dynamic_pivot"], 1, 4, 5, verify=True)
    con, ds = _db_with(items)
    runner.run(con, "r", ds, _cfg(mock="perfect", model="mock", n=1, ask_confidence=False))
    # metadata carries the intermediate (committed) grade
    md = json.loads(con.execute(
        "SELECT metadata FROM responses WHERE run_id='r' LIMIT 1").fetchone()["metadata"])
    assert "intermediate_correct" in md
    bt = metrics.compute(con, "r")["backtracking"]
    assert bt["n"] == len(items)
    # perfect mock commits AND revises every item correctly
    assert bt["intermediate_accuracy"] == pytest.approx(1.0)
    assert bt["final_accuracy"] == pytest.approx(1.0)
    assert bt["revision_success"] == pytest.approx(1.0)


def test_difficulty_axis_labels_are_honest():
    # bench-lop / E6: the difficulty axis is not the same quantity across families.
    # Tier/size families must be labelled distinctly from the default "reasoning steps".
    assert generators.difficulty_axis("arithmetic") == "reasoning steps"
    assert generators.difficulty_axis("sequences") == "rule tier"
    for fam in ("knights_knaves", "logic_grid", "unsat_csp"):
        assert "n" in generators.difficulty_axis(fam)
    # every generator family resolves to some axis label (default included)
    for fam in generators.GENERATORS:
        assert generators.difficulty_axis(fam)


def test_sequences_capped_at_tier_6():
    # tier 6 (cubic) is the hardest rule; asking for more must not silently reuse it
    ds = generators.build_dataset(["sequences"], 1, 9, 3, verify=True)
    assert max(p.difficulty for p in ds) == 6
    cubics = [p for p in ds if p.difficulty == 6]
    assert cubics and all(generators.verify_gold(p) for p in cubics)


def test_sequence_verifier_flags_ambiguous():
    # bench-kab: the ambiguity gate must REJECT a sequence that two simple rules
    # fit but disagree on. [3,6,12,24] is geometric (->48) OR two interleaved APs
    # ([3,12]+9, [6,24]+18 -> 21); with only 4 terms it is genuinely ambiguous,
    # so the verifier rejects EITHER candidate as the gold.
    amb = "Sequence: 3, 6, 12, 24, ...  What is the next number?"
    assert generators._verify_sequence(amb, "48") is False
    assert generators._verify_sequence(amb, "21") is False
    # The same rule with enough terms (6, as the generator emits) is unambiguous.
    clean = "Sequence: 3, 6, 12, 24, 48, 96, ...  What is the next number?"
    assert generators._verify_sequence(clean, "192") is True


@pytest.mark.parametrize("family", list(generators.GENERATORS))
def test_verifier_catches_wrong_gold(family):
    # The verifier must *independently re-derive* the answer, not trust the gold:
    # the true gold passes, a corrupted one fails — for every family.
    p = generators._mk(family, 3, 7, 0, False, "base", "g")
    assert generators.verify_gold(p) is True
    if p.answer_type == "int":
        bad = dataclasses.replace(p, gold=str(int(p.gold) + 1))
    else:
        bad = dataclasses.replace(p, gold=next(c for c in p.choices if c != p.gold))
    assert generators.verify_gold(bad) is False


def test_generation_is_deterministic():
    # item_id/gold/prompt must be byte-stable across builds, or resume, the mock,
    # and cross-run comparisons all silently break.
    a = generators.build_dataset(list(generators.GENERATORS), 1, 5, 3, surface_variants=2, verify=False)
    b = generators.build_dataset(list(generators.GENERATORS), 1, 5, 3, surface_variants=2, verify=False)
    assert [(p.item_id, p.gold, p.prompt) for p in a] == [(p.item_id, p.gold, p.prompt) for p in b]


# --------------------------------------------------- hard CSP families (#new)
# These are the genuine reasoning discriminators: every puzzle must have exactly
# ONE solution (else "accuracy" is meaningless), the gold must be that solution,
# and difficulty must really add entities. The verifier re-solves from the prompt
# text by brute force, so a passing verify already proves uniqueness + correctness.

def test_knights_knaves_is_uniquely_solvable_and_scales():
    small = generators._mk("knights_knaves", 1, 3, 0, False, "base", "g")
    big = generators._mk("knights_knaves", 6, 3, 0, False, "base", "g")
    n_small, _ = generators._kk_parse(small.prompt)
    n_big, _ = generators._kk_parse(big.prompt)
    assert len(n_small) == 3 and len(n_big) == 8           # difficulty + 2 islanders
    for p in (small, big):
        names, stmts = generators._kk_parse(p.prompt)
        sols = generators._kk_all_solutions(names, stmts)
        assert len(sols) == 1                              # unique => well-posed
        # gold is the sorted, comma-separated list of knave names (2^n space)
        knave_names = {n for n, is_knight in sols[0].items() if not is_knight}
        assert set(t.strip() for t in p.gold.split(",") if t.strip()) == knave_names


def test_knights_knaves_breaks_the_global_flip_symmetry():
    # The classic trap: with only "X says Y is a knight/knave" links, flipping every
    # type yields a second solution. A well-formed puzzle must rule that out.
    for r in range(8):
        p = generators._mk("knights_knaves", 4, r, 0, False, "base", "g")
        names, stmts = generators._kk_parse(p.prompt)
        sols = generators._kk_all_solutions(names, stmts)
        assert len(sols) == 1
        flipped = {n: (not v) for n, v in sols[0].items()}
        assert flipped not in generators._kk_all_solutions(names, stmts)


def test_logic_grid_is_uniquely_solvable_and_scales():
    small = generators._mk("logic_grid", 1, 5, 0, False, "base", "g")
    big = generators._mk("logic_grid", 5, 5, 0, False, "base", "g")
    ns, n_s, _, _ = generators._lg_parse(small.prompt)
    nb, n_b, _, _ = generators._lg_parse(big.prompt)
    assert len(ns) == 3 and len(nb) == 7                   # difficulty + 2 floors
    for p in (small, big):
        names, n, clues, q = generators._lg_parse(p.prompt)
        sols = generators._lg_solutions(names, n, clues)
        assert len(sols) == 1
        assert str(sols[0][q]) == p.gold
def _first_invariant_slot_unsat_csp(max_seed=400):
    """Find a dropped-clue unsat_csp item whose queried slot is invariant across
    the multiple solutions (determinate despite global ambiguity). Returns
    (prompt, gold, values) or None. Scans seeds so it survives generator-prior
    changes (bench-5zn) rather than hard-coding one seed."""
    for seed in range(max_seed):
        prompt, gold, _atype, _choices = generators.gen_unsat_csp(2, seed, 0, False)
        names, stmts = generators._kk_parse(prompt)
        sols = generators._kk_all_solutions(names, stmts)
        if len(sols) <= 1:
            continue
        query_name = re.search(r"Is (\w+) a knight or a knave\?", prompt).group(1)
        values = {s.get(query_name) for s in sols}
        if len(values) == 1:
            return prompt, gold, values
    return None


def test_unsat_csp_dropped_clue_with_invariant_queried_slot_is_determinate():
    # bench-le7.1 regression: when a clue is dropped and the puzzle becomes
    # under-constrained, the queried slot may still be invariant across all
    # remaining solutions. The gold must be the determinate knight/knave, not
    # the generic UNDETERMINED sentinel. bench-5zn keeps ~20% of the dropped-clue
    # class querying an invariant slot precisely to preserve this probe.
    found = _first_invariant_slot_unsat_csp()
    assert found is not None, "no dropped-clue item with an invariant queried slot found"
    _prompt, gold, values = found
    assert gold in ("knight", "knave"), gold
    assert gold == ("knight" if next(iter(values)) else "knave")


def test_unsat_csp_verifier_rejects_undetermined_on_invariant_slot():
    # bench-le7.2 regression: the verifier must independently check the set of
    # values for the queried slot, not blindly accept UNDETERMINED whenever there
    # are multiple solutions.
    found = _first_invariant_slot_unsat_csp()
    assert found is not None, "no dropped-clue item with an invariant queried slot found"
    prompt, gold, _values = found
    assert generators._verify_unsat_csp(prompt, gold) is True
    assert generators._verify_unsat_csp(prompt, "UNDETERMINED") is False

def test_unsat_csp_over_constrained_branch_is_determinate():
    # bench-le7.3: the 10% over-constrained branch adds redundant clues but must
    # still leave exactly one solution, so the gold is a determinate knight/knave.
    for seed in (2, 12, 14):
        prompt, gold, _atype, _choices = generators.gen_unsat_csp(2, seed, 0, False)
        names, stmts = generators._kk_parse(prompt)
        sols = generators._kk_all_solutions(names, stmts)
        assert len(sols) == 1, f"seed={seed}: expected unique solution, got {len(sols)}"
        assert gold in ("knight", "knave"), f"seed={seed}: gold={gold}"
        query_name = re.search(r"Is (\w+) a knight or a knave\?", prompt).group(1)
        values = {s.get(query_name) for s in sols}
        assert len(values) == 1
        assert gold == ("knight" if next(iter(values)) else "knave")


def test_unsat_csp_label_prior_is_roughly_balanced():
    # bench-5zn: rebalance the four-way label prior so UNDETERMINED is not
    # severely under-represented (it was ~17%; querying a varying slot on the
    # dropped-clue class lifts it toward ~25%).
    from collections import Counter
    labels = Counter()
    for diff in range(1, 6):
        for seed in range(120):
            p = generators._mk("unsat_csp", diff, seed, 0, False, "base", "g")
            labels[p.gold] += 1
    tot = sum(labels.values())
    for lab in ("knight", "knave", "UNDETERMINED", "NO_SOLUTION"):
        frac = labels[lab] / tot
        assert 0.15 < frac < 0.35, f"{lab} prior {frac:.2%} is too skewed"
    assert labels["UNDETERMINED"] / tot > 0.20    # the previously-rare class


def test_csp_puzzles_are_minimally_constrained():
    # Dropping any one clue should destroy uniqueness — proof the puzzle has no
    # redundant giveaways (a redundant clue would make it easier than its tier).
    p = generators._mk("logic_grid", 4, 6, 0, False, "base", "g")
    names, n, clues, _ = generators._lg_parse(p.prompt)
    assert len(generators._lg_solutions(names, n, clues)) == 1
    assert all(len(generators._lg_solutions(names, n, clues[:i] + clues[i + 1:])) != 1
               for i in range(len(clues)))


def test_csp_surface_variants_hold_gold_and_structure_fixed():
    # A surface variant must be the SAME puzzle with different names: the
    # underlying slot structure and the knave slot set must be invariant,
    # while the rendered name list and gold string change with renaming.
    for fam in ("knights_knaves", "logic_grid"):
        if fam == "knights_knaves":
            def shape(pr):
                _, stmts = generators._kk_parse(pr)
                return sorted(s[0] for s in stmts)
            def knave_count(pr):
                names, stmts = generators._kk_parse(pr)
                sols = generators._kk_all_solutions(names, stmts)
                assert len(sols) == 1
                return sum(1 for v in sols[0].values() if not v)
        else:
            def shape(pr):
                names, n, clues, _ = generators._lg_parse(pr)
                return (n, sorted(c[0] for c in clues))
            def knave_count(pr):
                # logic_grid gold is a floor number; surface variants
                # change the names but the queried floor must match.
                return None
        base = generators._mk(fam, 4, 2, 0, False, "base", "g")
        variants = [generators._mk(fam, 4, 2, s, False, "surface", "g") for s in (1, 2, 3)]
        assert all(shape(v.prompt) == shape(base.prompt) for v in variants)
        assert all(v.prompt != base.prompt for v in variants)
        if knave_count(base.prompt) is not None:
            base_kc = knave_count(base.prompt)
            assert all(knave_count(v.prompt) == base_kc for v in variants)
        else:
            # logic_grid: gold floor must be identical across variants
            assert {base.gold} | {v.gold for v in variants} == {base.gold}


def test_arithmetic_division_op_is_generated_and_verified():
    # The hardened arithmetic family must actually emit division, and the
    # independent verifier must handle it (integer-exact by construction).
    div = next((generators._mk("arithmetic", 4, r, 0, False, "base", "g")
                for r in range(200) if "divides the" in
                generators._mk("arithmetic", 4, r, 0, False, "base", "g").prompt), None)
    assert div is not None, "division operation was never generated"
    assert generators.verify_gold(div) is True
    assert generators.verify_gold(dataclasses.replace(div, gold=str(int(div.gold) + 1))) is False


@pytest.mark.parametrize("family,diff", [("knights_knaves", 5), ("logic_grid", 4),
                                         ("arithmetic", 5), ("sequences", 6)])
def test_no_degenerate_constant_gold(family, diff):
    # A generator that ignored its structure seed would emit one repeated answer;
    # require real answer diversity across structures at a fixed difficulty.
    golds = {generators._mk(family, diff, r, 0, False, "base", "g").gold for r in range(20)}
    min_div = 2 if family == "knights_knaves" else 4
    assert len(golds) >= min_div, (family, golds)



def test_composed_gold_verifies_end_to_end():
    # The composed cascade (knights -> arithmetic -> ordering -> two op-gates) must
    # produce a gold that the independent verifier re-derives from the prompt text.
    p = generators._mk("composed", 3, 7, 0, False, "base", "g")
    assert generators.verify_gold(p) is True
    # Sanity: prompt has all five (load-bearing) stages.
    for st in ("Stage 1:", "Stage 2:", "Stage 3:", "Stage 4:", "Stage 5:"):
        assert st in p.prompt, st
    # The final answer G is an unbounded integer >= 2 (K >= 1, C >= 2, F >= 2).
    assert p.answer_type == "int"
    assert int(p.gold) >= 2
    # bench-o6o: the knight count must be DEDUCED, never printed, and no stage
    # is a labelled distractor (both were exploitable shortcuts before).
    assert "starts with as many" in p.prompt
    assert re.search(r"starts with \d", p.prompt) is None
    assert "distractor" not in p.prompt.lower()


def test_composed_perturb_hop_a_changes_final_gold():
    # Perturbing the first hop (knights count) must propagate through the
    # arithmetic hop and change R: a different knight count yields a different R,
    # which feeds (and whose parity selects an operator in) the downstream gates.
    p = generators._mk("composed", 3, 7, 0, False, "base", "g")
    parsed = generators._composed_parse_hops(p.prompt)
    names, stmts = generators._kk_parse(parsed["knights_prompt"])
    # Flip the type of the first speaker in every statement: changes the
    # knight count.
    flipped = []
    for st in stmts:
        if st[0] == "ABS":
            flipped.append((st[0], st[1], st[2], not st[3]))
        else:
            flipped.append((st[0], st[1], st[2], st[3], not st[4]))
    sols = generators._kk_all_solutions(names, flipped)
    if len(sols) != 1:
        # If flipping breaks uniqueness, just verify that a different
        # hop-A seed usually changes the final gold by sampling.
        seen = {p.gold}
        for s in range(1, 30):
            q = generators._mk("composed", 3, s, 0, False, "base", "g")
            if q.gold not in seen:
                break
            seen.add(q.gold)
        else:
            assert False, "composed gold did not vary across 30 hop-A seeds"
        return
    new_knight_count = sum(1 for v in sols[0].values() if v)
    if new_knight_count == sum(1 for v in generators._kk_all_solutions(names, stmts)[0].values() if v):
        # If equal by bad luck, try the next speaker flip.
        for i in range(1, len(stmts)):
            flipped_i = stmts[:i] + [flipped[i]] + stmts[i + 1:]
            sols_i = generators._kk_all_solutions(names, flipped_i)
            if len(sols_i) == 1:
                new_knight_count = sum(1 for v in sols_i[0].values() if v)
                if new_knight_count != sum(1 for v in generators._kk_all_solutions(names, stmts)[0].values() if v):
                    break
    old_knight_count = sum(1 for v in generators._kk_all_solutions(names, stmts)[0].values() if v)
    r_old = generators._verify_arithmetic_raw(parsed["arith_prompt"], old_knight_count)
    r_new = generators._verify_arithmetic_raw(parsed["arith_prompt"], new_knight_count)
    # A 1-unit knight-count change propagates through the arithmetic hop, so the
    # final gold (R * C, C fixed) changes too.
    assert r_old is not None and r_new is not None
    assert int(r_new) != int(r_old), (r_old, r_new)


def test_composed_hop_c_is_load_bearing():
    # H6/bench-32g: the ordering hop (Stage 3) genuinely contributes. C feeds gate 4
    # (R op C) and its PARITY selects gate 5's operator, so a different ordering
    # outcome (different C) changes the gold -- Stage 3 cannot be skipped.
    p = generators._mk("composed", 4, 11, 0, False, "base", "g")
    assert generators.verify_gold(p) is True
    parsed = generators._composed_parse_hops(p.prompt)
    order, _ = generators._verify_order_raw(parsed["order_block"], "")
    assert order is not None and parsed["pivot"] in order
    c = order.index(parsed["pivot"]) + 1
    assert c >= 2
    kk_names, kk_stmts = generators._kk_parse(parsed["knights_prompt"])
    k = sum(1 for v in generators._kk_all_solutions(kk_names, kk_stmts)[0].values() if v)
    r = generators._verify_arithmetic_raw(parsed["arith_prompt"], k)
    assert r is not None and r >= 1

    def cascade(r_, c_):
        e4, o4 = generators._composed_gate_ops(parsed["gate4"], "R")
        f = (e4 if r_ % 2 == 0 else o4)(r_, c_)
        e5, o5 = generators._composed_gate_ops(parsed["gate5"], "C")
        return (e5 if c_ % 2 == 0 else o5)(f, r_)

    assert int(p.gold) == cascade(r, c)              # gold is the cascade's output
    assert cascade(r, c - 1) != int(p.gold)          # changing C by 1 changes the gold


def test_composed_early_slip_changes_a_later_operation():
    # H6/bench-32g: the defining property -- a 1-unit slip in an EARLY hop flips a
    # parity and so selects a DIFFERENT OPERATOR downstream, not just a shifted
    # magnitude. We verify both halves: (a) each gate assigns different operators to
    # its even/odd branches, and (b) recomputing with R off by one selects the other
    # operator and yields a gold that differs by more than a unit (so a magnitude-
    # only guesser collapses).
    amplified = 0
    for seed in range(40):
        p = generators._mk("composed", 4, seed, 0, False, "base", "g")
        parsed = generators._composed_parse_hops(p.prompt)
        e4, o4 = generators._composed_gate_ops(parsed["gate4"], "R")
        e5, o5 = generators._composed_gate_ops(parsed["gate5"], "C")
        assert e4(3, 5) != o4(3, 5)                  # even/odd branches use different ops
        assert e5(3, 5) != o5(3, 5)
        kk = generators._kk_parse(parsed["knights_prompt"])
        r = generators._verify_arithmetic_raw(
            parsed["arith_prompt"],
            sum(1 for v in generators._kk_all_solutions(*kk)[0].values() if v))
        order, _ = generators._verify_order_raw(parsed["order_block"], "")
        c = order.index(parsed["pivot"]) + 1

        def cascade(rr, cc):
            f = (e4 if rr % 2 == 0 else o4)(rr, cc)
            return (e5 if cc % 2 == 0 else o5)(f, rr)

        assert cascade(r, c) == int(p.gold)
        if abs(cascade(r + 1, c) - cascade(r, c)) > 1:   # parity slip -> operator change
            amplified += 1
    assert amplified > 0, "an early parity slip never amplified into an operator change"


# ------------------------------------------------------------------ grading (#1)
def test_confidence_value_not_used_as_answer():
    # The dangerous case: confidence equals gold and the model gave no numeric
    # answer. The old fallback grabbed the CONFIDENCE digits -> spurious "correct".
    parsed, correct, conf, _ = grading.grade("I am not sure of the count.\nCONFIDENCE: 13",
                                          "int", "13")
    assert conf == 13
    assert parsed is None
    assert correct is False


def test_int_fallback_still_finds_a_stated_number():
    # stripping the confidence line must not stop us finding a real trailing number
    parsed, correct, _, _ = grading.grade("so the result is 13.\nCONFIDENCE: 80", "int", "13")
    assert parsed == "13" and correct


def test_grading_well_formed():
    parsed, correct, conf, _ = grading.grade("work...\nANSWER: 13\nCONFIDENCE: 80", "int", "13")
    assert parsed == "13" and correct and conf == 80


def test_grading_choice_fallback():
    parsed, correct, _, _ = grading.grade("so the tallest is Diego.", "choice", "Diego",
                                       ["Diego", "Lena"])
    assert correct


def test_grading_choice_fallback_restricted_on_csp_sentinels():
    # Prompt echoes words like knight/knave; without an ANSWER line they must
    # NOT rescue a non-compliant response via whole-text fallback.
    choices = ["knight", "knave", "UNDETERMINED", "NO_SOLUTION"]
    parsed, src = grading.parse_answer(
        "The prompt mentions a knight and a knave.", "choice", choices)
    assert parsed is None and src == "none"

    # ANSWER line still wins.
    parsed, src = grading.parse_answer(
        "I think the speaker is a knight, not a knave.\nANSWER: knight",
        "choice", choices)
    assert parsed == "knight" and src == "marker"


def test_grading_choice_fallback_unrestricted_for_other_families():
    # Non-CSP choice families keep the whole-text fallback.
    parsed, src = grading.parse_answer(
        "Diego is clearly the tallest here.", "choice", ["Diego", "Lena"])
    assert parsed == "Diego" and src == "fallback"

def test_grading_int_strips_thousands_separator():
    # CSP/arithmetic answers can be large; "1,234" must read as 1234, not 1.
    parsed, correct, _, _ = grading.grade("After working it out,\nANSWER: 1,234", "int", "1234")
    assert parsed == "1234" and correct


def test_grading_int_handles_negative():
    parsed, correct, _, _ = grading.grade("the net change is -5.\nANSWER: -5", "int", "-5")
    assert parsed == "-5" and correct


def test_grading_marker_overrides_earlier_wrong_mention():
    # The model second-guesses itself in prose; the ANSWER line is authoritative,
    # so an earlier wrong name/number in the reasoning must not be scored.
    p_choice, ok_choice, _, _ = grading.grade("At first Diego looks tallest, but no.\nANSWER: Lena", "choice", "Lena", ["Diego", "Lena"])
    assert p_choice == "Lena" and ok_choice
    p_int, ok_int, _, _ = grading.grade("I initially get 7, but rechecking it is 4.\nANSWER: 4", "int", "4")
    assert p_int == "4" and ok_int


def test_grading_choice_marker_with_extra_words():
    # marker line carries justification around the choice token
    parsed, correct, _, _ = grading.grade("ANSWER: The knight is Lena.", "choice", "Lena", ["Diego", "Lena", "Omar"])
    assert parsed == "Lena" and correct


# ------------------------------------------------- mock end-to-end, determinism (#13)
def test_mock_perfect_is_full_accuracy():
    items = generators.build_dataset(["arithmetic", "ordering"], 1, 3, 4,
                                     with_distractor=True, surface_variants=2)
    con, ds = _db_with(items)
    runner.run(con, "perfect", ds, _cfg(mock="perfect"))
    res = metrics.compute(con, "perfect")
    assert res["overall_accuracy"] == 1.0
    assert res["coverage"]["errored"] == 0
    assert 0.0 <= res["invariance"]["answer_flip_rate"] <= 1.0


def test_mock_perfect_on_csp_families_flows_through_pipeline():
    # the hard CSP families (int answers, no distractor/surface) must run end-to-end
    items = generators.build_dataset(["knights_knaves", "logic_grid"], 1, 4, 3,
                                     surface_variants=2)
    con, ds = _db_with(items)
    runner.run(con, "csp", ds, _cfg(mock="perfect"))
    res = metrics.compute(con, "csp")
    assert res["overall_accuracy"] == 1.0
    assert res["coverage"]["errored"] == 0
    # surface variants were emitted, so invariance must be measurable on these families
    assert {"knights_knaves", "logic_grid"} <= set(res["invariance"]["by_family"])


def test_mock_is_deterministic():
    items = generators.build_dataset(["arithmetic"], 1, 4, 5)
    con1, ds1 = _db_with(items)
    runner.run(con1, "a", ds1, _cfg(mock="noisy"))
    con2, ds2 = _db_with(items)
    runner.run(con2, "b", ds2, _cfg(mock="noisy"))
    assert metrics.compute(con1, "a")["overall_accuracy"] == \
           metrics.compute(con2, "b")["overall_accuracy"]


def test_rerun_does_not_duplicate():
    items = generators.build_dataset(["arithmetic"], 1, 2, 4)
    con, ds = _db_with(items)
    runner.run(con, "r", ds, _cfg(mock="noisy"))
    n1 = con.execute("SELECT COUNT(*) FROM responses WHERE run_id='r'").fetchone()[0]
    runner.run(con, "r", ds, _cfg(mock="noisy"))
    n2 = con.execute("SELECT COUNT(*) FROM responses WHERE run_id='r'").fetchone()[0]
    assert n1 == n2 == len(ds)



# ------------------------------------------------- telemetry persistence (#mf4.1/2)
def test_mock_run_writes_telemetry_row_with_caps_and_source():
    items = generators.build_dataset(["arithmetic"], 1, 1, 2)
    con, ds = _db_with(items)
    cfg = _cfg(mock="perfect", capabilities=["stream", "mock"], n=2)
    runner.run(con, "r1", ds, cfg)
    row = storage.load_telemetry(con, "r1", ds[0]["item_id"], 0)
    assert row is not None
    assert row["capabilities"] == ["stream", "mock"]
    assert row["reasoning_token_source"] is not None
    # mock path has no honest producer for reasoning tokens
    assert row["reasoning_token_source"] == "unavailable"
    assert row["prompt_tokens"] is None


def test_mock_run_telemetry_marks_unobservable_fields():
    items = generators.build_dataset(["arithmetic"], 1, 1, 2)
    con, ds = _db_with(items)
    cfg = _cfg(mock="perfect", capabilities=["mock"], n=1)
    runner.run(con, "r2", ds, cfg)
    row = storage.load_telemetry(con, "r2", ds[0]["item_id"], 0)
    assert row is not None
    unobs = row["unobservable_fields"]
    # The mock runner does not emit native timings, so reasoning_wall_ms
    # is unavailable and gets the "ttft_or_answer_wall_unavailable" marker.
    assert row["reasoning_wall_ms"] is None
    assert unobs["reasoning_wall_ms"] == "ttft_or_answer_wall_unavailable"
    assert "token_entropy" in unobs
    assert "thinking_tps" in unobs
    assert "tot_branch_map" in unobs


def test_build_telemetry_reasonable_when_think_tokens_present():
    cfg = _cfg(capabilities=["native_anthropic", "stream"])
    telemetry = runner._build_telemetry(
        cfg, prompt_tokens=10, completion_tokens=100,
        timings={"ttft": 0.1, "first_reasoning": 0.2,
                 "answer_wall": 0.5, "total": 0.6},
        think_tokens=30, text="ANSWER: 42")
    assert telemetry["reasoning_token_source"] == "native_usage"
    assert telemetry["reasoning_tokens"] == 30
    # density proxy = (ct - (ct - rt)) / (ct - rt) = rt / (ct - rt)
    # reasoning_wall_ms proxy = answer_wall - ttft = 500 - 100 = 400
    assert telemetry["reasoning_wall_ms"] == 400
    assert "reasoning_wall_ms" not in telemetry["unobservable_fields"]
    assert telemetry["answer_wall_ms"] == 500




def test_build_telemetry_honest_null_when_think_tokens_absent():
    cfg = _cfg(capabilities=["native_anthropic"])
    telemetry = runner._build_telemetry(
        cfg, prompt_tokens=10, completion_tokens=100,
        timings={"ttft": 0.1, "first_reasoning": 0.2,
                 "answer_wall": 0.5, "total": 0.6},
        think_tokens=0, text="ANSWER: 42")
    assert telemetry["reasoning_token_source"] == "unavailable"
    assert telemetry["reasoning_tokens"] == 0
    assert telemetry["reasoning_density_proxy"] is None
    assert telemetry["unobservable_fields"]["reasoning_tokens"] == "not_exposed_by_provider"

def test_baseline_metrics_unchanged():
    # Full reference dataset snapshot: every family, difficulties 1..6, reps 12,
    # distractor + 3 surface variants, mock noisy with 3 samples. The deterministic
    # pipeline must reproduce the per-family metrics byte-for-byte.
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "baseline_metrics.json").read_text()
    )
    items = generators.build_dataset(list(generators.GENERATORS), 1, 6, 12,
                                     with_distractor=True, surface_variants=3,
                                     verify=True)
    con, ds = _db_with(items)
    runner.run(con, "baseline-regression", ds,
               _cfg(mock="noisy", model="mock", n=3, workers=1,
                    max_tokens=1024, timeout=30, ask_confidence=False))
    res = metrics.compute(con, "baseline-regression")

    def per_family(d):
        out = {
            "accuracy_by_family": d["accuracy_by_family"],
            "degradation": d["degradation"],
            "distractibility": d["distractibility"],
            "invariance_by_family": d["invariance"]["by_family"],
        }
        # bench-9eb.1 is a MIGRATION: knights_knaves moved from a binary
        # answer to a 2^n set answer, which changes the mock-noisy
        # accuracy and the surface-flip rate. Exclude it from the
        # byte-stable comparison; re-baseline after the migration tag
        # bump.
        for top in out.values():
            if isinstance(top, dict):
                top.pop("knights_knaves", None)
                # bench-9et.1 adds unsat_csp to SUPPORTS_SURFACE;
                # the new surface variants change the invariance metrics.
                top.pop("unsat_csp", None)
                # bench-7fo adds the dynamic_pivot family (genuine multi-turn
                # backtracking), absent from the baseline fixture.
                top.pop("dynamic_pivot", None)
                # bench-dqw adds the false_lemma and noise_haystack families.
                top.pop("false_lemma", None)
                top.pop("noise_haystack", None)
        return out

    assert json.dumps(per_family(res), sort_keys=True, indent=2) == \
           json.dumps(per_family(fixture), sort_keys=True, indent=2)


# ----------------------------------------- errors are coverage gaps, not wrong (#2, #3)
def test_errors_excluded_and_retried_on_resume(monkeypatch):
    items = generators.build_dataset(["arithmetic"], 1, 1, 3)
    con, ds = _db_with(items)
    monkeypatch.setattr(runner, "call_api", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    runner.run(con, "e", ds, _cfg(mock=None, retries=0))
    res = metrics.compute(con, "e")
    assert res["coverage"]["errored"] == len(ds)
    assert res["overall_accuracy"] == 0.0
    assert storage.done_items(con, "e") == set()   # nothing counts as done -> all retried


def test_malformed_200_response_does_not_crash(monkeypatch):
    items = generators.build_dataset(["arithmetic"], 1, 1, 2)
    con, ds = _db_with(items)

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return json.dumps({"error": "model loading"}).encode()

    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *a, **k: _Resp())
    runner.run(con, "m", ds, _cfg(mock=None, retries=1))   # must not raise
    assert metrics.compute(con, "m")["coverage"]["errored"] == len(ds)


# -------------------------------------------------- majority vote / self-consistency (#8)
def test_majority_vote_overrides_unlucky_first_sample():
    con = storage.connect(":memory:")
    p = generators._mk("arithmetic", 1, 0, 0, False, "base", "g")
    storage.save_dataset(con, [p])
    storage.new_run(con, "v", "m", "", {})
    g = p.gold
    storage.save_response(con, "v", p.item_id, 0, "x", str(int(g) + 1), 0, 50, 1, None, None)
    storage.save_response(con, "v", p.item_id, 1, "x", g, 1, 60, 1, None, None)
    storage.save_response(con, "v", p.item_id, 2, "x", g, 1, 60, 1, None, None)
    con.commit()
    pk = metrics.compute(con, "v")["passk"]
    assert pk["pass@1"] == 0.0
    assert pk["maj@k"] == 1.0
    assert pk["pass@k_oracle"] == 1.0


# -------------------------------------------------------- degradation uses a CI (#5)
def test_degradation_has_wilson_interval():
    items = generators.build_dataset(["arithmetic"], 1, 2, 6)
    con, ds = _db_with(items)
    runner.run(con, "c", ds, _cfg(mock="noisy"))
    cell = next(iter(next(iter(metrics.compute(con, "c")["degradation"].values())).values()))
    assert cell["lo"] <= cell["mean"] <= cell["hi"]
    assert {"mean", "lo", "hi", "n", "std"} <= set(cell)


# -------------------------------------------------------------- multi-turn inject (#WS7b)
def test_multi_turn_build_messages_emits_user_per_turn():
    p = generators._mk("multi_turn_inject", 2, 5, 0, False, "base", "g")
    item = p.row()
    msgs = runner.build_messages(item, ask_confidence=False)
    assert msgs[0]["role"] == "system"
    # Alternating transcript: system, user, assistant, user, assistant, ...
    # so consecutive same-role messages are never emitted (axk.1).
    turns = p.turns
    expected_len = 1 + 2 * len(turns) - 1
    assert len(msgs) == expected_len
    roles = [m["role"] for m in msgs]
    assert roles == ["system"] + [r for _ in turns
                                  for r in ("user", "assistant")][:2 * len(turns) - 1]
    assert turns[0] in msgs[1]["content"]
    assert "ANSWER:" in msgs[-1]["content"]


def test_multi_turn_gold_verifies_from_full_prompt():
    p = generators._mk("multi_turn_inject", 3, 7, 0, False, "base", "g")
    assert generators.verify_gold(p)
    bad = dataclasses.replace(p, gold=str(int(p.gold) + 1))
    assert generators.verify_gold(bad) is False


def test_multi_turn_runs_through_mock():
    items = generators.build_dataset(["multi_turn_inject"], 1, 3, 3)
    con, ds = _db_with(items)
    runner.run(con, "mt", ds, _cfg(mock="perfect"))
    res = metrics.compute(con, "mt")
    assert res["coverage"]["errored"] == 0
    assert res["overall_accuracy"] == 1.0


def test_single_turn_families_unaffected_by_turns_field():
    p = generators._mk("arithmetic", 2, 3, 0, False, "base", "g")
    assert p.turns is None
    msgs = runner.build_messages(p.row(), ask_confidence=False)
    assert len(msgs) == 2
    assert "ANSWER:" in msgs[-1]["content"]


# ----------------------------------------------------------- provider registry (feature)
_REG = {
    "providers": {
        "ollama": {"base_url": "http://localhost:11434/v1"},
        "openai": {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
    },
    "models": {
        "gpt4omini": {"provider": "openai", "model": "gpt-4o-mini",
                      "context_window": 128000, "max_tokens": 4096},
    },
}


def test_resolve_model_alias():
    ep = providers.resolve(_REG, "gpt4omini", None, None, "KEY")
    assert ep["base_url"].endswith("openai.com/v1")
    assert ep["model"] == "gpt-4o-mini"
    assert ep["context_window"] == 128000 and ep["max_tokens"] == 4096
    assert ep["api_key"] == "KEY"


def test_resolve_raw_model_falls_back_to_default_provider():
    ep = providers.resolve(_REG, "llama3.2:3b", None, None, None, default_provider="ollama")
    assert ep["base_url"].endswith("11434/v1")
    assert ep["model"] == "llama3.2:3b"


def test_resolve_base_url_override_wins():
    ep = providers.resolve(_REG, "gpt4omini", None, "http://local/v1", None)
    assert ep["base_url"] == "http://local/v1" and ep["model"] == "gpt-4o-mini"


def test_resolve_unknown_provider_raises():
    with pytest.raises(ValueError):
        providers.resolve(_REG, "x", "nope", None, None)


def test_shipped_providers_json_is_valid():
    reg = providers.load()                 # the real providers.json next to the code
    assert reg["providers"] and reg["models"]
    for name, m in reg["models"].items():
        assert m["provider"] in reg["providers"], f"{name} -> unknown provider"

def test_retroactive_edit_gold_verifies_end_to_end():
    p = generators._mk("retroactive_edit", 3, 7, 0, False, "base", "g")
    assert generators.verify_gold(p)


def test_retroactive_edit_changes_gold_vs_unedited():
    p = generators._mk("retroactive_edit", 3, 7, 0, False, "base", "g")
    # The prompt contains an "Actually..." edit clause.
    assert "Actually," in p.prompt
    # A corrupted gold must fail verification.
    assert not generators.verify_gold(dataclasses.replace(p, gold=str(int(p.gold) + 1)))


def test_retroactive_edit_factor_is_load_bearing():
    # bench-5zn: the queried container is the EDITED one, so the retroactive
    # factor always affects the answer -- re-deriving with the factor changed by
    # 1 must change the gold (the "Actually..." pivot is never a no-op).
    for seed in (7, 19, 23):
        p = generators._mk("retroactive_edit", 3, seed, 0, False, "base", "g")
        m = re.search(r"(held )(\d+)( times as many)", p.prompt)
        assert m, "expected an 'Actually... N times as many' edit clause"
        perturbed = p.prompt[:m.start(2)] + str(int(m.group(2)) + 1) + p.prompt[m.end(2):]
        assert generators._verify_retroactive_edit(perturbed, p.gold) is False


def test_token_entropy_stats_math():
    # bench-aes: entropy over the renormalized top-k distribution.
    import math as _m
    uniform = [{"token": "a", "logprob": _m.log(0.5),
                "top_logprobs": [{"token": "a", "logprob": _m.log(0.5)},
                                 {"token": "b", "logprob": _m.log(0.5)}]}]
    s = runner.token_entropy_stats(uniform)
    assert s["token_entropy_mean"] == pytest.approx(1.0, abs=1e-9)   # 2 equal -> 1 bit
    assert s["token_entropy_max"] == pytest.approx(1.0, abs=1e-9)
    assert s["logprob_divergence_spikes"] == 0
    # a low-probability committed token is a divergence spike
    spike = [{"token": "x", "logprob": -3.0,
              "top_logprobs": [{"token": "x", "logprob": -3.0},
                               {"token": "y", "logprob": -0.05}]}]
    assert runner.token_entropy_stats(spike)["logprob_divergence_spikes"] == 1
    assert runner.token_entropy_stats([]) is None


def test_logprob_telemetry_end_to_end(monkeypatch):
    # bench-aes: with a 'logprobs' capability and a server that returns logprobs,
    # token-entropy telemetry is computed and stored (observable on open weights).
    items = generators.build_dataset(["arithmetic"], 1, 1, 1, verify=False)
    con, ds = _db_with(items)
    body = {
        "choices": [{
            "message": {"content": f"ANSWER: {ds[0]['gold']}"},
            "logprobs": {"content": [
                {"token": "A", "logprob": -0.01,
                 "top_logprobs": [{"token": "A", "logprob": -0.01},
                                  {"token": "B", "logprob": -4.0}]},
                {"token": "7", "logprob": -3.0,
                 "top_logprobs": [{"token": "7", "logprob": -3.0},
                                  {"token": "8", "logprob": -0.05}]},
            ]},
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(body).encode()

    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *a, **k: _Resp())
    cfg = _cfg(model="m", base_url="http://x/v1", capabilities=["logprobs"], n=1)
    runner.run(con, "r", ds, cfg)
    tel = storage.load_telemetry(con, "r", ds[0]["item_id"], 0)
    assert tel is not None
    assert tel["token_entropy_mean"] is not None
    assert tel["logprob_divergence_spikes"] == 1            # the '7' token at logprob -3.0
    # and the unobservable fields are honestly annotated, not faked
    assert tel["unobservable_fields"].get("tot_branch_map") == "unobservable_without_raw_cot"
