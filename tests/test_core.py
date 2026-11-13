"""Interface, cards, equity, stats, and one-hand engine."""

import json

import pytest

from llm_poker_lab import __version__
from llm_poker_lab.cards import DECK, postflop_tier, preflop_score
from llm_poker_lab.engine import play_hand
from llm_poker_lab.equity import equity, mix_seed, multiway, rank_index
from llm_poker_lab.interface import ACTION_DOC, STATE_DOC, STATE_KEYS, Bot, coerce
from llm_poker_lab.stats import StatBook
from tests.support import SAMPLE_BOARD, SAMPLE_BURNS, SAMPLE_HOLES, Scripted


def test_version_and_docs():
    """The package version and the public contract are importable."""
    assert __version__ == "0.1.0"
    assert "hole_cards" in STATE_DOC
    assert "check_call" in ACTION_DOC


def test_base_bot_is_abstract():
    """The documented base refuses to act until a subclass implements it."""
    with pytest.raises(NotImplementedError):
        Bot().act({})


def test_coerce_actions():
    """Illegal output becomes a fold, a check, or a clamped raise."""
    priced = {"to_call": 6, "can_raise": True, "min_raise_to": 10, "max_raise_to": 40}
    free = {"to_call": 0, "can_raise": True, "min_raise_to": 2, "max_raise_to": 40}
    assert coerce({"kind": "fold"}, priced) == {"kind": "fold"}
    assert coerce({"kind": "fold"}, free) == {"kind": "check_call"}
    assert coerce({"kind": "check_call"}, priced) == {"kind": "check_call"}
    assert coerce({"kind": "raise", "amount": 1}, priced) == {"kind": "raise", "amount": 10}
    assert coerce({"kind": "raise", "amount": 99}, priced) == {"kind": "raise", "amount": 40}
    assert coerce({"kind": "raise", "amount": "12"}, priced) == {"kind": "raise", "amount": 12}
    assert coerce({"kind": "raise", "amount": "nope"}, priced) == {"kind": "raise", "amount": 10}
    assert coerce({"kind": "raise", "amount": True}, priced) == {"kind": "raise", "amount": 10}
    assert coerce({"kind": "raise", "amount": 4.9}, priced) == {"kind": "raise", "amount": 10}
    blocked = dict(priced, can_raise=False)
    assert coerce({"kind": "raise", "amount": 20}, blocked) == {"kind": "check_call"}
    missing = dict(priced, min_raise_to=None)
    assert coerce({"kind": "raise", "amount": 20}, missing) == {"kind": "check_call"}
    assert coerce("nope", priced) == {"kind": "fold"}
    assert coerce({"kind": "dance"}, free) == {"kind": "check_call"}
    assert coerce(None, priced) == {"kind": "fold"}


def test_deck_and_preflop_scores():
    """Chen scores order the hands a baseline uses to open or fold."""
    assert len(DECK) == 52
    assert len(set(DECK)) == 52
    assert preflop_score("As", "Ad") == 20
    assert preflop_score("Ks", "Kd") == 16
    assert preflop_score("Qs", "Qd") == 14
    assert preflop_score("Ts", "Td") == 10
    assert preflop_score("2s", "2d") == 5
    assert preflop_score("As", "Ks") == 12
    assert preflop_score("As", "Qd") == 9
    assert preflop_score("As", "Jd") == 8
    assert preflop_score("As", "Td") == 6
    assert preflop_score("As", "9d") == 5
    assert preflop_score("Jc", "Tc") == 9
    assert preflop_score("7c", "2d") == -1.5
    assert preflop_score("2d", "7c") == preflop_score("7c", "2d")
    assert preflop_score("As", "Ad") > preflop_score("As", "Ks") > preflop_score("7c", "2d")


def test_postflop_tiers():
    """Made-hand tiers separate air, weak pairs, top pair, and monsters."""
    assert postflop_tier(["As", "Kd"], []) == 0
    assert postflop_tier(["As", "Kd"], ["2c"]) == 0
    assert postflop_tier(["9c", "8d"], ["Ah", "Kd", "2c"]) == 0
    assert postflop_tier(["8c", "2d"], ["Ah", "Kd", "8s"]) == 1
    assert postflop_tier(["Ac", "Kd"], ["Ah", "7c", "2d"]) == 2
    assert postflop_tier(["As", "Ad"], ["Kh", "7c", "2d"]) == 2
    assert postflop_tier(["As", "Kd"], ["Ah", "Kd", "2c"]) == 3
    assert postflop_tier(["As", "Ad"], ["Ah", "7c", "2d"]) == 4
    assert postflop_tier(["9h", "8h"], ["7c", "6d", "5s"]) == 5
    assert postflop_tier(["Ah", "Kh"], ["Qh", "7h", "2h"]) == 5
    assert postflop_tier(["As", "Ad"], ["Ah", "Kd", "Kc"]) == 6
    assert postflop_tier(["As", "Ad"], ["Ah", "Ac", "2d"]) == 6
    assert postflop_tier(["9h", "8h"], ["7h", "6h", "5h"]) == 6


def test_equity_math_and_cache(monkeypatch):
    """Monte Carlo counts wins, ties, and losses, and caches preflop classes."""
    import llm_poker_lab.equity as eq

    calls = {"n": 0}

    def hero_wins(cards):
        """Alternate scores so the first evaluate (the hero) is higher."""
        calls["n"] += 1
        return 2 if calls["n"] % 2 == 1 else 1

    monkeypatch.setattr("eval7.evaluate", hero_wins)
    assert equity(["As", "Kd"], ["2c", "3d", "4h"], iterations=3, seed=1) == 1
    calls["n"] = 0

    def hero_loses(cards):
        """Alternate scores so the villain evaluate is higher."""
        calls["n"] += 1
        return 1 if calls["n"] % 2 == 1 else 2

    monkeypatch.setattr("eval7.evaluate", hero_loses)
    assert equity(["As", "Kd"], ["2c", "3d", "4h", "5s"], iterations=2, seed=2) == 0
    monkeypatch.setattr("eval7.evaluate", lambda cards: 5)
    assert equity(["As", "Kd"], ["2c", "3d", "4h", "5s", "9c"], iterations=4, seed=3) == 0.5
    with pytest.raises(ValueError):
        equity(["As", "Kd"], iterations=0)
    before = len(eq._CACHE)
    first = equity(["Ah", "Kh"], [], iterations=5, seed=4)
    second = equity(["Kh", "Ah"], [], iterations=5, seed=4)
    assert first == second
    assert len(eq._CACHE) == before + 1
    equity(["As", "Kd"], [], iterations=5, seed=4)
    equity(["As", "Ks"], [], iterations=5, seed=4)
    equity(["Ts", "Td"], [], iterations=5, seed=4)
    assert multiway(0.5, 0) == 0.5
    assert multiway(0.5, 2) == 0.25
    assert mix_seed(["As", "Kd"], 1) == mix_seed(["As", "Kd"], 1)
    assert mix_seed(["As", "Kd"], 1) != mix_seed(["As", "Kd"], 2)
    assert rank_index("2") == 0
    assert rank_index("A") == 12


def test_real_equity_orders_hands():
    """Aces beat seven-deuce against a random hand. This uses real eval7."""
    aces = equity(["As", "Ad"], [], iterations=40, seed=7)
    junk = equity(["7c", "2d"], [], iterations=40, seed=7)
    assert aces > 0.7
    assert junk < 0.45
    assert aces > junk


def test_stat_book_interval_and_hud():
    """bb/100, the sample interval, and aggression factor use the running sums."""
    book = StatBook("tag")
    empty = book.summary()
    assert empty["hands"] == 0
    assert empty["bb_per_100"] == 0
    assert empty["aggression_factor"] is None
    book.add(0, bb=2, won=False, vpip=False, pfr=False, aggressive=0, calls=0)
    book.add(4, bb=2, won=True, vpip=True, pfr=True, aggressive=4, calls=2)
    summary = book.summary()
    assert summary["bb_per_100"] == pytest.approx(100)
    assert summary["ci_low"] == pytest.approx(-96)
    assert summary["ci_high"] == pytest.approx(296)
    assert summary["vpip"] == pytest.approx(50)
    assert summary["pfr"] == pytest.approx(50)
    assert summary["aggression_factor"] == pytest.approx(2)
    assert summary["hands_won"] == 1
    assert summary["profit_chips"] == 4
    lonely = StatBook("one")
    lonely.add(10, bb=2, won=True, vpip=True, pfr=False, aggressive=3, calls=0)
    assert lonely.summary()["ci_low"] == lonely.summary()["ci_high"]
    assert lonely.summary()["aggression_factor"] is None
    with pytest.raises(ValueError):
        book.add(0, bb=0, won=False, vpip=False, pfr=False, aggressive=0, calls=0)


def test_fold_around_and_view():
    """Five folds award the small blind to the big blind, and the view stays private."""
    spy = Scripted("spy", {"kind": "fold"})
    bots = [Scripted(f"f{i}", {"kind": "fold"}) for i in range(6)]
    bots[2] = spy
    result = play_hand(bots, SAMPLE_HOLES, SAMPLE_BOARD, SAMPLE_BURNS, hand_id=9)
    assert result.payoffs == [-1, 1, 0, 0, 0, 0]
    assert sum(result.payoffs) == 0
    assert result.vpip == [False] * 6
    assert result.pfr == [False] * 6
    view = spy.seen[0]
    assert set(view) == set(STATE_KEYS)
    assert view["hole_cards"] == ["As", "Ad"]
    assert view["hero_position"] == "UTG"
    assert view["hero_index"] == 2
    assert view["positions"][0] == "SB"
    assert view["hand_id"] == 9
    assert view["street"] == "preflop"
    assert view["to_call"] == 2
    assert view["can_raise"] is True
    blob = json.dumps(view)
    assert "Kc" not in blob
    assert "Qd" not in blob


def test_showdown_all_in_and_coercion():
    """Aces hold on this board, and illegal raise sizes are clamped."""
    def shove(state):
        """Move all-in when a raise is legal."""
        if state["can_raise"]:
            return {"kind": "raise", "amount": state["max_raise_to"]}
        return {"kind": "fold"}

    bots = [
        Scripted("sb", {"kind": "fold"}),
        Scripted("bb", {"kind": "fold"}),
        Scripted("utg", shove),
        Scripted("hj", {"kind": "check_call"}),
        Scripted("co", {"kind": "fold"}),
        Scripted("btn", {"kind": "fold"}),
    ]
    result = play_hand(bots, SAMPLE_HOLES, SAMPLE_BOARD, SAMPLE_BURNS)
    assert result.payoffs == [-1, -2, 403, -400, 0, 0]
    assert result.actions[0]["kind"] == "raise"
    assert result.actions[0]["amount"] == 400
    assert result.vpip[2] is True and result.pfr[2] is True
    assert result.aggressive[2] == 1
    assert result.vpip[3] is True and result.pfr[3] is False
    assert result.calls[3] == 1

    tiny = [Scripted(f"p{i}", {"kind": "fold"}) for i in range(6)]
    tiny[2] = Scripted("utg", {"kind": "raise", "amount": 1})
    small = play_hand(tiny, SAMPLE_HOLES, SAMPLE_BOARD, SAMPLE_BURNS)
    assert small.actions[0] == {"seat": 2, "position": "UTG", "street": "preflop", "kind": "raise", "amount": 4}
    assert small.payoffs[2] == 3

    huge = [Scripted(f"p{i}", {"kind": "fold"}) for i in range(6)]
    huge[2] = Scripted("utg", {"kind": "raise", "amount": 9999})
    big = play_hand(huge, SAMPLE_HOLES, SAMPLE_BOARD, SAMPLE_BURNS, stack=100)
    assert big.actions[0]["amount"] == 100
    assert sum(big.payoffs) == 0


def test_free_fold_is_a_check_and_crashes_fold():
    """Folding when a check is free stays in the hand. A raising bot that throws folds."""
    def maybe_fold(state):
        """Try to fold on the flop and later streets."""
        if state["street"] != "preflop":
            return {"kind": "fold"}
        return {"kind": "check_call"}

    callers = [Scripted(f"c{i}", maybe_fold) for i in range(6)]
    result = play_hand(callers, SAMPLE_HOLES, SAMPLE_BOARD, SAMPLE_BURNS)
    flop_actions = [item for item in result.actions if item["street"] == "flop"]
    assert flop_actions
    assert all(item["kind"] == "check_call" for item in flop_actions)
    assert any(item["street"] == "river" for item in result.actions)
    assert sum(result.payoffs) == 0

    def explode(state):
        """Crash on purpose."""
        raise RuntimeError("nope")

    blown = play_hand([Scripted(f"x{i}", explode) for i in range(6)], SAMPLE_HOLES, SAMPLE_BOARD, SAMPLE_BURNS)
    assert blown.payoffs == [-1, 1, 0, 0, 0, 0]


def test_play_hand_rejects_bad_deals():
    """The engine only deals six seats, five board cards, and three burns."""
    bots = [Scripted(f"p{i}", {"kind": "fold"}) for i in range(6)]
    with pytest.raises(ValueError):
        play_hand(bots[:5], SAMPLE_HOLES, SAMPLE_BOARD, SAMPLE_BURNS)
    with pytest.raises(ValueError):
        play_hand(bots, SAMPLE_HOLES, SAMPLE_BOARD[:3], SAMPLE_BURNS)
