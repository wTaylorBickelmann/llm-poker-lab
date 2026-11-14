"""Baseline decisions and the duplicate-deal matcher."""

import pytest

from llm_poker_lab.bots import (
    EquityBot,
    LagBot,
    ManiacBot,
    RandomBot,
    RockBot,
    StationBot,
    TagBot,
    check_or_fold,
    make_baselines,
    position_bonus,
    pot_sized,
    sized_raise,
)
from llm_poker_lab.engine import play_hand
from llm_poker_lab.simulate import choose_lineup, deal_cards, plan_deals, simulate, sitout_cycle
from tests.support import SAMPLE_BOARD, SAMPLE_BURNS, SAMPLE_HOLES, Scripted, decision, fold_bot
import random


def test_sizing_helpers():
    """Raise sizes clamp into the legal interval and checks stay free."""
    state = decision()
    assert sized_raise(state, 1)["amount"] == 4
    assert sized_raise(state, 10_000)["amount"] == 400
    assert sized_raise(state, 12) == {"kind": "raise", "amount": 12}
    assert sized_raise(decision(can_raise=False), 12)["kind"] == "check_call"
    assert sized_raise(decision(min_raise_to=None), 12)["kind"] == "check_call"
    assert pot_sized(decision(pot=10, to_call=2, hero_bet=0, big_blind=2), 1.0) == 14
    assert check_or_fold(decision(to_call=0)) == {"kind": "check_call"}
    assert check_or_fold(decision(to_call=4)) == {"kind": "fold"}
    assert position_bonus("BTN") == 2
    assert position_bonus("unknown") == 0


def test_random_station_and_maniac():
    """The loose bots honor their probabilities and the station's price cap."""
    with pytest.raises(ValueError):
        RandomBot(fold_prob=0, call_prob=0, raise_prob=0)
    state = decision()
    assert RandomBot(fold_prob=1, call_prob=0, raise_prob=0, seed=1).act(state)["kind"] == "fold"
    assert RandomBot(fold_prob=0, call_prob=1, raise_prob=0, seed=1).act(state)["kind"] == "check_call"
    raised = RandomBot(fold_prob=0, call_prob=0, raise_prob=1, raise_bb=8, seed=1).act(state)
    assert raised == {"kind": "raise", "amount": 16}
    same = RandomBot(seed=3).act(state)
    assert same == RandomBot(seed=3).act(state)
    assert StationBot().act(state)["kind"] == "check_call"
    assert StationBot(max_call_bb=1).act(decision(to_call=4))["kind"] == "fold"
    assert StationBot(max_call_bb=1).act(decision(to_call=0))["kind"] == "check_call"
    assert ManiacBot(raise_prob=1, seed=1).act(state)["kind"] == "raise"
    assert ManiacBot(raise_prob=0, seed=1).act(state)["kind"] == "fold"
    assert ManiacBot(raise_prob=1, seed=1).act(decision(can_raise=False))["kind"] == "check_call"
    assert ManiacBot(raise_prob=0, seed=1).act(decision(to_call=0))["kind"] == "check_call"


def test_rock_and_tag_ranges():
    """Rock calls a strong hand; TAG raises it and gives up junk."""
    rock = RockBot()
    assert rock.act(decision(hole_cards=["As", "Ad"]))["kind"] == "raise"
    assert rock.act(decision(hole_cards=["Ts", "Td"]))["kind"] == "check_call"
    assert rock.act(decision(hole_cards=["7c", "2d"]))["kind"] == "fold"
    assert rock.act(decision(hole_cards=["7c", "2d"], to_call=0))["kind"] == "check_call"
    facing = decision(hole_cards=["As", "Ad"], to_call=10, pot=20, pot_odds=1 / 3)
    assert rock.act(facing)["kind"] == "check_call"
    flop = ["Ah", "7c", "2d"]
    assert rock.act(decision(street="flop", board=flop, hole_cards=["As", "Ad"], to_call=0))["kind"] == "raise"
    assert rock.act(decision(street="flop", board=flop, hole_cards=["As", "Kd"], to_call=10, pot=20))["kind"] == "check_call"
    assert rock.act(decision(street="flop", board=flop, hole_cards=["9c", "8d"], to_call=10))["kind"] == "fold"
    assert rock.act(decision(street="flop", board=flop, hole_cards=["9c", "8d"], to_call=0))["kind"] == "check_call"
    assert rock.act(decision(street="flop", board=flop, hole_cards=["As", "Ad"], can_raise=False, to_call=0))["kind"] == "check_call"

    tag = TagBot()
    assert tag.act(decision(hole_cards=["As", "Ad"]))["kind"] == "raise"
    assert tag.act(decision(hole_cards=["7c", "2d"]))["kind"] == "fold"
    assert tag.act(decision(hole_cards=["7c", "2d"], to_call=0, hero_position="BB"))["kind"] == "check_call"
    three = decision(hole_cards=["As", "Ad"], to_call=18, pot=24, pot_odds=0.4)
    assert tag.act(three)["kind"] == "raise"
    assert tag.act(decision(hole_cards=["Ts", "Td"], to_call=18, pot=24))["kind"] == "check_call"
    assert tag.act(decision(hole_cards=["7c", "2d"], to_call=18, pot=24))["kind"] == "fold"
    assert tag.act(decision(hole_cards=["As", "Ad"], can_raise=False))["kind"] == "check_call"
    assert tag.act(decision(street="flop", board=flop, hole_cards=["As", "Kd"], to_call=0))["kind"] == "raise"
    assert tag.act(decision(street="flop", board=flop, hole_cards=["9c", "8d"], to_call=0))["kind"] == "check_call"
    assert tag.act(decision(street="flop", board=flop, hole_cards=["As", "Ad"], to_call=10, pot=20))["kind"] == "raise"
    assert tag.act(decision(street="flop", board=flop, hole_cards=["As", "Kd"], to_call=10, pot=20))["kind"] == "check_call"
    weak = decision(street="flop", board=["Ah", "Kd", "8s"], hole_cards=["8c", "2d"], to_call=2, pot=20)
    assert tag.act(weak)["kind"] == "check_call"
    expensive = decision(street="flop", board=["Ah", "Kd", "8s"], hole_cards=["8c", "2d"], to_call=40, pot=20)
    assert tag.act(expensive)["kind"] == "fold"
    assert tag.act(decision(street="flop", board=flop, hole_cards=["9c", "8d"], to_call=10, pot=20))["kind"] == "fold"


def test_lag_bluffs_and_equity_thresholds(monkeypatch):
    """LAG bluffs when told to, and the equity bot follows the patched share."""
    quiet = LagBot(bluff_freq=0, seed=1)
    loud = LagBot(bluff_freq=1, seed=1)
    flop = ["Ah", "7c", "2d"]
    assert loud.act(decision(hole_cards=["7c", "2d"]))["kind"] == "raise"
    assert quiet.act(decision(hole_cards=["7c", "2d"]))["kind"] == "fold"
    assert quiet.act(decision(hole_cards=["7c", "2d"], to_call=0))["kind"] == "check_call"
    assert loud.act(decision(hole_cards=["7c", "2d"], to_call=18, pot=24))["kind"] == "raise"
    assert quiet.act(decision(hole_cards=["As", "Ad"], to_call=18, pot=24))["kind"] == "raise"
    assert quiet.act(decision(hole_cards=["As", "Jd"], to_call=18, pot=24))["kind"] == "check_call"
    assert quiet.act(decision(hole_cards=["7c", "2d"], to_call=18, pot=24))["kind"] == "fold"
    assert loud.act(decision(street="flop", board=flop, hole_cards=["9c", "8d"], to_call=10))["kind"] == "raise"
    assert loud.act(decision(street="flop", board=flop, hole_cards=["9c", "8d"], can_raise=False, to_call=0))["kind"] == "check_call"
    assert quiet.act(decision(street="flop", board=["Ah", "Kd", "8s"], hole_cards=["8c", "2d"], to_call=0))["kind"] == "raise"
    assert quiet.act(decision(street="flop", board=flop, hole_cards=["9c", "8d"], to_call=0))["kind"] == "check_call"
    assert quiet.act(decision(street="flop", board=["Ah", "Kd", "2c"], hole_cards=["As", "Kd"], to_call=8, pot=10))["kind"] == "raise"
    assert quiet.act(decision(street="flop", board=["Ah", "Kd", "8s"], hole_cards=["8c", "2d"], to_call=8, pot=10))["kind"] == "check_call"
    assert quiet.act(decision(street="flop", board=flop, hole_cards=["9c", "8d"], to_call=8, pot=10))["kind"] == "fold"

    shares = {"value": 0.9}

    def patched(*args, **kwargs):
        """Return the share the test is currently forcing."""
        return shares["value"]

    monkeypatch.setattr("llm_poker_lab.bots.equity", patched)
    bot = EquityBot(raise_equity=0.7, margin=1.0)
    assert bot.act(decision(hole_cards=["As", "Kd"]))["kind"] == "raise"
    shares["value"] = 0.1
    assert bot.act(decision(hole_cards=["7c", "2d"]))["kind"] == "fold"
    assert bot.act(decision(hole_cards=["7c", "2d"], to_call=0, hero_position="BB"))["kind"] == "check_call"
    shares["value"] = 0.9
    assert bot.act(decision(hole_cards=["As", "Kd"], can_raise=False))["kind"] == "check_call"
    shares["value"] = 0.9
    assert bot.act(decision(hole_cards=["As", "Kd"], to_call=20, pot=30, pot_odds=0.4))["kind"] == "raise"
    shares["value"] = 0.5
    assert bot.act(decision(hole_cards=["Ts", "Td"], to_call=20, pot=30, pot_odds=0.2, num_active=2))["kind"] == "check_call"
    shares["value"] = 0.05
    assert bot.act(decision(hole_cards=["7c", "2d"], to_call=20, pot=10, pot_odds=0.6))["kind"] == "fold"
    flop_state = decision(street="flop", board=flop, hole_cards=["As", "Kd"], to_call=0, num_active=3)
    shares["value"] = 0.95
    assert bot.act(flop_state)["kind"] == "raise"
    shares["value"] = 0.2
    assert bot.act(decision(street="flop", board=flop, hole_cards=["9c", "8d"], to_call=0, num_active=2))["kind"] == "check_call"
    shares["value"] = 0.5
    assert bot.act(decision(street="flop", board=flop, hole_cards=["As", "Kd"], to_call=4, pot=20, pot_odds=0.15, num_active=2))["kind"] == "check_call"
    shares["value"] = 0.05
    assert bot.act(decision(street="flop", board=flop, hole_cards=["9c", "8d"], to_call=20, pot=10, pot_odds=0.6, num_active=2))["kind"] == "fold"
    names = [bot.name for bot in make_baselines(2)]
    assert names == ["random", "station", "rock", "tag", "lag", "maniac", "equity"]


def test_schedule_and_duplicate_match():
    """Sit-outs rotate, duplicate deals balance seats, and chips are conserved."""
    rng = random.Random(0)
    holes, burns, board = deal_cards(rng)
    flat = [card for pair in holes for card in pair] + burns + board
    assert len(flat) == 20 and len(set(flat)) == 20
    bots = [fold_bot(f"b{i}") for i in range(6)]
    assert sitout_cycle(bots) == 1
    assert plan_deals(5) == 1
    with pytest.raises(ValueError):
        plan_deals(0)
    with pytest.raises(ValueError):
        choose_lineup(bots[:3], 0)
    seven = [fold_bot(f"s{i}") for i in range(7)]
    with pytest.raises(ValueError):
        choose_lineup(seven, 0, locked=tuple(bot.name for bot in seven))
    match = simulate(bots, hands=6, seed=2)
    assert match.hands_played == 6
    assert all(book.hands == 6 for book in match.books.values())
    assert sum(book.profit for book in match.books.values()) == 0
    assert all(book.summary()["bb_per_100"] == 0 for book in match.books.values())
    field = [fold_bot("hero")] + [fold_bot(f"f{i}") for i in range(6)]
    assert sitout_cycle(field, locked=("hero",)) == 6
    assert plan_deals(6, cycle=6) == 6
    seated = choose_lineup(field, 0, locked=("hero",))
    assert seated[0].name == "hero" and len(seated) == 6
    rotated = simulate(field, hands=6, seed=3, locked=("hero",))
    assert rotated.books["hero"].hands == 36
    assert rotated.books["f0"].hands == 30
    assert sum(book.profit for book in rotated.books.values()) == 0
    public = rotated.public()
    assert public["hands_played"] == 36
    assert public["hands_per_sec"] > 0
    assert public["locked"] == ["hero"]
    twins = [fold_bot("same") for _ in range(6)]
    with pytest.raises(ValueError):
        simulate(twins, hands=6)


def test_chip_conservation_and_fresh_stacks():
    """Payoffs are the chip delta from a stack that resets at the start of every hand."""
    stack = 400
    folders = [Scripted(f"f{index}", {"kind": "fold"}) for index in range(6)]
    folded = play_hand(folders, SAMPLE_HOLES, SAMPLE_BOARD, SAMPLE_BURNS, stack=stack)
    assert folded.payoffs == [-1, 1, 0, 0, 0, 0]
    assert sum(folded.payoffs) == 0
    assert folded.stacks == [stack + payoff for payoff in folded.payoffs]
    assert sum(folded.stacks) == stack * 6

    wider = play_hand(folders, SAMPLE_HOLES, SAMPLE_BOARD, SAMPLE_BURNS, sb=5, bb=10, stack=200)
    assert wider.payoffs == [-5, 5, 0, 0, 0, 0]
    assert wider.stacks == [200 + payoff for payoff in wider.payoffs]

    def mixed(state):
        """Shove from UTG, call from the hijack, and fold the rest."""
        if state["hero_index"] == 2 and state["can_raise"]:
            return {"kind": "raise", "amount": state["max_raise_to"]}
        if state["hero_index"] == 3:
            return {"kind": "check_call"}
        return {"kind": "fold"}

    swung = play_hand(
        [Scripted(f"m{index}", mixed) for index in range(6)],
        SAMPLE_HOLES,
        SAMPLE_BOARD,
        SAMPLE_BURNS,
        stack=stack,
    )
    assert sum(swung.payoffs) == 0
    assert swung.stacks == [stack + payoff for payoff in swung.payoffs]
    assert max(swung.stacks) > stack
    reset = play_hand(folders, SAMPLE_HOLES, SAMPLE_BOARD, SAMPLE_BURNS, stack=stack)
    assert reset.payoffs == folded.payoffs
    assert reset.stacks == folded.stacks

    styles = (
        {"kind": "fold"},
        {"kind": "check_call"},
        lambda state: (
            {"kind": "raise", "amount": state["max_raise_to"]}
            if state["can_raise"]
            else {"kind": "check_call"}
        ),
    )
    for seed in range(8):
        holes, burns, board = deal_cards(random.Random(seed))
        table = [Scripted(f"s{index}", styles[index % 3]) for index in range(6)]
        dealt = play_hand(table, holes, board, burns, stack=250, sb=2, bb=4)
        assert sum(dealt.payoffs) == 0
        assert dealt.stacks == [250 + payoff for payoff in dealt.payoffs]
        assert sum(dealt.stacks) == 250 * 6
        assert all(chips >= 0 for chips in dealt.stacks)

    match = simulate(make_baselines(1), hands=36, seed=7)
    assert sum(book.profit for book in match.books.values()) == 0
    assert all(book.hands > 0 for book in match.books.values())


def test_disqualification_stops_the_match():
    """A bot that flags itself is recorded and the remaining deals are skipped."""

    class Quitter:
        """Folds and then leaves the match."""

        def __init__(self, name, message):
            self.name = name
            self.disqualified = False
            self.last_error = message
            self.acts = 0

        def act(self, state):
            """Fold, and on the first decision mark the bot disqualified."""
            self.acts += 1
            self.disqualified = True
            return {"kind": "fold"}

    quitter = Quitter("evolved", "boom")
    bots = [quitter] + [fold_bot(f"f{i}") for i in range(5)]
    match = simulate(bots, hands=30, seed=1, locked=("evolved",))
    assert match.hands_played == 1
    assert match.books["evolved"].disqualified is True
    assert match.books["evolved"].error == "boom"
    silent = Scripted("quiet", {"kind": "fold"})
    silent.disqualified = True
    stopped = simulate([silent] + [fold_bot(f"g{i}") for i in range(5)], hands=12, seed=1)
    assert stopped.hands_played == 1
    assert stopped.books["quiet"].error == "disqualified"
