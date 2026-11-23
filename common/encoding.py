import re
import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"

INPUT_SIZE = 108


def card_to_index(card):
    """
    Convert PokerKit or RLCard cards into an index 0-51.
    """

    # PokerKit: "JACK OF CLUBS (Jc)"
    match = re.search(r"\(([2-9TJQKA][cdhs])\)", card)

    if match:
        code = match.group(1)

    # RLCard: "CJ" (club jack)
    elif len(card) == 2 and card[0] in "CDHS":
        code = card[1] + card[0].lower()

    else:
        raise ValueError(f"Unknown card format: {card}")

    rank = RANKS.index(code[0])
    suit = SUITS.index(code[1])

    return rank * 4 + suit


def encode_cards(cards):
    """Convert cards into a 52-element vector."""

    encoded = np.zeros(52, dtype=np.float32)

    for card in cards:
        encoded[card_to_index(card)] = 1.0

    return encoded


def make_features(hole_cards, board_cards,
                  my_chips, opponent_chips,
                  call_amount, big_blind):
    """
    Shared feature layout for ACPC and RLCard.
    """

    hole = encode_cards(hole_cards)
    board = encode_cards(board_cards)

    numeric = np.array([
        len(board_cards) / 5.0,
        my_chips / big_blind,
        opponent_chips / big_blind,
        call_amount / big_blind
    ], dtype=np.float32)

    features = np.concatenate([hole, board, numeric])

    assert len(features) == INPUT_SIZE

    return features


def encode_state(record):
    """Encode a historical ACPC decision."""

    player = record["player"]
    opponent = 1 - player

    # Starting stack = 2000, big blind = 10
    # These assumptions must match the ACPC hands.
    my_chips = 2000 - record["stacks"][player]
    opponent_chips = 2000 - record["stacks"][opponent]

    return make_features(
        record["hole_cards"],
        record["board_cards"],
        my_chips,
        opponent_chips,
        record["call_amount"],
        big_blind=10.0
    )


def encode_rlcard_state(state, player_id):
    """Encode a live RLCard observation."""

    raw = state["raw_obs"]

    my_chips = raw["all_chips"][player_id]
    opponent_chips = raw["all_chips"][1 - player_id]

    call_amount = max(
        0, opponent_chips - my_chips
    )

    return make_features(
        raw["hand"],
        raw["public_cards"],
        my_chips,
        opponent_chips,
        call_amount,
        big_blind=2.0
    )