import gzip
import json
from collections import Counter
from pathlib import Path

from pokerkit import HandHistory

# Location of your downloaded dataset
DATA_PATH = Path("data/raw/acpc-fixedlimit.phhs.gz")

# Check that the file exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

with gzip.open(DATA_PATH, "rb") as f:
    hands = list(HandHistory.load_all(f))

print("Total hands:", len(hands))

# Keep two-player Fixed-Limit Hold'em
heads_up_hands = [
    h for h in hands
    if h.variant == "FT"
    and len(h.starting_stacks) == 2
]

print("\nHeads-up Fixed-Limit hands:", len(heads_up_hands))

def snapshot(state):
    return {
        "actor_index": state.actor_index,
        "bets": list(state.bets),
        "stacks": list(state.stacks),
        "board_cards": [
            str(card)
            for group in state.board_cards
            for card in group
        ],
        "hole_cards": [
            [str(card) for card in cards]
            for cards in state.hole_cards
        ],
        "call_amount": state.checking_or_calling_amount,
    }

print("\n=== EXTRACTING TRAINING EXAMPLES ===")

# RLCard Limit Hold'em action IDs
ACTION_IDS = {
    "call": 0,
    "raise": 1,
    "fold": 2,
    "check": 3
}

def get_action_label(action, call_amount):
    parts = action.split()
    action_code = parts[1]

    if action_code == "f":
        return "fold"
    elif action_code == "cc":
        return "call" if call_amount > 0 else "check"
    elif action_code == "cbr":
        return "raise"

    return None


records = []


for hand_id, hand in enumerate(heads_up_hands):
    previous = None

    for state, action in hand.state_actions:

        # Only include actual betting decisions
        parts = action.split() if action else []

        is_betting_action = (
            len(parts) >= 2
            and parts[0].startswith("p")
            and parts[1] in {"f", "cc", "cbr"}
        )

        if is_betting_action:
            player = int(parts[0][1:]) - 1

            if previous is None:
                raise ValueError("Missing pre-action state")

            if previous["actor_index"] != player:
                raise ValueError(
                    f"Actor mismatch in hand {hand_id}: {action}"
                )

            label = get_action_label(
                action,
                previous["call_amount"]
            )

            if label is None:
                raise ValueError(f"Unknown action: {action}")

            record = {
                "hand_id": hand_id,
                "player": player,
                "hole_cards": previous["hole_cards"][player],
                "board_cards": previous["board_cards"],
                "bets": previous["bets"],
                "stacks": previous["stacks"],
                "call_amount": previous["call_amount"],
                "action": label,
                "action_id": ACTION_IDS[label]
            }

            records.append(record)

        # Capture values before the next replay transition
        previous = snapshot(state)


# Save the extracted decisions
output_path = Path("data/processed/decisions.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")


print("Total decisions:", len(records))
print("Action distribution:",
      Counter(r["action"] for r in records))
print("\nSaved to:", output_path)

assert len(records) > 0, "No training examples extracted"
assert all(0 <= r["action_id"] <= 3 for r in records)