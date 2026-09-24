"""Persistent Coworld decision bridge over the deterministic CitySim episode."""

import hashlib
import json
import sys

from citysim.policies.strategies import BalancedBaseline
from citysim.simulation.engine import Episode
from citysim.simulation.interventions import INTERVENTIONS, check_precondition
from citysim.simulation.world import BLOCK_ORDER


GLOBAL_VALUES = (
    "day", "days_remaining", "budget", "daily_upkeep", "action_points_remaining", "population",
    "average_mood", "median_rent", "average_rent_burden", "mobility", "cleanliness", "business_health", "health",
)
BLOCK_VALUES = (
    "population", "average_mood", "average_rent_burden", "cleanliness", "transit_access",
    "recreation_access", "healthcare_access", "food_access", "perceived_safety",
)
ACTIONS = [{"type": "end_day"}] + [
    {"type": "action", "action": action, "target_id": block_id}
    for action in sorted(INTERVENTIONS)
    for block_id in BLOCK_ORDER
]


class Bridge:
    def __init__(self):
        self.episode = None
        self.teacher = None
        self.decision_id = 0

    def legal(self, action):
        if action["type"] == "end_day":
            return True
        city = self.episode.city
        intervention = INTERVENTIONS[action["action"]]
        return (
            city.action_points > 0
            and intervention.cost <= city.budget
            and check_precondition(city, action["action"], city.blocks[action["target_id"]]) is None
        )

    def decision(self):
        dashboard = self.episode.dashboard()
        self.decision_id += 1
        legal_actions = [action for action in ACTIONS if self.legal(action)]
        return {
            "kind": "decision", "game": "citysim", "decision_id": self.decision_id,
            "seat": 0, "engine_seat": 0, "turn": dashboard["day"],
            "semantic_view": dashboard, "inbox": [],
            "messages": [
                {"role": "system", "content": "Manage CitySim. Choose one legal intervention or end the day."},
                {"role": "user", "content": json.dumps(dashboard, separators=(",", ":"))},
            ],
            "speech_messages": [], "action_schema": {"enum": legal_actions}, "typed_question": None,
        }

    def handle(self, command):
        kind = command["kind"]
        if kind == "reset":
            if command["players"] != 1:
                raise ValueError("CitySim has one manager seat")
            seed = int.from_bytes(hashlib.sha256(command["seed"].encode()).digest()[:4], "big")
            self.episode = Episode(seed=seed)
            self.teacher = BalancedBaseline(seed=seed)
            self.teacher.on_welcome(self.episode.handshake())
            self.decision_id = 0
            return self.decision()
        if self.episode is None or self.episode.finished:
            raise ValueError("Reset before requesting a decision")
        if kind == "encode":
            dashboard = self.episode.dashboard()
            blocks = {row["block_id"]: row for row in dashboard["blocks"]}
            return {
                "decision_id": self.decision_id,
                "values": [dashboard[field] for field in GLOBAL_VALUES] + [
                    blocks[block_id][field] for block_id in BLOCK_ORDER for field in BLOCK_VALUES
                ],
                "actions": [action if self.legal(action) else None for action in ACTIONS],
            }
        if kind == "teacher":
            candidates = self.teacher.plan(self.episode.dashboard())
            action = next(
                (candidate for candidate in candidates if self.legal({"type": "action", **candidate})),
                None,
            )
            return {"response": json.dumps({"type": "action", **action} if action else ACTIONS[0])}
        if kind != "step":
            raise ValueError("Unknown bridge command")
        if command["decision_id"] != self.decision_id:
            raise ValueError("Decision ID does not match the current CitySim state")
        action = json.loads(command["response"])
        if action not in ACTIONS or not self.legal(action):
            return {"kind": "rejected", "reason": "Action is not legal in the current city state"}
        if action["type"] == "action":
            result = self.episode.submit_action(action["action"], action["target_id"])
            if not result.ok:
                raise RuntimeError(result.payload)
            self.teacher.note_accepted(action["action"], action["target_id"])
        else:
            self.episode.end_day()
        if self.episode.finished:
            score = self.episode.results()["score"]
            observation = {"kind": "terminal", "scores": {"0": score}, "utilities": {"0": score / 50 - 1}}
        else:
            observation = self.decision()
        return {"kind": "accepted", "action": action, "observation": observation}


def main():
    bridge = Bridge()
    for line in sys.stdin:
        print(json.dumps(bridge.handle(json.loads(line)), separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
