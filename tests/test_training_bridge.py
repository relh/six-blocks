"""Exercise a complete CitySim training episode and its legal action mask."""

import json
import subprocess
import sys


def test_teacher_completes_game_with_stable_numeric_codec():
    process = subprocess.Popen(
        [sys.executable, "-m", "citysim.training_bridge"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    assert process.stdin is not None and process.stdout is not None

    def request(command):
        process.stdin.write(json.dumps(command) + "\n")
        process.stdin.flush()
        return json.loads(process.stdout.readline())

    try:
        observation = request({"kind": "reset", "seed": "citysim-training-smoke", "players": 1})
        decisions = 0
        while observation["kind"] == "decision":
            encoding = request({"kind": "encode"})
            assert len(encoding["values"]) == 67
            assert len(encoding["actions"]) == 73
            assert encoding["actions"][0] == {"type": "end_day"}
            response = request({"kind": "teacher"})["response"]
            assert json.loads(response) in encoding["actions"]
            step = request({"kind": "step", "decision_id": observation["decision_id"], "response": response})
            assert step["kind"] == "accepted"
            observation = step["observation"]
            decisions += 1
            assert decisions <= 120
        assert observation["kind"] == "terminal"
        assert 0 <= observation["scores"]["0"] <= 100
        assert -1 <= observation["utilities"]["0"] <= 1
    finally:
        process.stdin.close()
        process.wait(timeout=5)
    assert process.returncode == 0
