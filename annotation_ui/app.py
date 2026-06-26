#!/usr/bin/env python
"""annotation_ui/app.py — faithful Phase-2 human-perception study (15/30 rule).

A minimal Flask app that presents candidate fake face images one at a time
(blind), collects "looks real / looks fake" judgements, and exports a tally
``data/work/votes.json`` = {uid: n_real_votes} consumed by
``scripts/04_phase2_select.py`` in ``manual`` mode.

Run:
    python annotation_ui/app.py --images data/work/aligned --port 5000
Then have ~30 participants each judge the set. When done, visit /export.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_file, session

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "work"

app = Flask(__name__)
app.secret_key = os.environ.get("ANNO_SECRET", "id-test-anno-dev")
STATE_FILE = WORK / "anno_state.json"

# Populated at startup: uid -> absolute image path.
CANDIDATES: dict[str, str] = {}


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"participants": 0, "judgements": []}  # judgements: [{uid, pid, choice}]


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _pid() -> int:
    state = _load_state()
    if "pid" not in session:
        state["participants"] = state.get("participants", 0) + 1
        session["pid"] = state["participants"]
        _save_state(state)
    return int(session["pid"])


@app.route("/")
def index():
    return redirect("/judge")


@app.route("/img/<uid>")
def img(uid):
    path = CANDIDATES.get(uid)
    if not path or not Path(path).exists():
        return "not found", 404
    return send_file(path)


@app.route("/judge")
def judge():
    """Show the next image this participant has not yet judged (< 30 total)."""
    state = _load_state()
    pid = _pid()
    judged = {j["uid"] for j in state["judgements"] if j["pid"] == pid}
    counts: dict[str, int] = {}
    for j in state["judgements"]:
        counts[j["uid"]] = counts.get(j["uid"], 0) + 1
    target_n = app.config["TARGET_N"]
    for uid in CANDIDATES:  # deterministic order
        if uid in judged:
            continue
        if counts.get(uid, 0) >= target_n:
            continue
        return render_template("judge.html", uid=uid, pid=pid, n_done=len(judged),
                               n_total=len(CANDIDATES))
    return render_template("done.html", pid=pid)


@app.route("/vote", methods=["POST"])
def vote():
    uid = request.form.get("uid")
    choice = request.form.get("choice")  # "real" | "fake"
    if uid not in CANDIDATES or choice not in ("real", "fake"):
        return jsonify({"ok": False}), 400
    state = _load_state()
    state["judgements"].append({"uid": uid, "pid": _pid(), "choice": choice, "key": uuid.uuid4().hex})
    _save_state(state)
    return redirect("/judge")


@app.route("/results")
def results():
    state = _load_state()
    tally: dict[str, int] = {}
    for j in state["judgements"]:
        if j["choice"] == "real":
            tally[j["uid"]] = tally.get(j["uid"], 0) + 1
    return render_template("results.html", tally=tally, n_judgements=len(state["judgements"]))


@app.route("/export")
def export():
    """Write data/work/votes.json = {uid: n_real_votes} for Phase-2 manual mode."""
    state = _load_state()
    tally: dict[str, int] = {}
    for j in state["judgements"]:
        if j["choice"] == "real":
            tally[j["uid"]] = tally.get(j["uid"], 0) + 1
    out = WORK / "votes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tally, indent=2))
    return jsonify({"wrote": str(out), "uids": len(tally)})


def _load_candidates(images_dir: str, index_path: str | None):
    global CANDIDATES
    CANDIDATES = {}
    if index_path and Path(index_path).exists():
        for e in json.loads(Path(index_path).read_text()):
            if e.get("split") == "fake":
                CANDIDATES[e["uid"]] = e["path"]
    if images_dir:
        for p in sorted(Path(images_dir).glob("*.png")):
            CANDIDATES.setdefault(p.stem, str(p))
    print(f"[anno] loaded {len(CANDIDATES)} candidate images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(WORK / "aligned"))
    ap.add_argument("--index", default=str(WORK / "aligned_index.json"))
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--participants", type=int, default=30, help="target judgements per image")
    args = ap.parse_args()
    app.config["TARGET_N"] = args.participants
    _load_candidates(args.images, args.index)
    app.run(port=args.port, debug=False)
