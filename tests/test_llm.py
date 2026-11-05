import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from informatics import llm
from informatics.cli import main
from informatics.keywords import THEMES
from informatics.llm import (
    THEME_DEFINITIONS,
    LlmClassifier,
    LlmConfig,
    custom_id,
    decisions_from_answers,
    parse_result,
    rubric_text,
)


# A stand-in for anthropic.Anthropic().messages.batches ----------------------------------------
def _text_result(request_id: str, payload: object, stop_reason: str = "end_turn"):
    content = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(
            type="text", text=payload if isinstance(payload, str) else json.dumps(payload)
        ),
    ]
    message = SimpleNamespace(content=content, stop_reason=stop_reason, stop_details=None)
    return SimpleNamespace(
        custom_id=request_id, result=SimpleNamespace(type="succeeded", message=message)
    )


def _refusal(request_id: str):
    details = SimpleNamespace(category="bio", explanation="declined")
    message = SimpleNamespace(content=[], stop_reason="refusal", stop_details=details)
    return SimpleNamespace(
        custom_id=request_id, result=SimpleNamespace(type="succeeded", message=message)
    )


def _error(request_id: str, kind: str):
    response = SimpleNamespace(
        type="error", error=SimpleNamespace(type=kind)
    )  # as the SDK nests it
    outcome = SimpleNamespace(type="errored", error=response)
    return SimpleNamespace(custom_id=request_id, result=outcome)


def judge(request: dict):
    """Answers like a careful reader of the synthetic corpus would."""
    text = request["params"]["messages"][0]["content"]
    themes = [t for t in ("privacy", "bias") if t in text.lower()]
    relevant = "gradient boosting" not in text
    payload = {"rationale": "synthetic", "relevant": relevant, "themes": themes + ["nonsense"]}
    return _text_result(request["custom_id"], payload)


class FakeBatches:
    def __init__(self, answer=judge, polls_before_end: int = 0):
        self.answer, self.polls_before_end = answer, polls_before_end
        self.created: dict[str, list[dict]] = {}
        self.remaining: dict[str, int] = {}

    def create(self, requests):
        batch_id = f"msgbatch_{len(self.created):03d}"
        self.created[batch_id] = list(requests)
        self.remaining[batch_id] = self.polls_before_end
        return SimpleNamespace(id=batch_id, processing_status="in_progress")

    def retrieve(self, batch_id):
        ended = self.remaining[batch_id] <= 0
        self.remaining[batch_id] -= 1
        return SimpleNamespace(id=batch_id, processing_status="ended" if ended else "in_progress")

    def results(self, batch_id):
        yield from (self.answer(r) for r in reversed(self.created[batch_id]))  # any order

    def sent(self) -> list[str]:
        return [r["custom_id"] for requests in self.created.values() for r in requests]


def fake_client(batches: FakeBatches):
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


@pytest.fixture
def candidates(corpus):
    kinds = corpus["kind"]
    papers = corpus[kinds.isin(["ai_medicine", "wide_relevant", "wide_irrelevant"])]
    papers = papers.assign(keyword_rule=papers["kind"] == "ai_medicine")
    return papers.reset_index(drop=True)


def _classifier(tmp_path: Path, batches: FakeBatches, **config) -> LlmClassifier:
    return LlmClassifier(
        tmp_path / "answers.parquet",
        tmp_path / "batches.json",
        LlmConfig(**{"batch_size": 50, **config}),
        client=fake_client(batches),
        sleep=lambda seconds: None,
    )


# Tests -----------------------------------------------------------------------------------------
def test_every_theme_is_defined_in_the_rubric():
    assert list(THEME_DEFINITIONS) == list(THEMES)
    text = rubric_text()
    assert all(f"- {name}: " in text for name in THEMES) and "{themes}" not in text


def test_request_follows_the_api_shape():
    request = LlmConfig().request("cs/0501001v2", "Title: x\n\nAbstract: y")
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request["custom_id"])
    assert request["custom_id"] == custom_id("cs/0501001v2") != custom_id("cs/0501002v2")
    params = request["params"]
    assert params["model"] == "claude-opus-5-5" and params["max_tokens"] == 8000
    assert "thinking" not in params and "temperature" not in params
    assert params["messages"] == [{"role": "user", "content": "Title: x\n\nAbstract: y"}]
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    schema = params["output_config"]["format"]["schema"]
    assert params["output_config"]["effort"] == "low"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["themes"]["items"]["enum"] == list(THEMES)


def test_run_submits_collects_and_resumes(tmp_path, candidates):
    batches = FakeBatches()
    classifier = _classifier(tmp_path, batches)
    counts = classifier.run(candidates)
    assert counts == {"ok": len(candidates), "waiting": 0, "unanswered": 0}
    assert len(batches.created) == 4  # 200 papers in batches of 50
    answers = classifier.answers().set_index("id")
    kinds = candidates.set_index("id")["kind"]
    assert (answers["relevant"] == (kinds[answers.index] != "wide_irrelevant")).all()
    assert all(set(t) <= set(THEMES) for t in answers["themes"])  # unknown names dropped
    assert (answers["model"] == "claude-opus-5-5").all() and answers["batch_id"].notna().all()

    classifier.run(candidates)  # nothing new to send
    assert len(batches.created) == 4
    # A changed rubric (or model, or abstract) means new answers are needed.
    changed = _classifier(tmp_path, batches, rubric="Only imaging papers are relevant. {themes}")
    changed.run(candidates.head(10))
    assert len(batches.sent()) == len(candidates) + 10


def test_waiting_for_batches(tmp_path, candidates):
    batches = FakeBatches(polls_before_end=2)
    papers = candidates.head(20)
    counts = _classifier(tmp_path, batches).run(papers)
    assert counts["waiting"] == 20 and "ok" not in counts  # submitted, not finished
    sleeps = []
    classifier = _classifier(tmp_path, batches)
    classifier.sleep = sleeps.append
    counts = classifier.run(papers, wait=True, poll_seconds=5)
    assert counts["ok"] == 20 and counts["waiting"] == 0 and sleeps == [5]
    assert len(batches.created) == 1  # the pending batch was collected, not sent again


def test_failures_are_recorded_and_retried(tmp_path, candidates):
    papers = candidates.head(5)
    ids = papers["id"].tolist()
    outcomes = {
        custom_id(ids[0]): lambda r: _refusal(r["custom_id"]),
        custom_id(ids[1]): lambda r: _error(r["custom_id"], "invalid_request_error"),
        custom_id(ids[2]): lambda r: _error(r["custom_id"], "api_error"),
        custom_id(ids[3]): lambda r: _text_result(r["custom_id"], "{", "max_tokens"),
    }
    calls: dict[str, int] = {}

    def answer(request):
        calls[request["custom_id"]] = calls.get(request["custom_id"], 0) + 1
        return outcomes.get(request["custom_id"], judge)(request)

    batches = FakeBatches(answer)
    counts = _classifier(tmp_path, batches).run(papers)
    status = _classifier(tmp_path, batches).answers().set_index("id")["status"]
    assert status[ids[0]] == "refused" and status[ids[1]] == "invalid"
    assert status[ids[2]] == status[ids[3]] == "retry" and status[ids[4]] == "ok"
    assert calls[custom_id(ids[2])] == 3  # retried up to 3 rounds in one run
    assert calls[custom_id(ids[0])] == 1  # refusals are final ...
    assert counts["refused"] == 1 and counts["ok"] == 1
    _classifier(tmp_path, batches).run(papers, retry_failed=True)  # ... unless asked
    assert calls[custom_id(ids[0])] == 2 and calls[custom_id(ids[1])] == 2


def test_parse_result_reads_the_json_answer():
    good = parse_result(_text_result("p1", {"rationale": "r", "relevant": True,
                                            "themes": ["bias", "bias"]}))  # fmt: skip
    assert good == {"status": "ok", "relevant": True, "themes": ["bias"], "rationale": "r"}
    assert parse_result(_text_result("p1", "not json"))["status"] == "retry"
    assert parse_result(_refusal("p1")) == {"status": "refused", "refusal_category": "bio"}
    expired = SimpleNamespace(custom_id="p1", result=SimpleNamespace(type="expired"))
    assert parse_result(expired) == {"status": "retry", "error": "expired"}


def test_decisions_fall_back_to_the_keyword_rule():
    candidates = pd.DataFrame({"id": ["a", "b", "c"], "keyword_rule": [False, True, True]})
    answers = pd.DataFrame(
        {"id": ["a", "b", "c"], "status": ["ok", "refused", "ok"], "relevant": [True, None, False]}
    )
    table, info = decisions_from_answers(candidates, answers, pd.Series({"c": True}), "m")
    assert table["predicted"].tolist() == [True, True, False]  # b: keyword rule
    assert table["relevant"].tolist() == [True, True, True]  # c: the hand label wins
    assert info["keyword_rule_fallback"] == 1 and info["refused"] == 1
    assert info["labels_used"] == 1 and info["method"] == "llm"


def test_missing_sdk_has_install_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_anthropic(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_anthropic)
    with pytest.raises(RuntimeError, match=r"\.\[llm\]"):
        llm.make_client()


def test_language_model_through_the_cli(layout, corpus, monkeypatch, capsys):
    monkeypatch.chdir(layout.data_dir.parent)
    batches = FakeBatches()
    monkeypatch.setattr(llm, "make_client", lambda: fake_client(batches))
    assert main(["filter", "--set", "cs"]) == 0
    assert main(["classify", "--set", "cs", "--method", "llm", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would be sent to claude-opus-5-5" in out and '"custom_id"' in out
    assert not batches.created  # a dry run sends nothing

    # Label some papers, check the model on them first, then classify everything.
    assert main(["label-sample", "--set", "cs", "--n-strict", "20", "--n-wide", "20",
                 "--n-per-stratum", "3"]) == 0  # fmt: skip
    sheet_path = Path("labels/cs_to_label.csv")
    sheet = pd.read_csv(sheet_path, dtype=str, keep_default_na=False)
    kinds = corpus.set_index("id")["kind"]
    relevant = kinds[sheet["id"]].isin(["ai_medicine", "duplicate", "wide_relevant"])
    sheet["relevant"] = relevant.map({True: "1", False: "0"}).to_numpy()
    sheet.to_csv(sheet_path, index=False)
    assert main(["classify", "--set", "cs", "--method", "llm", "--labelled-only"]) == 0
    assert "selection is unchanged" in capsys.readouterr().out
    assert not Path("data/interim/cs_relevance.parquet").exists()
    assert main(["label-evaluate", "--set", "cs"]) == 0
    assert "Language model: precision 100.0%" in capsys.readouterr().out

    assert main(["classify", "--set", "cs", "--method", "llm"]) == 0
    assert "candidates relevant" in capsys.readouterr().out
    decisions = pd.read_parquet("data/interim/cs_relevance.parquet").set_index("id")
    assert not kinds[decisions.index[decisions["relevant"]]].eq("wide_irrelevant").any()
    assert main(["analyze", "--set", "cs", "--themes", "llm", "--min-df", "3",
                 "--cluster-min-df", "3", "--k", "4"]) == 0  # fmt: skip
    themes = pd.read_csv("results/cs/themes.csv").set_index("tag")
    assert themes.loc["privacy", "papers"] > 0 and themes.loc["regulation", "papers"] == 0
    report = Path("results/cs/report.md").read_text(encoding="utf-8")
    assert "a language model (claude-opus-5-5)" in report and "judged by a language model" in report


def test_against_the_real_sdk(tmp_path):
    """The real anthropic SDK, talking to a local server that answers like the Batches API."""
    anthropic = pytest.importorskip("anthropic")
    import http.server
    import threading

    received: dict = {}

    def batch(status: str) -> dict:
        return {
            "id": "msgbatch_1", "type": "message_batch", "processing_status": status,
            "request_counts": {"processing": 0, "succeeded": 2, "errored": 1, "canceled": 0,
                               "expired": 1},
            "ended_at": None, "created_at": "2026-11-01T00:00:00Z",
            "expires_at": "2026-11-02T00:00:00Z", "cancel_initiated_at": None,
            "archived_at": None,
            "results_url": f"{base}/v1/messages/batches/msgbatch_1/results",
        }  # fmt: skip

    def message(text: str, stop_reason: str = "end_turn", **extra) -> dict:
        return {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": text}], "stop_reason": stop_reason,
            "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}, **extra,
        }  # fmt: skip

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body: bytes, kind: str = "application/json") -> None:
            self.send_response(200)
            self.send_header("content-type", kind)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            received["body"] = json.loads(self.rfile.read(int(self.headers["content-length"])))
            self._send(json.dumps(batch("in_progress")).encode())

        def do_GET(self):  # noqa: N802
            if not self.path.endswith("/results"):
                self._send(json.dumps(batch("ended")).encode())
                return
            ids = [r["custom_id"] for r in received["body"]["requests"]]
            answer = json.dumps({"rationale": "r", "relevant": True, "themes": ["privacy"]})
            refusal = {"type": "refusal", "category": "bio", "explanation": "no"}
            invalid = {"type": "error", "error": {"type": "invalid_request_error", "message": "x"}}
            lines = [
                {"custom_id": ids[0], "result": {"type": "succeeded", "message": message(answer)}},
                {"custom_id": ids[1], "result": {"type": "succeeded", "message": message(
                    "", "refusal", stop_details=refusal)}},
                {"custom_id": ids[2], "result": {"type": "errored", "error": invalid}},
                {"custom_id": ids[3], "result": {"type": "expired"}},
            ]  # fmt: skip
            self._send("\n".join(json.dumps(x) for x in lines).encode(), "application/binary")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = anthropic.Anthropic(api_key="test-key", base_url=base, max_retries=0)
        papers = pd.DataFrame(
            {"id": ["2101.00001", "cs/0501001", "pmid:1", "2101.00004"], "title": list("ABCD"),
             "abstract": list("abcd")}
        )  # fmt: skip
        classifier = LlmClassifier(
            tmp_path / "a.parquet", tmp_path / "b.json", LlmConfig(), client=client
        )
        counts = classifier.run(papers, max_rounds=1)
    finally:
        server.shutdown()
    status = classifier.answers().set_index("id")
    assert status.loc["2101.00001", "status"] == "ok" and status.loc["2101.00001", "relevant"]
    assert status.loc["cs/0501001", "refusal_category"] == "bio"
    assert status.loc["pmid:1", "status"] == "invalid"
    assert status.loc["2101.00004", "error"] == "expired"
    assert counts["ok"] == 1 and counts["retry"] == 1
    params = received["body"]["requests"][0]["params"]
    assert params["output_config"]["format"]["type"] == "json_schema"


def test_a_lost_connection_to_the_api_loses_nothing(layout, monkeypatch, capsys):
    anthropic = pytest.importorskip("anthropic")

    class ConnectionLost(anthropic.APIError):
        def __init__(self):
            Exception.__init__(self, "Connection error.")

    class Offline(FakeBatches):
        def retrieve(self, batch_id):
            raise ConnectionLost()

    batches = Offline()
    monkeypatch.chdir(layout.data_dir.parent)
    monkeypatch.setattr(llm, "make_client", lambda: fake_client(batches))
    assert main(["filter", "--set", "cs"]) == 0
    assert main(["classify", "--set", "cs", "--method", "llm", "--max-papers", "5"]) == 1
    err = capsys.readouterr().err
    assert "Nothing is lost: submitted batches keep running" in err
    assert len(batches.created) == 1  # the batch was submitted and recorded before the failure
    recorded = json.loads(Path("data/interim/cs_llm_batches.json").read_text())["batches"]
    assert recorded[0]["id"] == "msgbatch_000"
