"""Second selection stage, option 2: a language model reads each candidate and decides, from a
written rubric, whether it is about AI in medicine and which themes it discusses.

``informatics classify --set cs --method llm`` sends every candidate's title and abstract to
Claude through the Message Batches API (half the price of single requests; most batches finish
within an hour). The rubric below is the system prompt, cached across requests, and the answer
is constrained to a JSON schema: ``relevant`` (true/false), ``themes`` (names from
keywords.THEMES) and a one-sentence ``rationale``.

Runs are resumable. Submitted batches are recorded in
``data/interim/<set>_llm_batches.json``, and answers are kept in
``data/interim/<set>_llm.parquet`` together with a hash of the model, rubric and paper text, so
a paper is only sent again when one of them changes. Run the command again to collect batches
that have finished, or pass ``--wait``. Once every candidate has an answer, the decisions are
written to ``<set>_relevance.parquet`` (hand labels override them, as for the trained
classifier). Papers the model refuses or cannot answer fall back to the keyword rule and are
counted in ``<set>_relevance.json``.

Check the model against hand labels before trusting it: ``informatics label-evaluate`` reports
its precision, recall and agreement (Cohen's kappa) with your labels, and
``label-themes-sample --random N`` / ``label-themes-evaluate`` do the same for its themes.

Needs ``pip install -e ".[llm]"`` and an API key in ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

from informatics.keywords import THEMES
from informatics.normalize import clean_characters
from informatics.relevance import decisions_table
from informatics.storage import read_json, read_parquet, write_json, write_parquet

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
BATCH_SIZE = 5_000  # requests per batch; well under the 100,000-request and 256 MB limits
MAX_TOKENS = 8_000  # thinking is always on and counts toward this; batches bill actual use

THEME_DEFINITIONS: dict[str, str] = {
    "adoption": "uptake of AI in real clinical practice or health systems: how much it is used, "
    "acceptance by clinicians or patients, barriers and facilitators. Not 'we adopt a method'.",
    "diagnostics": "AI that diagnoses, detects, screens for or classifies disease, or supports "
    "diagnostic tests.",
    "personalize": "personalised or precision medicine: tailoring predictions, treatment or care "
    "to individual patients.",
    "operational efficiency": "efficiency of health-care operations: workflow, workload, time, "
    "costs, staffing or resource use.",
    "ethical": "ethical questions raised by the AI: consent, autonomy, responsibility, harm, "
    "justice. A passing 'ethics approval was obtained' does not count.",
    "legal": "law and liability: legal responsibility, malpractice, intellectual property, legal "
    "status of AI decisions.",
    "privacy": "privacy and confidentiality of patient data: de-identification, data sharing "
    "restrictions, privacy-preserving methods such as federated learning used for privacy.",
    "cybersecurity": "security of AI systems or health data against attacks or breaches, "
    "including adversarial attacks meant to cause harm.",
    "transparency": "transparency, explainability or interpretability of the model, its data or "
    "its decisions.",
    "accuracy": "reports how accurate the AI is (accuracy, AUC, sensitivity, error rates). Most "
    "papers that evaluate a model qualify.",
    "strategic implementation": "plans or strategies for implementing or deploying AI at the "
    "level of an organisation, health system or policy.",
    "maturity": "maturity of the technology or of organisations using it: maturity models, "
    "technology readiness levels, early-stage versus established use.",
    "readiness": "readiness of organisations, staff, data or infrastructure to adopt AI.",
    "bias": "unfair differences in how AI performs for or treats groups of people (sex, race, "
    "age, income, hospital), or biased data that leads to them. Statistical terms such as "
    "'inductive bias' or an MRI 'bias field' do not count.",
    "compliance": "compliance with laws, regulations, standards or guidelines (GDPR, HIPAA, "
    "clinical guidelines). Not physiological compliance such as lung compliance.",
    "regulation": "regulation and oversight of AI or medical devices: approval (e.g. FDA, CE "
    "marking), the EU AI Act, regulators. Not biological regulation such as gene regulation.",
}

RUBRIC = """You classify scientific papers for a study of research on artificial intelligence (AI)
in medicine. You will read one paper's title and abstract.

Decide "relevant": true when the paper's main contribution develops, evaluates, applies or
studies AI for human health or health care. AI includes machine learning, deep learning,
language models, computer vision, and statistical learning models used to make predictions.
Health or health care includes diagnosis, prognosis, treatment, clinical care, medical imaging,
clinical text and records, physiological or biomedical signals used for health, public health,
mental health, health services, and the ethics, regulation and adoption of AI in health care.

Decide "relevant": false when
- medicine or health is only one example among many unrelated applications;
- the paper uses no AI (for example, a clinical study analysed with conventional statistics);
- it is basic biology, chemistry or genomics without a stated application to human health;
- it applies AI outside human health (machine fault diagnosis, veterinary or plant science,
  finance), even if it borrows medical words.

Then list in "themes" every theme below that the paper substantively discusses, whether or not
it is relevant. A word in passing is not enough: the paper must say something about the theme.
Use only these names:
{themes}

Give a one-sentence "rationale" for the relevance decision."""


def rubric_text(rubric: str = RUBRIC) -> str:
    themes = "\n".join(f"- {name}: {text}" for name, text in THEME_DEFINITIONS.items())
    return rubric.replace("{themes}", themes)


def answer_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "relevant": {"type": "boolean"},
            "themes": {"type": "array", "items": {"type": "string", "enum": list(THEMES)}},
        },
        "required": ["rationale", "relevant", "themes"],
        "additionalProperties": False,
    }


def paper_message(title: object, abstract: object) -> str:
    clean = [clean_characters(v) if isinstance(v, str) else "" for v in (title, abstract)]
    return f"Title: {clean[0]}\n\nAbstract: {clean[1]}"


def custom_id(paper_id: str) -> str:
    """A batch request ID for a paper (paper IDs contain '.', '/' or ':', which batch IDs may
    not): a hash, so the mapping back needs no stored table."""
    return "p" + hashlib.sha1(paper_id.encode("utf-8")).hexdigest()[:30]


@dataclass(frozen=True)
class LlmConfig:
    model: str = DEFAULT_MODEL
    rubric: str = RUBRIC
    effort: str = "low"  # a short classification needs little thinking
    max_tokens: int = MAX_TOKENS
    batch_size: int = BATCH_SIZE

    @property
    def system_prompt(self) -> str:
        return rubric_text(self.rubric)

    def fingerprint(self, message: str) -> str:
        """Changes when the model, the rubric or the paper's text changes."""
        text = "\x00".join([self.model, self.effort, self.system_prompt, message])
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def request(self, paper_id: str, message: str) -> dict:
        """One Message Batches request (the SDK's ``Request`` is this dict)."""
        return {
            "custom_id": custom_id(paper_id),
            "params": {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": [
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [{"role": "user", "content": message}],
                "output_config": {
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": answer_schema()},
                },
            },
        }


ANSWER_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("status", pa.string()),  # ok, refused, invalid, or retry (sent again on the next run)
        ("relevant", pa.bool_()),
        ("themes", pa.list_(pa.string())),
        ("rationale", pa.string()),
        ("model", pa.string()),
        ("fingerprint", pa.string()),  # model, rubric and paper text the answer is for
        ("batch_id", pa.string()),
        ("refusal_category", pa.string()),
        ("error", pa.string()),
    ]
)
ANSWER_COLUMNS = ANSWER_SCHEMA.names
FINAL = {"ok", "refused", "invalid"}  # not sent again unless --retry-failed


def make_client() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            'the language model needs the anthropic package: pip install -e ".[llm]"'
        ) from exc
    return anthropic.Anthropic()


def parse_result(result: Any) -> dict:
    """One batch result -> status ("ok", "refused", "invalid" or "retry") and the answer."""
    outcome = result.result
    if outcome.type != "succeeded":
        # The SDK nests the error: result.error is the response (type "error"), and its .error
        # says what went wrong (e.g. "invalid_request_error", "api_error").
        response = getattr(outcome, "error", None)
        error_type = getattr(getattr(response, "error", response), "type", None) or outcome.type
        # Invalid requests fail again unchanged; errors, expiry and cancellation are retried.
        invalid = error_type in ("invalid_request_error", "invalid_request")
        return {"status": "invalid" if invalid else "retry", "error": str(error_type)}
    message = outcome.message
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        return {"status": "refused", "refusal_category": getattr(details, "category", None)}
    if message.stop_reason == "max_tokens":
        return {"status": "retry", "error": "max_tokens"}
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        answer = json.loads(text)
        relevant = bool(answer["relevant"])
        themes = [t for t in answer.get("themes", []) if t in THEMES]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return {"status": "retry", "error": f"unreadable answer: {exc}"}
    return {
        "status": "ok",
        "relevant": relevant,
        "themes": list(dict.fromkeys(themes)),
        "rationale": str(answer.get("rationale", "")),
    }


class LlmClassifier:
    """Submits, tracks and collects the batches for one set."""

    def __init__(
        self,
        answers_path: str | Path,
        state_path: str | Path,
        config: LlmConfig | None = None,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.answers_path, self.state_path = Path(answers_path), Path(state_path)
        self.config = config or LlmConfig()
        self._client = client
        self.sleep = sleep

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_client()
        return self._client

    # Stored answers and batches ---------------------------------------------------------------
    def answers(self) -> pd.DataFrame:
        if not self.answers_path.exists():
            return pd.DataFrame(columns=ANSWER_COLUMNS)
        return read_parquet(self.answers_path)

    def _state(self) -> dict:
        return read_json(self.state_path) if self.state_path.exists() else {"batches": []}

    def _save(self, answers: pd.DataFrame, state: dict | None = None) -> None:
        answers = answers.drop_duplicates("id", keep="last").sort_values("id")
        answers = answers.reset_index(drop=True).reindex(columns=ANSWER_COLUMNS)
        answers["themes"] = answers["themes"].map(lambda v: v if isinstance(v, list) else [])
        answers["relevant"] = answers["relevant"].astype("boolean")
        write_parquet(answers, self.answers_path, schema=ANSWER_SCHEMA)
        if state is not None:
            write_json(state, self.state_path)

    # Steps ------------------------------------------------------------------------------------
    def collect(self, ids: Iterable[str]) -> int:
        """Store the results of every batch that has ended; returns batches still running."""
        state = self._state()
        by_custom_id = {custom_id(i): i for i in ids}
        answers, running = self.answers(), 0
        for batch in state["batches"]:
            if batch["status"] == "collected":
                continue
            info = self.client.messages.batches.retrieve(batch["id"])
            if info.processing_status != "ended":
                running += 1
                continue
            rows = []
            sent = batch["fingerprints"]  # paper ID -> fingerprint of what was sent
            for result in self.client.messages.batches.results(batch["id"]):
                paper_id = by_custom_id.get(result.custom_id)
                if paper_id is None or paper_id not in sent:  # no longer a candidate
                    continue
                row = {"id": paper_id, **parse_result(result)}
                row.update(model=batch["model"], fingerprint=sent[paper_id], batch_id=batch["id"])
                rows.append(row)
            if rows:
                answers = pd.concat([answers, pd.DataFrame(rows)], ignore_index=True)
            batch["status"] = "collected"
            batch["collected_utc"] = _now()
            log.info("collected %d answers from batch %s", len(rows), batch["id"])
            self._save(answers, state)
        return running

    def pending(self) -> set[str]:
        """Papers in batches that have been submitted but not collected."""
        return {
            paper_id
            for batch in self._state()["batches"]
            if batch["status"] != "collected"
            for paper_id in batch["fingerprints"]
        }

    def to_send(self, messages: dict[str, str], retry_failed: bool = False) -> list[str]:
        """Papers without a current answer (none yet, one for another model, rubric or text,
        or a failure worth retrying) that are not already waiting in a batch."""
        answers = self.answers().set_index("id")
        pending = self.pending()
        out = []
        for paper_id, message in messages.items():
            if paper_id in pending:
                continue
            if paper_id in answers.index:
                row = answers.loc[paper_id]
                current = row["fingerprint"] == self.config.fingerprint(message)
                final = row["status"] == "ok" or (row["status"] in FINAL and not retry_failed)
                if current and final:
                    continue
            out.append(paper_id)
        return out

    def submit(self, messages: dict[str, str], paper_ids: list[str]) -> list[str]:
        """Submit ``paper_ids`` in batches of ``batch_size``; returns the batch IDs."""
        state, created = self._state(), []
        size = self.config.batch_size
        for start in range(0, len(paper_ids), size):
            chunk = paper_ids[start : start + size]
            requests = [self.config.request(i, messages[i]) for i in chunk]
            batch = self.client.messages.batches.create(requests=requests)
            state["batches"].append(
                {
                    "id": batch.id,
                    "submitted_utc": _now(),
                    "model": self.config.model,
                    "status": "submitted",
                    "fingerprints": {i: self.config.fingerprint(messages[i]) for i in chunk},
                }
            )
            write_json(state, self.state_path)  # recorded at once, so a crash loses nothing
            created.append(batch.id)
            log.info("submitted batch %s with %d papers", batch.id, len(chunk))
        return created

    def run(
        self,
        papers: pd.DataFrame,
        wait: bool = False,
        poll_seconds: float = 60.0,
        retry_failed: bool = False,
        max_rounds: int = 3,
    ) -> dict:
        """Collect finished batches and submit papers still without an answer; with ``wait``,
        poll until every batch has ended. Papers whose answer came back as a retryable failure
        are sent again, up to ``max_rounds`` submissions per run. Returns counts by status."""
        messages = {
            i: paper_message(t, a)
            for i, t, a in zip(papers["id"], papers["title"], papers["abstract"], strict=True)
        }
        rounds = 0
        while True:
            running = self.collect(messages)
            todo = self.to_send(messages, retry_failed and rounds == 0)
            if todo and rounds < max_rounds:
                self.submit(messages, todo)
                rounds += 1
                continue
            if not wait or not running:
                break
            log.info("%d batch(es) still running; checking again in %.0f s", running, poll_seconds)
            self.sleep(poll_seconds)
        answers = self.answers()
        answers = answers[answers["id"].isin(messages)]
        counts = answers["status"].value_counts().to_dict()
        counts["waiting"] = len(self.pending() & set(messages))
        counts["unanswered"] = len(set(messages) - set(answers["id"]) - self.pending())
        return counts


def decisions_from_answers(
    candidates: pd.DataFrame, answers: pd.DataFrame, labels: pd.Series, model: str
) -> tuple[pd.DataFrame, dict]:
    """Relevance decisions for every candidate: the model's answer, else the keyword rule."""
    ok = answers[answers["status"] == "ok"].set_index("id")["relevant"]
    predicted = candidates["id"].map(ok)
    fallback = predicted.isna()
    predicted = predicted.fillna(candidates["keyword_rule"]).astype(bool)
    table = decisions_table(candidates["id"].tolist(), predicted, None, labels, "llm")
    statuses = answers.set_index("id")["status"]
    info = {
        "method": "llm",
        "model": model,
        "candidates": len(candidates),
        "answered": int((~fallback).sum()),
        "keyword_rule_fallback": int(fallback.sum()),
        "refused": int((candidates["id"].map(statuses) == "refused").sum()),
        "labels_used": int(table["label"].notna().sum()),
        "selected": int(table["relevant"].sum()),
        "created_utc": _now(),
    }
    return table, info


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
