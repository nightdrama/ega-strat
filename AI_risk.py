"""
Minimal Prompt + Python Scorer for AI Moat Risk (5 dimensions)

Dimensions (each scored 0–5):
- ISR: Interface Substitution Risk
- DRR: Data Replicability Risk
- WDR: Workflow Disintermediation Risk
- PPCR: Pricing Power Compression Risk
- RA: Re-Anchoring Ability (offset)

Composite:
AI_MOAT_RISK = (ISR + DRR + WDR + PPCR) - (1.5 * RA)

This file provides:
1) MINIMAL_PROMPT: a compact instruction string for an LLM
2) build_prompt(): injects transcript + filing texts
3) score_ai_moat_risk(): validates + computes composite + 0–100 normalization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import re


# -----------------------------
# 1) Minimal prompt
# -----------------------------

MINIMAL_SYSTEM = (
    "You are an investment text analyst. Output ONLY valid JSON. "
    "Use ONLY the provided transcript and filing text. "
    "Do not invent facts. Quotes must be exact and <= 25 words."
)

MINIMAL_PROMPT = r"""
Task:
Score five AI moat dimensions from the provided texts.

Return ONLY JSON with exactly this shape:
{
  "scores": {
    "ISR": {"value": 0-5, "confidence": 0-1, "evidence": [{"source":"transcript|filing","quote":"<=25 words","where":"..."}]},
    "DRR": {"value": 0-5, "confidence": 0-1, "evidence": [...]},
    "WDR": {"value": 0-5, "confidence": 0-1, "evidence": [...]},
    "PPCR":{"value": 0-5, "confidence": 0-1, "evidence": [...]},
    "RA":  {"value": 0-5, "confidence": 0-1, "evidence": [...]}
  },
  "notes": {
    "key_assumptions": [string,...],
    "limitations": [string,...],
    "contradictions": [string,...]
  }
}

Definitions (score 0=low, 5=high):

ISR (Interface Substitution Risk):
- Higher when value is mostly UI/workflow usage and AI agents can bypass the interface (chat/agent layer).
- Evidence: agent/copilot overlay, seat consolidation, UI differentiation claims, automation replacing clicks.

DRR (Data Replicability Risk):
- Higher when data advantage is non-exclusive or easily reproduced (public/partner data, weak rights).
- Lower when proprietary rights-cleared compounding data, strong governance/permissioning.
- Evidence: data rights, exclusivity, licensing, privacy/reg constraints, reliance on 3rd-party models.

WDR (Workflow Disintermediation Risk):
- Higher when AI can collapse multi-step workflows or remove intermediaries/integrations.
- Evidence: vendor rationalization, consolidation, tool sprawl, “single workflow”, agent orchestration.

PPCR (Pricing Power Compression Risk):
- Higher when pricing faces compression (discounting, procurement pressure, bundling “for free”).
- Evidence: discounting, price compression, competitive pricing, bundling, renewals pressure, ROI scrutiny.

RA (Re-Anchoring Ability) [OFFSET]:
- Higher when company successfully shifts moat to harder-to-bypass anchors: system of record, compliance/audit,
  distribution/bundling, outcome-based pricing, deep embedding, SKU/pricing/packaging changes that monetize AI.
- Evidence: GA launches tied to paid SKUs, attach rates, repricing, customer willingness-to-pay, system-of-record claims.

Evidence requirements:
- Provide 2–5 evidence quotes per dimension when possible; otherwise fewer + note limitation.
- Quotes must be exact <= 25 words and include "where" (e.g., "Q&A", "Prepared remarks", "Risk Factors", "MD&A").

Now score using ONLY the provided texts.
""".strip()


def build_prompt(
    transcript_text: str,
    current_filing_text: str,
    company: str = "",
    ticker: str = "",
    period: str = "",
) -> List[Dict[str, str]]:
    meta = {"company": company, "ticker": ticker, "period": period}
    user = (
        MINIMAL_PROMPT
        + "\n\nMETADATA:\n" + json.dumps(meta, ensure_ascii=False)
        + "\n\nTRANSCRIPT_TEXT:\n" + (transcript_text or "")
        + "\n\nCURRENT_FILING_TEXT:\n" + (current_filing_text or "")
    )
    return [
        {"role": "system", "content": MINIMAL_SYSTEM},
        {"role": "user", "content": user},
    ]


# -----------------------------
# 2) Python scorer (post-LLM)
# -----------------------------

_DIMENSIONS = ("ISR", "DRR", "WDR", "PPCR", "RA")


class ScoreValidationError(ValueError):
    pass


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(x: Any) -> float:
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x)
    raise ScoreValidationError(f"Expected number, got: {type(x).__name__} {x!r}")


def _validate_evidence_list(evs: Any) -> None:
    if not isinstance(evs, list):
        raise ScoreValidationError("evidence must be a list")
    for e in evs:
        if not isinstance(e, dict):
            raise ScoreValidationError("each evidence item must be an object")
        for k in ("source", "quote", "where"):
            if k not in e:
                raise ScoreValidationError(f"evidence item missing key: {k}")
        if e["source"] not in ("transcript", "filing"):
            raise ScoreValidationError(f"evidence.source invalid: {e['source']}")
        if not isinstance(e["quote"], str) or not e["quote"].strip():
            raise ScoreValidationError("evidence.quote must be non-empty string")
        # enforce <=25 words (approx by whitespace split)
        if len(e["quote"].strip().split()) > 25:
            raise ScoreValidationError("evidence.quote exceeds 25 words")
        if not isinstance(e["where"], str) or not e["where"].strip():
            raise ScoreValidationError("evidence.where must be non-empty string")


def compute_ai_moat_risk(
    llm_json: Dict[str, Any],
    normalize_0_to_100: bool = True,
) -> Dict[str, Any]:
    """
    Input: parsed JSON from the LLM matching the minimal shape.
    Output: dict with:
      - raw dimension scores
      - composite AI_MOAT_RISK (raw)
      - AI_MOAT_RISK_0_100 (optional)
      - confidence (aggregate)
    """
    if not isinstance(llm_json, dict):
        raise ScoreValidationError("Top-level must be a JSON object")

    scores = llm_json.get("scores")
    if not isinstance(scores, dict):
        raise ScoreValidationError("Missing or invalid 'scores' object")

    dim_values: Dict[str, float] = {}
    dim_conf: Dict[str, float] = {}

    for d in _DIMENSIONS:
        if d not in scores or not isinstance(scores[d], dict):
            raise ScoreValidationError(f"Missing dimension object: {d}")
        obj = scores[d]

        v = _safe_float(obj.get("value"))
        c = _safe_float(obj.get("confidence"))

        if not (0.0 <= v <= 5.0):
            raise ScoreValidationError(f"{d}.value out of range 0–5: {v}")
        if not (0.0 <= c <= 1.0):
            raise ScoreValidationError(f"{d}.confidence out of range 0–1: {c}")

        evs = obj.get("evidence", [])
        _validate_evidence_list(evs)

        dim_values[d] = v
        dim_conf[d] = c

    # Composite (raw)
    raw = (dim_values["ISR"] + dim_values["DRR"] + dim_values["WDR"] + dim_values["PPCR"]) - (1.5 * dim_values["RA"])

    # Range:
    # min when ISR=DRR=WDR=PPCR=0 and RA=5 => raw = -7.5
    # max when ISR=DRR=WDR=PPCR=5 and RA=0 => raw = 20
    raw_min, raw_max = -7.5, 20.0

    out: Dict[str, Any] = {
        "dimensions_0_to_5": dim_values,
        "ai_moat_risk_raw": raw,
        "ai_moat_risk_raw_range": {"min": raw_min, "max": raw_max},
        "aggregate_confidence_0_to_1": sum(dim_conf.values()) / len(dim_conf),
        "formula": "AI_MOAT_RISK = (ISR + DRR + WDR + PPCR) - (1.5 * RA)",
    }

    if normalize_0_to_100:
        # linear mapping to 0–100
        norm = 100.0 * (raw - raw_min) / (raw_max - raw_min)
        out["ai_moat_risk_0_to_100"] = _clamp(norm, 0.0, 100.0)

    return out


# -----------------------------
# 3) Example usage
# -----------------------------

if __name__ == "__main__":
    # 1) Build prompt for your LLM call
    msgs = build_prompt(
        transcript_text="(paste transcript)",
        current_filing_text="(paste 10-K/10-Q)",
        company="ExampleCo",
        ticker="EXM",
        period="2025Q4",
    )
    print("SYSTEM:\n", msgs[0]["content"][:200], "...\n")
    print("USER length:", len(msgs[1]["content"]))

    # 2) After LLM returns JSON, parse and score:
    example_llm_output = {
        "scores": {
            "ISR": {"value": 4, "confidence": 0.6, "evidence": [{"source":"transcript","quote":"We are seeing seat consolidation as customers standardize workflows.","where":"Q&A"}]},
            "DRR": {"value": 3, "confidence": 0.5, "evidence": [{"source":"filing","quote":"We rely on third-party models and data sources for AI features.","where":"Risk Factors"}]},
            "WDR": {"value": 4, "confidence": 0.6, "evidence": [{"source":"transcript","quote":"Customers want fewer tools; agents can orchestrate tasks across systems.","where":"Prepared remarks"}]},
            "PPCR":{"value": 5, "confidence": 0.7, "evidence": [{"source":"transcript","quote":"Pricing remains competitive and discounts increased in larger renewals.","where":"Q&A"}]},
            "RA":  {"value": 2, "confidence": 0.5, "evidence": [{"source":"transcript","quote":"We plan to introduce AI add-ons; packaging is under evaluation.","where":"Q&A"}]},
        },
        "notes": {"key_assumptions": [], "limitations": [], "contradictions": []},
    }
    scored = compute_ai_moat_risk(example_llm_output, normalize_0_to_100=True)
    print(json.dumps(scored, indent=2))
