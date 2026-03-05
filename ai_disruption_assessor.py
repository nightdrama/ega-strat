"""
AI Disruption Vulnerability Assessor
=====================================
Batch-assesses companies against the four-dimension framework from
"AI Disruption Assessment: A First-Principles Framework" (Hua & Zhu, 2026).

For each company it:
  1. Fetches the most recent 10-K from SEC EDGAR and extracts only the
     sections that carry analytical signal for the framework:
       • Item 1    — Business (products, competition, data assets)
       • Item 1A   — Risk Factors (AI threats, regulatory, liability)
       • Item 7    — MD&A narrative (revenue mix, cost structure)
       • Item 2    — Properties (physical footprint / last-mile gate)
       • Item 3    — Legal Proceedings (active regulatory/liability disputes)
  2. Passes the extracted sections — labelled and separated — to Claude
     with a system prompt that maps each section to the dimensions it
     evidences.
  3. Scores the four vulnerability dimensions (0–5 each) and six gates.
  4. Identifies the gate most likely to open first and what triggers it.
  5. Writes results to a CSV and prints a ranked summary table.

Usage
-----
  pip install anthropic requests pandas tabulate beautifulsoup4 lxml

  export ANTHROPIC_API_KEY=sk-ant-...
  python ai_disruption_assessor.py

Configuration
-------------
  Edit COMPANIES, PRINT_DETAIL, OUTPUT_CSV, and MODEL near the bottom.
"""

import os
import re
import json
import time
import textwrap
import requests
import anthropic
import pandas as pd
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from typing import Optional
from tabulate import tabulate

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from datetime import date


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompanyInput:
    name: str
    ticker: Optional[str] = None           # SEC ticker → auto-fetches 10-K
    custom_description: Optional[str] = None    # Override with free-form text
    file_path: Optional[str] = None             # Path to local SEC submission
                                                # (.txt bundle or .htm). Takes
                                                # priority over ticker fetch.


@dataclass
class DimensionScore:
    score: float       # 0–5
    rationale: str


@dataclass
class GateAssessment:
    name: str
    strength: str      # None / Low / Medium / High / Very High
    rationale: str


@dataclass
class AssessmentResult:
    company: str
    ticker: Optional[str]
    source_used: str
    sections_extracted: dict = field(default_factory=dict)  # name → char count

    dim1_cognitive_rent:            DimensionScore = field(default_factory=lambda: DimensionScore(0, ""))
    dim2_dgf_properties:            DimensionScore = field(default_factory=lambda: DimensionScore(0, ""))
    dim3_reward_verifiability:      DimensionScore = field(default_factory=lambda: DimensionScore(0, ""))
    dim4_data_availability:         DimensionScore = field(default_factory=lambda: DimensionScore(0, ""))
    dim5_workflow_disintermediation:DimensionScore = field(default_factory=lambda: DimensionScore(0, ""))
    dim6_interface_substitution:    DimensionScore = field(default_factory=lambda: DimensionScore(0, ""))

    composite_score:     float = 0.0
    vulnerability_label: str   = ""

    gates:                  list = field(default_factory=list)
    gate_composite_score:   float = 0.0
    binding_gate:           str = ""
    binding_gate_rationale: str = ""

    raw_llm_response: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Section configuration
#
# Each entry defines:
#   name          — display label included in the prompt
#   start_re      — regex to locate the section header in plain text
#   end_re        — regex marking where the section ends
#   char_budget   — max characters to keep from this section
#   why           — annotation sent in the prompt explaining what to look for
#
# Dimension / gate mapping (sent to Claude in the system prompt):
#   Item 1  Business   → D1, D2, D4, G3, G4, G5, G6
#   Item 1A Risk Fac   → D2, D3, D4, G1, G2, G4  ← most candid AI risk signal
#   Item 7  MD&A       → D1, D2, G3, G6
#   Item 2  Properties → G5
#   Item 3  Legal      → G1, G2
# ─────────────────────────────────────────────────────────────────────────────

SECTION_CONFIG = [
    {
        "name": "ITEM 1 — BUSINESS",
        # Handles dot, colon, or space separators; negative lookahead prevents
        # matching "Item 1A".  e.g. "Item 1: Business", "Item 1. Business"
        "start_re": re.compile(
            r"(?:^|\n)\s*item\s+1(?!a\b)[:\.\s]+(?:business|overview)",
            re.IGNORECASE,
        ),
        "end_re": re.compile(
            r"(?:^|\n)\s*item\s+1a\b",
            re.IGNORECASE,
        ),
        "char_budget": 25_000,
        "why": (
            "Core business model, product descriptions, data assets, "
            "expert networks, competition. Primary evidence for D1, D2, D4 "
            "and for the Institutional Inertia (G3), Trust (G4), "
            "Physical Last-Mile (G5), and Network Effects (G6) gates."
        ),
    },
    {
        "name": "ITEM 1A — RISK FACTORS",
        # e.g. "Item 1A: Risk Factors", "Item 1A. Risk Factors"
        "start_re": re.compile(
            r"(?:^|\n)\s*item\s+1a[:\.\s]+risk\s*factors",
            re.IGNORECASE,
        ),
        # End at Item 1B OR Item 2 — many 10-Ks omit Item 1B entirely,
        # so anchoring on 1B alone would let Item 1A absorb the rest of the doc.
        "end_re": re.compile(
            r"(?:^|\n)\s*item\s+(?:1b|2)\b",
            re.IGNORECASE,
        ),
        "char_budget": 20_000,
        "why": (
            "Most candid disclosure in the filing. Companies that fear AI "
            "substitution disclose it here. Look for: AI/technology "
            "competition risks (D2, D3), government-provided alternatives "
            "(D3), data privacy/portability risks (D4), regulatory risks "
            "(G1), liability risks (G2), reputational/trust risks (G4)."
        ),
    },
    {
        "name": "ITEM 7 — MD&A (NARRATIVE)",
        # Negative lookahead avoids matching "Item 7A".
        # Does NOT require "management" on same line — handles filings where
        # the section title wraps to the next line.
        # e.g. "Item 7: Management's Discussion...", "Item 7. Management's..."
        "start_re": re.compile(
            r"(?:^|\n)\s*item\s+7(?!a\b)[:\.\s]",
            re.IGNORECASE,
        ),
        "end_re": re.compile(
            r"(?:^|\n)\s*item\s+7a\b",
            re.IGNORECASE,
        ),
        "char_budget": 15_000,
        "why": (
            "Revenue and cost structure, segment operating margins, "
            "subscription vs. transaction mix, growth drivers. "
            "Primary evidence for D1 (cognitive rent share), G3 "
            "(Institutional Inertia — subscription stickiness), and G6 "
            "(Network Effects — network scale and growth dynamics)."
        ),
    },
    {
        "name": "ITEM 2 — PROPERTIES",
        # e.g. "Item 2: Properties", "Item 2. Properties"
        "start_re": re.compile(
            r"(?:^|\n)\s*item\s+2[:\.\s]+properties",
            re.IGNORECASE,
        ),
        "end_re": re.compile(
            r"(?:^|\n)\s*item\s+3\b",
            re.IGNORECASE,
        ),
        "char_budget": 3_000,
        "why": (
            "Physical footprint: owned/leased locations, service centers, "
            "field offices. Primary evidence for the Physical Last-Mile "
            "gate (G5). A dense network of physical service locations is a "
            "strong gate; a few HQ office leases is effectively no gate."
        ),
    },
    {
        "name": "ITEM 3 — LEGAL PROCEEDINGS",
        # e.g. "Item 3: Legal Proceedings", "Item 3. Legal Proceedings"
        "start_re": re.compile(
            r"(?:^|\n)\s*item\s+3[:\.\s]+legal\s*proceedings",
            re.IGNORECASE,
        ),
        "end_re": re.compile(
            r"(?:^|\n)\s*item\s+4\b",
            re.IGNORECASE,
        ),
        "char_budget": 3_000,
        "why": (
            "Active regulatory investigations and material lawsuits. "
            "Evidence for the Regulatory (G1) and Liability (G2) gates — "
            "an active government investigation signals the regulatory gate "
            "is real and contested; significant error-liability suits "
            "signal the liability gate has substance."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# EDGAR helpers
# ─────────────────────────────────────────────────────────────────────────────

EDGAR_HEADERS = {"User-Agent": "disruption-research contact@example.com"}

# Hardcoded CIK table — used first so the script works in sandboxed environments
# where sec.gov may be unreachable. Add entries here for any ticker that fails.
# CIKs are zero-padded to 10 digits as required by the EDGAR API.
KNOWN_CIKS: dict[str, str] = {
    "SHOP":  "0001594805",   # Shopify Inc.
    "PLTR":  "0001321655",   # Palantir Technologies Inc.
    "KD":    "0001803482",   # Kyndryl Holdings Inc.
    "AFRM":  "0001820175",   # Affirm Holdings Inc.
    "SNOW":  "0001640147",   # Snowflake Inc.
    "DDOG":  "0001726445",   # Datadog Inc.
    "TOST":  "0001679273",   # Toast Inc.
    "XYZ":   "0001512673",   # Block Inc. (formerly Square)
    "GOOGL": "0001652044",   # Alphabet Inc.
    "PYPL":  "0001633917",   # PayPal Holdings Inc.
    "OKTA":  "0001660134",   # Okta Inc.
    "TWLO":  "0001447362",   # Twilio Inc.
    "NET":   "0001477333",   # Cloudflare Inc.
    "PAYC":  "0001590714",   # Paycom Software Inc.
    "SSNC":  "0000880562",   # SS&C Technologies Holdings Inc.
    "V":     "0001403161",   # Visa Inc.
    "MSFT":  "0000789019",   # Microsoft Corporation
    "INTU":  "0000896878",   # Intuit Inc.
    "IBM":   "0000051143",   # International Business Machines Corp.
    "IT":    "0000749251",   # Gartner Inc.
    "ACN":   "0001467373",   # Accenture plc
    "FISV":  "0000798354",   # Fiserv Inc.
    "EXLS":  "0001289490",   # ExlService Holdings Inc.
    "G":     "0001321732",   # Genpact Ltd.
    "ADP":   "0000012006",   # Automatic Data Processing Inc.
    "APLD":  "0001817175",   # Applied Digital Corporation
    "MDB":   "0001441816",   # MongoDB Inc.
    "RELY":  "0001837107",   # Remitly Global Inc.
    "VRSN":  "0001014473",   # VeriSign Inc.
    "CPAY":  "0001175922",   # Corpay Inc. (formerly FleetCor)
}


def resolve_cik(ticker: str) -> Optional[str]:
    """
    Map a ticker symbol to a zero-padded 10-digit CIK.
    Checks KNOWN_CIKS first (works offline / sandboxed), then falls back
    to the live SEC EDGAR company_tickers.json endpoint.
    """
    t = ticker.upper()

    # 1 — fast local lookup
    if t in KNOWN_CIKS:
        return KNOWN_CIKS[t]

    # 2 — live EDGAR lookup (requires sec.gov network access)
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=EDGAR_HEADERS, timeout=15,
        )
        for v in r.json().values():
            if v.get("ticker", "").upper() == t:
                return str(v["cik_str"]).zfill(10)
    except Exception:
        pass

    return None


def get_latest_10k_info(cik: str) -> tuple:
    """
    Return (filing_index_url, filing_date) for the most recent 10-K.
    We return the index URL so we can inspect the full file list and pick
    the best HTML document (not always the primaryDocument).
    """
    try:
        r = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=EDGAR_HEADERS, timeout=15,
        )
        filings = r.json().get("filings", {}).get("recent", {})
        for i, form in enumerate(filings.get("form", [])):
            if form == "10-K":
                acc  = filings["accessionNumber"][i].replace("-", "")
                date = filings["filingDate"][i]
                idx_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{acc}/{filings['accessionNumber'][i]}-index.htm"
                )
                return idx_url, date
    except Exception:
        pass
    return None, None


def get_10k_document_url(index_url: str, cik: str) -> Optional[str]:
    """
    Parse the filing index page to find the best HTML document for the 10-K.
    Preference order:
      1. Document described as "10-K" with .htm extension
      2. Any .htm file whose name contains "10k" or "annual"
      3. Fallback to index URL with -index.htm replaced by .htm
    """
    try:
        r    = requests.get(index_url, headers=EDGAR_HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "lxml")

        candidates = []
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            doc_type = cells[0].get_text(strip=True)
            doc_name = cells[2].get_text(strip=True).lower()
            link_tag = cells[2].find("a")
            if not link_tag:
                continue
            href = link_tag.get("href", "")
            if not href.endswith((".htm", ".html")):
                continue

            full_url = (
                href if href.startswith("http")
                else "https://www.sec.gov" + href
            )

            if doc_type == "10-K":
                candidates.insert(0, full_url)
            elif any(k in doc_name for k in ("10k", "annual", "form10")):
                candidates.append(full_url)

        return candidates[0] if candidates else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HTML → plain text
# ─────────────────────────────────────────────────────────────────────────────

def html_to_text(html: str) -> str:
    """
    Convert 10-K HTML to clean plain text.
    - Preserves paragraph breaks so section-header regexes can match on \\n
    - Collapses runs of whitespace within a paragraph
    - Removes XBRL / inline data that precedes the readable narrative
    """
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "ix:header", "ix:nonnumeric",
                     "ix:nonfraction", "head"]):
        tag.decompose()

    text  = soup.get_text(separator="\n")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Drop everything before the first "ITEM 1" to skip XBRL preamble
    m = re.search(r"(?:^|\n)\s*item\s+1[\.\s]", text, re.IGNORECASE)
    if m:
        text = text[m.start():]

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Section extractor
# ─────────────────────────────────────────────────────────────────────────────

def _find_section_start(text: str, start_re: re.Pattern) -> int:
    """
    Return the character position of the ACTUAL section content, not a
    table-of-contents entry.

    Strategy: collect all matches of start_re. A TOC entry has very little
    text before the next section header. We require at least MIN_CONTENT_CHARS
    of content before accepting a match as the real section.
    """
    MIN_CONTENT_CHARS = 500
    matches = list(start_re.finditer(text))
    if not matches:
        return -1

    # Match ALL separator styles (dot, colon, space) so TOC entries in
    # colon-format filings (e.g. "Item 1: Business") are detected and skipped.
    all_header_re = re.compile(
        r"(?:^|\n)\s*item\s+\d+[a-z]?[:\.\s]",
        re.IGNORECASE,
    )
    all_positions = [m.start() for m in all_header_re.finditer(text)]

    for m in matches:
        pos         = m.end()
        next_header = next((p for p in all_positions if p > pos), len(text))
        if (next_header - pos) >= MIN_CONTENT_CHARS:
            return pos

    return matches[-1].end()   # fallback: last match


def extract_sections(full_text: str) -> dict:
    """
    Extract each target section from the full plain-text 10-K.
    Returns dict: section_name → extracted_text (truncated to char_budget).
    """
    extracted = {}

    for cfg in SECTION_CONFIG:
        start = _find_section_start(full_text, cfg["start_re"])
        if start == -1:
            extracted[cfg["name"]] = ""
            continue

        end_match = cfg["end_re"].search(full_text, start)
        end       = end_match.start() if end_match else len(full_text)

        raw    = full_text[start:end]
        budget = cfg["char_budget"]

        if len(raw) > budget:
            # Prefer a clean paragraph break near the budget
            cutoff = raw.rfind("\n\n", 0, budget)
            if cutoff == -1 or cutoff < budget * 0.7:
                cutoff = budget
            raw = raw[:cutoff] + "\n\n[... truncated to section budget ...]"

        extracted[cfg["name"]] = raw.strip()

    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# Main 10-K fetch entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_10k_html_from_submission(file_path: str) -> str:
    """
    Parse a full SEC submission text file (the .txt bundle from EDGAR that
    wraps all exhibits) and return the HTML of the primary 10-K document.

    The SEC submission format looks like:
        <DOCUMENT>
        <TYPE>10-K
        ...
        <TEXT>
        ... html ...
        </DOCUMENT>

    Raises ValueError if no 10-K block is found.
    """
    with open(file_path, "r", errors="replace") as f:
        lines = f.readlines()

    start = end = None
    in_10k = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "<DOCUMENT>":
            # Look ahead up to 5 lines for the TYPE tag
            for j in range(i + 1, min(i + 6, len(lines))):
                if lines[j].strip():
                    if lines[j].strip() == "<TYPE>10-K":
                        start = i
                        in_10k = True
                    break
        if in_10k and stripped == "</DOCUMENT>":
            end = i
            break

    if start is None or end is None:
        raise ValueError(f"No <TYPE>10-K document block found in {file_path}")

    return "".join(lines[start:end])


def read_10k_sections_from_file(file_path: str) -> tuple:
    """
    Read a local 10-K file and return (sections_dict, source_label).

    Accepts:
      • Full SEC submission .txt bundle  (e.g. 0001594805-26-000007.txt)
      • Standalone 10-K HTML file        (e.g. shop-20251231.htm)
    """
    try:
        lower = file_path.lower()
        if lower.endswith(".htm") or lower.endswith(".html"):
            with open(file_path, "r", errors="replace") as f:
                html = f.read()
        else:
            html = extract_10k_html_from_submission(file_path)
    except Exception as e:
        return {}, f"error reading file: {e}"

    plain    = html_to_text(html)
    sections = extract_sections(plain)

    found   = [k for k, v in sections.items() if v]
    missing = [k for k, v in sections.items() if not v]
    fname   = file_path.split("/")[-1]
    label   = f"local file ({fname}) — {len(found)}/{len(SECTION_CONFIG)} sections"
    if missing:
        label += f" | missing: {', '.join(s.split('—')[0].strip() for s in missing)}"

    return sections, label


def fetch_10k_sections(ticker: str) -> tuple:
    """
    Fetch the most recent 10-K for *ticker* and return:
      (sections_dict, source_label)
    """
    cik = resolve_cik(ticker)
    if not cik:
        return {}, f"error: could not resolve CIK for {ticker}"

    idx_url, filing_date = get_latest_10k_info(cik)
    if not idx_url:
        return {}, f"error: no 10-K found for CIK {cik}"

    doc_url = get_10k_document_url(idx_url, cik)
    if not doc_url:
        doc_url = idx_url.replace("-index.htm", ".htm")

    try:
        r = requests.get(doc_url, headers=EDGAR_HEADERS, timeout=60)
        r.raise_for_status()
    except Exception as e:
        return {}, f"error fetching 10-K document: {e}"

    plain    = html_to_text(r.text)
    sections = extract_sections(plain)

    found   = [k for k, v in sections.items() if v]
    missing = [k for k, v in sections.items() if not v]
    label   = f"10-K ({filing_date}) — {len(found)}/{len(SECTION_CONFIG)} sections"
    if missing:
        label += f" | missing: {', '.join(s.split('—')[0].strip() for s in missing)}"

    return sections, label


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — framework + section→dimension mapping
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a rigorous financial analyst applying the AI Disruption Vulnerability
Framework (Hua & Zhu, 2026) to assess how vulnerable a company's business
model is to AI-driven disruption.

════════════════════════════════════════
FRAMEWORK
════════════════════════════════════════

AI substitutes capital for cognitive labor. Vulnerability is determined by
four structural dimensions, each scored 0–5 (0 = low risk, 5 = high risk),
then filtered through six gating mechanisms.

FOUR DIMENSIONS
───────────────
D1 — Cognitive Rent Share
  What fraction of the industry's value creation depends on human cognitive
  labor vs. physical labor or physical asset deployment?
  Observable indicator: labor cost as a fraction of revenue, weighted by
  cognitive intensity. High revenue per employee with low physical capital
  requirements = high cognitive rent share.
  Evidence in filing: cost structure (R&D + labor-heavy COGS vs. capex),
  product descriptions, asset-light language, revenue per employee.
  Scoring note: high cognitive rent share in a platform/marketplace context
  may overstate standalone vulnerability — cross-reference with G5 and G6.
  0 = capital-intensive physical business (manufacturing, logistics, mining)
  5 = pure knowledge service; near-zero physical capital; 100% cognitive rent

D2 — Data Generating Function (DGF) Properties
  How compressible is the industry's data generating function? Score along
  two axes:
    Axis 1 — Rule stability: half-life of the ruleset. Software compilation
             rules are essentially permanent. Tax codes change annually.
             Market microstructure evolves continuously.
    Axis 2 — Outcome determinism: how deterministic is input→output under
             fixed rules? Compiling code is fully deterministic. Medical
             diagnosis has significant stochastic residual. Stock return
             prediction has massive stochastic residual even under fixed rules.
  Stable rules + deterministic outcomes = most compressible (score 5).
  Unstable rules + stochastic outcomes = least compressible (score 0).
  Key refinement: distinguish bounded non-stationarity (new data within
  known rules — AI handles well) from unbounded non-stationarity (rules
  themselves change through reflexivity, policy, competition — AI handles
  poorly).
  Evidence in filing: product descriptions, competition section, risk factors
  citing regulatory change or AI substitution.
  0 = volatile rules + stochastic outcomes (hardest to compress)
  5 = stable rules + deterministic outputs (easiest to compress)

D3 — Reward Function Verifiability
  How cheaply, quickly, and unambiguously can the correctness of an
  AI-generated outcome be verified?
  High verifiability (4–5): binary, immediate, machine-readable feedback
  (code compiles/fails, tax return accepted/rejected, audit upheld).
  Medium verifiability (2–3): available but delayed or expert-dependent
  (legal contract assessed against precedent, financial trade P&L over
  months).
  Low verifiability (0–1): slow, ambiguous, confounded, or subjective
  (patient outcomes, policy impact, therapeutic relationship value).
  Stakes asymmetry: reversible errors (bug fix) tolerate faster adoption;
  irreversible errors (misdiagnosis) demand higher verification, slowing
  adoption regardless of capability.
  Evidence in filing: risk factors mentioning government alternatives or AI
  substitution (management implicitly acknowledges output is verifiable);
  product accuracy guarantees; professional licensing requirements.
  0 = output quality essentially unverifiable by a third party
  5 = output verified against objective external ground truth

D4 — Data Availability  ⚠ HIGH availability = HIGH risk
  Can the AI be trained effectively given available data? Score both sides:
  (1) Attacker's access — is training data available through public sources,
  purchasable datasets, or synthesis? Consider three scarcity types:
    • Structural scarcity: domain complexity, few cases per condition
    • Regulatory scarcity: data exists but blocked by HIPAA, GDPR, etc.
    • Institutional scarcity: data siloed across organizations
  (2) Defender's data moat — does the incumbent hold exclusive operational
  data (transaction histories, routing, customer behavior) that compounds
  its AI advantage and can't be replicated from public sources?
  Synthetic data feasibility also matters: industries where synthetic data
  is safe and effective will be disrupted faster.
  Evidence in filing: data asset descriptions in Business section; data
  portability / privacy risk factors; scale of accumulated records.
  0 = highly proprietary data + strong defender moat no one can replicate
  5 = tasks trainable on cheap abundant public data, no defender data moat

D5 — Workflow Disintermediation  ⚠ HIGH score = HIGH risk
  Does AI collapse or bypass the multi-step workflow this product
  orchestrates? Score across three sub-dimensions:
  (1) Product role — is it a connector, aggregator, or orchestrator sitting
  between other systems rather than a terminal endpoint? Intermediary
  products are exposed when AI agents can call upstream and downstream
  systems directly, removing the need for the middleman.
  (2) Workflow collapsibility — can an AI agent compress the multi-step
  process the product currently manages into a single prompt-to-output
  sequence? Count the discrete human handoffs or system hops the product
  orchestrates. The more hops that can be collapsed, the higher the score.
  (3) End-to-end automation — can the full workflow be automated without
  the product's involvement? If inputs and outputs are both machine-readable
  and the transformation logic is learnable, the intermediary is structurally
  exposed.
  Evidence in filing: product descriptions (integration, connector, API,
  workflow, orchestration language); competitive risk factors discussing AI
  agents or workflow automation; revenue model (per-seat vs. per-transaction
  suggests orchestration depth).
  0 = product IS the terminal endpoint; deeply embedded proprietary workflow;
      not bypassable; each step requires unique contextual judgment
  5 = pure connector or aggregator; workflow trivially collapsible by AI
      agents; no proprietary orchestration logic; inputs/outputs fully
      machine-readable

D6 — Interface Substitution  ⚠ HIGH score = HIGH risk
  Risk that AI agents bypass the product's UI or interaction layer entirely.
  Distinct from D5 (workflow collapse across systems): D6 asks whether the
  value of THIS PRODUCT'S OWN INTERFACE gets substituted by agents or
  copilots that interact with underlying APIs or data directly, removing
  the need for a human to navigate the UI.
  Score across four sub-dimensions:
  (1) UI-centric value — does the product's value primarily reside in UI
  navigation, clicks, and workflow steps rather than in proprietary
  underlying data or computation? When the UI is the only moat, agents
  can bypass it. When deep data or logic sits behind the UI, the interface
  is less exposed.
  (2) Agent/copilot overlap — can AI agents or vendor-supplied copilots
  replicate or replace the interaction layer? Look for: agents calling
  product APIs directly, other vendors building AI layers on top, risk
  factors discussing AI automation of manual steps.
  (3) Seat/license consolidation — does the pricing model depend on human
  operators (per-seat, per-user licensing)? AI that reduces the number of
  human operators directly compresses revenue. Look for per-seat pricing
  language, risk factors about seat count pressure, agent-per-seat models.
  (4) UI commoditization signals — filing language acknowledging AI
  automation replacing manual workflow steps, platform consolidation risk,
  or vendor consolidation reducing distinct tool count.
  Evidence in filing: pricing model descriptions (per-seat vs. usage-based);
  risk factors citing AI agents, copilots, or automation displacing human
  operators; competitive risks from platform consolidation; product
  descriptions of UI-centric value vs. data/logic-centric value.
  0 = value is in irreplaceable underlying data, logic, or computation;
      UI is a thin disposable access layer; no per-seat exposure
  5 = value almost entirely in UI/workflow navigation; agents can fully
      bypass via APIs; per-seat model directly threatened by AI reducing
      operator count; multiple vendors building copilots on top

COMPOSITE SCORE = average of D1–D6
  0.0–1.5 → Low Vulnerability
  1.5–2.5 → Low-Moderate Vulnerability
  2.5–3.5 → Moderate-High Vulnerability
  3.5–4.5 → High Vulnerability
  4.5–5.0 → Very High Vulnerability

SIX GATING MECHANISMS
─────────────────────
Even a high composite-score business may be protected if gates are strong.
Rate each gate: None / Low / Medium / High / Very High.
Each gate has a DURABILITY CLASS that determines how it erodes over time:
  Hard (×1.5)       — requires external authority (policy, judicial) to change;
                       does not erode over time
  Structural (×1.25) — requires physical infrastructure or network buildout;
                       doesn't erode from market forces alone
  Soft (×1.0)        — erodes continuously through competition and
                       generational change

G1 — Regulatory Gate  [HARD]
  Does AI deployment require explicit government approval, or do judicial/
  structural rulings bar or slow new AI entrants? Includes licensing regimes,
  approval processes, and court-ordered market restructuring.
  Hard gate: holds until a policy or judicial decision flips it.
  Evidence: Item 1 regulatory environment, Item 1A regulatory risks, Item 3.

G2 — Liability Gate  [STRUCTURAL]
  Who bears the cost of AI error, and does unclear liability create
  human-in-the-loop friction? Includes financial infrastructure embeddedness
  where the transaction must route through the incumbent regardless of who
  initiates it. Opens when liability frameworks clarify through legislation,
  case law, or insurance pricing.
  Evidence: Item 1A liability risks, Item 3 legal proceedings, Item 1
  product guarantee language, payment/settlement infrastructure descriptions.

G3 — Institutional Inertia Gate  [SOFT]
  Switching costs, procurement cycles, legacy system integration,
  professional guild resistance, long contract durations, and deep
  integration into customer operations. Soft gate: erodes continuously as
  contracts expire and AI-native decision-makers enter authority.
  Evidence: Item 1 customer/contract descriptions, Item 7 subscription vs.
  transaction revenue mix.

G4 — Trust Gate  [SOFT]
  Psychological or cultural requirement for a trusted party — human or
  established brand — at the point of delivery or decision. Erodes
  generationally; among the slowest gates to open but erosion is largely
  irreversible.
  Evidence: Item 1A reputational risks, human expert network descriptions,
  professional licensing language.

G5 — Physical Last-Mile Gate  [STRUCTURAL]
  Must the service be delivered in person or at a physical location?
  Includes multi-sided physical coordination across customers, suppliers,
  and workers. Most structurally durable gate: persists until general-purpose
  robotics matures.
  Evidence: Item 2 properties (field office network vs. HQ-only), Item 1
  product delivery descriptions, any field service / on-site language.

G6 — Network Effects Gate  [STRUCTURAL]
  Multi-sided market network effects requiring simultaneous competitive
  displacement across all market sides. A three-sided market is harder to
  displace than two-sided, which is harder than single-sided. Scores the
  structural difficulty of bootstrapping a competing network from scratch,
  even if the cognitive tasks are fully replicable.
  Evidence: Item 1 marketplace/platform descriptions, number of market
  sides served, Item 7 discussion of network scale and growth dynamics.

COMPOSITE GATE SCORE
────────────────────
Convert gate strengths to numeric (None=0, Low=1, Medium=2, High=3,
Very High=4), multiply each by its durability weight (Hard=1.5,
Structural=1.25, Soft=1.0), average across all six gates, then normalize
to a 0–5 scale by multiplying by (5/4). Higher = better protected.
Report this as "gate_composite_score" in the JSON output.

BINDING GATE
────────────
The binding gate is the gate with the highest EFFECTIVE STRENGTH on the
critical attack path. Effective strength = numeric strength × durability
multiplier. A "High" hard gate (3 × 1.5 = 4.5) outranks a "Very High"
soft gate (4 × 1.0 = 4.0) because it doesn't erode over time.
When identifying the binding gate:
  1. Consider the realistic attack path — which gates must an AI disruptor
     actually pass through to compete?
  2. Among those gates, select the one with highest effective strength.
  3. If an open/low gate is shielded by a strong gate behind it, the
     strong gate is binding, not the weak one.
Report this as "binding_gate" and "binding_gate_rationale" in the JSON.

════════════════════════════════════════
SOURCE MATERIAL AND SECTION MAPPING
════════════════════════════════════════

You will receive sections extracted from the company's most recent 10-K,
each annotated with which dimensions and gates it primarily evidences:

  ITEM 1 — BUSINESS         → D1, D2, D4, D5, D6  |  gates G3, G4, G5, G6
  ITEM 1A — RISK FACTORS    → D2, D3, D4, D5, D6  |  gates G1, G2, G4
  ITEM 7 — MD&A (NARRATIVE) → D1, D2, D5, D6      |  gates G3, G6
  ITEM 2 — PROPERTIES       →                 |  gate  G5
  ITEM 3 — LEGAL PROCEEDINGS→                 |  gates G1, G2

CRITICAL RULES:
  • Ground EVERY score and gate rating in specific evidence from the filing.
    Quote or closely paraphrase the relevant passage.
  • If a section is marked [NOT FOUND], note the absence and score
    conservatively for the related dimensions.
  • The Risk Factors section is the most candid part of any 10-K. Explicit
    AI substitution risk disclosure is direct evidence for D2 and D3 — treat
    it with full weight.
  • Do not import material outside knowledge beyond what is needed to
    interpret the filing language.

════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════

Respond ONLY with valid JSON. No preamble, no markdown fences, no trailing
commentary.

{
  "dim1_cognitive_rent": {
    "score": <float 0-5>,
    "rationale": "<2-4 sentences citing specific evidence from the filing>"
  },
  "dim2_dgf_properties": {
    "score": <float 0-5>,
    "rationale": "<2-4 sentences>"
  },
  "dim3_reward_verifiability": {
    "score": <float 0-5>,
    "rationale": "<2-4 sentences>"
  },
  "dim4_data_availability": {
    "score": <float 0-5>,
    "rationale": "<2-4 sentences>"
  },
  "dim5_workflow_disintermediation": {
    "score": <float 0-5>,
    "rationale": "<2-4 sentences addressing: (1) whether the product is a connector/aggregator/orchestrator vs. terminal endpoint, (2) whether its multi-step workflow can be collapsed by AI agents, (3) whether the end-to-end workflow can be fully automated without it>"
  },
  "dim6_interface_substitution": {
    "score": <float 0-5>,
    "rationale": "<2-4 sentences addressing: (1) whether product value resides in UI vs. underlying data/logic, (2) whether agents/copilots can bypass the interface, (3) per-seat pricing exposure to seat consolidation, (4) any filing signals about UI commoditization or AI automation of manual steps>"
  },
  "composite_score": <float — mean of six scores, 2 decimal places>,
  "vulnerability_label": "<Low|Low-Moderate|Moderate-High|High|Very High>",
  "gates": [
    {
      "name": "Regulatory",
      "strength": "<None|Low|Medium|High|Very High>",
      "rationale": "<1-2 sentences citing Item 1A or Item 3>"
    },
    {
      "name": "Liability",
      "strength": "<None|Low|Medium|High|Very High>",
      "rationale": "<1-2 sentences>"
    },
    {
      "name": "Institutional Inertia",
      "strength": "<None|Low|Medium|High|Very High>",
      "rationale": "<1-2 sentences citing Item 1 or Item 7>"
    },
    {
      "name": "Trust",
      "strength": "<None|Low|Medium|High|Very High>",
      "rationale": "<1-2 sentences>"
    },
    {
      "name": "Physical Last-Mile",
      "strength": "<None|Low|Medium|High|Very High>",
      "rationale": "<1-2 sentences citing Item 2>"
    },
    {
      "name": "Network Effects",
      "strength": "<None|Low|Medium|High|Very High>",
      "rationale": "<1-2 sentences citing Item 1 marketplace/platform structure>"
    }
  ],
  "gate_composite_score": <float — weighted average of gate strengths normalized to 0-5, 2 decimal places>,
  "binding_gate": "<name of gate with highest effective strength (strength × durability) on the critical attack path>",
  "binding_gate_rationale": "<2-3 sentences: why this gate is binding, what effective strength calculation supports it, and what would need to change to open it>"
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(company_name: str, sections: dict) -> str:
    """
    Assemble the user-turn prompt from extracted sections.
    Each section is labelled with its name, analytical purpose, and content.
    """
    lines = [
        f"Company: {company_name}",
        "",
        "=" * 70,
        "10-K SECTIONS EXTRACTED FOR ANALYSIS",
        "=" * 70,
    ]

    for cfg in SECTION_CONFIG:
        name = cfg["name"]
        text = sections.get(name, "")
        lines += [
            "",
            "─" * 60,
            name,
            f"Analytical purpose: {cfg['why']}",
            "─" * 60,
            text if text else "[NOT FOUND — score conservatively for related dimensions]",
        ]

    lines += [
        "",
        "=" * 70,
        "Assess the above using the framework. Return only valid JSON.",
        "=" * 70,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Claude call + response parsing
# ─────────────────────────────────────────────────────────────────────────────

def assess_company(
    company: CompanyInput,
    client: anthropic.Anthropic,
    model: str = "claude-opus-4-6",
    retry_delay: float = 6.0,
) -> AssessmentResult:

    result = AssessmentResult(
        company=company.name,
        ticker=company.ticker,
        source_used="",
    )

    # ── Acquire source material ────────────────────────────────────────────
    if company.file_path:
        sections, label = read_10k_sections_from_file(company.file_path)
        result.source_used = label
        if not any(sections.values()):
            sections = {"ITEM 1 — BUSINESS": (
                f"No sections could be extracted from {company.file_path}. "
                "Use general knowledge and flag this limitation in every rationale."
            )}
            result.source_used += " | fallback: model knowledge"

    elif company.custom_description:
        sections = {
            "ITEM 1 — BUSINESS":          company.custom_description,
            "ITEM 1A — RISK FACTORS":     "",
            "ITEM 7 — MD&A (NARRATIVE)":  "",
            "ITEM 2 — PROPERTIES":        "",
            "ITEM 3 — LEGAL PROCEEDINGS": "",
        }
        result.source_used = "custom description"

    elif company.ticker:
        sections, label = fetch_10k_sections(company.ticker)
        result.source_used = label
        if not any(sections.values()):
            sections = {
                "ITEM 1 — BUSINESS": (
                    f"No 10-K text could be retrieved for {company.name} "
                    f"({company.ticker}). Use your general knowledge of this "
                    f"company's business model, and explicitly flag this "
                    f"limitation in every rationale field."
                ),
            }
            result.source_used += " | fallback: model knowledge"
    else:
        result.source_used = "error: no ticker or custom description provided"
        return result

    result.sections_extracted = {k: len(v) for k, v in sections.items()}

    # ── Build prompt and call Claude ───────────────────────────────────────
    user_prompt = build_user_prompt(company.name, sections)
    raw = ""

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            result.raw_llm_response = raw
            break
        except Exception as e:
            if attempt < 2:
                print(f"\n  ⚠  API error ({e}), retrying in {retry_delay}s…",
                      end=" ", flush=True)
                time.sleep(retry_delay)
            else:
                result.source_used += f" | API error: {e}"
                return result

    # ── Parse JSON ─────────────────────────────────────────────────────────
    try:
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw,
                       flags=re.MULTILINE).strip()
        data, _ = json.JSONDecoder().raw_decode(clean)  # ignores trailing text after JSON

        result.dim1_cognitive_rent            = DimensionScore(**data["dim1_cognitive_rent"])
        result.dim2_dgf_properties            = DimensionScore(**data["dim2_dgf_properties"])
        result.dim3_reward_verifiability      = DimensionScore(**data["dim3_reward_verifiability"])
        result.dim4_data_availability         = DimensionScore(**data["dim4_data_availability"])
        result.dim5_workflow_disintermediation= DimensionScore(**data["dim5_workflow_disintermediation"])
        result.dim6_interface_substitution    = DimensionScore(**data["dim6_interface_substitution"])
        result.composite_score                = round(float(data["composite_score"]), 2)
        result.vulnerability_label       = data["vulnerability_label"]
        result.gates                     = [GateAssessment(**g) for g in data["gates"]]
        result.gate_composite_score      = round(float(data["gate_composite_score"]), 2)
        result.binding_gate              = data["binding_gate"]
        result.binding_gate_rationale    = data["binding_gate_rationale"]

    except Exception as e:
        result.source_used += f" | JSON parse error: {e}"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

SCORE_BAR = {0: "░░░░░", 1: "█░░░░", 2: "██░░░",
             3: "███░░", 4: "████░", 5: "█████"}


def score_bar(score: float) -> str:
    return SCORE_BAR.get(round(score), f"{score:.1f}")


def print_summary_table(results: list) -> None:
    rows = []
    for r in sorted(results, key=lambda x: -x.composite_score):
        extracted = sum(1 for v in r.sections_extracted.values() if v > 0)
        rows.append([
            r.company,
            r.ticker or "—",
            f"{r.composite_score:.2f}",
            r.vulnerability_label,
            (f"D1:{score_bar(r.dim1_cognitive_rent.score)} "
             f"D2:{score_bar(r.dim2_dgf_properties.score)} "
             f"D3:{score_bar(r.dim3_reward_verifiability.score)} "
             f"D4:{score_bar(r.dim4_data_availability.score)} "
             f"D5:{score_bar(r.dim5_workflow_disintermediation.score)} "
             f"D6:{score_bar(r.dim6_interface_substitution.score)}"),
            f"{r.gate_composite_score:.2f}",
            r.binding_gate or "—",
            f"{extracted}/{len(SECTION_CONFIG)}",
        ])

    print("\n" + "=" * 112)
    print("AI DISRUPTION VULNERABILITY RANKING")
    print("=" * 112)
    print(tabulate(
        rows,
        headers=["Company", "Ticker", "Vuln Score", "Label",
                 "D1–D6 (higher = riskier)", "Gate Score", "Binding gate", "Sections"],
        tablefmt="rounded_outline",
    ))
    print("D1=Cognitive Rent  D2=DGF Properties  D3=Reward Verifiability  "
          "D4=Data Availability  D5=Workflow Disintermediation  D6=Interface Substitution  "
          "│  Vuln Score=avg(D1–D6)  Gate Score=durability-weighted avg(G1–G6)\n")


def print_detail(r: AssessmentResult) -> None:
    print(f"\n{'─' * 84}")
    print(f"  {r.company}  ({r.ticker or 'no ticker'})  "
          f"│  Vuln: {r.composite_score:.2f}  │  {r.vulnerability_label}")
    print(f"  Source: {r.source_used}")
    print("  Sections: " + "  ".join(
        f"{k.split('—')[0].strip()}:{v:,}c"
        for k, v in r.sections_extracted.items()
    ))
    print(f"{'─' * 84}")

    for label, dim in [
        ("D1 Cognitive Rent Share",         r.dim1_cognitive_rent),
        ("D2 DGF Properties",               r.dim2_dgf_properties),
        ("D3 Reward Verifiability",          r.dim3_reward_verifiability),
        ("D4 Data Availability",             r.dim4_data_availability),
        ("D5 Workflow Disintermediation",    r.dim5_workflow_disintermediation),
        ("D6 Interface Substitution",        r.dim6_interface_substitution),
    ]:
        print(f"\n  {label}: {dim.score}/5  {score_bar(dim.score)}")
        for line in textwrap.wrap(dim.rationale, width=76):
            print(f"    {line}")

    print("\n  GATES:")
    for g in r.gates:
        print(f"    {g.name:<26}  {g.strength:<12}  {g.rationale[:70]}")

    print(f"\n  GATE COMPOSITE SCORE: {r.gate_composite_score:.2f}/5.00")
    print(f"\n  BINDING GATE: {r.binding_gate}")
    for line in textwrap.wrap(r.binding_gate_rationale, width=76):
        print(f"    {line}")


def save_csv(results: list, path: str) -> None:
    rows = []
    for r in results:
        gate_dict = {g.name: g.strength  for g in r.gates}
        gate_rat  = {g.name: g.rationale for g in r.gates}
        rows.append({
            "company":                r.company,
            "ticker":                 r.ticker,
            "source":                 r.source_used,
            "composite_score":        r.composite_score,
            "vulnerability":          r.vulnerability_label,
            "d1_cognitive_rent":      r.dim1_cognitive_rent.score,
            "d2_dgf_properties":      r.dim2_dgf_properties.score,
            "d3_reward_verif":        r.dim3_reward_verifiability.score,
            "d4_data_avail":          r.dim4_data_availability.score,
            "d5_workflow_disint":     r.dim5_workflow_disintermediation.score,
            "d6_interface_subst":     r.dim6_interface_substitution.score,
            "d1_rationale":           r.dim1_cognitive_rent.rationale,
            "d2_rationale":           r.dim2_dgf_properties.rationale,
            "d3_rationale":           r.dim3_reward_verifiability.rationale,
            "d4_rationale":           r.dim4_data_availability.rationale,
            "d5_rationale":           r.dim5_workflow_disintermediation.rationale,
            "d6_rationale":           r.dim6_interface_substitution.rationale,
            "gate_regulatory":        gate_dict.get("Regulatory", ""),
            "gate_liability":         gate_dict.get("Liability", ""),
            "gate_inertia":           gate_dict.get("Institutional Inertia", ""),
            "gate_trust":             gate_dict.get("Trust", ""),
            "gate_physical":          gate_dict.get("Physical Last-Mile", ""),
            "gate_network":           gate_dict.get("Network Effects", ""),
            "gate_regulatory_rat":    gate_rat.get("Regulatory", ""),
            "gate_liability_rat":     gate_rat.get("Liability", ""),
            "gate_inertia_rat":       gate_rat.get("Institutional Inertia", ""),
            "gate_trust_rat":         gate_rat.get("Trust", ""),
            "gate_physical_rat":      gate_rat.get("Physical Last-Mile", ""),
            "gate_network_rat":       gate_rat.get("Network Effects", ""),
            "gate_composite_score":   r.gate_composite_score,
            "binding_gate":           r.binding_gate,
            "binding_gate_rationale": r.binding_gate_rationale,
            "item1_chars":            r.sections_extracted.get("ITEM 1 — BUSINESS", 0),
            "item1a_chars":           r.sections_extracted.get("ITEM 1A — RISK FACTORS", 0),
            "item7_chars":            r.sections_extracted.get("ITEM 7 — MD&A (NARRATIVE)", 0),
            "item2_chars":            r.sections_extracted.get("ITEM 2 — PROPERTIES", 0),
            "item3_chars":            r.sections_extracted.get("ITEM 3 — LEGAL PROCEEDINGS", 0),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n  ✓ Results saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit here
# ─────────────────────────────────────────────────────────────────────────────

COMPANIES = [
    CompanyInput(name="DoorDash",   ticker="DASH",
                 file_path="/Users/yuyuanzhu/QuantResearch/.claude/skills/ai-disruption-assessment/filings/0001628280-25-005715.txt"),
    CompanyInput(name="Shopify",    ticker="SHOP",
                 file_path="/Users/yuyuanzhu/QuantResearch/.claude/skills/ai-disruption-assessment/filings/0001594805-26-000007.txt"),
    CompanyInput(name="Palantir",   ticker="PLTR",
                 file_path="/Users/yuyuanzhu/QuantResearch/.claude/skills/ai-disruption-assessment/filings/0001321655-26-000011.txt"),
    CompanyInput(name="FICO",       ticker="FICO",
                 file_path="/Users/yuyuanzhu/QuantResearch/.claude/skills/ai-disruption-assessment/filings/0000814547-25-000030.txt"),
    CompanyInput(name="Veeva",      ticker="VEEV",
                 file_path="/Users/yuyuanzhu/QuantResearch/.claude/skills/ai-disruption-assessment/filings/0001393052-25-000022.txt"),
    CompanyInput(name="Datadog",    ticker="DDOG",
                 file_path="/Users/yuyuanzhu/QuantResearch/.claude/skills/ai-disruption-assessment/filings/0001628280-26-008819.txt"),
    CompanyInput(name="Toast",      ticker="TOST",
                 file_path="/Users/yuyuanzhu/QuantResearch/.claude/skills/ai-disruption-assessment/filings/0001650164-26-000057.txt"),
    CompanyInput(name="Trade Desk", ticker="TTD",
                 file_path="/Users/yuyuanzhu/QuantResearch/.claude/skills/ai-disruption-assessment/filings/0001671933-26-000014.txt"),
    CompanyInput(name="Robinhood",  ticker="HOOD",
                 file_path="/Users/yuyuanzhu/QuantResearch/.claude/skills/ai-disruption-assessment/filings/0001783879-26-000023.txt"),
    CompanyInput(name="DraftKings", ticker="DKNG",
                 file_path="/Users/yuyuanzhu/QuantResearch/.claude/skills/ai-disruption-assessment/filings/0001883685-26-000013.txt"),
]

PRINT_DETAIL = True                      # False = summary table only
OUTPUT_CSV   = "/Users/yuyuanzhu/QuantResearch/.claude/skills/ai-disruption-assessment/outputs/disruption_scores.csv"
OUTPUT_PDF   = "/Users/yuyuanzhu/QuantResearch/.claude/skills/ai-disruption-assessment/outputs/disruption_report.pdf"
MODEL        = "claude-opus-4-6"



# ─────────────────────────────────────────────────────────────────────────────
# PDF report generation
# ─────────────────────────────────────────────────────────────────────────────

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY       = colors.HexColor("#0D1B2A")
SLATE      = colors.HexColor("#1F3A5F")
TEAL       = colors.HexColor("#1A7A8A")
LIGHT_TEAL = colors.HexColor("#E8F4F6")
MID_GRAY   = colors.HexColor("#6B7280")
LIGHT_GRAY = colors.HexColor("#F3F4F6")
BORDER     = colors.HexColor("#D1D5DB")
WHITE      = colors.white

# Vulnerability label colours
VULN_COLORS = {
    "Low":           colors.HexColor("#D1FAE5"),
    "Low-Moderate":  colors.HexColor("#FEF9C3"),
    "Moderate-High": colors.HexColor("#FED7AA"),
    "High":          colors.HexColor("#FECACA"),
    "Very High":     colors.HexColor("#F87171"),
}
VULN_TEXT = {
    "Low":           colors.HexColor("#065F46"),
    "Low-Moderate":  colors.HexColor("#713F12"),
    "Moderate-High": colors.HexColor("#7C2D12"),
    "High":          colors.HexColor("#7F1D1D"),
    "Very High":     colors.HexColor("#450A0A"),
}

# ── Score bar helper ──────────────────────────────────────────────────────────
def score_bar_text(score):
    filled = round(float(score))
    return "█" * filled + "░" * (5 - filled)

# ── Custom bar chart flowable ─────────────────────────────────────────────────
class ScoreBar(Flowable):
    """Renders a horizontal filled bar for a 0–5 score."""
    def __init__(self, score, width=80, height=8):
        super().__init__()
        self.score  = float(score)
        self.width  = width
        self.height = height

    def wrap(self, *args):
        return self.width, self.height

    def draw(self):
        c = self.canv
        # Background
        c.setFillColor(colors.HexColor("#E5E7EB"))
        c.roundRect(0, 0, self.width, self.height, 2, fill=1, stroke=0)
        # Fill
        fill_w = (self.score / 5.0) * self.width
        if fill_w > 0:
            c.setFillColor(TEAL)
            c.roundRect(0, 0, fill_w, self.height, 2, fill=1, stroke=0)

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# ── Styles ────────────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title", fontSize=28, fontName="Helvetica-Bold",
        textColor=WHITE, alignment=TA_LEFT, spaceAfter=6,
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub", fontSize=13, fontName="Helvetica",
        textColor=colors.HexColor("#B0C8D8"), alignment=TA_LEFT, spaceAfter=4,
    )
    styles["cover_date"] = ParagraphStyle(
        "cover_date", fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor("#7FA8C0"), alignment=TA_LEFT,
    )
    styles["section_head"] = ParagraphStyle(
        "section_head", fontSize=16, fontName="Helvetica-Bold",
        textColor=NAVY, spaceBefore=18, spaceAfter=8,
    )
    styles["company_head"] = ParagraphStyle(
        "company_head", fontSize=14, fontName="Helvetica-Bold",
        textColor=WHITE, spaceAfter=0,
    )
    styles["company_sub"] = ParagraphStyle(
        "company_sub", fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#B0C8D8"), spaceAfter=0,
    )
    styles["dim_label"] = ParagraphStyle(
        "dim_label", fontSize=9, fontName="Helvetica-Bold",
        textColor=NAVY, spaceAfter=1,
    )
    styles["body"] = ParagraphStyle(
        "body", fontSize=8.5, fontName="Helvetica",
        textColor=colors.HexColor("#374151"), leading=13,
        spaceAfter=6,
    )
    styles["gate_label"] = ParagraphStyle(
        "gate_label", fontSize=8.5, fontName="Helvetica-Bold",
        textColor=NAVY,
    )
    styles["gate_body"] = ParagraphStyle(
        "gate_body", fontSize=8, fontName="Helvetica",
        textColor=MID_GRAY, leading=12,
    )
    styles["trigger_head"] = ParagraphStyle(
        "trigger_head", fontSize=9, fontName="Helvetica-Bold",
        textColor=TEAL, spaceBefore=6, spaceAfter=2,
    )
    styles["footer"] = ParagraphStyle(
        "footer", fontSize=7.5, fontName="Helvetica",
        textColor=MID_GRAY, alignment=TA_CENTER,
    )
    styles["caption"] = ParagraphStyle(
        "caption", fontSize=8, fontName="Helvetica-Oblique",
        textColor=MID_GRAY, alignment=TA_CENTER, spaceBefore=2,
    )
    styles["toc_company"] = ParagraphStyle(
        "toc_company", fontSize=10, fontName="Helvetica-Bold",
        textColor=NAVY,
    )
    return styles


# ── Cover page ────────────────────────────────────────────────────────────────
class CoverBackground(Flowable):
    """Draws the dark navy cover background panel."""
    def __init__(self, page_w, page_h):
        super().__init__()
        self.page_w = page_w
        self.page_h = page_h

    def wrap(self, *args):
        return 0, 0

    def draw(self):
        c = self.canv
        c.setFillColor(NAVY)
        c.rect(0, 0, self.page_w, self.page_h, fill=1, stroke=0)
        c.setFillColor(SLATE)
        c.rect(0, self.page_h * 0.38, self.page_w, self.page_h * 0.62, fill=1, stroke=0)
        # Accent bar
        c.setFillColor(TEAL)
        c.rect(0.6 * inch, 0.38 * self.page_h - 4, 1.5 * inch, 6, fill=1, stroke=0)


def _result_to_dict(r) -> dict:
    """Convert AssessmentResult to the dict format the PDF functions expect."""
    if isinstance(r, dict):
        return r
    gate_dict = {g.name: g.strength  for g in r.gates}
    gate_rat  = {g.name: g.rationale for g in r.gates}
    return {
        "company":             r.company,
        "ticker":              r.ticker or "",
        "source":              r.source_used,
        "composite_score":     r.composite_score,
        "vulnerability":       r.vulnerability_label,
        "d1_cognitive_rent":   r.dim1_cognitive_rent.score,
        "d2_dgf_properties":   r.dim2_dgf_properties.score,
        "d3_reward_verif":     r.dim3_reward_verifiability.score,
        "d4_data_avail":       r.dim4_data_availability.score,
        "d5_workflow_disint":  r.dim5_workflow_disintermediation.score,
        "d6_interface_subst":  r.dim6_interface_substitution.score,
        "d1_rationale":        r.dim1_cognitive_rent.rationale,
        "d2_rationale":        r.dim2_dgf_properties.rationale,
        "d3_rationale":        r.dim3_reward_verifiability.rationale,
        "d4_rationale":        r.dim4_data_availability.rationale,
        "d5_rationale":        r.dim5_workflow_disintermediation.rationale,
        "d6_rationale":        r.dim6_interface_substitution.rationale,
        "gate_regulatory":     gate_dict.get("Regulatory", ""),
        "gate_liability":      gate_dict.get("Liability", ""),
        "gate_inertia":        gate_dict.get("Institutional Inertia", ""),
        "gate_trust":          gate_dict.get("Trust", ""),
        "gate_physical":       gate_dict.get("Physical Last-Mile", ""),
        "gate_network":        gate_dict.get("Network Effects", ""),
        "gate_regulatory_rat": gate_rat.get("Regulatory", ""),
        "gate_liability_rat":  gate_rat.get("Liability", ""),
        "gate_inertia_rat":    gate_rat.get("Institutional Inertia", ""),
        "gate_trust_rat":      gate_rat.get("Trust", ""),
        "gate_physical_rat":   gate_rat.get("Physical Last-Mile", ""),
        "gate_network_rat":    gate_rat.get("Network Effects", ""),
        "gate_composite_score":   r.gate_composite_score,
        "binding_gate":           r.binding_gate,
        "binding_gate_rationale": r.binding_gate_rationale,
        "item1_chars":         r.sections_extracted.get("ITEM 1 — BUSINESS", 0),
        "item1a_chars":        r.sections_extracted.get("ITEM 1A — RISK FACTORS", 0),
        "item7_chars":         r.sections_extracted.get("ITEM 7 — MD&A (NARRATIVE)", 0),
        "item2_chars":         r.sections_extracted.get("ITEM 2 — PROPERTIES", 0),
        "item3_chars":         r.sections_extracted.get("ITEM 3 — LEGAL PROCEEDINGS", 0),
    }


def build_cover(styles, companies):
    companies = [_result_to_dict(r) for r in companies]
    story = []
    story.append(CoverBackground(letter[0], letter[1]))
    story.append(Spacer(1, 2.8 * inch))
    story.append(Paragraph("AI Disruption", styles["cover_title"]))
    story.append(Paragraph("Risk Assessment", styles["cover_title"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        f"Scoring {len(companies)} companies across four structural dimensions "
        "and six gating mechanisms",
        styles["cover_sub"],
    ))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        f"Generated {date.today().strftime('%B %d, %Y')}",
        styles["cover_date"],
    ))
    story.append(PageBreak())

    # ── Framework definitions page ─────────────────────────────────────────
    def def_head(text):
        return Paragraph(text, ParagraphStyle(
            "defh", fontSize=11, fontName="Helvetica-Bold",
            textColor=NAVY, spaceBefore=14, spaceAfter=4,
        ))

    def def_item(label, body):
        """Returns a two-row mini-table: bold label row + body text."""
        label_p = Paragraph(f"<b>{label}</b>", ParagraphStyle(
            "dil", fontSize=8.5, fontName="Helvetica-Bold", textColor=WHITE,
        ))
        body_p = Paragraph(body, ParagraphStyle(
            "dib", fontSize=8, fontName="Helvetica",
            textColor=colors.HexColor("#374151"), leading=12,
        ))
        tbl = Table(
            [[label_p], [body_p]],
            colWidths=[6.5 * inch],
        )
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0), SLATE),
            ("BACKGROUND",    (0, 1), (0, 1), LIGHT_TEAL),
            ("TOPPADDING",    (0, 0), (0, 0), 5),
            ("BOTTOMPADDING", (0, 0), (0, 0), 5),
            ("TOPPADDING",    (0, 1), (0, 1), 6),
            ("BOTTOMPADDING", (0, 1), (0, 1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ]))
        return tbl

    story.append(Paragraph("Framework Overview", ParagraphStyle(
        "foh", fontSize=16, fontName="Helvetica-Bold",
        textColor=NAVY, spaceAfter=4,
    )))
    story.append(Paragraph(
        "AI replaces capital for cognitive labor. A company's vulnerability is "
        "determined by four structural dimensions (how much of its revenue derives "
        "from cognitive tasks AI can compress) filtered through six gating mechanisms "
        "(structural barriers that slow or block AI substitution). Each dimension is "
        "scored 0–5; the composite is their average.",
        ParagraphStyle("fob", fontSize=9, fontName="Helvetica",
                       textColor=MID_GRAY, leading=13, spaceAfter=6),
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))

    # ── Dimensions ─────────────────────────────────────────────────────────
    story.append(def_head("Vulnerability Dimensions  (0 = low risk · 5 = high risk)"))

    dims = [
        ("D1 — Cognitive Rent Share",
         "What fraction of revenue is earned by applying cognitive labor to structured "
         "problems, vs. deploying physical capital? Pure software and knowledge services "
         "score high; capital-intensive physical businesses (manufacturing, mining, "
         "logistics) score low."),
        ("D2 — Data Generating Function (DGF) Properties",
         "Where do the company's core tasks sit on two axes — rule stability (governed "
         "by stable external rules such as tax code or GAAP, vs. volatile judgment-driven "
         "domains) and outcome determinism (objectively specifiable output vs. stochastic "
         "or creative)? Tasks in the stable + deterministic quadrant are maximally "
         "compressible by AI and score high."),
        ("D3 — Reward Function Verifiability",
         "Can the quality of the cognitive output be verified objectively and quickly "
         "against an external ground truth? High-verifiability outputs (code compiles, "
         "tax return accepted, audit upheld) score high. Low-verifiability outputs "
         "(strategy advice, therapy, creative direction) score low."),
        ("D4 — Data Availability  \u26a0 high availability = high risk",
         "How difficult is it for a competitor to assemble training data to replicate "
         "the company's core cognitive tasks? Tasks trainable on abundant public data "
         "score high (risky). Companies with decades of proprietary, hard-to-replicate "
         "transaction records score low (protective data moat)."),
        ("D5 — Workflow Disintermediation  \u26a0 high score = high risk",
         "Does AI collapse or bypass the multi-step workflow this product orchestrates? "
         "Score across three sub-dimensions: (1) Product role — connector, aggregator, or "
         "orchestrator vs. terminal endpoint; intermediaries are exposed when AI agents can "
         "call upstream and downstream systems directly. (2) Workflow collapsibility — can "
         "an AI agent compress the multi-step process into a single prompt-to-output "
         "sequence? (3) End-to-end automation — can the full workflow be automated without "
         "the product? Score 0 if deeply embedded terminal endpoint; score 5 if pure "
         "connector with trivially collapsible workflow."),
        ("D6 — Interface Substitution  \u26a0 high score = high risk",
         "Risk that AI agents bypass the product's UI or interaction layer entirely. "
         "Score across four sub-dimensions: (1) UI-centric value — does value primarily "
         "reside in UI navigation and workflow steps vs. proprietary underlying data/logic? "
         "(2) Agent/copilot overlap — can AI agents or copilots replicate the interaction "
         "layer by calling underlying APIs directly? (3) Seat/license consolidation — "
         "per-seat pricing threatened when AI reduces the number of human operators needed. "
         "(4) UI commoditization signals — filing language about AI automation replacing "
         "manual steps or platform consolidation. Score 0 if value is in irreplaceable "
         "underlying logic; score 5 if value is almost entirely in the UI/workflow."),
    ]
    for label, body in dims:
        story.append(def_item(label, body))
        story.append(Spacer(1, 0.05 * inch))

    # Composite score legend
    score_rows = [
        ["0.0 – 1.5", "Low Vulnerability"],
        ["1.5 – 2.5", "Low-Moderate Vulnerability"],
        ["2.5 – 3.5", "Moderate-High Vulnerability"],
        ["3.5 – 4.5", "High Vulnerability"],
        ["4.5 – 5.0", "Very High Vulnerability"],
    ]
    legend_data = [[
        Paragraph("<b>Composite Score</b>", ParagraphStyle(
            "lth", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("<b>Label</b>", ParagraphStyle(
            "lth2", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE)),
    ]] + [
        [Paragraph(s, ParagraphStyle("ls", fontSize=8, alignment=TA_CENTER)),
         Paragraph(l, ParagraphStyle("ll", fontSize=8,
                                     textColor=VULN_TEXT.get(l.replace(" Vulnerability",""), NAVY),
                                     fontName="Helvetica-Bold"))]
        for s, l in score_rows
    ]
    legend_tbl = Table(legend_data, colWidths=[1.4*inch, 2.4*inch])
    legend_style = [
        ("BACKGROUND",   (0,0),(-1,0), NAVY),
        ("GRID",         (0,0),(-1,-1), 0.5, BORDER),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LIGHT_GRAY]),
    ]
    for i, (_, label) in enumerate(score_rows, 1):
        key = label.replace(" Vulnerability", "")
        legend_style.append(("BACKGROUND", (1,i),(1,i), VULN_COLORS.get(key, LIGHT_GRAY)))
    legend_tbl.setStyle(TableStyle(legend_style))
    story.append(Spacer(1, 0.08*inch))
    story.append(legend_tbl)
    story.append(Spacer(1, 0.15 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=6))

    # ── Gates ──────────────────────────────────────────────────────────────
    story.append(def_head("Gating Mechanisms  (None · Low · Medium · High · Very High)"))
    story.append(Paragraph(
        "Even a high composite-score business may be protected if gates are strong. "
        "Gates are rated by the structural depth of the barrier, not its current height.",
        ParagraphStyle("gob", fontSize=9, fontName="Helvetica",
                       textColor=MID_GRAY, leading=13, spaceAfter=6),
    ))

    gates = [
        ("G1 — Regulatory Gate  [Hard ×1.5]",
         "Does AI deployment require explicit government approval, or do judicial/"
         "structural rulings bar or slow new AI entrants? Hard gate: holds "
         "until a policy or judicial decision flips it."),
        ("G2 — Liability Gate  [Structural ×1.25]",
         "Who bears the cost of AI error, and does unclear liability create "
         "human-in-the-loop friction? Includes financial infrastructure embeddedness. "
         "Opens when liability frameworks clarify through legislation or case law."),
        ("G3 — Institutional Inertia Gate  [Soft ×1.0]",
         "Switching costs, procurement cycles, legacy system integration, professional "
         "guild resistance, and deep integration into customer operations. "
         "Soft gate: erodes continuously."),
        ("G4 — Trust Gate  [Soft ×1.0]",
         "Psychological or cultural requirement for a trusted party — human or "
         "established brand — at the point of delivery or decision. Erodes "
         "generationally; erosion is largely irreversible."),
        ("G5 — Physical Last-Mile Gate  [Structural ×1.25]",
         "Must the service be delivered in person or at a physical location? Includes "
         "multi-sided physical coordination. Most structurally durable gate: persists "
         "until general-purpose robotics matures."),
        ("G6 — Network Effects Gate  [Structural ×1.25]",
         "Multi-sided market network effects requiring simultaneous competitive "
         "displacement across all market sides. Scores the structural difficulty "
         "of bootstrapping a competing network from scratch."),
    ]
    for label, body in gates:
        story.append(def_item(label, body))
        story.append(Spacer(1, 0.05 * inch))

    # Durability class legend
    dur_data = [
        [Paragraph("<b>Durability</b>", ParagraphStyle(
            "dth", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER)),
         Paragraph("<b>Weight</b>", ParagraphStyle(
            "dth2", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER)),
         Paragraph("<b>Erosion</b>", ParagraphStyle(
            "dth3", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE))],
        [Paragraph("Hard", ParagraphStyle("dl1", fontSize=8, fontName="Helvetica-Bold", alignment=TA_CENTER)),
         Paragraph("×1.5", ParagraphStyle("dl2", fontSize=8, alignment=TA_CENTER)),
         Paragraph("Requires external authority (policy/judicial) to change", ParagraphStyle("dl3", fontSize=8))],
        [Paragraph("Structural", ParagraphStyle("dl1", fontSize=8, fontName="Helvetica-Bold", alignment=TA_CENTER)),
         Paragraph("×1.25", ParagraphStyle("dl2", fontSize=8, alignment=TA_CENTER)),
         Paragraph("Requires infrastructure/network buildout; doesn't erode from market forces", ParagraphStyle("dl3", fontSize=8))],
        [Paragraph("Soft", ParagraphStyle("dl1", fontSize=8, fontName="Helvetica-Bold", alignment=TA_CENTER)),
         Paragraph("×1.0", ParagraphStyle("dl2", fontSize=8, alignment=TA_CENTER)),
         Paragraph("Erodes continuously through competition and generational change", ParagraphStyle("dl3", fontSize=8))],
    ]
    dur_tbl = Table(dur_data, colWidths=[1.0*inch, 0.7*inch, 4.0*inch])
    dur_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0), NAVY),
        ("GRID",         (0,0),(-1,-1), 0.5, BORDER),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LIGHT_GRAY]),
    ]))
    story.append(Spacer(1, 0.08*inch))
    story.append(dur_tbl)

    story.append(PageBreak())
    return story


# ── Summary table ─────────────────────────────────────────────────────────────
def build_summary(styles, rows):
    rows = [_result_to_dict(r) for r in rows]
    story = []
    story.append(Paragraph("Summary Ranking", styles["section_head"]))
    story.append(Paragraph(
        "Companies sorted by composite vulnerability score (highest to lowest). "
        "D1–D4 bars represent each dimension on a 0–5 scale.",
        styles["body"],
    ))
    story.append(Spacer(1, 0.1 * inch))

    sorted_rows = sorted(rows, key=lambda r: -float(r["composite_score"]))

    # Header
    header = [
        Paragraph("<b>#</b>", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold",
                                              textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("<b>Company</b>", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold",
                                                    textColor=WHITE)),
        Paragraph("<b>Vuln Score</b>", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold",
                                                     textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("<b>Label</b>", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold",
                                                  textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("<b>D1</b>", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold",
                                               textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("<b>D2</b>", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold",
                                               textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("<b>D3</b>", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold",
                                               textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("<b>D4</b>", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold",
                                               textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("<b>D5</b>", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold",
                                               textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("<b>D6</b>", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold",
                                               textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("<b>Gate Score</b>", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold",
                                                     textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("<b>Binding Gate</b>", ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold",
                                                         textColor=WHITE)),
    ]
    table_data = [header]

    label_style = ParagraphStyle("lbl", fontSize=7.5, fontName="Helvetica-Bold",
                                  alignment=TA_CENTER, leading=10)

    for i, r in enumerate(sorted_rows, 1):
        label     = r["vulnerability"]
        bg_color  = VULN_COLORS.get(label, LIGHT_GRAY)
        txt_color = VULN_TEXT.get(label, NAVY)
        lbl_style = ParagraphStyle(f"lbl{i}", fontSize=7.5, fontName="Helvetica-Bold",
                                    alignment=TA_CENTER, leading=10, textColor=txt_color)

        table_data.append([
            Paragraph(str(i), ParagraphStyle("n", fontSize=8.5, fontName="Helvetica",
                                              alignment=TA_CENTER)),
            Paragraph(f"<b>{r['company']}</b><br/>"
                      f"<font size='7' color='#6B7280'>{r['ticker']}</font>",
                      ParagraphStyle("co", fontSize=8.5, fontName="Helvetica", leading=12)),
            Paragraph(f"<b>{float(r['composite_score']):.2f}</b>",
                      ParagraphStyle("sc", fontSize=9, fontName="Helvetica-Bold",
                                     alignment=TA_CENTER)),
            Paragraph(label, lbl_style),
            Paragraph(f"{float(r['d1_cognitive_rent']):.1f}",
                      ParagraphStyle("d", fontSize=8.5, alignment=TA_CENTER)),
            Paragraph(f"{float(r['d2_dgf_properties']):.1f}",
                      ParagraphStyle("d", fontSize=8.5, alignment=TA_CENTER)),
            Paragraph(f"{float(r['d3_reward_verif']):.1f}",
                      ParagraphStyle("d", fontSize=8.5, alignment=TA_CENTER)),
            Paragraph(f"{float(r['d4_data_avail']):.1f}",
                      ParagraphStyle("d", fontSize=8.5, alignment=TA_CENTER)),
            Paragraph(f"{float(r['d5_workflow_disint']):.1f}",
                      ParagraphStyle("d", fontSize=8.5, alignment=TA_CENTER)),
            Paragraph(f"{float(r['d6_interface_subst']):.1f}",
                      ParagraphStyle("d", fontSize=8.5, alignment=TA_CENTER)),
            Paragraph(f"<b>{float(r['gate_composite_score']):.2f}</b>",
                      ParagraphStyle("gc", fontSize=9, fontName="Helvetica-Bold",
                                     alignment=TA_CENTER)),
            Paragraph(r["binding_gate"],
                      ParagraphStyle("fg", fontSize=7.5, fontName="Helvetica",
                                     textColor=MID_GRAY)),
        ])

    col_widths = [0.22*inch, 1.0*inch, 0.4*inch, 0.95*inch,
                  0.28*inch, 0.28*inch, 0.28*inch, 0.28*inch, 0.28*inch, 0.28*inch, 0.45*inch, 1.1*inch]

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)

    # Build per-row label background commands
    style_cmds = [
        ("BACKGROUND",  (0, 0), (-1, 0),  NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("GRID",        (0, 0), (-1, -1),  0.5, BORDER),
        ("VALIGN",      (0, 0), (-1, -1),  "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1),  5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1),  5),
        ("RIGHTPADDING",(0, 0), (-1, -1),  5),
    ]

    for i, r in enumerate(sorted_rows, 1):
        label    = r["vulnerability"]
        bg_color = VULN_COLORS.get(label, LIGHT_GRAY)
        style_cmds.append(("BACKGROUND", (3, i), (3, i), bg_color))

    tbl.setStyle(TableStyle(style_cmds))
    story.append(tbl)

    # Legend
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "D1 = Cognitive Rent Share  ·  D2 = DGF Properties  ·  D3 = Reward Verifiability  "
        "·  D4 = Data Availability  ·  D5 = Workflow Disintermediation  "
        "·  D6 = Interface Substitution  (all scored 0–5, higher = more vulnerable)",
        styles["caption"],
    ))
    return story


# ── Individual company detail ─────────────────────────────────────────────────
def gate_strength_color(strength):
    return {
        "None":       colors.HexColor("#D1FAE5"),
        "Low":        colors.HexColor("#FEF9C3"),
        "Medium":     colors.HexColor("#FED7AA"),
        "High":       colors.HexColor("#FECACA"),
        "Very High":  colors.HexColor("#F87171"),
    }.get(strength, LIGHT_GRAY)


def build_company_detail(styles, r):
    r = _result_to_dict(r)
    story = []

    # ── Company header banner ──────────────────────────────────────────────
    vuln   = r["vulnerability"]
    header_table = Table([[
        Paragraph(f"<b>{r['company']}</b>", styles["company_head"]),
        Paragraph(
            f"Vuln: <b>{float(r['composite_score']):.2f}</b> / 5.00",
            ParagraphStyle("hs", fontSize=11, fontName="Helvetica-Bold",
                           textColor=WHITE, alignment=TA_RIGHT),
        ),
    ]], colWidths=[4.2*inch, 2.3*inch])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), SLATE),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))

    sub_table = Table([[
        Paragraph(
            f"Ticker: {r['ticker']}  ·  "
            f"Vulnerability: <b>{vuln}</b>  ·  "
            f"Source: {r['source'].split('|')[0].strip()}",
            styles["company_sub"],
        ),
    ]], colWidths=[6.5*inch])
    sub_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
    ]))

    story.append(KeepTogether([header_table, sub_table]))
    story.append(Spacer(1, 0.12 * inch))

    # ── Four dimensions ────────────────────────────────────────────────────
    # Map each dimension key to the 10-K sections that evidence it, paired
    # with the CSV column that records how many chars were extracted.
    DIM_SOURCES = {
        "D1": [("Item 1 — Business",      "item1_chars"),
               ("Item 7 — MD&A",          "item7_chars")],
        "D2": [("Item 1 — Business",      "item1_chars"),
               ("Item 1A — Risk Factors", "item1a_chars")],
        "D3": [("Item 1A — Risk Factors", "item1a_chars"),
               ("Item 1 — Business",      "item1_chars")],
        "D4": [("Item 1 — Business",      "item1_chars"),
               ("Item 1A — Risk Factors", "item1a_chars")],
        "D5": [("Item 1 — Business",      "item1_chars"),
               ("Item 1A — Risk Factors", "item1a_chars")],
        "D6": [("Item 1 — Business",      "item1_chars"),
               ("Item 1A — Risk Factors", "item1a_chars")],
    }

    dims = [
        ("D1", "D1 — Cognitive Rent Share",        r["d1_cognitive_rent"],    r["d1_rationale"]),
        ("D2", "D2 — DGF Properties",              r["d2_dgf_properties"],    r["d2_rationale"]),
        ("D3", "D3 — Reward Verifiability",         r["d3_reward_verif"],      r["d3_rationale"]),
        ("D4", "D4 — Data Availability",            r["d4_data_avail"],        r["d4_rationale"]),
        ("D5", "D5 — Workflow Disintermediation",   r["d5_workflow_disint"],   r["d5_rationale"]),
        ("D6", "D6 — Interface Substitution",       r["d6_interface_subst"],   r["d6_rationale"]),
    ]

    story.append(Paragraph("Dimension Scores", ParagraphStyle(
        "dsh", fontSize=10, fontName="Helvetica-Bold", textColor=SLATE,
        spaceBefore=2, spaceAfter=6,
    )))

    src_found_style = ParagraphStyle(
        "src_found", fontSize=7, fontName="Helvetica",
        textColor=colors.HexColor("#065F46"),      # dark green — section had content
    )
    src_missing_style = ParagraphStyle(
        "src_miss", fontSize=7, fontName="Helvetica",
        textColor=colors.HexColor("#9CA3AF"),      # gray — section not found
    )

    for dim_key, label, score, rationale in dims:
        score_f = float(score)
        clean   = rationale.lstrip("⚠ ").lstrip("NOTE: ").strip()

        # Build source-section tag string
        source_parts = []
        for section_name, char_col in DIM_SOURCES[dim_key]:
            chars = int(r.get(char_col, 0) or 0)
            if chars > 0:
                source_parts.append(
                    (f"✓ {section_name} ({chars:,} chars)", True)
                )
            else:
                source_parts.append(
                    (f"✗ {section_name} (not extracted)", False)
                )

        # Header row: label + score bar
        row_tbl = Table([[
            Paragraph(f"<b>{label}</b>",
                      ParagraphStyle("dl", fontSize=9, fontName="Helvetica-Bold",
                                     textColor=NAVY)),
            Paragraph(f"<b>{score_f:.1f} / 5.0</b>  {score_bar_text(score_f)}",
                      ParagraphStyle("ds", fontSize=9, fontName="Helvetica-Bold",
                                     textColor=TEAL, alignment=TA_RIGHT)),
        ]], colWidths=[3.8*inch, 2.7*inch])
        row_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), LIGHT_TEAL),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 8),
            ("RIGHTPADDING",  (0,0),(-1,-1), 8),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ]))
        story.append(row_tbl)

        # Source section tags row
        tag_cells = []
        for text, found in source_parts:
            bg   = colors.HexColor("#D1FAE5") if found else colors.HexColor("#F3F4F6")
            cell = Table([[Paragraph(text, src_found_style if found else src_missing_style)]],
                         colWidths=[2.8*inch])
            cell.setStyle(TableStyle([
                ("BACKGROUND",    (0,0),(-1,-1), bg),
                ("TOPPADDING",    (0,0),(-1,-1), 3),
                ("BOTTOMPADDING", (0,0),(-1,-1), 3),
                ("LEFTPADDING",   (0,0),(-1,-1), 6),
                ("RIGHTPADDING",  (0,0),(-1,-1), 6),
                ("BOX",           (0,0),(-1,-1), 0.5, BORDER),
            ]))
            tag_cells.append(cell)

        tag_row = Table([tag_cells], colWidths=[2.85*inch] * len(tag_cells))
        tag_row.setStyle(TableStyle([
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 4),
        ]))
        story.append(tag_row)

        # Rationale text
        story.append(Paragraph(clean, styles["body"]))
        story.append(Spacer(1, 0.04 * inch))

    story.append(Spacer(1, 0.08 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 0.08 * inch))

    # ── Gates ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Gating Mechanisms", ParagraphStyle(
        "gsh", fontSize=10, fontName="Helvetica-Bold", textColor=SLATE,
        spaceBefore=2, spaceAfter=6,
    )))

    gates = [
        ("Regulatory",          r["gate_regulatory"],  r["gate_regulatory_rat"]),
        ("Liability",           r["gate_liability"],   r["gate_liability_rat"]),
        ("Institutional Inertia", r["gate_inertia"],   r["gate_inertia_rat"]),
        ("Trust",               r["gate_trust"],       r["gate_trust_rat"]),
        ("Physical Last-Mile",  r["gate_physical"],    r["gate_physical_rat"]),
        ("Network Effects",     r["gate_network"],     r["gate_network_rat"]),
    ]

    gate_data = []
    for name, strength, rationale in gates:
        bg = gate_strength_color(strength)
        strength_p = Paragraph(
            f"<b>{strength}</b>",
            ParagraphStyle("gs", fontSize=8, fontName="Helvetica-Bold",
                           textColor=NAVY, alignment=TA_CENTER),
        )
        gate_data.append([
            Paragraph(f"<b>{name}</b>", styles["gate_label"]),
            strength_p,
            Paragraph(rationale.lstrip("⚠ ").lstrip("NOTE: ").strip(),
                      styles["gate_body"]),
        ])

    gate_tbl = Table(gate_data, colWidths=[1.35*inch, 0.75*inch, 4.4*inch])
    gate_style_cmds = [
        ("GRID",         (0,0), (-1,-1), 0.5, BORDER),
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [WHITE, LIGHT_GRAY]),
    ]
    for i, (name, strength, _) in enumerate(gates):
        gate_style_cmds.append(("BACKGROUND", (1, i), (1, i), gate_strength_color(strength)))

    gate_tbl.setStyle(TableStyle(gate_style_cmds))
    story.append(gate_tbl)

    # ── Gate composite score + Binding gate ─────────────────────────────────
    story.append(Spacer(1, 0.1 * inch))
    gate_score_box = Table([[
        Paragraph(
            f"Gate Composite Score: <b>{float(r['gate_composite_score']):.2f} / 5.00</b>"
            f"&nbsp;&nbsp;&nbsp;·&nbsp;&nbsp;&nbsp;"
            f"Binding Gate: <b>{r['binding_gate']}</b>",
            ParagraphStyle("fgt", fontSize=9, fontName="Helvetica-Bold",
                           textColor=WHITE),
        ),
    ]], colWidths=[6.5*inch])
    gate_score_box.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), TEAL),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
    ]))
    story.append(gate_score_box)
    story.append(Paragraph(r["binding_gate_rationale"], ParagraphStyle(
        "ftrig", fontSize=8.5, fontName="Helvetica", textColor=colors.HexColor("#374151"),
        leading=13, spaceBefore=5, spaceAfter=4,
        leftIndent=10, rightIndent=10,
        borderPad=8, borderColor=TEAL, borderWidth=0.5,
        backColor=colors.HexColor("#F0FAFB"),
    )))

    story.append(PageBreak())
    return story


# ── Page template with footer ─────────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawCentredString(
        letter[0] / 2, 0.45 * inch,
        f"AI Disruption Risk Assessment  ·  Page {doc.page}",
    )
    # Top accent line (skip cover)
    if doc.page > 1:
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(2)
        canvas.line(0.6*inch, letter[1] - 0.4*inch,
                    letter[0] - 0.6*inch, letter[1] - 0.4*inch)
    canvas.restoreState()


def generate_pdf_report(results: list, output_path: str) -> None:
    """
    Generate a PDF report from a list of AssessmentResult objects.
    Called automatically at the end of main() when OUTPUT_PDF is set.
    """
    styles = make_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.6*inch,
        rightMargin=0.6*inch,
        topMargin=0.55*inch,
        bottomMargin=0.6*inch,
        title="AI Disruption Risk Assessment",
        author="AI Disruption Framework",
    )

    story = []
    story += build_cover(styles, results)
    story.append(Paragraph("Summary Rankings", styles["section_head"]))
    story += build_summary(styles, results)
    story.append(PageBreak())
    story.append(Paragraph("Company Assessments", styles["section_head"]))
    story.append(Spacer(1, 0.05 * inch))

    for r in sorted(results, key=lambda x: -x.composite_score):
        story += build_company_detail(styles, r)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"  ✓ PDF report saved to {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n"
            "Run:  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    client  = anthropic.Anthropic(api_key=api_key)
    results = []

    print(f"\nAssessing {len(COMPANIES)} companies  │  model: {MODEL}")
    print("Targeted sections: " +
          " | ".join(c["name"] for c in SECTION_CONFIG))
    print("─" * 80)

    for i, company in enumerate(COMPANIES, 1):
        print(f"[{i}/{len(COMPANIES)}] {company.name} …", end=" ", flush=True)
        r = assess_company(company, client, model=MODEL)
        results.append(r)

        extracted = sum(1 for v in r.sections_extracted.values() if v > 0)
        label     = r.vulnerability_label or "parse error"
        print(f"score={r.composite_score:.2f}  {label}  "
              f"({extracted}/{len(SECTION_CONFIG)} sections)")

        time.sleep(1)

    print_summary_table(results)

    if PRINT_DETAIL:
        for r in sorted(results, key=lambda x: -x.composite_score):
            print_detail(r)

    save_csv(results, OUTPUT_CSV)

    if OUTPUT_PDF:
        print(f"\nGenerating PDF report …")
        generate_pdf_report(results, OUTPUT_PDF)


if __name__ == "__main__":
    main()
