---
name: ai-disruption-assessment
description: >
  Runs a structured AI disruption vulnerability assessment on one or more companies
  using their SEC 10-K filings. Extracts five key 10-K sections (Item 1 Business,
  Item 1A Risk Factors, Item 7 MD&A, Item 2 Properties, Item 3 Legal Proceedings),
  scores each company across four vulnerability dimensions (Cognitive Rent Share,
  DGF Properties, Reward Verifiability, Data Availability) and six gating mechanisms
  (Regulatory, Liability, Institutional Inertia, Trust, Physical Last-Mile, Network
  Effects), and generates a professional PDF report. Use this skill whenever a user
  uploads a 10-K filing and asks for AI disruption analysis, vulnerability scoring,
  or wants to understand how exposed a company is to AI substitution. Also trigger
  when the user asks to "assess" or "score" a company using the disruption framework,
  or when comparing multiple companies' AI vulnerability from their filings.
---

# AI Disruption Assessment Skill

## What this skill does

Given one or more SEC 10-K filings (as uploaded files), this skill:

1. Extracts five targeted sections from each filing using HTML parsing + regex
2. Sends the extracted text to Claude via the Anthropic API with a structured scoring prompt
3. Scores each company on four vulnerability dimensions (D1–D4) and six gates (G1–G6)
4. Produces a console summary table, a CSV of raw scores, and a multi-page PDF report

## Framework summary

**Vulnerability dimensions** (each scored 0–5, higher = more vulnerable to AI disruption):
- **D1 Cognitive Rent Share** — what fraction of the industry's value creation depends on human cognitive labor vs. physical labor or physical asset deployment? Observable indicator: labor cost as a fraction of revenue, weighted by cognitive intensity. High revenue per employee with low physical capital = high cognitive rent share. When scoring, note that high cognitive rent share in a platform/marketplace context may overstate standalone vulnerability — cross-reference with G5 and G6.
- **D2 DGF Properties** — how compressible is the industry's data generating function? Score along two axes: rule stability (half-life of the ruleset) and outcome determinism (how deterministic is input→output under fixed rules). Stable rules + deterministic outcomes = most compressible (score 5). Unstable rules + stochastic outcomes = least compressible (score 0). Key refinement: distinguish bounded non-stationarity (new data within known rules — AI handles well) from unbounded non-stationarity (rules themselves change through reflexivity, policy, competition — AI handles poorly).
- **D3 Reward Verifiability** — how cheaply, quickly, and unambiguously can the correctness of an AI-generated outcome be verified? High verifiability (4–5): binary, immediate, machine-readable feedback. Medium (2–3): available but delayed or expert-dependent. Low (0–1): slow, ambiguous, confounded, or subjective. Also consider stakes asymmetry: reversible errors tolerate faster adoption; irreversible errors demand higher verification, slowing adoption regardless of capability.
- **D4 Data Availability** — can the AI be trained effectively given available data? Score both sides: (1) Attacker's access — is training data available through public sources, purchasable datasets, or synthesis? Consider three scarcity types: structural (domain complexity, few cases), regulatory (HIPAA, GDPR), institutional (data siloed across organizations). (2) Defender's data moat — does the incumbent hold exclusive operational data (transaction histories, routing, customer behavior) that compounds its AI advantage and can't be replicated from public sources? High D4 means attacker can realistically assemble comparable data and defender has no exclusive moat. Low D4 means data is scarce or incumbent's proprietary data creates a compounding advantage.
- **D5 Workflow Disintermediation** — does AI collapse or bypass the multi-step workflow this product orchestrates? Risk is higher when: (1) Product role — is it a connector, aggregator, or orchestrator sitting between other systems rather than a terminal endpoint? Intermediary products are exposed when AI agents can call upstream and downstream systems directly. (2) Workflow collapsibility — can an AI agent compress the multi-step process the product currently manages into a single prompt-to-output sequence? Count the number of discrete human handoffs or system hops the product currently orchestrates. (3) End-to-end automation — can the full workflow be automated without the product's involvement? If inputs and outputs are both machine-readable and the transformation logic is learnable, the intermediary is structurally exposed. Score 0 = product IS the terminal endpoint, not an intermediary; workflows are proprietary and deeply embedded. Score 5 = pure connector or aggregator; multi-step workflow trivially collapsible by AI agents; no proprietary orchestration logic.
- **D6 Interface Substitution** — does AI bypass the product's UI or interaction layer entirely? Distinct from D5 (workflow collapse across systems): D6 asks whether the value of *this product's own interface* gets substituted by agents or copilots that interact with underlying APIs or data directly. Risk is higher when: (1) UI-centric value — product value primarily resides in UI navigation, clicks, and workflow steps rather than in proprietary underlying data or logic; the glass between user and data becomes the product's only moat. (2) Agent/copilot overlap — AI agents or vendor copilots can replicate or replace the interaction layer (e.g., an agent calling Salesforce APIs directly bypasses the CRM UI); other vendors building AI layers on top erode the incumbent's interface moat. (3) Seat consolidation — per-seat pricing models are directly threatened when AI reduces the number of human operators needed; risk factors mentioning seat count pressure, license compression, or agent-per-seat substitution. (4) UI commoditization signals — filing language acknowledging AI automation replacing manual steps, workflow automation risk, or platform consolidation. Score 0 = product value is in irreplaceable underlying logic, data, or computation; UI is a thin access layer. Score 5 = value almost entirely in the UI/workflow experience; agents can fully bypass via APIs; per-seat model directly threatened.

**Gates** (structural barriers that slow AI substitution regardless of dimension scores, rated None / Low / Medium / High / Very High). Each gate is tagged with a **durability class** that determines how it erodes:
- **G1 Regulatory** [Hard] — does AI deployment require explicit government approval, or do judicial/structural rulings bar or slow new AI entrants? Hard gate: holds until a policy or judicial decision flips it.
- **G2 Liability** [Structural] — who bears the cost of AI error, and does unclear liability create human-in-the-loop friction? Includes financial infrastructure embeddedness where the transaction must route through the incumbent regardless of who initiates it.
- **G3 Institutional Inertia** [Soft] — switching costs, procurement cycles, legacy system integration, professional guild resistance, long contract durations, and deep integration into customer operations. Soft gate: erodes continuously.
- **G4 Trust** [Soft] — psychological or cultural requirement for a trusted party — human or established brand — at the point of delivery or decision. Erodes generationally; among the slowest gates to open but erosion is largely irreversible.
- **G5 Physical Last-Mile** [Structural] — must the service be delivered in person or at a physical location? Includes multi-sided physical coordination across customers, suppliers, and workers. Most structurally durable gate.
- **G6 Network Effects** [Structural] — multi-sided market network effects requiring simultaneous competitive displacement across all market sides. A three-sided market is harder to displace than two-sided, which is harder than single-sided.

**Durability classes:**
- **Hard** (×1.5) — requires external authority (policy, judicial) to change; does not erode over time
- **Structural** (×1.25) — requires physical infrastructure or network buildout to overcome; doesn't erode from market forces alone but can be built with sufficient capital and time
- **Soft** (×1.0) — erodes continuously through competition and generational change

**Composite vulnerability score** = average of D1–D6.

**Composite gate score** = weighted average of gate strengths (None=0, Low=1, Medium=2, High=3, Very High=4) multiplied by durability weights, normalized to 0–5. Higher = better protected.

**Binding gate** = the gate with the highest effective strength (strength × durability multiplier) on the critical attack path. A "High" hard gate outranks a "Very High" soft gate because it doesn't erode.

## How to run the assessment

### Step 1 — Identify inputs

Collect from the user:
- One or more 10-K files (SEC `.txt` submission bundle or standalone `.htm`)
- Company names and tickers
- Optionally: a preferred model (default `claude-opus-4-6`)

Files uploaded by the user are available at `/mnt/user-data/uploads/<filename>`.

### Step 2 — Install dependencies

\`\`\`bash
pip install anthropic requests pandas tabulate beautifulsoup4 lxml reportlab --break-system-packages -q
\`\`\`

### Step 3 — Configure and run the script

The script lives at `scripts/ai_disruption_assessor.py`. Edit the `COMPANIES` list near the bottom of the file to point at the uploaded files:

\`\`\`python
COMPANIES = [
    CompanyInput(
        name="Company Name",
        ticker="TICK",
        file_path="/mnt/user-data/uploads/<filename>.txt",
    ),
]

MODEL = "claude-opus-4-6"
OUTPUT_CSV = "disruption_scores.csv"
OUTPUT_PDF = "disruption_report.pdf"
\`\`\`

Then run:

\`\`\`bash
cd /home/claude
ANTHROPIC_API_KEY="<key>" python /path/to/scripts/ai_disruption_assessor.py
\`\`\`

### Step 4 — Copy outputs and present to user

\`\`\`bash
cp /home/claude/disruption_report.pdf /mnt/user-data/outputs/disruption_report.pdf
cp /home/claude/disruption_scores.csv /mnt/user-data/outputs/disruption_scores.csv
\`\`\`

...and so on (full file as shown above)