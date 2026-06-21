"""
rank.py - Redrob AI Hackathon 2026: Candidate Ranking System
==============================================================

CPU-only, offline, <=5 min runtime, <=16GB RAM.
No GPU, no network, no hosted LLM APIs.

Usage:
    python rank.py --input candidates.jsonl --output submission.csv
    python rank.py --input candidates.jsonl --output submission.csv --debug debug.csv
"""

import argparse
import csv
import json
import math
import random
import re
import time
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# ---------------------------------------------------------------------------
# 1. JOB DESCRIPTION CONFIG
# ---------------------------------------------------------------------------

JD_TEXT = """
Senior AI Engineer Founding Team at Redrob AI, based in Pune or Noida,
hybrid, open to relocation from Tier-1 Indian cities. Deeply technical in
modern machine learning systems and product focused, capable of shipping
quickly. Production experience with embeddings based retrieval systems,
retrieval systems, ranking systems, recommendation systems, vector search,
hybrid search, written in python. Familiar with sentence transformers,
openai embeddings, bge, e5, pinecone, weaviate, qdrant, milvus, opensearch,
elasticsearch, faiss. Strong evaluation knowledge including ndcg, map, mrr,
a b testing and offline evaluation. Preferred experience with llm
fine-tuning, lora, qlora, peft, learning to rank, xgboost, hrtech,
marketplace products, distributed systems, open source contributions.
"""

# Core required skills (expanded to cover modern RAG / LLM tooling)
REQUIRED_SKILLS = [
    "python",
    "embedding",
    "embeddings",
    "retrieval",
    "ranking",
    "recommendation",
    "vector search",
    "hybrid search",

    "sentence transformer",
    "sentence transformers",
    "openai embedding",

    "bge",
    "e5",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "opensearch",
    "elasticsearch",
    "faiss",

    "rag",
    "llm",
    "transformers",
    "langchain",
    "llamaindex",
    "chromadb",
    "hnsw",
    "reranking",
    "cross encoder"
]

EVAL_SKILLS = [
    "ndcg",
    "map@",
    "mean average precision",
    "mrr",
    "a/b test",
    "ab test",
    "offline evaluation",
]

PREFERRED_SKILLS = [
    "lora",
    "qlora",
    "peft",
    "fine-tun",
    "finetun",
    "fine tun",

    "learning to rank",
    "learning-to-rank",
    "ltr",
    "lambda mart",
    "xgb ranker",
    "xgboost",
    "lightgbm",
    "catboost",

    "rag evaluation",
    "cross encoder",
    "reranker",
    "prompt engineering",
    "langgraph",
    "agentic ai",

    "hrtech",
    "marketplace",
    "distributed system",

    "open source",
    "open-source",
    "github",
    "github.com"
]

PRODUCTION_SIGNAL_TERMS = [
    "production",
    "deployed",
    "shipped",
    "scale",
    "scaled",
    "live",
    "users",
    "latency",
    "throughput",
    "in prod",
    "rolled out",
    "served",
    "microservice",
    "docker",
    "kubernetes",
    "api",
    "real time",
    "monitoring",
    "airflow"
]

NEGATIVE_TITLE_TERMS = [
    "marketing manager",
    "digital marketing",
    "seo specialist",
    "hr manager",
    "human resources",
    "graphic designer",
    "accountant",
    "content writer",
    "copywriter",
    "talent acquisition",
    "recruiter",
    "sales executive",
    "office admin",
    "customer support",
    "customer success",
    "business analyst",
    "ui designer",
    "ux designer",
    "sales manager",
    "finance manager",
]

# Updated ML signal terms (improved recall for modern GenAI + Retrieval stacks)
ML_SIGNAL_TERMS = [
    "machine learning",
    "ml engineer",
    "ai engineer",
    "data scientist",
    "deep learning",
    "nlp",
    "llm",
    "transformer",
    "retrieval",
    "search",
    "ranking",
    "recommendation",
    "rag",
    "vector database",
    "embedding",
    "applied scientist",
    "research engineer",
    "genai",
]

CONSULTING_COMPANIES = [
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "cognizant",
    "capgemini",
    "accenture",
    "hcl",
"tech mahindra",
"deloitte",
"ibm",
"ltimindtree",
]

CORE_LOCATIONS = ["pune", "noida"]
TIER1_CITIES = [
    "pune",
    "noida",
    "bangalore",
    "bengaluru",
    "delhi",
    "new delhi",
    "mumbai",
    "hyderabad",
    "chennai",
    "gurgaon",
    "gurugram",
    "ncr",
]

POSITIVE_TITLE_TERMS = [
    "ai engineer",
    "ml engineer",
    "machine learning engineer",
    "senior ml engineer",
    "senior ai engineer",
    "applied scientist",
    "research scientist",
    "research engineer",
    "data scientist",
    "nlp engineer",
    "llm engineer",
    "genai engineer",
    "gen ai engineer",
    "search engineer",
    "retrieval engineer",
    "ranking engineer",
    "recommendation engineer",
    "backend engineer",
    "software engineer",
    "senior software engineer",
    "staff engineer",
    "founding engineer",
    "platform engineer",
]

EXPERT_LEVEL_TERMS = ["expert", "advanced", "proficient", "specialist"]

CURRENT_YEAR = datetime.now().year

# ---------------------------------------------------------------------------
# 2. GENERIC HELPERS
# ---------------------------------------------------------------------------

def g(d, *keys, default=None):
    """Safely fetch the first present key from a dict, case-insensitive."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
        for actual in d:
            if actual.lower() == k.lower() and d[actual] not in (None, ""):
                return d[actual]
    return default


def to_text(val):
    """Flatten any value (list/dict/str/None) into a lowercase string."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.lower()
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, list):
        return " ".join(to_text(v) for v in val)
    if isinstance(val, dict):
        return " ".join(to_text(v) for v in val.values()) if val else ""
    return str(val).lower()


_DATE_RE = re.compile(r"(\d{4})")


def parse_year(val):
    """Best-effort extraction of a 4-digit year from messy date fields."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        y = int(val)
        return y if 1950 <= y <= CURRENT_YEAR + 1 else None
    s = str(val).strip().lower()
    if s in ("present", "current", "now", "ongoing", ""):
        return CURRENT_YEAR
    m = _DATE_RE.search(s)
    if m:
        y = int(m.group(1))
        if 1950 <= y <= CURRENT_YEAR + 1:
            return y
    return None


def safe_float(val, default=0.0):
    try:
        if isinstance(val, str):
            val = val.strip().rstrip("%")
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# 3. FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def get_career_spans(career_history):
    """Return list of (start_year, end_year, title, company) tuples."""
    spans = []
    if not isinstance(career_history, list):
        return spans
    for role in career_history:
        if not isinstance(role, dict):
            continue
        start = parse_year(g(role, "start_date", "from", "start_year", "startdate"))
        end = parse_year(g(role, "end_date", "to", "end_year", "enddate", default="present"))
        title = to_text(g(role, "title", "job_title", "position", "designation"))
        company = to_text(g(role, "company", "company_name", "employer"))
        if start is not None:
            if end is None or end < start:
                end = max(start, CURRENT_YEAR)
            spans.append((start, end, title, company))
    spans.sort(key=lambda x: x[0])
    return spans


def compute_experience_years(profile, career_history):
    """Returns (exp_years, stated_exp, derived_exp)."""
    stated_exp = None
    raw = g(profile, "experience", "total_experience", "years_of_experience")
    if raw is not None:
        if isinstance(raw, (int, float)):
            yrs = float(raw)
            if 0 <= yrs <= 50:
                stated_exp = yrs
        else:
            m = re.search(r"(\d+(\.\d+)?)", str(raw))
            if m:
                yrs = float(m.group(1))
                if 0 <= yrs <= 50:
                    stated_exp = yrs

    spans = get_career_spans(career_history)
    if spans:
        earliest = min(s[0] for s in spans)
        latest = max(s[1] for s in spans)
        derived_exp = max(0.0, float(latest - earliest))
    else:
        derived_exp = 0.0

    exp_years = stated_exp if stated_exp is not None else derived_exp
    return exp_years, stated_exp, derived_exp


def compute_skill_text(candidate, profile):
    """Combine all textual signal fields used for skill / semantic matching."""
    parts = [
        to_text(g(profile, "headline", "title", "current_title")),
        to_text(g(profile, "industry")),
        to_text(candidate.get("skills")),
        to_text(candidate.get("certifications")),
    ]
    for role in candidate.get("career_history", []) or []:
        if isinstance(role, dict):
            parts.append(to_text(g(role, "title", "job_title", "position")))
            parts.append(to_text(g(role, "description", "summary", "responsibilities")))
    return " ".join(p for p in parts if p)


def keyword_score(text, vocab, cap):
    """Fraction of vocab terms present, scaled with diminishing returns."""
    matched = [term for term in vocab if term in text]
    n = len(matched)
    score = min(n, cap) / cap
    return score, matched


def compute_title_features(profile, career_history):
    current_title = to_text(g(profile, "headline", "title", "current_title"))
    spans = get_career_spans(career_history)
    if not current_title and spans:
        current_title = spans[-1][2]

    negative_hit = any(t in current_title for t in NEGATIVE_TITLE_TERMS)
    positive_hit = any(t in current_title for t in POSITIVE_TITLE_TERMS)
    ml_signal = any(t in current_title for t in ML_SIGNAL_TERMS)

    if positive_hit:
        title_score = 1.0
    elif ml_signal:
        title_score = 0.6
    elif negative_hit:
        title_score = 0.0
    else:
        title_score = 0.35

    return current_title, title_score, negative_hit


def compute_company_features(profile, career_history):
    companies = set()
    current_company = to_text(g(profile, "company", "current_company"))
    if current_company:
        companies.add(current_company)
    for role in career_history or []:
        if isinstance(role, dict):
            c = to_text(g(role, "company", "company_name", "employer"))
            if c:
                companies.add(c)

    consulting_hits = sum(1 for c in companies if any(cc in c for cc in CONSULTING_COMPANIES))
    is_consulting_only = len(companies) > 0 and consulting_hits == len(companies)
    has_consulting = consulting_hits > 0

    if is_consulting_only:
        company_score = 0.2
    elif has_consulting:
        company_score = 0.7
    else:
        company_score = 1.0

    return company_score, is_consulting_only, current_company


def compute_location_features(profile, redrob_signals):
    loc = to_text(g(profile, "location", "city", "current_location"))
    open_to_reloc = bool(g(redrob_signals, "open_to_relocate", "open_to_relocation", default=False))

    if any(c in loc for c in CORE_LOCATIONS):
        score = 1.0
    elif any(c in loc for c in TIER1_CITIES):
        score = 0.85 if open_to_reloc else 0.55
    else:
        score = 0.6 if open_to_reloc else 0.25

    return score, loc


def compute_redrob_features(redrob_signals):
    if not isinstance(redrob_signals, dict):
        return 0.5, False

    open_to_work = bool(g(redrob_signals, "open_to_work", default=False))
    response_rate = safe_float(
        g(redrob_signals, "recruiter_response_rate", "response_rate", default=None), default=None
    )
    activity = g(redrob_signals, "recruiter_activity", "last_active", "activity_score")

    components = [1.0 if open_to_work else 0.4]

    if response_rate is not None:
        rr = response_rate / 100.0 if response_rate > 1 else response_rate
        components.append(clamp(rr))

    if activity is not None:
        if isinstance(activity, (int, float)):
            components.append(clamp(activity / 100.0 if activity > 1 else activity))
        else:
            act_text = to_text(activity)
            if any(t in act_text for t in ["active", "recent", "high"]):
                components.append(1.0)
            elif any(t in act_text for t in ["inactive", "low", "dormant"]):
                components.append(0.2)
            else:
                components.append(0.5)

    score = sum(components) / len(components)
    return score, open_to_work


# ---------------------------------------------------------------------------
# 4. HONEYPOT DETECTION
# ---------------------------------------------------------------------------

def detect_honeypot(
    candidate,
    profile,
    career_history,
    education,
    exp_years,
    stated_exp,
    derived_exp,
    skill_text,
):
    """Returns (honeypot_score 0-1, list_of_flag_names)."""
    flags = []
    spans = get_career_spans(career_history)

    if stated_exp is not None and spans:
        if stated_exp - derived_exp >= 4:
            flags.append("stated_vs_derived_experience_mismatch")

    expert_mentions = sum(1 for t in EXPERT_LEVEL_TERMS if t in skill_text)
    n_skills = len(candidate.get("skills") or [])
    if exp_years < 1.5 and (expert_mentions >= 2 or n_skills >= 8):
        flags.append("expert_skills_no_experience")

    bad_order = any(end < start for start, end, _, _ in spans)
    overlap_years = 0
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            s1, e1, _, _ = spans[i]
            s2, e2, _, _ = spans[j]
            overlap = min(e1, e2) - max(s1, s2)
            if overlap > 0:
                overlap_years += overlap
    if bad_order or overlap_years >= 3:
        flags.append("career_history_inconsistent")

    grad_years = []
    if isinstance(education, list):
        for ed in education:
            if isinstance(ed, dict):
                gy = parse_year(g(ed, "end_date", "end_year", "graduation_year", "to"))
                if gy:
                    grad_years.append(gy)

    if grad_years:
        earliest_grad = min(grad_years)
        years_since_grad = CURRENT_YEAR - earliest_grad
        if exp_years > years_since_grad + 1.5 and years_since_grad >= 0:
            flags.append("experience_exceeds_education_timeline")

    if spans:
        first_title = spans[0][2]
        first_start = spans[0][0]
        is_junior_start = any(t in first_title for t in ["intern", "junior", "fresher", "trainee"])
        for start, end, title, _ in spans:
            is_senior_title = any(t in title for t in ["vp", "vice president", "director", "head of", "chief"])
            if is_junior_start and is_senior_title and (start - first_start) <= 2:
                flags.append("implausible_title_jump")
                break

    rs = candidate.get("redrob_signals") or {}
    founded_year = parse_year(g(rs, "current_company_founded_year", "company_founded_year"))
    if founded_year and spans:
        last_start = spans[-1][0]
        tenure = spans[-1][1] - spans[-1][0]
        if last_start < founded_year and tenure >= 2:
            flags.append("company_age_mismatch")

    if exp_years > 25 and not spans and not grad_years:
        flags.append("unverifiable_long_experience")

    score = min(1.0, 0.3 * len(flags))
    return score, flags


# ---------------------------------------------------------------------------
# 5. SEMANTIC SIMILARITY (offline, no model download)
# ---------------------------------------------------------------------------

_VECTORIZER = HashingVectorizer(
    n_features=2**16,
    alternate_sign=False,
    ngram_range=(1, 2),
    norm="l2",
)


def build_jd_vector():
    return _VECTORIZER.transform([JD_TEXT.lower()])


# ---------------------------------------------------------------------------
# 6. SCORING WEIGHTS
# ---------------------------------------------------------------------------

WEIGHTS = {
    "semantic_sim": 0.22,
    "required_skill": 0.18,
    "eval_skill": 0.08,
    "preferred_skill": 0.07,
    "production_signal": 0.07,
    "experience_fit": 0.16,
    "title_score": 0.09,
    "company_score": 0.05,
    "location_score": 0.04,
    "redrob_score": 0.04,
}


def experience_fit_score(exp_years):
    if 6 <= exp_years <= 8:
        return 1.0
    if 5 <= exp_years <= 9:
        return 0.75
    sigma = 3.0
    return clamp(math.exp(-((exp_years - 7) ** 2) / (2 * sigma**2)) * 0.7)


# ---------------------------------------------------------------------------
# 7. REASONING GENERATION
# ---------------------------------------------------------------------------

OPENERS = [
    "With {exp:.0f} years of experience, {pron} has hands-on production work in {skills}.",
    "{Pron} brings {exp:.0f} years of experience and demonstrable production exposure to {skills}.",
    "{exp:.0f} years in the field, with practical depth in {skills}.",
    "A {exp:.0f}-year track record that includes real production usage of {skills}.",
]

ROLE_CLAUSES = [
    "Currently a {title} at {company}, closely aligned with the founding AI engineer mandate.",
    "Their role as {title} at {company} maps well onto a product-focused, ship-fast engineering team.",
    "As {title} at {company}, the background fits a hands-on, full-stack ML engineering role.",
]

EVAL_CLAUSES = [
    "Evaluation rigor is evident through familiarity with {evals}.",
    "Shows offline/online evaluation maturity, referencing {evals}.",
]

LOCATION_CLAUSES = [
    "Based in {loc}, matching the preferred hybrid location for this role.",
    "Located in {loc}, which fits the Pune/Noida hybrid setup well.",
    "Located in {loc}; open to relocating to a Tier-1 hub for the role.",
]

GENERIC_CLAUSES = [
    "Profile signals (recruiter responsiveness / openness to work) suggest strong engagement potential.",
    "Recent activity and responsiveness indicate this candidate is actively reachable.",
]

CONCERN_CLAUSES = [
    "Some consulting-heavy background reduces confidence in direct product shipping experience.",
    "Experience sits slightly outside the ideal 6-8 year band but remains within an acceptable range.",
]


def generate_reasoning(rec):
    rnd = random.Random(rec["candidate_id"])
    pron, Pron = "they", "They"

    skills_for_text = rec["matched_required"][:3] or rec["matched_preferred"][:3] or ["python", "ml systems"]
    skills_text = ", ".join(skills_for_text)

    opener = rnd.choice(OPENERS).format(exp=rec["exp_years"], pron=pron, Pron=Pron, skills=skills_text)
    sentences = [opener]

    if rec["current_title"] and rec["current_company"]:
        sentences.append(
            rnd.choice(ROLE_CLAUSES).format(
                title=rec["current_title"].title(),
                company=rec["current_company"].title(),
            )
        )
    elif rec["matched_eval"]:
        sentences.append(rnd.choice(EVAL_CLAUSES).format(evals=", ".join(rec["matched_eval"][:3])))
    elif rec["location"]:
        sentences.append(rnd.choice(LOCATION_CLAUSES).format(loc=rec["location"].title()))
    else:
        sentences.append(rnd.choice(GENERIC_CLAUSES))

    if rec["is_consulting_only"] or not (5 <= rec["exp_years"] <= 9):
        if len(sentences) < 2:
            sentences.append(rnd.choice(CONCERN_CLAUSES))

    return " ".join(sentences[:2])


# ---------------------------------------------------------------------------
# 8. SCORING + PIPELINE
# ---------------------------------------------------------------------------


def composite_score(rec):
    score = (
        WEIGHTS["semantic_sim"] * rec["semantic_sim"]
        + WEIGHTS["required_skill"] * rec["req_score"]
        + WEIGHTS["eval_skill"] * rec["eval_score"]
        + WEIGHTS["preferred_skill"] * rec["pref_score"]
        + WEIGHTS["production_signal"] * rec["prod_score"]
        + WEIGHTS["experience_fit"] * rec["exp_fit"]
        + WEIGHTS["title_score"] * rec["title_score"]
        + WEIGHTS["company_score"] * rec["company_score"]
        + WEIGHTS["location_score"] * rec["location_score"]
        + WEIGHTS["redrob_score"] * rec["redrob_score"]
    )

    if rec["neg_title"] and rec["prod_score"] < 0.5 and rec["req_score"] < 0.25:
        score *= 0.25

    score *= (1 - 0.6 * rec["honeypot_score"])
    return clamp(score, 0.0, 1.0)


def process_file(input_path):
    records = []
    texts = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue

            candidate_id = candidate.get("candidate_id") or f"UNKNOWN_{line_no}"
            profile = candidate.get("profile") or {}
            career_history = candidate.get("career_history") or []
            education = candidate.get("education") or []
            redrob_signals = candidate.get("redrob_signals") or {}

            skill_text = compute_skill_text(candidate, profile)
            exp_years, stated_exp, derived_exp = compute_experience_years(profile, career_history)

            req_score, matched_req = keyword_score(skill_text, REQUIRED_SKILLS, cap=8)
            eval_score, matched_eval = keyword_score(skill_text, EVAL_SKILLS, cap=3)
            pref_score, matched_pref = keyword_score(skill_text, PREFERRED_SKILLS, cap=4)
            prod_score, _ = keyword_score(skill_text, PRODUCTION_SIGNAL_TERMS, cap=2)

            current_title, title_score, neg_title = compute_title_features(profile, career_history)
            company_score, is_consulting_only, current_company = compute_company_features(profile, career_history)
            location_score, location = compute_location_features(profile, redrob_signals)
            redrob_score, open_to_work = compute_redrob_features(redrob_signals)
            exp_fit = experience_fit_score(exp_years)

            honeypot_score, honeypot_flags = detect_honeypot(
                candidate,
                profile,
                career_history,
                education,
                exp_years,
                stated_exp,
                derived_exp,
                skill_text,
            )

            records.append(
                {
                    "candidate_id": candidate_id,
                    "exp_years": exp_years,
                    "req_score": req_score,
                    "eval_score": eval_score,
                    "pref_score": pref_score,
                    "prod_score": prod_score,
                    "title_score": title_score,
                    "neg_title": neg_title,
                    "company_score": company_score,
                    "is_consulting_only": is_consulting_only,
                    "location_score": location_score,
                    "location": location,
                    "redrob_score": redrob_score,
                    "open_to_work": open_to_work,
                    "exp_fit": exp_fit,
                    "honeypot_score": honeypot_score,
                    "honeypot_flags": honeypot_flags,
                    "current_title": current_title,
                    "current_company": current_company,
                    "matched_required": matched_req,
                    "matched_eval": matched_eval,
                    "matched_preferred": matched_pref,
                }
            )
            texts.append(skill_text if skill_text else " ")

    return records, texts


def add_semantic_similarity(records, texts):
    jd_vec = build_jd_vector()
    batch_size = 50000

    if not texts:
        return

    sims = np.zeros(len(texts), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        mat = _VECTORIZER.transform(chunk)
        sim = (mat @ jd_vec.T).toarray().ravel()
        sims[start : start + len(chunk)] = sim

    denom = float(sims.max()) if len(sims) else 0.0
    sims_norm = sims / denom if denom > 0 else sims

    for rec, s in zip(records, sims_norm):
        rec["semantic_sim"] = float(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="candidates.jsonl")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--debug", default=None, help="optional path to write full feature debug CSV")
    args = parser.parse_args()

    t0 = time.time()
    records, texts = process_file(args.input)
    print(f"Loaded {len(records)} candidates in {time.time() - t0:.1f}s")

    if not records:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        print(f"No candidates to rank. Wrote 0 rows to {args.output}")
        return

    add_semantic_similarity(records, texts)
    print(f"Semantic similarity computed in {time.time() - t0:.1f}s total")

    for rec in records:
        rec["score"] = composite_score(rec)

    eligible = [r for r in records if r["honeypot_score"] < 0.6]
    eligible.sort(key=lambda r: r["score"], reverse=True)
    top100 = eligible[:100]

    rows = []
    prev_score = None
    for i, rec in enumerate(top100, start=1):
        score = rec["score"]
        if prev_score is not None and score >= prev_score:
            score = prev_score - 1e-4
        prev_score = score

        rec["score"] = max(score, 0.0)
        rec["rank"] = i
        rec["reasoning"] = generate_reasoning(rec)
        rows.append(rec)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rec in rows:
            writer.writerow([rec["candidate_id"], rec["rank"], f"{rec['score']:.6f}", rec["reasoning"]])

    if args.debug:
        debug_fields = [
            "candidate_id",
            "rank",
            "score",
            "exp_years",
            "semantic_sim",
            "req_score",
            "eval_score",
            "pref_score",
            "prod_score",
            "title_score",
            "company_score",
            "location_score",
            "redrob_score",
            "exp_fit",
            "honeypot_score",
            "honeypot_flags",
            "current_title",
            "current_company",
            "location",
            "matched_required",
            "matched_eval",
            "matched_preferred",
        ]
        with open(args.debug, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(debug_fields)
            for rec in rows:
                writer.writerow([rec.get(k) for k in debug_fields])

    print(f"Done in {time.time() - t0:.1f}s. Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

