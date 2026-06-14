"""
streamlit_app.py - Demo UI for the Redrob AI Hackathon ranking pipeline.

Run with:
    streamlit run streamlit_app.py

Lets you upload candidates.jsonl, runs the ranking pipeline from rank.py,
and shows the Top 100 with score breakdowns and honeypot diagnostics.
"""

import io
import time
import tempfile
import os

import pandas as pd
import streamlit as st

import rank as ranker

st.set_page_config(page_title="Redrob AI Candidate Ranker", layout="wide")

st.title("Redrob AI — Senior AI Engineer Candidate Ranker")
st.caption(
    "Upload candidates.jsonl to generate the Top 100 ranked candidates for "
    "the Senior AI Engineer — Founding Team role."
)

uploaded = st.file_uploader("candidates.jsonl", type=["jsonl"])

col1, col2 = st.columns(2)
with col1:
    show_debug = st.checkbox("Show full feature breakdown", value=True)
with col2:
    sample_limit = st.number_input(
        "Limit candidates processed (0 = all)", min_value=0, value=0, step=1000
    )

if uploaded is not None:
    if st.button("Run Ranking", type="primary"):
        with st.spinner("Processing candidates..."):
            # Write upload to a temp file so rank.py can stream it.
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".jsonl", delete=False
            ) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            t0 = time.time()
            try:
                records, texts = ranker.process_file(tmp_path)

                if sample_limit and sample_limit < len(records):
                    records = records[:sample_limit]
                    texts = texts[:sample_limit]

                ranker.add_semantic_similarity(records, texts)

                for rec in records:
                    rec["score"] = ranker.composite_score(rec)

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
                    rec["reasoning"] = ranker.generate_reasoning(rec)
                    rows.append(rec)

                elapsed = time.time() - t0
            finally:
                os.unlink(tmp_path)

        st.success(
            f"Processed {len(records)} candidates, ranked top {len(rows)} "
            f"in {elapsed:.1f}s."
        )

        flagged = [r for r in records if r["honeypot_score"] >= 0.3]
        st.info(
            f"Honeypot heuristics flagged {len(flagged)} candidates "
            f"({len(flagged)/max(1,len(records))*100:.1f}% of pool); "
            f"{sum(1 for r in rows if r['honeypot_score'] > 0)} of these "
            f"slipped into the Top 100 (must be <10)."
        )

        # Build submission dataframe
        sub_df = pd.DataFrame(
            [
                {
                    "candidate_id": r["candidate_id"],
                    "rank": r["rank"],
                    "score": round(r["score"], 6),
                    "reasoning": r["reasoning"],
                }
                for r in rows
            ]
        )

        st.subheader("Top 100 Submission")
        st.dataframe(sub_df, use_container_width=True, height=400)

        csv_bytes = sub_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download submission.csv",
            data=csv_bytes,
            file_name="submission.csv",
            mime="text/csv",
        )

        if show_debug:
            st.subheader("Feature Breakdown (Top 100)")
            debug_df = pd.DataFrame(
                [
                    {
                        "candidate_id": r["candidate_id"],
                        "rank": r["rank"],
                        "score": round(r["score"], 4),
                        "exp_years": r["exp_years"],
                        "semantic_sim": round(r["semantic_sim"], 3),
                        "required_skill": round(r["req_score"], 2),
                        "eval_skill": round(r["eval_score"], 2),
                        "preferred_skill": round(r["pref_score"], 2),
                        "production_signal": round(r["prod_score"], 2),
                        "experience_fit": round(r["exp_fit"], 2),
                        "title_score": round(r["title_score"], 2),
                        "company_score": round(r["company_score"], 2),
                        "location_score": round(r["location_score"], 2),
                        "redrob_score": round(r["redrob_score"], 2),
                        "honeypot_score": r["honeypot_score"],
                        "honeypot_flags": ", ".join(r["honeypot_flags"]),
                        "current_title": r["current_title"],
                        "current_company": r["current_company"],
                        "location": r["location"],
                    }
                    for r in rows
                ]
            )
            st.dataframe(debug_df, use_container_width=True, height=500)

            st.subheader("Score Distribution — Top 100")
            st.bar_chart(sub_df.set_index("rank")["score"])
else:
    st.write("Upload `candidates.jsonl` to begin.")
