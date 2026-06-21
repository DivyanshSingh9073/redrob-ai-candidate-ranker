# TODO - Candidate ranker tuning (Redrob AI Hackathon 2026)

## Planned edits (from approved plan)
- [ ] Update `PREFERRED_SKILLS` to the improved list (normalize to lowercase)
- [ ] Update `EVAL_SKILLS` to include the full evaluation terms list (lowercase)
- [ ] Add `VERY_NEGATIVE_TITLE_TERMS` and apply strong `title_score = -0.5` override in `compute_title_features`
- [ ] Rebalance `WEIGHTS` to the provided table
- [ ] Penalize consulting-only profiles more: `company_score` 0.2 → 0.1
- [ ] Step 10: Add `DESCRIPTION_SIGNAL_TERMS`, compute `desc_score` from career-history descriptions, and include it in `composite_score`
- [ ] Smoke test: run `python rank.py --input candidates.jsonl --output submission.csv --debug debug.csv`

