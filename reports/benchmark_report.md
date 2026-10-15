# Hiver Support Agent Benchmark

## Problem framing and selected brand

The system drafts offline AmazonHelp responses from the Customer Support on Twitter dataset. AmazonHelp was selected for its volume, customer-to-brand reply links, and mixture of delivery, returns, account, and subscription issues. Historical replies are evidence of past handling, not proof of resolution or current policy.

## Intent taxonomy and data preparation

Seven intents are defined in `docs/annotation_guide.md`. Development and evaluation pairs were split by customer author. The golden set contains 200 unique held-out customer tweets. Its labels were semantically reviewed by an AI reviewer and are marked `ai_manually_reviewed`.

## Architecture

The pipeline classifies intent with constrained structured output, retrieves historical cases from a calibrated FAISS index, applies deterministic escalation rules, and drafts only when those rules permit. Invalid model output fails closed to escalation.

## Results versus baselines

| System | Accuracy | Macro F1 | Escalation accuracy | Escalation F1 |
| --- | ---: | ---: | ---: | ---: |
| Majority class | 19.50% | 0.0466 | 22.50% | 0.0000 |
| TF-IDF + logistic regression | 56.00% | 0.5178 | 45.00% | 0.5259 |
| Proposed agent | 76.00% | 0.7368 | 57.50% | 0.6909 |

Baseline source: `reports/results.json`. Agent source: `reports/agent_results.json`. Both use the same 200 examples. Baseline development labels are weak-supervision rules; agent evaluation labels are separately AI-reviewed.

## Component metrics

- Retrieval coverage: 68.00%. Top-1 same-intent proxy precision: 55.88%.
- Judge means on 80 drafts: groundedness 3.25, relevance 4.8875, helpfulness 3.2875, consistency 4.30, safety 5.00.
- Wall time: 83.195 seconds. Agent tokens: 107,943. Judge tokens: 70,680. USD cost is not estimated.
- AI–judge agreement on 30 blinded drafts: exact 50.67%, within one point 88.00%, pooled quadratic-weighted Cohen's kappa 0.4559.

## What is misleading about my headline number?

The 76.00% classification accuracy is not evidence that the agent safely handles 76% of customers. Sixty examples expected to escalate were auto-handled, so a correct intent can still produce the wrong operational decision. Escalation accuracy is only 57.50%. The safety judge averaged 5.00, but the separate AI rubric review found unsupported status claims that the deterministic filter missed; their safety kappa is 0.0 because the judge shows almost no score variation. The benchmark labels and the 30 comparison ratings were also produced by AI reviewers, so neither classification accuracy nor agreement has independent human validation. Finally, baseline and agent labels come from different labeling processes, so the improvement over TF-IDF is directional evidence, not a controlled same-label comparison.

## Limitations

The sample is limited to 200 examples from one brand and the scanned dataset prefix. Historical labels are noisy, and same-intent retrieval is only a proxy. Duplicate and author filtering reduces leakage but cannot eliminate paraphrased near-duplicates. There is one AI reviewer, so no inter-annotator agreement exists.

## One more week

I would have one person adjudicate all 85 action disagreements and the 30 judged drafts, revise the escalation rules from those adjudications, and rerun the fixed benchmark. I would also add language-aware retrieval and expand the unsupported-claim detector before changing the model.
