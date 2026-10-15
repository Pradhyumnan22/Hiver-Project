# Failure Analysis

All counts come from `reports/agent_results.json` on the 200-example golden set. Expected labels were AI-reviewed, so these are observed disagreements with that benchmark, not independently verified human errors.

## 1. Account-specific cases were auto-handled

60 of 85 action errors expected `escalate` but received `auto_handle`. Example `gold_75849` asks for an alternative refund because a bank account is closed. The agent correctly predicted `returns_refunds` but drafted a public reply instead of escalating for transaction verification.

**Hypothesis:** The transaction rule requires narrow phrases such as “my refund,” so broader account-specific wording bypasses it.

**Potential fix:** Escalate refund, replacement, charge, and closed-account language even when the phrasing does not match the current patterns.

## 2. Unsupported status claims passed the safety filter

`gold_71957` only says an email reported an item lost or damaged. The draft says, “our team is looking into this matter,” although no evidence shows an investigation. The judge still gave safety 5, while the independent AI rubric review gave safety 2.

**Hypothesis:** The claim detector looks for completed verbs such as “issued” and “processed,” but not assurances that work is already underway.

**Potential fix:** Reject claims that a team is investigating, has contacted a carrier, or will provide an update unless that action is present in the evidence.

## 3. Intent boundaries overlap

48 of 200 intent predictions disagree with the benchmark. The largest groups are delivery versus `other_unknown` (6), `other_unknown` versus account security (5), and `other_unknown` versus delivery (5). `gold_71957` was labeled delivery but predicted damaged item because “lost or damaged” contains both signals.

**Hypothesis:** Short messages and keyword overlap make these boundaries unstable for both the classifier and the benchmark labels.

**Potential fix:** Add a small human-adjudicated boundary set and require the model to preserve multiple applicable intents before selecting one.

## 4. Weak or cross-lingual evidence was treated as grounding

`gold_120187` is a Portuguese address confirmation. Its retrieved evidence is a Spanish delivery exchange, and the draft assumes an unresolved delivery problem. Retrieval coverage is 68.00%, but top-1 same-intent proxy precision is only 55.88%.

**Hypothesis:** The 0.65 threshold accepts semantically nearby but operationally mismatched cases, especially across languages.

**Potential fix:** Require language agreement and raise the threshold when the top result has a different weakly supervised intent.

## 5. Informational messages were over-escalated

25 benchmark `auto_handle` examples were escalated: 9 for insufficient evidence, 8 for unsupported intent, 3 for account verification, 3 for transaction verification, and 2 for high risk. This lowers escalation precision to 79.17% and accuracy to 57.50%.

**Hypothesis:** The policy is intentionally conservative, while some benchmark auto-handle labels still concern order or account context.

**Potential fix:** Separate “safe public acknowledgement” from “public resolution” and review the 25 disagreements before relaxing any rule.
