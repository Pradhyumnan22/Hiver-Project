# Customer Support Intent Annotation Guide — AmazonHelp

## 1. Objectives & Overview
This guide defines the taxonomy and annotation protocols for customer messages directed to **AmazonHelp**.
Every example in the golden evaluation set is labeled with:
- `intent`: One of 7 mutually exclusive intents.
- `expected_action`: `auto_handle` or `escalate`.
- `escalation_reason`: Explicit rationale if escalated (empty/none if `auto_handle`).
- `notes`: Specific context or edge-case handling considerations.
- `provenance`: Tweet and author identifiers ensuring provenance without exposing PII.

---

## 2. Intent Taxonomy Definitions

### 1. `order_status_delivery`
- **Definition**: Inquiries regarding package location, delivery dates, tracking status, shipping carrier delays, or packages marked delivered but not received.
- **Example**: *"Where is my package? The tracking has not updated in 3 days."*
- **Auto-handle vs Escalate**:
  - `auto_handle`: General questions about standard delivery windows, tracking link guidance, or holiday shipping policies.
  - `escalate`: Specific missing packages, packages marked delivered that cannot be found, carrier dispute requiring order/tracking lookup.

### 2. `returns_refunds`
- **Definition**: Requests to return an item, obtain a return QR/barcode label, drop-off locations, refund status, or reimbursement timelines.
- **Example**: *"I dropped off my return at the locker yesterday, when will I get my refund?"*
- **Auto-handle vs Escalate**:
  - `auto_handle`: Return policy window questions, generic instructions on how to start a return in 'Your Orders'.
  - `escalate`: Specific refund discrepancies, missing return confirmation, fee disputes, requests for manual override.

### 3. `damaged_defective_item`
- **Definition**: Products arriving broken, defective, expired, missing parts, or completely wrong item received.
- **Example**: *"Received my blender today and the glass pitcher was completely shattered."*
- **Auto-handle vs Escalate**:
  - `auto_handle`: Standard replacement policy explanation (e.g. 30-day replacement window).
  - `escalate`: Immediate replacement/refund requests, safety hazards, damaged high-value goods requiring photo inspection or replacement dispatch.

### 4. `account_access_security`
- **Definition**: Inability to log in, 2-factor authentication / OTP failure, compromised account, unauthorized charges, or closing an account.
- **Example**: *"I see an unknown charge of $79 on my card from Amazon, please help."*
- **Auto-handle vs Escalate**:
  - `auto_handle`: None. Account security issues ALWAYS require human or secure channel verification.
  - `escalate`: Escalate 100% of the time (`EscalationReason.ACCOUNT_SECURITY_PII`).

### 5. `subscription_digital`
- **Definition**: Questions or complaints regarding Amazon Prime, Prime Video, Kindle Unlimited, Music, or Audible subscriptions (billing, auto-renew, benefits).
- **Example**: *"Why was I charged for Prime when I cancelled my free trial?"*
- **Auto-handle vs Escalate**:
  - `auto_handle`: Benefit explanations (e.g. what is included in Prime Video).
  - `escalate`: Unauthorized subscription charges, refund requests for auto-renewed subscriptions, plan cancellations requiring account changes.

### 6. `product_inquiry_availability`
- **Definition**: Questions regarding item specifications, stock availability, pricing match, warranty details, or seller questions prior to purchase.
- **Example**: *"Will this case fit the iPhone 13 Pro or only the standard 13?"*
- **Auto-handle vs Escalate**:
  - `auto_handle`: Directing user to product details page, seller Q&A section, or stock notification toggle.
  - `escalate`: Complex manufacturer warranty claims or pricing error disputes.

### 7. `other_unknown`
- **Definition**: Unclear messages, gibberish, general praise/venting without actionable issue, or completely unrelated inquiries.
- **Example**: *"Amazon is great thanks!"* or *"Hey check this out"*
- **Auto-handle vs Escalate**:
  - `auto_handle`: Polite acknowledgement for pure greetings/praise.
  - `escalate`: Incomprehensible messages with frustrated tone or ambiguous claims (`EscalationReason.AMBIGUOUS_UNSUPPORTED`).

---

## 3. Escalation Policy Criteria

| Action | When Applicable | Rationale |
| :--- | :--- | :--- |
| `auto_handle` | Pure informational questions, generic policy queries, standard how-to guidance that requires NO private customer data. | Can be safely resolved using publicly documented knowledge and guidance. |
| `escalate` | Order-specific lookup, account credentials, payment disputes, refunds, damaged replacements, angry repeated complaints. | Requires private customer lookup (DM), sensitive PII, or financial authorization. |

### Valid Escalation Reasons
1. `account_security_pii`: Involves passwords, unauthorized charges, account takeover, or personal data.
2. `order_lookup_required`: Requires accessing customer order records, courier tracking details, or address verification.
3. `refund_replacement_dispute`: Involves financial transactions, refund approvals, or replacement shipments.
4. `ambiguous_unsupported`: Message cannot be mapped with confidence to supported knowledge or requires clarification.
5. `high_sentiment_risk`: Severe customer frustration, legal threats, or safety hazards.

---

## 4. Review Procedure

Run `python scripts/build_golden_set.py prepare` to create
`data/annotations/golden_review.csv`. The command uses seed 42, removes
duplicate customer tweet IDs, stratifies by preliminary intent, and backfills
sparse categories to produce exactly 200 rows.

The preliminary labels are keyword-assisted suggestions, not ground truth.
For every row, a reviewer must:

1. Read the cleaned customer message, available context, and historical reply.
2. Confirm or correct `expected_intent`, `expected_action`,
   `escalation_reason`, and `annotation_notes`.
3. Record genuinely ambiguous choices in `review_notes`.
4. Set `review_status` to `verified`, enter a stable non-secret
   `reviewer_id`, and enter an ISO-8601 timestamp in `reviewed_at`.

The historical reply may clarify what the customer meant, but it must not be
treated as proof that the issue was resolved or that the response remains
current policy.

For the current benchmark, four independent AI review batches each received
50 rows and applied the guide row by row. Their corrections were merged only
after complete ID coverage and taxonomy validation. Final rows therefore use
`annotation_method=ai_manually_reviewed`; they are not represented as
human-verified. A human-reviewed subset and judge-human agreement remain work
for Checkpoint 7, and no inter-annotator agreement is claimed here.

After all rows are reviewed, run:

```powershell
python scripts/build_golden_set.py apply-ai-reviews
python scripts/build_golden_set.py finalize
```

Finalization fails if the file does not contain exactly 200 unique customer
tweets or any row lacks review provenance. The review CSV and four batch files
are retained as the annotation audit trail.
