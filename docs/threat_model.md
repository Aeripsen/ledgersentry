# Threat model

Scope: this repo as shipped - a research/portfolio fraud scorer (FastAPI service,
Streamlit dashboard, training pipeline). It is not a production payment system and
this page does not pretend to secure one; it states what the shipped code defends
against, what it deliberately leaves to deployment, and the one rule that must
never be broken.

## Assets

- The trained artifact (`artifacts/ledgersentry.joblib`): model + preprocessor +
  drift reference. Gitignored, produced locally by `scripts/train.py`.
- The service itself (availability, correctness of decisions).
- No PII exists in the repo: no dataset is committed, ULB features are
  PCA-anonymized by the dataset authors, and the synthetic fixture is generated.

## The one hard rule: the artifact is code

`joblib.load` is pickle: loading an artifact EXECUTES whatever is inside it.
The service therefore loads exactly one artifact, from the configured local
`artifact_dir`, which only `scripts/train.py` writes. Never point it at an
artifact you did not train yourself, never download artifacts, never accept an
artifact path from a request. This repo ships no artifact-fetching code on
purpose - adding some would move the trust boundary, and the README will not
tell anyone to `curl` a model.

## Request surface (what the shipped code handles)

| Threat | Shipped handling |
|---|---|
| Malformed request bodies | pydantic validation; unknown shapes are 422, never a stack trace |
| Type confusion (string where a number goes) | explicit coercion; failure is 422, not 500 |
| Oversized batch (memory/CPU per request) | `/predict/batch` and `/drift` capped at `max_batch` (config, default 10,000); over-cap is 422 |
| Missing model at startup | `/predict` 503s with a clear message; `/ready` gates traffic; `/health` stays alive for the orchestrator |
| Sensitive data in logs | structured logs carry decisions/latencies/counts only - transaction feature values are never logged |

## Adversarial ML surface (stated, mostly NOT mitigated here)

- **Evasion / score probing:** an attacker who can query `/predict` freely can
  binary-search the decision boundary and shape transactions to auto-clear. The
  shipped code does not rate-limit or authenticate - those are deployment
  controls (put the service behind auth; alert on high-volume probing).
- **Model stealing:** unlimited scoring access lets an attacker fit a surrogate.
  Same deployment story.
- **Drift as attack:** fraud drifts adversarially by nature. The PSI monitor
  (`/drift`) catches marginal feature shift and null-spikes; it does not catch
  joint-distribution shifts or label drift, and it never proves the model wrong,
  only that its inputs moved. See `src/ledgersentry/drift.py`.
- **Training-data poisoning:** out of scope; training reads local files you
  chose to drop into `data/`.

## Deliberately left to deployment

Authentication, TLS, rate limiting, request quotas, network segmentation,
secrets management. The service binds to localhost by default via uvicorn; the
Docker compose file publishes ports for local demo use. None of this is a
production posture and the README does not claim otherwise.

## Money boundary

This system scores transactions for fraud and routes them to fraud/legit/review.
It holds no funds, moves no money, and emits no financial advice. Nothing in
this repo may be extended to do so.
