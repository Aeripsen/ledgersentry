# ADR 004: serving shape - one FastAPI service, one artifact, and the cuts

Status: accepted

## Context

The serving problem is small and sharp: one model, dataset-dependent feature
schema, single-row latency that must fit inside payment authorization, plus a
batch path. The temptation in "make it production-grade" work is to add the
infrastructure of a system ten times this size.

## Decision

One FastAPI service loading one locally-trained joblib artifact
({preprocessor, model, drift_reference} - the drift reference travels WITH the
model it describes). Endpoints: /predict, /predict/batch, /drift, /health,
/ready, /curve. The expected input schema is read off the fitted preprocessor,
never hardcoded, so retraining on a different source reshapes the API's
expectations automatically. Scoring goes through the compiled scorer (ADR 005).
Configuration via pydantic-settings (env + optional yaml). Docker/compose for
the local two-container demo (api + dashboard).

## Cuts (each considered, each rejected as speculative for THIS system)

- **Kubernetes / microservices:** one process serves one model. There is no
  second service to orchestrate.
- **Message queue / streaming platform:** the "stream" here is a replay for
  honest latency measurement. No producer exists; a queue would be theater.
- **Database:** no state outlives a request except the artifact and metrics
  files. Adding one creates state without a customer for it.
- **ONNX / model-compilation export:** a second serialization format means a
  permanent parity-testing surface. The measured bottleneck was pandas overhead,
  not the model (see benchmark); compiled transform + native sklearn predict
  already meets the stated p99 budget 5x over.
- **A Sink/alert-output abstraction:** the CLI replay prints, the dashboard
  renders. Two consumers, but no third destination exists or is named; a
  protocol there would have one hypothetical implementer. If a webhook/queue
  consumer ever arrives, extract it then.
- **Plugin system for loaders/models:** the LoaderSpec table and the model
  registry ARE the extension points, each with multiple real implementations
  today. Anything more general has no second use case in the repo.

## Consequences

- The whole system remains readable in one sitting, which is a feature.
- Horizontal scale is "run more replicas behind a load balancer" and is a
  deployment concern, documented as such in the threat model.
