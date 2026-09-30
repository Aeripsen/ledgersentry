// Closed-loop load against POST /predict through the ClusterIP Service, run
// in-cluster as a Job (job.yaml) so it measures the Service path, not a
// port-forward. VUS virtual users each send one request, wait for the answer,
// then send the next. Prints one machine-readable summary line that
// scripts/k8s_report.py turns into artifacts/k8s_kind_ledgersentry.json.
//
// Besides the phase totals, requests and failures are counted per 10 s of wall
// clock (setup() records the epoch the phase started at), so the report can
// line them up with the moment the rolling restart began and ended instead of
// comparing whole-phase averages.
import http from "k6/http";
import { Counter } from "k6/metrics";

const status2xx = new Counter("status_2xx");
const status5xx = new Counter("status_5xx");
const statusOther = new Counter("status_other");
const connErrors = new Counter("conn_errors");

// k6 only allows metrics declared in the init context, so the buckets are
// declared up front. 60 x 10 s covers any phase up to 10 minutes; later
// requests land in the last bucket.
const BUCKET_S = 10;
const MAX_BUCKETS = 60;
const bucketReqs = [];
const bucketFails = [];
for (let i = 0; i < MAX_BUCKETS; i++) {
  const n = String(i).padStart(2, "0");
  bucketReqs.push(new Counter(`bucket_${n}_requests`));
  bucketFails.push(new Counter(`bucket_${n}_failed`));
}

const TARGET = __ENV.TARGET;
const BODY = open("/scripts/payload.json");

export const options = {
  scenarios: {
    load: {
      executor: "constant-vus",
      vus: Number(__ENV.VUS || 16),
      duration: __ENV.DURATION || "60s",
    },
  },
  summaryTrendStats: ["avg", "med", "p(95)", "p(99)", "max"],
};

export function setup() {
  return { t0: Date.now() };
}

export default function (data) {
  const r = http.post(`${TARGET}/predict`, BODY, {
    headers: { "Content-Type": "application/json" },
    timeout: "10s",
  });
  const b = Math.min(MAX_BUCKETS - 1, Math.floor((Date.now() - data.t0) / (BUCKET_S * 1000)));
  bucketReqs[b].add(1);
  if (r.status === 0) connErrors.add(1);
  else if (r.status >= 200 && r.status < 300) status2xx.add(1);
  else if (r.status >= 500) status5xx.add(1);
  else statusOther.add(1);
  if (r.status < 200 || r.status >= 300) bucketFails[b].add(1);
}

function count(data, name) {
  const m = data.metrics[name];
  return m ? m.values.count : 0;
}

export function handleSummary(data) {
  const d = data.metrics.http_req_duration.values;
  const buckets = [];
  for (let i = 0; i < MAX_BUCKETS; i++) {
    const n = String(i).padStart(2, "0");
    const req = count(data, `bucket_${n}_requests`);
    if (req) buckets.push([i * BUCKET_S, req, count(data, `bucket_${n}_failed`)]);
  }
  const out = {
    phase: __ENV.PHASE,
    vus: Number(__ENV.VUS || 16),
    duration: __ENV.DURATION || "60s",
    requests: data.metrics.http_reqs.values.count,
    req_per_s: data.metrics.http_reqs.values.rate,
    latency_ms: { avg: d.avg, p50: d.med, p95: d["p(95)"], p99: d["p(99)"], max: d.max },
    ok_2xx: count(data, "status_2xx"),
    failed_5xx: count(data, "status_5xx"),
    failed_other_status: count(data, "status_other"),
    failed_connection: count(data, "conn_errors"),
    t0_epoch_ms: data.setup_data ? data.setup_data.t0 : null,
    bucket_seconds: BUCKET_S,
    // [seconds since t0 at the bucket start, requests finished in it, of which failed]
    buckets: buckets,
  };
  return { stdout: "\nK6_SUMMARY " + JSON.stringify(out) + "\n" };
}
