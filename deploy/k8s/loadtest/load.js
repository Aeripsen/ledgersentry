// Closed-loop load against POST /predict through the ClusterIP Service, run
// in-cluster as a Job (job.yaml) so it measures the Service path, not a
// port-forward. VUS virtual users each send one request, wait for the answer,
// then send the next. Prints one machine-readable summary line that
// scripts/k8s_report.py turns into artifacts/k8s_kind_ledgersentry.json.
import http from "k6/http";
import { Counter } from "k6/metrics";

const status2xx = new Counter("status_2xx");
const status5xx = new Counter("status_5xx");
const statusOther = new Counter("status_other");
const connErrors = new Counter("conn_errors");

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

export default function () {
  const r = http.post(`${TARGET}/predict`, BODY, {
    headers: { "Content-Type": "application/json" },
    timeout: "10s",
  });
  if (r.status === 0) connErrors.add(1);
  else if (r.status >= 200 && r.status < 300) status2xx.add(1);
  else if (r.status >= 500) status5xx.add(1);
  else statusOther.add(1);
}

function count(data, name) {
  const m = data.metrics[name];
  return m ? m.values.count : 0;
}

export function handleSummary(data) {
  const d = data.metrics.http_req_duration.values;
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
  };
  return { stdout: "\nK6_SUMMARY " + JSON.stringify(out) + "\n" };
}
