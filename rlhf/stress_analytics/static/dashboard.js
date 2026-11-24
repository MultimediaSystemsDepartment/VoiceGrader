const palette = {
  ink: "#182225", teal: "#087f7a", coral: "#d6533c", amber: "#c28116",
  green: "#477b48", blue: "#356aa0", muted: "#7b8784", grid: "#e1e5e2",
};

const state = {
  runs: [], runId: null, overview: null, timeline: null, routes: [], issues: [],
  rangeStart: null, rangeEnd: null, charts: {},
};

const $ = (id) => document.getElementById(id);

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function number(value, digits = 0) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function percent(value, digits = 1) {
  if (value === null || value === undefined) return "—";
  return `${number(Number(value) * 100, digits)}%`;
}

function milliseconds(value) {
  if (value === null || value === undefined) return "—";
  if (value >= 1000) return `${number(value / 1000, 2)} s`;
  return `${number(value, value < 10 ? 2 : 1)} ms`;
}

function bytesPerSecond(value) {
  if (value === null || value === undefined) return "—";
  if (value >= 1024 * 1024) return `${number(value / 1024 / 1024, 2)} MiB/s`;
  if (value >= 1024) return `${number(value / 1024, 1)} KiB/s`;
  return `${number(value, 0)} B/s`;
}

function dateText(value) {
  if (!value) return "—";
  return new Date(value).toLocaleString([], { dateStyle: "medium", timeStyle: "medium" });
}

function durationText(start, end) {
  if (!start || !end) return "—";
  let seconds = Math.max(0, (new Date(end) - new Date(start)) / 1000);
  const days = Math.floor(seconds / 86400); seconds %= 86400;
  const hours = Math.floor(seconds / 3600); seconds %= 3600;
  const minutes = Math.floor(seconds / 60);
  return days ? `${days}d ${hours}h` : hours ? `${hours}h ${minutes}m` : `${minutes}m`;
}

function localInputValue(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  const shifted = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return shifted.toISOString().slice(0, 19);
}

function axisDefaults(name = "") {
  return {
    type: "value", name, nameTextStyle: { color: palette.muted },
    axisLabel: { color: palette.muted, fontSize: 11 },
    splitLine: { lineStyle: { color: palette.grid } },
  };
}

function baseChartOption() {
  return {
    animationDuration: 350,
    textStyle: { fontFamily: "Bahnschrift, Arial Narrow, sans-serif" },
    grid: { left: 58, right: 55, top: 86, bottom: 70 },
    tooltip: { trigger: "axis", confine: true, backgroundColor: "#182225", borderWidth: 0, textStyle: { color: "#fff" } },
    legend: { top: 15, left: 52, itemWidth: 18, textStyle: { color: palette.muted } },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 20, borderColor: "transparent", fillerColor: "rgba(8,127,122,.15)" }],
    xAxis: { type: "time", axisLabel: { color: palette.muted }, axisLine: { lineStyle: { color: "#aeb8b5" } } },
  };
}

function chart(id) {
  if (!state.charts[id]) state.charts[id] = echarts.init($(id), null, { renderer: "canvas" });
  return state.charts[id];
}

function pairs(rows, field) {
  return rows.map((row) => [row.ts, row[field]]).filter((item) => item[1] !== null);
}

function renderCharts() {
  const host = state.timeline.host || [];
  const platform = state.timeline.platform || [];
  const http = state.timeline.http || [];

  chart("hostChart").setOption({
    ...baseChartOption(),
    yAxis: [{ ...axisDefaults("Utilization (%)"), min: 0, max: 100, axisLabel: { formatter: "{value}%", color: palette.muted } }, axisDefaults("Load average (tasks)")],
    series: [
      { name: "CPU average (%)", type: "line", showSymbol: false, data: pairs(host, "cpu_avg"), lineStyle: { color: palette.teal, width: 2 }, areaStyle: { color: "rgba(8,127,122,.08)" } },
      { name: "CPU peak (%)", type: "line", showSymbol: false, data: pairs(host, "cpu_max"), lineStyle: { color: palette.coral, width: 1, type: "dashed" } },
      { name: "Memory used (%)", type: "line", showSymbol: false, data: pairs(host, "memory_used_pct"), lineStyle: { color: palette.blue, width: 2 } },
      { name: "Load 1m (tasks)", type: "line", yAxisIndex: 1, showSymbol: false, data: pairs(host, "load_avg"), lineStyle: { color: palette.amber, width: 1.5 } },
    ],
  }, true);

  chart("platformChart").setOption({
    ...baseChartOption(),
    yAxis: axisDefaults("Records"),
    series: [
      { name: "Pending pairs", type: "line", showSymbol: false, step: "end", data: pairs(platform, "pending_pairs"), lineStyle: { color: palette.amber, width: 2 }, areaStyle: { color: "rgba(194,129,22,.09)" } },
      { name: "Annotated pairs", type: "line", showSymbol: false, step: "end", data: pairs(platform, "annotated_pairs"), lineStyle: { color: palette.teal, width: 2 } },
      { name: "Actionable annotations", type: "line", showSymbol: false, step: "end", data: pairs(platform, "actionable_annotations"), lineStyle: { color: palette.green, width: 1.5 } },
    ],
  }, true);

  chart("httpChart").setOption({
    ...baseChartOption(),
    yAxis: [axisDefaults("Requests / bucket (count)"), { ...axisDefaults("Latency (ms)"), axisLabel: { formatter: "{value} ms", color: palette.muted } }],
    series: [
      { name: "Requests", type: "bar", data: pairs(http, "requests"), itemStyle: { color: "rgba(53,106,160,.55)" }, barMaxWidth: 12 },
      { name: "Errors", type: "bar", data: pairs(http, "errors"), itemStyle: { color: palette.coral }, barMaxWidth: 12 },
      { name: "Average latency (ms)", type: "line", yAxisIndex: 1, showSymbol: false, data: pairs(http, "latency_avg"), lineStyle: { color: palette.teal, width: 2 } },
      { name: "Maximum latency (ms)", type: "line", yAxisIndex: 1, showSymbol: false, data: pairs(http, "latency_max"), lineStyle: { color: palette.amber, width: 1, type: "dashed" } },
    ],
  }, true);
}

function renderOverview() {
  const { run, bounds, host, platform, http, issues } = state.overview;
  $("runTitle").textContent = run.run_id;
  $("runState").textContent = run.state;
  $("runState").className = `state-badge state-badge--${run.state}`;
  $("runPeriod").textContent = `${dateText(bounds.start)} — ${dateText(bounds.end)}`;
  $("kpiSpan").textContent = durationText(bounds.start, bounds.end);
  $("kpiSamples").textContent = `${number(run.samples)} declared samples`;
  $("kpiRequests").textContent = number(http.requests);
  $("kpiErrorRate").textContent = `${percent(http.error_rate, 2)} error rate`;
  $("kpiP95").textContent = milliseconds(http.latency_p95);
  $("kpiMaxLatency").textContent = `${milliseconds(http.latency_max)} maximum`;
  $("kpiIssues").textContent = number(issues.opened);
  $("kpiErrorIssues").textContent = `${number(issues.errors)} error openings`;
  $("kpiPending").textContent = number(platform.pending_pairs);
  $("kpiAnnotated").textContent = `${number(platform.annotated_pairs)} annotated pairs`;
  $("kpiCpu").textContent = `${number(host.cpu_max, 1)}%`;
  $("kpiMemory").textContent = `${number(host.memory_available_min_pct, 1)}% min memory free`;

  const flags = [];
  if (run.partial) flags.push(["Partial capture", "warning"]);
  if (!run.gpu_rows) flags.push(["GPU unavailable", "warning"]);
  if (!run.git_commit || run.git_commit === "unknown") flags.push(["Commit unknown", "warning"]);
  if (state.overview.http.errors) flags.push([`${number(state.overview.http.errors)} HTTP errors`, "error"]);
  $("qualityFlags").innerHTML = flags.map(([label, level]) => `<span class="quality-flag quality-flag--${level}">${label}</span>`).join("");

  const metadata = [
    ["Artifact", run.file_name], ["Run ID", run.run_id], ["State", run.state],
    ["Started", dateText(run.started_at)], ["Last sample", dateText(run.last_sample_at)],
    ["Interval", `${number(run.sample_interval_s, 1)} s`], ["Host rows", number(run.host_rows)],
    ["Platform rows", number(run.platform_rows)], ["HTTP events", number(run.web_rows)],
    ["Issue events", number(run.issue_rows)], ["GPU rows", number(run.gpu_rows)],
    ["Platform", run.platform || "—"], ["Python", run.python_version || "—"],
    ["Git commit", run.git_commit || "—"], ["Indexed", dateText(run.indexed_at)],
  ];
  $("metadataList").innerHTML = metadata.map(([key, value]) => `<dt>${escapeHTML(key)}</dt><dd>${escapeHTML(value)}</dd>`).join("");
}

function renderRoutes() {
  const query = $("routeFilter").value.trim().toLowerCase();
  const rows = state.routes.filter((row) => `${row.method} ${row.route}`.toLowerCase().includes(query));
  $("routeRows").innerHTML = rows.map((row) => `
    <tr>
      <td><span class="method">${escapeHTML(row.method || "—")}</span></td>
      <td><code>${escapeHTML(row.route)}</code></td>
      <td class="number">${number(row.requests)}</td>
      <td class="number ${row.errors ? "rate-high" : ""}">${number(row.errors)}</td>
      <td class="number ${row.error_rate >= .01 ? "rate-high" : ""}">${percent(row.error_rate, 2)}</td>
      <td class="number">${milliseconds(row.latency_avg)}</td>
      <td class="number">${milliseconds(row.latency_p95)}</td>
      <td class="number">${milliseconds(row.latency_max)}</td>
    </tr>`).join("");
}

function renderIssues() {
  const severity = $("issueSeverity").value;
  const rows = state.issues.filter((row) => severity === "all" || row.severity === severity);
  $("issueList").innerHTML = rows.length ? rows.map((row) => `
    <div class="issue-row">
      <span class="issue-marker issue-marker--${row.severity}"></span>
      <time>${escapeHTML(dateText(row.ts))}</time>
      <div><span class="issue-name">${escapeHTML(row.issue)}</span> <span class="issue-event">${escapeHTML(row.event.replace("issue_", ""))}</span></div>
      <span class="issue-detail">${escapeHTML(row.detail || "—")}</span>
    </div>`).join("") : `<div class="issue-row"><span></span><div>No issue events in this filter.</div></div>`;
}

function rangeQuery() {
  const params = new URLSearchParams({ points: "900" });
  if (state.rangeStart) params.set("start", String(state.rangeStart));
  if (state.rangeEnd) params.set("end", String(state.rangeEnd));
  return params.toString();
}

async function loadWindow() {
  if (!state.runId) return;
  const query = rangeQuery();
  const routeParams = new URLSearchParams();
  if (state.rangeStart) routeParams.set("start", String(state.rangeStart));
  if (state.rangeEnd) routeParams.set("end", String(state.rangeEnd));
  [state.timeline, state.routes] = await Promise.all([
    fetchJSON(`/api/runs/${state.runId}/timeline?${query}`),
    fetchJSON(`/api/runs/${state.runId}/routes?${routeParams}`),
  ]);
  renderCharts();
  renderRoutes();
}

async function loadRun(runId) {
  state.runId = Number(runId);
  $("loadingState").classList.remove("is-hidden");
  $("dashboard").classList.add("is-hidden");
  [state.overview, state.issues] = await Promise.all([
    fetchJSON(`/api/runs/${state.runId}/overview`),
    fetchJSON(`/api/runs/${state.runId}/issues`),
  ]);
  const { start_epoch: start, end_epoch: end } = state.overview.bounds;
  state.rangeStart = start;
  state.rangeEnd = end;
  $("rangeStart").value = localInputValue(state.overview.bounds.start);
  $("rangeEnd").value = localInputValue(state.overview.bounds.end);
  document.querySelectorAll("#rangePresets button").forEach((button) => button.classList.toggle("is-active", button.dataset.range === "all"));
  await loadWindow();
  renderOverview();
  renderIssues();
  $("loadingState").classList.add("is-hidden");
  $("dashboard").classList.remove("is-hidden");
  requestAnimationFrame(() => Object.values(state.charts).forEach((instance) => instance.resize()));
}

async function loadRuns() {
  state.runs = await fetchJSON("/api/runs");
  $("runSelect").innerHTML = state.runs.map((run) => `<option value="${run.id}">${escapeHTML(run.run_id)} · ${escapeHTML(run.state)} · ${number(run.samples)} samples</option>`).join("");
  $("indexStatus").className = "status-dot status-dot--ready";
  $("indexStatusText").textContent = `${state.runs.length} artifact${state.runs.length === 1 ? "" : "s"} indexed`;
  if (!state.runs.length) {
    $("loadingState").classList.add("is-hidden");
    $("emptyState").classList.remove("is-hidden");
    return;
  }
  await loadRun(state.runs[0].id);
}

function bindEvents() {
  $("runSelect").addEventListener("change", (event) => loadRun(event.target.value).catch(showError));
  $("routeFilter").addEventListener("input", renderRoutes);
  $("issueSeverity").addEventListener("change", renderIssues);
  $("applyRange").addEventListener("click", () => {
    state.rangeStart = $("rangeStart").value ? new Date($("rangeStart").value).getTime() / 1000 : null;
    state.rangeEnd = $("rangeEnd").value ? new Date($("rangeEnd").value).getTime() / 1000 : null;
    document.querySelectorAll("#rangePresets button").forEach((button) => button.classList.remove("is-active"));
    loadWindow().catch(showError);
  });
  $("rangePresets").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-range]");
    if (!button || !state.overview) return;
    const end = state.overview.bounds.end_epoch;
    state.rangeEnd = end;
    state.rangeStart = button.dataset.range === "all" ? state.overview.bounds.start_epoch : end - Number(button.dataset.range);
    $("rangeStart").value = localInputValue(new Date(state.rangeStart * 1000).toISOString());
    $("rangeEnd").value = localInputValue(new Date(state.rangeEnd * 1000).toISOString());
    document.querySelectorAll("#rangePresets button").forEach((item) => item.classList.toggle("is-active", item === button));
    loadWindow().catch(showError);
  });
  $("refreshButton").addEventListener("click", async () => {
    $("indexStatus").className = "status-dot status-dot--idle";
    $("indexStatusText").textContent = "Rescanning artifacts";
    await fetchJSON("/api/refresh", { method: "POST" });
    await loadRuns();
  });
  document.querySelectorAll(".export-button").forEach((button) => button.addEventListener("click", () => {
    const instance = state.charts[button.dataset.chart];
    if (!instance) return;
    const link = document.createElement("a");
    link.download = `${button.dataset.chart}-${state.overview.run.run_id}.png`;
    link.href = instance.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: "#fbfbf8" });
    link.click();
  }));
  window.addEventListener("resize", () => Object.values(state.charts).forEach((instance) => instance.resize()));
}

function showError(error) {
  console.error(error);
  $("indexStatus").className = "status-dot status-dot--error";
  $("indexStatusText").textContent = error.message;
  $("loadingState").classList.add("is-hidden");
}

bindEvents();
loadRuns().catch(showError);