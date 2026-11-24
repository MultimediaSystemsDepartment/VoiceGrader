const byId = (id) => document.getElementById(id);
let profileDefaults = {};

function optionalNumber(id) {
  const value = byId(id).value;
  return value === "" ? undefined : Number(value);
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || `Request failed: ${response.status}`);
  return body;
}

function running(status) {
  return ["starting", "running", "stopping"].includes(status);
}

function applyProfile(profile) {
  const defaults = profileDefaults[profile];
  if (!defaults) return;
  byId("rounds").value = defaults.rounds;
  byId("samples").value = defaults.samples;
  byId("sftEpochs").value = defaults.sft_epochs;
  byId("dpoEpochs").value = defaults.dpo_epochs;
  byId("profileDescription").textContent = defaults.description;
  byId("profileDetails").textContent =
    `Batch size: ${defaults.batch_size} · Gradient accumulation: ${defaults.gradient_accumulation_steps} · ` +
    `Candidates: ${defaults.sampled_candidates} · Temperature: ${defaults.temperature}`;
}

async function loadProfiles() {
  const result = await request("/api/profiles");
  profileDefaults = result.profiles;
  const selected = document.querySelector('input[name="profile"]:checked').value;
  applyProfile(selected);
}

async function refreshStatus() {
  try {
    const state = await request("/api/status");
    const status = state.status || "idle";
    byId("statusBadge").textContent = status;
    byId("statusBadge").dataset.status = status;
    byId("stage").textContent = (state.stage || "idle").replaceAll("-", " ");
    byId("runId").textContent = state.run_id || "--";
    byId("round").textContent = state.round ? `${state.round} / ${state.rounds}` : "--";
    const current = state.progress_current || 0;
    const total = state.progress_total || 0;
    byId("progress").textContent = total ? `${current} / ${total}` : "--";
    byId("progressBar").style.width = total ? `${Math.min(100, current / total * 100)}%` : "0%";
    byId("dataset").textContent = state.dataset_ready ? "Ready" : "Not ready";
    byId("datasetNotice").textContent = state.dataset_ready
      ? ""
      : (state.dataset_error || "Start unavailable: ADMEDVOICE archive.zip is missing.");
    const gpu = state.gpu;
    byId("gpuName").textContent = gpu?.name || state.gpu_name || "Unavailable";
    byId("gpuUtil").textContent = gpu ? `${gpu.utilization_percent}%` : "--";
    byId("gpuMemory").textContent = gpu ? `${gpu.memory_used_mib} / ${gpu.memory_total_mib} MiB` : "--";
    byId("gpuTemperature").textContent = gpu ? `${gpu.temperature_c} C` : "--";
    byId("gpuPower").textContent = gpu ? `${gpu.power_w} W` : "--";
    byId("startButton").disabled = running(status) || !state.dataset_ready;
    byId("startButton").title = state.dataset_ready
      ? "Start the selected workload"
      : "ADMEDVOICE archive.zip must be mounted before a run can start";
    byId("stopButton").disabled = !running(status);
    byId("forceButton").disabled = !running(status);
  } catch (error) {
    byId("message").textContent = error.message;
  }
}

async function refreshLogs() {
  try {
    const result = await request("/api/logs?lines=300");
    const output = byId("logOutput");
    output.textContent = result.lines.length ? result.lines.join("") : "Waiting for a run.";
    if (byId("followLog").checked) output.scrollTop = output.scrollHeight;
  } catch (error) {
    byId("message").textContent = error.message;
  }
}

async function perform(path, body) {
  byId("message").textContent = "";
  try {
    await request(path, {
      method: "POST",
      headers: body ? {"Content-Type": "application/json"} : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    await refreshStatus();
    await refreshLogs();
  } catch (error) {
    byId("message").textContent = error.message;
  }
}

byId("startButton").addEventListener("click", () => {
  const profile = document.querySelector('input[name="profile"]:checked').value;
  perform("/api/start", {
    profile,
    rounds: optionalNumber("rounds"),
    samples: optionalNumber("samples"),
    sft_epochs: optionalNumber("sftEpochs"),
    dpo_epochs: optionalNumber("dpoEpochs"),
  });
});
document.querySelectorAll('input[name="profile"]').forEach((input) => {
  input.addEventListener("change", () => applyProfile(input.value));
});
byId("stopButton").addEventListener("click", () => perform("/api/stop"));
byId("forceButton").addEventListener("click", () => {
  if (window.confirm("Force-stop the active training process? Partial artifacts will remain.")) {
    perform("/api/force-stop");
  }
});

loadProfiles().catch((error) => { byId("message").textContent = error.message; });
refreshStatus();
refreshLogs();
setInterval(refreshStatus, 2000);
setInterval(refreshLogs, 2000);
