// RT-DETR Interactive Studio Client Engine

let currentConfig = {};
let selectedFile = null;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initVideoManager();
  initConfig();
  initGallery();
  initCalibration();
  initArtifacts();
  initInferenceLab();

  // Start continuous polling loop
  pollStateAndLogs();
  setInterval(pollStateAndLogs, 1500);
});

/* ================= TAB NAVIGATION ================= */
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(`tab-${targetId}`);
      if (targetPane) targetPane.classList.add("active");

      // Refresh data when switching tabs
      if (targetId === "gallery") initGallery();
      if (targetId === "defense") initCalibration();
      if (targetId === "artifacts") initArtifacts();
      if (targetId === "infer") loadExistingVideosForInference();
    });
  });
}

/* ================= POLLING STATE & LOGS ================= */
async function pollStateAndLogs() {
  try {
    const [stateRes, logsRes] = await Promise.all([
      fetch("/api/state"),
      fetch("/api/logs"),
    ]);

    if (stateRes.ok) {
      const state = await stateRes.json();
      updateStepper(state);
      updateHeaderStatus(state);
    }

    if (logsRes.ok) {
      const logsData = await logsRes.json();
      updateTerminal(logsData.logs || []);
    }
  } catch (err) {
    console.debug("State poll silent fail:", err);
  }
}

let previousRunningState = false;

function updateHeaderStatus(state) {
  const pill = document.getElementById("systemStatusPill");
  const text = document.getElementById("systemStatusText");
  const btnFull = document.getElementById("btnRunFullPipeline");
  const btnStop = document.getElementById("btnStopPipeline");

  if (state.is_running) {
    pill.style.background = "rgba(59, 130, 246, 0.15)";
    pill.style.borderColor = "rgba(59, 130, 246, 0.4)";
    pill.style.color = "#3B82F6";
    text.textContent = `PROCESSING: ${state.running_stage ? state.running_stage.toUpperCase() : 'STAGE'}`;

    if (btnFull) {
      btnFull.disabled = true;
      btnFull.innerHTML = `<span class="spinner" style="width: 14px; height: 14px; border-width: 2px; margin-right: 0.5rem; vertical-align: middle;"></span> Running: ${state.running_stage ? state.running_stage.toUpperCase() : 'STAGE'}...`;
    }
    if (btnStop) {
      btnStop.style.display = "inline-flex";
    }
    previousRunningState = true;
  } else {
    pill.style.background = "rgba(16, 185, 129, 0.1)";
    pill.style.borderColor = "rgba(16, 185, 129, 0.25)";
    pill.style.color = "#10B981";
    text.textContent = "ENGINE READY";

    if (btnFull) {
      btnFull.disabled = false;
      btnFull.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Full Pipeline (Auto-Resume)`;
    }
    if (btnStop) {
      btnStop.style.display = "none";
      btnStop.textContent = "Stop Pipeline";
      btnStop.disabled = false;
    }

    // Auto-refresh galleries & artifacts when pipeline transitions from running to finished
    if (previousRunningState) {
      previousRunningState = false;
      if (typeof initGallery === "function") initGallery();
      if (typeof initCalibration === "function") initCalibration();
      if (typeof initArtifacts === "function") initArtifacts();
      if (typeof loadExistingVideosForInference === "function") loadExistingVideosForInference();
    }
  }
}

function updateStepper(state) {
  const stages = state.stages || {};
  const stageKeys = ["extract", "annotate", "prepare", "train", "evaluate", "export"];

  stageKeys.forEach((key) => {
    const node = document.getElementById(`step-${key}`);
    if (!node) return;

    const data = stages[key] || {};
    const status = (data.status || "PENDING").toLowerCase();
    const duration = data.duration_sec ? `${data.duration_sec.toFixed(1)}s` : "0.0s";

    node.className = `step-node ${status}`;
    const badge = node.querySelector(".step-status-badge");
    const timeEl = node.querySelector(".step-time");

    if (badge) badge.textContent = status.toUpperCase();
    if (timeEl) timeEl.textContent = duration;
  });
}

function updateTerminal(logs) {
  const term = document.getElementById("terminalOutput");
  if (!term || !logs.length) return;

  const atBottom = term.scrollHeight - term.scrollTop <= term.clientHeight + 50;

  term.innerHTML = "";
  logs.forEach((l) => {
    const div = document.createElement("div");
    div.className = "terminal-line";
    if (l.includes("[ERROR") || l.includes("Failed")) div.classList.add("error");
    else if (l.includes("Complete") || l.includes("success") || l.includes("passed")) div.classList.add("success");
    else if (l.includes(">>>") || l.includes("Initializing")) div.classList.add("info");
    div.textContent = l;
    term.appendChild(div);
  });

  if (atBottom) {
    term.scrollTop = term.scrollHeight;
  }
}

function clearLogs() {
  const term = document.getElementById("terminalOutput");
  if (term) term.innerHTML = "";
}

/* ================= STAGE ACTIONS ================= */
async function triggerStage(stageName, payload = {}) {
  try {
    const res = await fetch(`/api/stage/run/${stageName}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      alert(`Could not start stage: ${err.detail || "Server error"}`);
    }
  } catch (e) {
    alert("Network error starting stage: " + e.message);
  }
}

async function resetStage(stageName) {
  const stageLabels = {
    extract: "Stage 1 (Frame Extraction)",
    annotate: "Stage 2 (Annotation & Auto-Labeling)",
    prepare: "Stage 3 (Dataset Preparation)",
    train: "Stage 4 (Model Training)",
    evaluate: "Stage 5 (Calibration & Evaluation)",
    export: "Stage 6 (Model Export)"
  };
  const label = stageLabels[stageName] || `Stage '${stageName}'`;
  if (!confirm(`⚠️ RESET CONFIRMATION:\n\nAre you sure you want to reset ${label} and all downstream stages?\n\nThis will permanently delete all generated frames, annotations, datasets, weights, and models for these stages from disk.`)) {
    return;
  }
  try {
    const res = await fetch(`/api/stage/reset/${stageName}`, { method: "POST" });
    if (res.ok) {
      await refreshAllViewsAfterReset();
      alert(`✅ ${label} and all downstream data have been cleared successfully.`);
    } else {
      const err = await res.json();
      alert(`Failed to reset stage: ${err.detail || "Unknown error"}`);
    }
  } catch (e) {
    alert("Network error resetting stage: " + e.message);
  }
}

async function resetEntirePipeline() {
  if (!confirm("⚠️ DANGER: RESET ENTIRE PIPELINE\n\nAre you sure you want to reset all 6 stages and wipe ALL generated workspace data?\n\nThis will permanently delete all extracted frames, annotations, datasets, trained weights, calibration curves, and exported models.")) {
    return;
  }
  try {
    const res = await fetch("/api/pipeline/reset", { method: "POST" });
    if (res.ok) {
      await refreshAllViewsAfterReset();
      alert("✅ Entire pipeline reset. All generated workspace data has been cleared.");
    } else {
      const err = await res.json();
      alert(`Failed to reset pipeline: ${err.detail || "Unknown error"}`);
    }
  } catch (e) {
    alert("Network error resetting pipeline: " + e.message);
  }
}

async function refreshAllViewsAfterReset() {
  await pollStateAndLogs();
  if (typeof initConfig === "function") await initConfig();
  if (typeof initGallery === "function") initGallery();
  if (typeof initCalibration === "function") initCalibration();
  if (typeof initArtifacts === "function") initArtifacts();
  if (typeof loadExistingVideosForInference === "function") loadExistingVideosForInference();
  if (typeof resetInferPreview === "function") resetInferPreview();
}

async function startFullPipeline() {
  const btn = document.getElementById("btnRunFullPipeline");
  try {
    const stateRes = await fetch("/api/state");
    const state = await stateRes.json();
    const stageKeys = ["extract", "annotate", "prepare", "train", "evaluate", "export"];
    let nextStage = stageKeys.find((k) => state.stages[k]?.status !== "completed");
    let force = false;

    if (!nextStage) {
      if (confirm("All 6 stages are already completed. Do you want to re-run the entire pipeline from Stage 1 (Extract)?")) {
        nextStage = "extract";
        force = true;
      } else {
        return;
      }
    }

    if (btn) {
      btn.disabled = true;
      btn.innerHTML = `<span class="spinner" style="width: 14px; height: 14px; border-width: 2px; margin-right: 0.5rem; vertical-align: middle;"></span> Starting Automated Pipeline...`;
    }

    const res = await fetch("/api/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ from_stage: nextStage, force: force }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Could not start full pipeline: ${err.detail || "Server error"}`);
    }
  } catch (e) {
    alert("Network error starting full pipeline: " + e.message);
  }
}

async function stopPipeline() {
  const btnStop = document.getElementById("btnStopPipeline");
  if (btnStop) {
    btnStop.textContent = "Stopping...";
    btnStop.disabled = true;
  }
  try {
    const res = await fetch("/api/pipeline/stop", { method: "POST" });
    if (!res.ok) {
      const err = await res.json();
      alert(`Could not stop pipeline: ${err.detail || "Server error"}`);
    }
  } catch (e) {
    alert("Network error requesting pipeline stop: " + e.message);
  }
}

document.getElementById("btnRunFullPipeline")?.addEventListener("click", startFullPipeline);
document.getElementById("btnStopPipeline")?.addEventListener("click", stopPipeline);

/* ================= CONFIGURATION ================= */
async function initConfig() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) return;
    currentConfig = await res.json();

    const ext = currentConfig.extraction || {};
    const blur = ext.blur_filter || {};
    const dedup = ext.deduplication || {};
    const ds = currentConfig.dataset || {};
    const fp = ds.false_positive_mitigation || {};
    const trn = currentConfig.training || {};
    const ev = currentConfig.evaluation || {};
    const ann = currentConfig.annotation || {};

    const primaryClass = (ann.class_names && ann.class_names.length) ? ann.class_names[0] : "scooter";
    setValue("cfg_class_name", primaryClass);

    setValue("cfg_sample_fps", ext.sample_fps || 2.0);
    setValue("cfg_blur_var", blur.min_laplacian_variance || 80.0);
    setValue("cfg_dedup_thresh", dedup.similarity_threshold || 0.96);
    setValue("cfg_bg_ratio", fp.target_background_ratio || 0.15);

    setValue("cfg_arch", trn.model_architecture || "rtdetr-l.pt");
    setValue("cfg_imgsz", trn.imgsz || 640);
    setValue("cfg_epochs", trn.epochs || 50);
    setValue("cfg_batch_size", trn.batch_size || 4);

    setValue("cfg_precision", ev.target_precision || 0.99);
    setValue("cfg_min_area", ev.min_box_area || 100);
  } catch (e) {
    console.error("Failed to load config:", e);
  }
}

function setValue(id, val) {
  const el = document.getElementById(id);
  if (el) el.value = val;
}

function getValue(id, defaultVal) {
  const el = document.getElementById(id);
  return el ? el.value : defaultVal;
}

async function saveConfig() {
  if (!currentConfig.extraction) currentConfig.extraction = {};
  if (!currentConfig.extraction.blur_filter) currentConfig.extraction.blur_filter = {};
  if (!currentConfig.extraction.deduplication) currentConfig.extraction.deduplication = {};
  if (!currentConfig.dataset) currentConfig.dataset = {};
  if (!currentConfig.dataset.false_positive_mitigation) currentConfig.dataset.false_positive_mitigation = {};
  if (!currentConfig.training) currentConfig.training = {};
  if (!currentConfig.evaluation) currentConfig.evaluation = {};
  if (!currentConfig.annotation) currentConfig.annotation = {};

  const cName = getValue("cfg_class_name", "scooter").trim();
  currentConfig.annotation.class_names = [cName];

  currentConfig.extraction.sample_fps = parseFloat(getValue("cfg_sample_fps", 2.0));
  currentConfig.extraction.blur_filter.min_laplacian_variance = parseFloat(getValue("cfg_blur_var", 80.0));
  currentConfig.extraction.deduplication.similarity_threshold = parseFloat(getValue("cfg_dedup_thresh", 0.96));
  currentConfig.dataset.false_positive_mitigation.target_background_ratio = parseFloat(getValue("cfg_bg_ratio", 0.15));

  currentConfig.training.model_architecture = getValue("cfg_arch", "rtdetr-l.pt");
  currentConfig.training.imgsz = parseInt(getValue("cfg_imgsz", 640));
  currentConfig.training.epochs = parseInt(getValue("cfg_epochs", 50));
  currentConfig.training.batch_size = parseInt(getValue("cfg_batch_size", 4));

  currentConfig.evaluation.target_precision = parseFloat(getValue("cfg_precision", 0.99));
  currentConfig.evaluation.min_box_area = parseFloat(getValue("cfg_min_area", 100));

  const btn = document.getElementById("btnSaveConfig");
  if (btn) btn.textContent = "Saving...";

  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: currentConfig }),
    });
    if (res.ok) {
      if (btn) btn.textContent = "Saved Successfully!";
      setTimeout(() => { if (btn) btn.textContent = "Save Changes to config.yaml"; }, 2000);
    }
  } catch (e) {
    alert("Failed to save config: " + e.message);
    if (btn) btn.textContent = "Save Changes to config.yaml";
  }
}

async function resetConfigToDefault() {
  if (!confirm("Reset configuration parameters in config.yaml back to system defaults?")) return;
  const btn = document.getElementById("btnResetConfig");
  if (btn) btn.textContent = "Resetting...";
  try {
    const res = await fetch("/api/config/reset", { method: "POST" });
    if (res.ok) {
      await initConfig();
      alert("✅ config.yaml has been restored to default template.");
    } else {
      alert("Failed to reset configuration.");
    }
  } catch (e) {
    alert("Network error resetting config: " + e.message);
  } finally {
    if (btn) btn.textContent = "Reset to Defaults";
  }
}

/* ================= DATASET & INSPECTION GALLERY ================= */
async function initGallery() {
  // 1. Inspection samples
  try {
    const res = await fetch("/api/gallery/inspection");
    const data = await res.json();
    const g = document.getElementById("inspectionGallery");
    if (g) {
      g.innerHTML = "";
      if (data.images && data.images.length > 0) {
        data.images.forEach((url) => {
          const card = document.createElement("div");
          card.className = "gallery-card";
          card.innerHTML = `
            <img class="gallery-thumb" src="${url}" alt="Inspection Sample">
            <div class="gallery-caption">${url.split("/").pop()}</div>
          `;
          card.onclick = () => window.open(url, "_blank");
          g.appendChild(card);
        });
      } else {
        g.innerHTML = `<div style="color: var(--text-muted); grid-column: 1/-1; text-align: center; padding: 2rem;">No inspection overlays found. Run Stage 2 (Annotate) to generate overlays.</div>`;
      }
    }
  } catch (e) {
    console.error("Failed loading inspection images:", e);
  }

  // 2. Negative samples
  try {
    const res2 = await fetch("/api/gallery/frames");
    const data2 = await res2.json();
    const ng = document.getElementById("negativesGallery");
    const countBadge = document.getElementById("negativeCountBadge");

    if (countBadge) {
      countBadge.textContent = `${data2.negative_count} background frame(s) harvested`;
    }

    if (ng) {
      ng.innerHTML = "";
      if (data2.negative_samples && data2.negative_samples.length > 0) {
        data2.negative_samples.forEach((url) => {
          const card = document.createElement("div");
          card.className = "gallery-card";
          card.innerHTML = `
            <img class="gallery-thumb" src="${url}" alt="Background Sample">
            <div class="gallery-caption">${url.split("/").pop()}</div>
          `;
          card.onclick = () => window.open(url, "_blank");
          ng.appendChild(card);
        });
      } else {
        ng.innerHTML = `<div style="color: var(--text-muted); grid-column: 1/-1; text-align: center; padding: 2rem;">No background scenes extracted yet. Place videos in data/negative_videos and run Stage 1.</div>`;
      }
    }
  } catch (e) {
    console.error("Failed loading negative samples:", e);
  }
}

/* ================= CALIBRATION & FP DEFENSE ================= */
async function initCalibration() {
  try {
    const res = await fetch("/api/calibration");
    const data = await res.json();

    const curveImg = document.getElementById("calibrationCurveImg");
    const noCurveMsg = document.getElementById("noCurveMessage");
    const tbody = document.getElementById("calibrationTableBody");
    const pill = document.getElementById("globalThresholdPill");

    // Curve image
    if (data.curve_image) {
      curveImg.src = `${data.curve_image}?t=${new Date().getTime()}`;
      curveImg.style.display = "block";
      noCurveMsg.style.display = "none";
    } else {
      curveImg.style.display = "none";
      noCurveMsg.style.display = "block";
    }

    // Threshold breakdown
    const calib = data.calibrated_thresholds || {};
    if (pill) {
      if (calib.global_calibrated_threshold) {
        pill.textContent = `Global τ* = ${calib.global_calibrated_threshold}`;
      } else {
        pill.textContent = "Global τ* = Not Calibrated";
      }
    }

    if (calib.class_thresholds && Object.keys(calib.class_thresholds).length > 0) {
      tbody.innerHTML = "";
      for (const [cname, info] of Object.entries(calib.class_thresholds)) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td style="font-weight: 700; color: #fff;">${cname}</td>
          <td><span style="font-family: var(--font-mono); font-weight: 700; color: var(--accent-cyan);">≥ ${info.calibrated_conf}</span></td>
          <td style="color: var(--accent-green); font-weight: 700;">${(info.precision * 100).toFixed(1)}%</td>
          <td><span style="background: rgba(16,185,129,0.15); color: var(--accent-green); padding: 0.15rem 0.5rem; border-radius: 4px; font-weight: 700;">${info.false_positives} FP</span></td>
        `;
        tbody.appendChild(tr);
      }
    } else {
      tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted); padding: 2rem;">Run Stage 5 (Calibrate) to compute optimal operating profile.</td></tr>`;
    }
  } catch (e) {
    console.error("Failed loading calibration:", e);
  }
}

/* ================= ARTIFACTS LIST ================= */
async function initArtifacts() {
  try {
    const res = await fetch("/api/artifacts");
    const data = await res.json();
    const tbody = document.getElementById("artifactsTableBody");
    if (!tbody) return;

    tbody.innerHTML = "";
    if (data.artifacts && data.artifacts.length > 0) {
      data.artifacts.forEach((art) => {
        const ext = art.name.split(".").pop().toUpperCase();
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td style="font-weight: 700; color: #fff;">${art.name}</td>
          <td><span class="status-pill" style="font-size: 0.72rem; padding: 0.2rem 0.6rem;">${ext}</span></td>
          <td style="font-family: var(--font-mono);">${art.size_mb > 0 ? `${art.size_mb} MB` : 'Directory'}</td>
          <td style="text-align: right;">
            <a href="${art.url}" download class="btn btn-secondary" style="text-decoration: none; display: inline-flex;">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              Download
            </a>
          </td>
        `;
        tbody.appendChild(tr);
      });
    } else {
      tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted); padding: 2rem;">No exported artifacts yet. Run Stage 6 (Export) to build models.</td></tr>`;
    }
  } catch (e) {
    console.error("Failed loading artifacts:", e);
  }
}

/* ================= LIVE INFERENCE LAB ================= */
let inferSourceMode = "existing"; // "existing" or "upload"
let selectedExistingVideo = null;
let selectedInferenceFile = null;

function showInferNotice(msg, isError = true) {
  const box = document.getElementById("inferAlertNotice");
  const txt = document.getElementById("inferAlertText");
  if (!box || !txt) return;
  txt.textContent = msg;
  box.style.display = "flex";
  if (isError) {
    box.style.background = "rgba(239, 68, 68, 0.15)";
    box.style.borderColor = "rgba(239, 68, 68, 0.4)";
    box.style.color = "#fca5a5";
  } else {
    box.style.background = "rgba(16, 185, 129, 0.15)";
    box.style.borderColor = "rgba(16, 185, 129, 0.4)";
    box.style.color = "#86efac";
  }
}

function hideInferNotice() {
  const box = document.getElementById("inferAlertNotice");
  if (box) box.style.display = "none";
}

function switchInferSource(mode) {
  inferSourceMode = mode;
  hideInferNotice();
  const btnExist = document.getElementById("inferSourceExistingBtn");
  const btnUp = document.getElementById("inferSourceUploadBtn");
  const containerExist = document.getElementById("inferExistingVideoContainer");
  const dropZone = document.getElementById("dropZone");

  if (mode === "existing") {
    if (btnExist) {
      btnExist.classList.add("btn-primary");
      btnExist.classList.remove("btn-secondary");
    }
    if (btnUp) {
      btnUp.classList.add("btn-secondary");
      btnUp.classList.remove("btn-primary");
    }
    if (containerExist) containerExist.style.display = "block";
    if (dropZone) dropZone.style.display = "none";
    onExistingVideoChange();
  } else {
    if (btnUp) {
      btnUp.classList.add("btn-primary");
      btnUp.classList.remove("btn-secondary");
    }
    if (btnExist) {
      btnExist.classList.add("btn-secondary");
      btnExist.classList.remove("btn-primary");
    }
    if (containerExist) containerExist.style.display = "none";
    if (dropZone) dropZone.style.display = "block";
    if (selectedInferenceFile) {
      handleFileSelected(selectedInferenceFile);
    } else {
      resetInferPreview();
    }
  }
}

async function loadExistingVideosForInference() {
  try {
    const res = await fetch("/api/videos");
    if (!res.ok) return;
    const data = await res.json();
    const select = document.getElementById("inferExistingVideoSelect");
    if (!select) return;

    select.innerHTML = '<option value="">-- Choose an uploaded workspace video --</option>';
    const allVideos = [];
    (data.positive_videos || []).forEach((v) => allVideos.push({ ...v, tag: "Training Video" }));
    (data.negative_videos || []).forEach((v) => allVideos.push({ ...v, tag: "Negative Background" }));

    allVideos.forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v.filename;
      opt.textContent = `${v.filename} (${v.tag} • ${v.size_mb} MB • ${v.duration_sec}s)`;
      select.appendChild(opt);
    });

    if (allVideos.length > 0 && !selectedExistingVideo) {
      select.value = allVideos[0].filename;
      onExistingVideoChange();
    }
  } catch (e) {
    console.debug("Failed loading existing videos for inference:", e);
  }
}

function onExistingVideoChange() {
  hideInferNotice();
  const select = document.getElementById("inferExistingVideoSelect");
  if (!select) return;
  selectedExistingVideo = select.value || null;

  const placeholder = document.getElementById("inferPlaceholder");
  const resultImg = document.getElementById("inferResultImg");
  const resultVideo = document.getElementById("inferResultVideo");

  if (resultVideo) {
    resultVideo.pause();
    resultVideo.style.display = "none";
  }
  if (resultImg) resultImg.style.display = "none";

  if (selectedExistingVideo && placeholder) {
    placeholder.innerHTML = `
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom: 0.75rem; opacity: 0.9; color: var(--accent-cyan);"><rect x="2" y="2" width="20" height="20" rx="2.18"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
      <div style="font-weight: 700; color: var(--text-primary); font-size: 1.05rem;">${selectedExistingVideo}</div>
      <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.35rem;">Workspace video selected & ready for calibrated evaluation.</div>
      <div style="font-size: 0.78rem; color: var(--accent-green); margin-top: 0.35rem; font-weight: 600;">Click "Run Calibrated Detection" below to evaluate zero false positives across all frames.</div>
    `;
    placeholder.style.display = "block";
  } else if (placeholder) {
    resetInferPreview();
  }
}

function resetInferPreview() {
  const placeholder = document.getElementById("inferPlaceholder");
  const resultImg = document.getElementById("inferResultImg");
  const resultVideo = document.getElementById("inferResultVideo");
  if (resultImg) resultImg.style.display = "none";
  if (resultVideo) {
    resultVideo.pause();
    resultVideo.style.display = "none";
  }
  if (placeholder) {
    placeholder.innerHTML = `
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom: 0.75rem; opacity: 0.5;"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
      <div>Select a workspace video or upload an image/video to run calibrated detection.</div>
    `;
    placeholder.style.display = "block";
  }
}

function initInferenceLab() {
  loadExistingVideosForInference();
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("inferFileInput");

  if (!dropZone || !fileInput) return;

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      selectedInferenceFile = e.dataTransfer.files[0];
      handleFileSelected(selectedInferenceFile);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      selectedInferenceFile = e.target.files[0];
      handleFileSelected(selectedInferenceFile);
    }
  });
}

function handleFileSelected(file) {
  if (!file) return;
  hideInferNotice();
  selectedInferenceFile = file;
  const placeholder = document.getElementById("inferPlaceholder");
  const resultImg = document.getElementById("inferResultImg");
  const resultVideo = document.getElementById("inferResultVideo");

  if (resultVideo) {
    resultVideo.pause();
    resultVideo.style.display = "none";
  }

  const isVideo = file.type.startsWith("video/") || /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name);
  if (isVideo) {
    if (placeholder) {
      placeholder.innerHTML = `
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom: 0.75rem; opacity: 0.85; color: var(--accent-cyan);"><rect x="2" y="2" width="20" height="20" rx="2.18"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
        <div style="font-weight: 700; color: var(--text-primary); font-size: 1rem;">${file.name}</div>
        <div style="font-size: 0.78rem; color: var(--text-secondary); margin-top: 0.35rem;">File selected (${(file.size / (1024 * 1024)).toFixed(2)} MB).</div>
        <div style="font-size: 0.75rem; color: var(--accent-green); margin-top: 0.25rem;">Click "Run Calibrated Detection" below to analyze every frame.</div>
      `;
      placeholder.style.display = "block";
    }
    if (resultImg) resultImg.style.display = "none";
  } else {
    const reader = new FileReader();
    reader.onload = (e) => {
      if (resultImg) {
        resultImg.src = e.target.result;
        resultImg.style.display = "block";
      }
      if (placeholder) placeholder.style.display = "none";
    };
    reader.readAsDataURL(file);
  }
}

async function submitInference() {
  hideInferNotice();
  const btn = document.getElementById("btnRunInference");
  const fileInput = document.getElementById("inferFileInput");

  // If in upload mode, check file input
  if (inferSourceMode === "upload") {
    if (!selectedInferenceFile && fileInput && fileInput.files && fileInput.files.length > 0) {
      selectedInferenceFile = fileInput.files[0];
    }
  }

  const hasExisting = inferSourceMode === "existing" && Boolean(selectedExistingVideo);
  const hasFile = inferSourceMode === "upload" && Boolean(selectedInferenceFile);

  if (!hasExisting && !hasFile) {
    const msg = inferSourceMode === "existing"
      ? "Please select a workspace video from the dropdown first."
      : "Please select or drop an image or video file first.";
    showInferNotice(msg, true);
    if (inferSourceMode === "upload") {
      const dropZone = document.getElementById("dropZone");
      if (dropZone) {
        dropZone.classList.add("shake-element");
        setTimeout(() => dropZone.classList.remove("shake-element"), 500);
      }
      if (fileInput) fileInput.click();
    }
    return;
  }

  const modelChoice = document.getElementById("inferModelSelect") ? document.getElementById("inferModelSelect").value : "onnx";
  const confOverrideVal = document.getElementById("inferConfOverride") ? document.getElementById("inferConfOverride").value : "";

  let isVideo = false;
  let targetName = "";
  if (hasExisting) {
    isVideo = true;
    targetName = selectedExistingVideo;
  } else if (selectedInferenceFile) {
    isVideo = selectedInferenceFile.type.startsWith("video/") || /\.(mp4|avi|mov|mkv|webm)$/i.test(selectedInferenceFile.name);
    targetName = selectedInferenceFile.name;
  }

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner" style="width: 14px; height: 14px; border-width: 2px; margin-right: 0.5rem; vertical-align: middle;"></span> Calibrating Every Frame...`;
  }

  const placeholder = document.getElementById("inferPlaceholder");
  const resultImg = document.getElementById("inferResultImg");
  const resultVideo = document.getElementById("inferResultVideo");

  if (isVideo && placeholder) {
    placeholder.innerHTML = `
      <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2.5rem 1rem;">
        <div class="spinner" style="width: 44px; height: 44px; border-width: 4px; margin-bottom: 1.25rem;"></div>
        <div style="font-weight: 700; color: var(--text-primary); font-size: 1.1rem;">Calibrating Every Video Frame...</div>
        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.5rem; max-width: 420px; text-align: center; line-height: 1.5;">
          Running RT-DETR Zero-False-Positive model on <strong>${targetName}</strong> across all frames. Live progress streaming to terminal below.
        </div>
      </div>
    `;
    placeholder.style.display = "block";
    if (resultImg) resultImg.style.display = "none";
    if (resultVideo) resultVideo.style.display = "none";
  }

  const formData = new FormData();
  if (hasExisting) {
    formData.append("existing_video", selectedExistingVideo);
  } else {
    formData.append("file", selectedInferenceFile);
  }
  formData.append("model_choice", modelChoice);
  if (confOverrideVal) {
    formData.append("conf_override", confOverrideVal);
  }

  try {
    const res = await fetch("/api/infer", {
      method: "POST",
      body: formData,
    });

    if (res.ok) {
      const data = await res.json();
      const downloadBtn = document.getElementById("btnDownloadVideo");
      const badgeCount = document.getElementById("inferBadgeCount");
      const metricsRow = document.getElementById("inferMetricsRow");
      const framesBadge = document.getElementById("metricFramesBadge");
      const totalFramesVal = document.getElementById("metricTotalFrames");

      if (placeholder) placeholder.style.display = "none";

      if (data.is_video) {
        if (resultImg) resultImg.style.display = "none";
        if (resultVideo) {
          resultVideo.src = `${data.annotated_video_url}?t=${Date.now()}`;
          resultVideo.style.display = "block";
          resultVideo.load();
          resultVideo.play().catch(() => {});
        }

        if (downloadBtn) {
          downloadBtn.href = data.annotated_video_url;
          downloadBtn.download = data.annotated_video_url.split("/").pop();
          downloadBtn.style.display = "inline-flex";
        }

        if (framesBadge && totalFramesVal) {
          framesBadge.style.display = "flex";
          totalFramesVal.textContent = `${data.total_frames} (${data.duration_sec}s)`;
        }

        if (badgeCount) {
          badgeCount.textContent = `${data.total_frames} Frames • ${data.detections_count} Calibrated Objects`;
          badgeCount.style.display = "inline-flex";
        }
      } else {
        if (resultVideo) {
          resultVideo.pause();
          resultVideo.style.display = "none";
        }
        if (downloadBtn) downloadBtn.style.display = "none";
        if (framesBadge) framesBadge.style.display = "none";

        if (resultImg) {
          resultImg.src = `${data.annotated_image_url}?t=${Date.now()}`;
          resultImg.style.display = "block";
        }

        if (badgeCount) {
          badgeCount.textContent = `${data.detections_count} Calibrated Object(s)`;
          badgeCount.style.display = "inline-flex";
        }
      }

      if (metricsRow) metricsRow.style.display = "flex";
      const elModel = document.getElementById("metricModelUsed");
      if (elModel) elModel.textContent = data.model_used;
      const elFloor = document.getElementById("metricCalibFloor");
      if (elFloor) elFloor.textContent = `≥ ${data.calibrated_threshold}`;
      const elCount = document.getElementById("metricDetCount");
      if (elCount) elCount.textContent = data.detections_count;

      showInferNotice(`Detection complete: ${data.detections_count} total detections found with zero false positives.`, false);

    } else {
      let errMsg = "Server error";
      try {
        const err = await res.json();
        errMsg = err.detail || errMsg;
      } catch (_) {}
      showInferNotice(`Inference failed: ${errMsg}`, true);
    }
  } catch (e) {
    showInferNotice("Network error running inference: " + e.message, true);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Run Calibrated Detection";
    }
  }
}

/* ================= VIDEO INGESTION & UPLOAD ================= */
let currentVideoUploadType = "positive";

function setVideoUploadType(type) {
  currentVideoUploadType = type;
  const togglePos = document.getElementById("toggleTypePos");
  const toggleNeg = document.getElementById("toggleTypeNeg");

  if (type === "positive") {
    togglePos.classList.add("active");
    toggleNeg.classList.remove("active");
  } else {
    toggleNeg.classList.add("active");
    togglePos.classList.remove("active");
  }
}

function initVideoManager() {
  const dropZone = document.getElementById("videoDropZone");
  const fileInput = document.getElementById("videoFileInput");

  if (!dropZone || !fileInput) return;

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleVideoUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleVideoUpload(e.target.files[0]);
    }
  });

  loadVideoInventory();
}

function handleVideoUpload(file) {
  const wrap = document.getElementById("videoUploadProgressWrap");
  const nameEl = document.getElementById("videoUploadFileName");
  const percentEl = document.getElementById("videoUploadPercent");
  const barEl = document.getElementById("videoUploadProgressBar");

  wrap.style.display = "block";
  nameEl.textContent = `Uploading: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
  percentEl.textContent = "0%";
  barEl.style.width = "0%";

  const formData = new FormData();
  formData.append("file", file);
  formData.append("video_type", currentVideoUploadType);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload/video", true);

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      percentEl.textContent = `${pct}%`;
      barEl.style.width = `${pct}%`;
    }
  };

  xhr.onload = () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      percentEl.textContent = "Complete!";
      setTimeout(() => {
        wrap.style.display = "none";
        loadVideoInventory();
      }, 1200);
    } else {
      let errMsg = "Upload failed";
      try {
        const res = JSON.parse(xhr.responseText);
        errMsg = res.detail || errMsg;
      } catch (err) {}
      alert(`Video upload failed: ${errMsg}`);
      wrap.style.display = "none";
    }
  };

  xhr.onerror = () => {
    alert("Network error during video upload.");
    wrap.style.display = "none";
  };

  xhr.send(formData);
}

async function loadVideoInventory() {
  const container = document.getElementById("videoInventoryList");
  const badge = document.getElementById("videoCountBadge");
  if (!container) return;

  try {
    const res = await fetch("/api/videos");
    const data = await res.json();

    if (badge) {
      badge.textContent = `${data.total_count} video(s) ready in project`;
    }

    const allVideos = [
      ...(data.positive_videos || []),
      ...(data.negative_videos || []),
    ];

    if (allVideos.length === 0) {
      container.innerHTML = `
        <div style="color: var(--text-muted); text-align: center; padding: 2rem; font-size: 0.85rem;">
          No video files uploaded yet. Drag & drop or browse an MP4/AVI/MOV file above to get started.
        </div>
      `;
      return;
    }

    container.innerHTML = "";
    allVideos.forEach((v) => {
      const isNeg = v.type === "negative";
      const badgeClass = isNeg ? "negative" : "positive";
      const badgeText = isNeg ? "Background (0-FP)" : "Target Training";

      const div = document.createElement("div");
      div.className = "video-item-row";
      div.innerHTML = `
        <div class="video-meta-left">
          <span class="video-type-badge ${badgeClass}">${badgeText}</span>
          <div>
            <div class="video-name" title="${v.filename}">${v.filename}</div>
            <div class="video-sub-meta">${v.size_mb} MB • ${v.resolution} • ${v.fps} FPS • ${v.duration_sec}s</div>
          </div>
        </div>
        <div class="btn-group">
          <button class="btn btn-primary" style="padding: 0.35rem 0.7rem; font-size: 0.75rem;" onclick="extractVideo('${v.filename}', '${v.type}')">
            Extract Frames
          </button>
          <button class="btn btn-danger" style="padding: 0.35rem 0.6rem; font-size: 0.75rem; min-width: 65px;" onclick="handleDeleteClick(this, '${v.type}', '${encodeURIComponent(v.filename)}')">
            Delete
          </button>
        </div>
      `;
      container.appendChild(div);
    });
  } catch (e) {
    container.innerHTML = `<div style="color: var(--accent-rose); text-align: center; padding: 1.5rem;">Failed loading videos: ${e.message}</div>`;
  }
}

async function extractVideo(filename, videoType) {
  const folder = videoType === "negative" ? "data/negative_videos" : "data/videos";
  const path = `${folder}/${filename}`;
  triggerStage("extract");
}

function handleDeleteClick(btn, videoType, encodedFilename) {
  const filename = decodeURIComponent(encodedFilename);

  if (btn.dataset.confirming === "true") {
    // Second click: execute deletion!
    btn.disabled = true;
    btn.textContent = "Deleting...";

    fetch(`/api/videos/${encodeURIComponent(videoType)}/${encodeURIComponent(filename)}`, { method: "DELETE" })
      .then(async (res) => {
        if (res.ok) {
          loadVideoInventory();
        } else {
          const err = await res.json();
          alert(`Failed to delete: ${err.detail || "Server error"}`);
          btn.disabled = false;
          btn.textContent = "Delete";
          delete btn.dataset.confirming;
        }
      })
      .catch((err) => {
        alert("Network error: " + err.message);
        btn.disabled = false;
        btn.textContent = "Delete";
        delete btn.dataset.confirming;
      });
  } else {
    // First click: ask for confirmation inline (never blocked by browser popups)
    btn.dataset.confirming = "true";
    btn.textContent = "Confirm?";
    btn.style.background = "#E11D48";
    btn.style.color = "#FFF";
    btn.style.borderColor = "#F43F5E";

    // Auto-reset after 3.5s if not clicked
    setTimeout(() => {
      if (btn && btn.dataset.confirming === "true") {
        delete btn.dataset.confirming;
        btn.textContent = "Delete";
        btn.style.background = "";
        btn.style.color = "";
        btn.style.borderColor = "";
      }
    }, 3500);
  }
}

