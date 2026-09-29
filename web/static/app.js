/**
 * CottonGuard AI - Farmer Web Dashboard Client Logic
 * Handles camera capture, uploads, samples, edge inference, GenAI advisory, TTS voice, and history.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const tabBtnCamera = document.getElementById("tabBtnCamera");
  const tabBtnUpload = document.getElementById("tabBtnUpload");
  const tabBtnSamples = document.getElementById("tabBtnSamples");
  const tabCamera = document.getElementById("tab-camera");
  const tabUpload = document.getElementById("tab-upload");
  const tabSamples = document.getElementById("tab-samples");

  const btnCapture = document.getElementById("btnCapture");
  const fileInput = document.getElementById("fileInput");
  const dropzone = document.getElementById("dropzone");
  const samplesContainer = document.getElementById("samplesContainer");

  const activeLeafImg = document.getElementById("activeLeafImg");
  const scanLaser = document.getElementById("scanLaser");
  const previewDims = document.getElementById("previewDims");

  const diagnosisCard = document.getElementById("diagnosisCard");
  const diagnosisTitle = document.getElementById("diagnosisTitle");
  const pathogenText = document.getElementById("pathogenText");
  const confidenceVal = document.getElementById("confidenceVal");
  const confidenceGauge = document.getElementById("confidenceGauge");
  const latencyVal = document.getElementById("latencyVal");
  const severityVal = document.getElementById("severityVal");
  const advisorySource = document.getElementById("advisorySource");
  const probBarsContainer = document.getElementById("probBarsContainer");

  // Validation Alert & Dual-AI Elements
  const validationAlertCard = document.getElementById("validationAlertCard");
  const alertIcon = document.getElementById("alertIcon");
  const alertTitle = document.getElementById("alertTitle");
  const alertMessage = document.getElementById("alertMessage");
  const alertDetected = document.getElementById("alertDetected");
  const alertTier = document.getElementById("alertTier");
  const alertDetails = document.getElementById("alertDetails");
  const dualAiBadge = document.getElementById("dualAiBadge");
  const dualAiText = document.getElementById("dualAiText");
  const dualAiTier = document.getElementById("dualAiTier");
  const genaiInsightBox = document.getElementById("genaiInsightBox");
  const genaiInsightText = document.getElementById("genaiInsightText");
  const advisorySection = document.querySelector(".advisory-section");

  // Telemetry & Gen AI Key Elements
  const genaiBadge = document.getElementById("genaiBadge");
  const genaiDot = document.getElementById("genaiDot");
  const btnOpenKeyModal = document.getElementById("btnOpenKeyModal");
  const keyModalOverlay = document.getElementById("keyModalOverlay");
  const btnCloseKeyModal = document.getElementById("btnCloseKeyModal");
  const btnCancelKey = document.getElementById("btnCancelKey");
  const btnSaveKey = document.getElementById("btnSaveKey");
  const geminiKeyInput = document.getElementById("geminiKeyInput");
  const keyStatusMsg = document.getElementById("keyStatusMsg");

  const langSelect = document.getElementById("langSelect");
  const advImmediate = document.getElementById("advImmediate");
  const advOrganic = document.getElementById("advOrganic");
  const advChemical = document.getElementById("advChemical");
  const advPrevention = document.getElementById("advPrevention");
  const btnAudioSpeak = document.getElementById("btnAudioSpeak");
  const btnAudioText = document.getElementById("btnAudioText");
  const btnPrintReport = document.getElementById("btnPrintReport");

  const historyGrid = document.getElementById("historyGrid");
  const historyCount = document.getElementById("historyCount");
  const emptyHistoryMsg = document.getElementById("emptyHistoryMsg");
  const btnClearHistory = document.getElementById("btnClearHistory");

  const deviceBadge = document.getElementById("deviceBadge");
  const cameraBadge = document.getElementById("cameraBadge");
  const modelBadge = document.getElementById("modelBadge");
  const camSensorName = document.getElementById("camSensorName");
  const btnRefreshStatus = document.getElementById("btnRefreshStatus");

  // State
  let currentDiagnosis = null;
  let currentAdvisory = null;
  let isSpeaking = false;
  let speechUtterance = null;

  // Initialize
  initTabs();
  initDropzone();
  initKeyModal();
  loadTelemetry();
  loadSamples();
  loadHistory();


  // ============================================================================
  // TAB NAVIGATION
  // ============================================================================
  function initTabs() {
    const tabs = [
      { btn: tabBtnCamera, content: tabCamera },
      { btn: tabBtnUpload, content: tabUpload },
      { btn: tabBtnSamples, content: tabSamples },
    ];

    tabs.forEach(({ btn, content }) => {
      btn.addEventListener("click", () => {
        tabs.forEach(t => {
          t.btn.classList.remove("active");
          t.content.classList.add("hidden");
        });
        btn.classList.add("active");
        content.classList.remove("hidden");
      });
    });
  }

  // ============================================================================
  // TELEMETRY & HARDWARE STATUS
  // ============================================================================
  async function loadTelemetry() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (data.status === "success" && data.telemetry) {
        const t = data.telemetry;
        deviceBadge.textContent = `${t.device} (${t.arch})`;
        cameraBadge.textContent = t.is_hardware_camera ? `${t.camera_backend.toUpperCase()} Ready` : "Simulation/Demo";
        camSensorName.textContent = t.is_hardware_camera 
          ? `Hardware Sensor Active (${t.camera_backend.toUpperCase()})` 
          : "Pi Camera Module 3 Simulation Ready";
        modelBadge.textContent = t.model_loaded.split("(")[0].trim();

        if (genaiBadge && genaiDot) {
          if (t.genai_active) {
            genaiBadge.textContent = "Gemini Active";
            genaiDot.className = "status-dot purple pulse";
            if (btnOpenKeyModal) {
              btnOpenKeyModal.textContent = "🔑 Key Active";
              btnOpenKeyModal.classList.add("active");
            }
          } else {
            genaiBadge.textContent = "Offline Guard";
            genaiDot.className = "status-dot orange";
            if (btnOpenKeyModal) {
              btnOpenKeyModal.textContent = "🔑 Set Key";
              btnOpenKeyModal.classList.remove("active");
            }
          }
        }
      }
    } catch (err) {
      console.warn("Failed to load telemetry:", err);
    }
  }

  btnRefreshStatus.addEventListener("click", () => {
    btnRefreshStatus.style.transform = "rotate(360deg)";
    loadTelemetry().finally(() => {
      setTimeout(() => { btnRefreshStatus.style.transform = "none"; }, 500);
    });
  });

  // ============================================================================
  // GEMINI API KEY MODAL
  // ============================================================================
  function initKeyModal() {
    if (!btnOpenKeyModal || !keyModalOverlay) return;

    btnOpenKeyModal.addEventListener("click", () => {
      keyModalOverlay.classList.remove("hidden");
      keyStatusMsg.className = "key-status-msg hidden";
      geminiKeyInput.value = "";
      geminiKeyInput.focus();
    });

    const closeModal = () => {
      keyModalOverlay.classList.add("hidden");
      geminiKeyInput.value = "";
    };

    if (btnCloseKeyModal) btnCloseKeyModal.addEventListener("click", closeModal);
    if (btnCancelKey) btnCancelKey.addEventListener("click", closeModal);
    keyModalOverlay.addEventListener("click", (e) => {
      if (e.target === keyModalOverlay) closeModal();
    });

    if (btnSaveKey) {
      btnSaveKey.addEventListener("click", async () => {
        const key = geminiKeyInput.value.trim();
        if (!key) {
          keyStatusMsg.className = "key-status-msg error";
          keyStatusMsg.textContent = "Please enter a valid Gemini API key (starts with AIzaSy...).";
          keyStatusMsg.classList.remove("hidden");
          return;
        }

        btnSaveKey.disabled = true;
        btnSaveKey.textContent = "Activating Gen AI...";

        try {
          const res = await fetch("/api/config/gemini_key", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: key }),
          });
          const data = await res.json();
          if (data.status === "success") {
            keyStatusMsg.className = "key-status-msg success";
            keyStatusMsg.textContent = "✅ " + data.message;
            keyStatusMsg.classList.remove("hidden");
            loadTelemetry();
            setTimeout(closeModal, 1400);
          } else {
            keyStatusMsg.className = "key-status-msg error";
            keyStatusMsg.textContent = data.message || "Failed to activate API key.";
            keyStatusMsg.classList.remove("hidden");
          }
        } catch (err) {
          keyStatusMsg.className = "key-status-msg error";
          keyStatusMsg.textContent = "Network error: " + err.message;
          keyStatusMsg.classList.remove("hidden");
        } finally {
          btnSaveKey.disabled = false;
          btnSaveKey.textContent = "Activate Gen AI Engine";
        }
      });
    }
  }


  // ============================================================================
  // DEMO SAMPLES LOADER
  // ============================================================================
  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      if (data.status === "success" && data.samples) {
        samplesContainer.innerHTML = "";
        data.samples.forEach(sample => {
          const card = document.createElement("div");
          card.className = "sample-card";
          card.dataset.id = sample.id;
          card.innerHTML = `
            <img src="${sample.url}" alt="${sample.name}" class="sample-thumb">
            <span class="sample-name" title="${sample.name}">${sample.name}</span>
          `;
          card.addEventListener("click", () => runSampleDiagnosis(sample));
          samplesContainer.appendChild(card);
        });

        // Run default diagnosis on initial sample
        if (data.samples.length > 0) {
          runSampleDiagnosis(data.samples[0], false);
        }
      }
    } catch (err) {
      console.error("Failed to load samples:", err);
    }
  }

  async function runSampleDiagnosis(sample, triggerScan = true) {
    document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
    const activeCard = document.querySelector(`.sample-card[data-id="${sample.id}"]`);
    if (activeCard) activeCard.classList.add("active");

    activeLeafImg.src = sample.url;
    previewDims.textContent = sample.name;

    if (triggerScan) {
      startScanLaser();
    }

    try {
      const formData = new FormData();
      formData.append("sample_id", sample.id);
      formData.append("language", langSelect.value);

      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.status === "success" || data.status === "rejected") {
        renderResults(data);
      } else {
        alert("Sample analysis failed: " + (data.message || "Unknown error"));
      }
    } catch (err) {
      alert("Sample analysis failed: " + err.message);
    } finally {
      stopScanLaser();
    }
  }

  // ============================================================================
  // CAMERA CAPTURE
  // ============================================================================
  btnCapture.addEventListener("click", async () => {
    btnCapture.disabled = true;
    btnCapture.innerHTML = `<span>Focusing & Capturing...</span>`;
    startScanLaser();

    try {
      const res = await fetch("/api/capture", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language: langSelect.value }),
      });
      const data = await res.json();
      if (data.status === "success" || data.status === "rejected") {
        activeLeafImg.src = data.image_url;
        if (data.capture_meta) {
          previewDims.textContent = `Captured via ${data.capture_meta.backend_used} (${data.capture_meta.capture_latency_ms} ms)`;
        }
        renderResults(data);
        if (data.status === "success") {
          loadHistory();
        }
      } else {
        alert("Camera error: " + (data.message || "Failed to capture"));
      }
    } catch (err) {
      alert("Capture request failed: " + err.message);
    } finally {
      btnCapture.disabled = false;
      btnCapture.innerHTML = `
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <circle cx="12" cy="12" r="4"></circle>
        </svg>
        <span>Capture & Diagnose Leaf</span>
      `;
      stopScanLaser();
    }
  });

  // ============================================================================
  // FILE UPLOAD DROPZONE
  // ============================================================================
  function initDropzone() {
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files.length > 0) {
        handleFileUpload(fileInput.files[0]);
      }
    });
  }

  async function handleFileUpload(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      activeLeafImg.src = e.target.result;
      previewDims.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
    };
    reader.readAsDataURL(file);

    startScanLaser();
    const formData = new FormData();
    formData.append("file", file);
    formData.append("language", langSelect.value);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.status === "success" || data.status === "rejected") {
        renderResults(data);
        if (data.status === "success") {
          loadHistory();
        }
      } else {
        alert("Upload error: " + (data.message || "Failed to analyze"));
      }
    } catch (err) {
      alert("Prediction error: " + err.message);
    } finally {
      stopScanLaser();
    }
  }

  // ============================================================================
  // SCANNING LASER ANIMATION
  // ============================================================================
  function startScanLaser() {
    scanLaser.classList.remove("hidden");
  }

  function stopScanLaser() {
    scanLaser.classList.add("hidden");
  }

  // ============================================================================
  // RENDER DIAGNOSIS & ADVISORY
  // ============================================================================
  function renderResults(data) {
    if (data.status === "rejected") {
      // Out-of-Distribution rejection: No leaf found OR Not a cotton leaf
      currentDiagnosis = null;
      currentAdvisory = null;

      if (validationAlertCard) validationAlertCard.classList.remove("hidden");
      if (diagnosisCard) diagnosisCard.classList.add("hidden");
      if (advisorySection) advisorySection.classList.add("hidden");

      if (alertIcon) alertIcon.textContent = data.validation_status === "NO_LEAF_FOUND" ? "🚫" : "🍃";
      if (alertTitle) alertTitle.textContent = data.title || "No Leaf Detected";
      if (alertMessage) alertMessage.textContent = data.message || "Please point the camera directly at a plant leaf.";
      if (alertDetected) alertDetected.textContent = data.plant_detected || "Non-Plant Surface";
      if (alertTier) alertTier.textContent = data.tier_used || "Dual-Tier Botanical Guard";
      if (alertDetails) alertDetails.textContent = data.visual_details || data.genai_assessment || "Image lacks characteristic cotton crop foliage traits.";

      stopAudio();
      return;
    }

    // Success: Valid Cotton Leaf Confirmed
    if (validationAlertCard) validationAlertCard.classList.add("hidden");
    if (diagnosisCard) diagnosisCard.classList.remove("hidden");
    if (advisorySection) advisorySection.classList.remove("hidden");

    currentDiagnosis = data.prediction;
    currentAdvisory = data.advisory;

    // Dual-AI consensus badge
    if (data.validation) {
      if (dualAiText) {
        dualAiText.textContent = data.validation.plant_detected 
          ? `${data.validation.plant_detected} Confirmed` 
          : "Cotton Leaf Verified (Gossypium hirsutum)";
      }
      if (dualAiTier) {
        dualAiTier.textContent = data.validation.tier_used 
          ? data.validation.tier_used.split("(")[0].trim() 
          : "Dual-AI Consensus";
      }
      if (genaiInsightText) {
        genaiInsightText.textContent = data.validation.genai_assessment || data.validation.visual_details || "Palmate lobing, venation, and spectral chlorophyll match cotton crop foliage.";
      }
    }

    // Header & Titles
    diagnosisTitle.textContent = currentAdvisory.disease_name || currentDiagnosis.predicted_label.replace(/_/g, " ");
    pathogenText.textContent = currentAdvisory.pathogen || "Plant Pathogen Assessment";

    // Confidence
    const conf = currentDiagnosis.confidence;
    confidenceVal.textContent = `${conf.toFixed(1)}%`;
    confidenceGauge.style.setProperty("--conf-pct", conf);

    // Metrics
    latencyVal.textContent = `${currentDiagnosis.inference_time_ms.toFixed(1)} ms`;
    severityVal.textContent = currentAdvisory.severity || "Standard";
    advisorySource.textContent = currentAdvisory.source.includes("Gemini") ? "Gemini Multimodal" : "Dual-AI Consensus";

    // Probability Bars
    renderProbabilityBars(currentDiagnosis.top_predictions);

    // Advisory Content Cards
    advImmediate.textContent = currentAdvisory.immediate_action || "Continue monitoring crop.";
    advOrganic.textContent = currentAdvisory.organic_remedy || "Apply neem oil formulation.";
    advChemical.textContent = currentAdvisory.chemical_control || "Follow standard dosage.";
    advPrevention.textContent = currentAdvisory.preventive_measures || "Practice crop rotation.";

    // Stop ongoing speech
    stopAudio();
  }


  function renderProbabilityBars(topPreds) {
    probBarsContainer.innerHTML = "";
    if (!topPreds || !topPreds.length) return;

    topPreds.forEach((item, index) => {
      const isWinner = index === 0;
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <span class="prob-label">${item.label.replace(/_/g, " ")}</span>
        <div class="prob-track">
          <div class="prob-fill ${isWinner ? 'winner' : ''}" style="width: ${item.confidence}%"></div>
        </div>
        <span class="prob-value">${item.confidence.toFixed(1)}%</span>
      `;
      probBarsContainer.appendChild(row);
    });
  }

  // ============================================================================
  // LANGUAGE SWITCHER
  // ============================================================================
  langSelect.addEventListener("change", async () => {
    if (!currentDiagnosis) return;

    try {
      const res = await fetch("/api/advisory", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          disease: currentDiagnosis.predicted_label,
          confidence: currentDiagnosis.confidence,
          language: langSelect.value,
        }),
      });
      const data = await res.json();
      if (data.status === "success" && data.advisory) {
        currentAdvisory = data.advisory;
        diagnosisTitle.textContent = currentAdvisory.disease_name;
        pathogenText.textContent = currentAdvisory.pathogen;
        severityVal.textContent = currentAdvisory.severity;
        advImmediate.textContent = currentAdvisory.immediate_action;
        advOrganic.textContent = currentAdvisory.organic_remedy;
        advChemical.textContent = currentAdvisory.chemical_control;
        advPrevention.textContent = currentAdvisory.preventive_measures;
        stopAudio();
      }
    } catch (err) {
      console.error("Language translation failed:", err);
    }
  });

  // ============================================================================
  // VOICE READ-ALOUD (SPEECH SYNTHESIS)
  // ============================================================================
  btnAudioSpeak.addEventListener("click", () => {
    if (isSpeaking) {
      stopAudio();
    } else {
      playAudio();
    }
  });

  function playAudio() {
    if (!currentAdvisory || !('speechSynthesis' in window)) {
      alert("Voice speech is not supported in this browser.");
      return;
    }

    const textToSpeak = currentAdvisory.voice_summary || 
      `${currentAdvisory.disease_name}. ${currentAdvisory.immediate_action}`;

    speechUtterance = new SpeechSynthesisUtterance(textToSpeak);
    
    // Set appropriate language code
    const langMap = {
      en: "en-US",
      hi: "hi-IN",
      mr: "mr-IN",
      te: "te-IN",
      gu: "gu-IN",
    };
    speechUtterance.lang = langMap[langSelect.value] || "en-US";
    speechUtterance.rate = 0.95;

    speechUtterance.onstart = () => {
      isSpeaking = true;
      btnAudioText.textContent = "Stop Audio";
      btnAudioSpeak.classList.add("btn-primary");
      btnAudioSpeak.classList.remove("btn-secondary");
    };

    speechUtterance.onend = () => {
      stopAudio();
    };

    speechUtterance.onerror = () => {
      stopAudio();
    };

    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(speechUtterance);
  }

  function stopAudio() {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    isSpeaking = false;
    btnAudioText.textContent = "Read Aloud";
    btnAudioSpeak.classList.remove("btn-primary");
    btnAudioSpeak.classList.add("btn-secondary");
  }

  // ============================================================================
  // PRINT DIAGNOSIS CERTIFICATE REPORT
  // ============================================================================
  btnPrintReport.addEventListener("click", () => {
    if (!currentDiagnosis || !currentAdvisory) {
      alert("No diagnosis available to print.");
      return;
    }

    document.getElementById("printMetaDate").textContent = `Generated: ${new Date().toLocaleString()}`;
    document.getElementById("printLeafImg").src = activeLeafImg.src;
    document.getElementById("printTitle").textContent = currentAdvisory.disease_name;
    document.getElementById("printPathogen").textContent = currentAdvisory.pathogen;
    document.getElementById("printConfidence").textContent = `${currentDiagnosis.confidence.toFixed(1)}%`;
    document.getElementById("printSeverity").textContent = currentAdvisory.severity;

    document.getElementById("printImmediate").textContent = currentAdvisory.immediate_action;
    document.getElementById("printOrganic").textContent = currentAdvisory.organic_remedy;
    document.getElementById("printChemical").textContent = currentAdvisory.chemical_control;
    document.getElementById("printPrevention").textContent = currentAdvisory.preventive_measures;

    window.print();
  });

  // ============================================================================
  // DIAGNOSIS HISTORY LOG
  // ============================================================================
  async function loadHistory() {
    try {
      const res = await fetch("/api/history");
      const data = await res.json();
      if (data.status === "success" && data.history) {
        renderHistoryList(data.history);
      }
    } catch (err) {
      console.error("Failed to load history:", err);
    }
  }

  function renderHistoryList(history) {
    historyCount.textContent = `${history.length} Scans`;

    if (history.length === 0) {
      emptyHistoryMsg.classList.remove("hidden");
      historyGrid.innerHTML = "";
      historyGrid.appendChild(emptyHistoryMsg);
      return;
    }

    emptyHistoryMsg.classList.add("hidden");
    historyGrid.innerHTML = "";

    history.forEach(item => {
      const card = document.createElement("div");
      card.className = "history-card";
      card.innerHTML = `
        <img src="${item.image_url}" alt="${item.prediction}" class="history-thumb">
        <div class="history-meta">
          <h4>${item.prediction.replace(/_/g, " ")}</h4>
          <p>${item.timestamp} • ${item.source || 'Scan'}</p>
          <div class="history-conf">${item.confidence.toFixed(1)}% Confidence (${item.inference_time_ms.toFixed(1)} ms)</div>
        </div>
      `;
      card.addEventListener("click", () => {
        activeLeafImg.src = item.image_url;
        previewDims.textContent = `${item.prediction} (${item.timestamp})`;
        renderResults({
          prediction: {
            predicted_label: item.prediction,
            confidence: item.confidence,
            inference_time_ms: item.inference_time_ms,
            top_predictions: item.prediction ? [{ label: item.prediction, confidence: item.confidence }] : []
          },
          advisory: item.advisory
        });
      });
      historyGrid.appendChild(card);
    });
  }

  btnClearHistory.addEventListener("click", async () => {
    if (!confirm("Clear all field diagnosis history logs?")) return;
    try {
      await fetch("/api/history/clear", { method: "POST" });
      loadHistory();
    } catch (err) {
      alert("Failed to clear history: " + err.message);
    }
  });
});
