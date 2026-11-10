/* Voice-first public chat interaction.
 *
 * One tap starts a MediaRecorder. A local Web Audio AnalyserNode watches the
 * microphone level; after speech has been detected and a sustained pause
 * follows, the client stops recording and uploads the turn automatically. A
 * visible Detener button remains available, and a hard maximum recording time
 * bounds every listening session.
 *
 * There is no wake word, no audio streaming and no dependency: the recorded
 * blob is sent once to the same-origin voice endpoint. Raw audio is kept in
 * memory just long enough to upload it; the app never stores it.
 */
(function () {
  "use strict";

  var CONSENT_KEY = "memo_voice_consent_v1";
  var FALLBACK_CONFIG = {
    maxRecordingMs: 60000,
    initialGraceMs: 2500,
    silenceDurationMs: 1600,
    silenceRmsThreshold: 0.012,
    analyserFftSize: 2048,
    pollIntervalMs: 100
  };
  var silenceApi =
    window.MemoVoiceSilence &&
    typeof window.MemoVoiceSilence.createSilenceDetector === "function"
      ? window.MemoVoiceSilence
      : null;
  var root = document.querySelector("[data-voice-endpoint]");
  if (!root) {
    return;
  }

  var endpoint = root.getAttribute("data-voice-endpoint");
  var conversation = document.getElementById("conversation");
  var liveTranscript = document.getElementById("live-transcript");
  var statusEl = document.getElementById("voice-status");
  var errorEl = document.getElementById("voice-error");
  var micButton = document.getElementById("voice-mic");
  var stopButton = document.getElementById("voice-stop");
  var replayButton = document.getElementById("voice-replay");
  var consentDialog = document.getElementById("voice-consent");
  var consentAccept = document.getElementById("voice-consent-accept");
  var consentCancel = document.getElementById("voice-consent-cancel");
  if (!endpoint || !conversation || !liveTranscript || !statusEl || !micButton) {
    return;
  }

  var config = voiceConfig();
  var mediaRecorder = null;
  var mediaStream = null;
  var audioContext = null;
  var mediaSource = null;
  var analyser = null;
  var analyserBuffer = null;
  var detector = null;
  var silenceTimer = null;
  var maxTimer = null;
  var chunks = [];
  var stopping = false;
  var manualStopOnly = false;
  var currentPhase = "idle";
  var playback = null;
  var lastAudioUrl = null;
  var busy = false;

  function voiceConfig() {
    var defaults =
      (silenceApi && silenceApi.DEFAULT_CONFIG) || FALLBACK_CONFIG;
    var overrides =
      window.MEMO_VOICE_CONFIG && typeof window.MEMO_VOICE_CONFIG === "object"
        ? window.MEMO_VOICE_CONFIG
        : {};
    var merged = Object.assign({}, defaults, overrides);
    if (silenceApi && typeof silenceApi.normalizeConfig === "function") {
      return silenceApi.normalizeConfig(merged);
    }
    return merged;
  }

  function nowMs() {
    if (
      window.performance &&
      typeof window.performance.now === "function"
    ) {
      return window.performance.now();
    }
    return Date.now();
  }

  function setStatus(message) {
    statusEl.textContent = message;
  }

  function clearError() {
    if (!errorEl) {
      return;
    }
    errorEl.textContent = "";
    errorEl.hidden = true;
  }

  function showError(message) {
    if (!errorEl) {
      return;
    }
    errorEl.textContent = message;
    errorEl.hidden = false;
    if (typeof errorEl.focus === "function") {
      errorEl.focus();
    }
  }

  function supportsRecording() {
    return Boolean(
      window.MediaRecorder &&
        navigator.mediaDevices &&
        typeof navigator.mediaDevices.getUserMedia === "function"
    );
  }

  function consentGranted() {
    try {
      return window.localStorage.getItem(CONSENT_KEY) === "granted";
    } catch (error) {
      return false;
    }
  }

  function rememberConsent() {
    try {
      window.localStorage.setItem(CONSENT_KEY, "granted");
    } catch (error) {
      /* Private mode may block storage; consent then applies per visit. */
    }
  }

  function openConsent() {
    if (!consentDialog) {
      return;
    }
    if (typeof consentDialog.showModal === "function") {
      consentDialog.showModal();
    } else {
      consentDialog.setAttribute("open", "");
    }
  }

  function closeConsent() {
    if (!consentDialog) {
      return;
    }
    if (typeof consentDialog.close === "function") {
      consentDialog.close();
    } else {
      consentDialog.removeAttribute("open");
    }
  }

  function preferredMimeType() {
    var candidates = [
      "audio/webm;codecs=opus",
      "audio/ogg;codecs=opus",
      "audio/webm",
      "audio/ogg",
      "audio/mp4",
      "audio/wav"
    ];
    if (typeof window.MediaRecorder.isTypeSupported !== "function") {
      return "";
    }
    for (var index = 0; index < candidates.length; index += 1) {
      if (window.MediaRecorder.isTypeSupported(candidates[index])) {
        return candidates[index];
      }
    }
    return "";
  }

  function setPhase(phase, message) {
    currentPhase = phase;
    root.classList.toggle("is-listening", phase === "listening");
    root.classList.toggle("is-processing", phase === "thinking");
    root.classList.toggle("is-speaking", phase === "speaking");
    root.setAttribute("data-voice-state", phase);
    micButton.setAttribute(
      "aria-pressed",
      phase === "listening" ? "true" : "false"
    );
    var label = micButton.querySelector(".js-mic-label");
    if (label) {
      label.textContent = phase === "listening" ? "Escuchando…" : "Hablar";
    }
    if (message) {
      setStatus(message);
    }
    updateStopButton();
  }

  function releaseStream() {
    if (!mediaStream) {
      return;
    }
    mediaStream.getTracks().forEach(function (track) {
      track.stop();
    });
    mediaStream = null;
  }

  function updateStopButton() {
    if (!stopButton) {
      return;
    }
    var listening =
      currentPhase === "listening" &&
      mediaRecorder !== null &&
      mediaRecorder.state === "recording";
    stopButton.hidden = !(listening || playback !== null);
  }

  function stopPlayback() {
    if (playback === null) {
      return;
    }
    try {
      playback.pause();
      playback.currentTime = 0;
    } catch (error) {
      /* Ignore a media element that has already been released. */
    }
    playback = null;
    updateStopButton();
  }

  function clearLastAudio() {
    if (lastAudioUrl) {
      window.URL.revokeObjectURL(lastAudioUrl);
      lastAudioUrl = null;
    }
    if (replayButton) {
      replayButton.hidden = true;
    }
  }

  function storeAnswerAudio(encoded, contentType) {
    clearLastAudio();
    if (!encoded) {
      return;
    }
    var binary = window.atob(encoded);
    var bytes = new Uint8Array(binary.length);
    for (var index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    var blob = new Blob([bytes], {
      type: contentType || "audio/mpeg"
    });
    lastAudioUrl = window.URL.createObjectURL(blob);
    if (replayButton) {
      replayButton.hidden = false;
    }
  }

  function playAnswer() {
    if (!lastAudioUrl) {
      return;
    }
    stopPlayback();
    playback = new Audio(lastAudioUrl);
    playback.addEventListener("ended", function () {
      playback = null;
      setPhase("idle", "Listo para escucharte.");
    });
    playback.addEventListener("error", function () {
      playback = null;
      setPhase(
        "idle",
        "La voz no está disponible. Puedes leer la respuesta."
      );
    });
    setPhase("speaking", "Memo está respondiendo…");
    var started = playback.play();
    if (started && typeof started.catch === "function") {
      started.catch(function () {
        setPhase("idle", "Pulsa Repetir para escuchar la respuesta.");
      });
    }
    updateStopButton();
  }

  function appendTurn(speaker, label, text) {
    var empty = document.getElementById("empty-state");
    if (empty) {
      empty.hidden = true;
    }
    var turn = document.createElement("div");
    turn.className = "turn " + speaker;

    var who = document.createElement("p");
    who.className = "who";
    who.textContent = label;

    var bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    turn.appendChild(who);
    turn.appendChild(bubble);
    liveTranscript.appendChild(turn);
    if (typeof turn.scrollIntoView === "function") {
      turn.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }

  function renderResult(payload) {
    var spokenText = payload.transcript || payload.text;
    if (spokenText) {
      appendTurn("user", "Tú", spokenText);
    }
    if (payload.answer) {
      appendTurn("memo", "Memo", payload.answer);
    }
    if (payload.conversation_id) {
      root.setAttribute("data-conversation-id", payload.conversation_id);
    }
    storeAnswerAudio(payload.audio, payload.audio_content_type);
    if (payload.audio) {
      playAnswer();
    } else {
      setPhase(
        "idle",
        "Respuesta lista. La voz no está disponible ahora mismo; puedes leerla."
      );
    }
  }

  function showHttpError(status) {
    if (status === 400) {
      setPhase(
        "idle",
        "No he podido entenderte. Prueba a hablar más cerca del micrófono."
      );
      showError("No se ha entendido el audio.");
      return;
    }
    if (status === 413) {
      setPhase("idle", "La grabación es demasiado larga.");
      showError(
        "La grabación es demasiado larga. Prueba con una frase más corta."
      );
      return;
    }
    if (status === 415) {
      setPhase("idle", "Formato de audio no compatible.");
      showError(
        "Este navegador usa un formato de audio no compatible. Prueba con otro navegador."
      );
      return;
    }
    if (status === 404) {
      setPhase("idle", "Conversación nueva.");
      showError(
        "No he podido recuperar esta conversación. Empezamos una conversación nueva."
      );
      return;
    }
    setPhase("idle", "No se ha podido completar la respuesta.");
    showError("Algo ha ido mal. Inténtalo de nuevo.");
  }

  function buildForm(blob, includeConversation) {
    var form = new FormData();
    var filename = "voice.webm";
    if (blob.type.indexOf("ogg") !== -1) {
      filename = "voice.ogg";
    } else if (blob.type.indexOf("mp4") !== -1) {
      filename = "voice.m4a";
    } else if (blob.type.indexOf("wav") !== -1) {
      filename = "voice.wav";
    }
    form.append("audio", blob, filename);
    var conversationId = root.getAttribute("data-conversation-id");
    if (includeConversation && conversationId) {
      form.append("conversation_id", conversationId);
    }
    return form;
  }

  function postVoice(blob, includeConversation) {
    return window.fetch(endpoint, {
      method: "POST",
      body: buildForm(blob, includeConversation),
      credentials: "same-origin",
      headers: { Accept: "application/json" }
    });
  }

  async function sendAudio(blob) {
    if (busy) {
      return;
    }
    busy = true;
    clearError();
    setPhase("thinking", "Procesando lo que has dicho…");
    try {
      var hadConversation =
        Boolean(root.getAttribute("data-conversation-id"));
      var response = await postVoice(blob, hadConversation);
      if (response.status === 404 && hadConversation) {
        /* A stale or foreign pointer must not block the turn. Retry once as a
         * fresh conversation; the backend then sets a new ownership cookie. */
        response = await postVoice(blob, false);
      }
      if (!response.ok) {
        showHttpError(response.status);
        return;
      }
      var payload = await response.json();
      renderResult(payload);
    } catch (error) {
      setPhase("idle", "Sin conexión.");
      showError("No he podido conectar con Memo. Inténtalo otra vez.");
    } finally {
      busy = false;
    }
  }

  function clearMaxTimer() {
    if (maxTimer !== null) {
      window.clearTimeout(maxTimer);
      maxTimer = null;
    }
  }

  function stopSilenceMonitor() {
    if (silenceTimer !== null) {
      window.clearInterval(silenceTimer);
      silenceTimer = null;
    }
    detector = null;
    analyser = null;
    analyserBuffer = null;
    mediaSource = null;
    if (audioContext) {
      try {
        var closed = audioContext.close();
        if (closed && typeof closed.catch === "function") {
          closed.catch(function () {
            /* The context may already be closed. */
          });
        }
      } catch (error) {
        /* Ignore a context that cannot be closed twice. */
      }
      audioContext = null;
    }
  }

  function sampleSilence() {
    if (!analyser || !detector || stopping || !silenceApi) {
      return;
    }
    try {
      var rms;
      if (typeof analyser.getFloatTimeDomainData === "function") {
        if (!analyserBuffer || analyserBuffer.length !== analyser.fftSize) {
          analyserBuffer = new Float32Array(analyser.fftSize);
        }
        analyser.getFloatTimeDomainData(analyserBuffer);
        rms = silenceApi.rmsFromFloatSamples(analyserBuffer);
      } else if (typeof analyser.getByteTimeDomainData === "function") {
        if (!analyserBuffer || analyserBuffer.length !== analyser.fftSize) {
          analyserBuffer = new Uint8Array(analyser.fftSize);
        }
        analyser.getByteTimeDomainData(analyserBuffer);
        rms = silenceApi.rmsFromByteSamples(analyserBuffer);
      } else {
        manualStopOnly = true;
        stopSilenceMonitor();
        setStatus("Te escucho. Pulsa Detener cuando acabes.");
        return;
      }

      var decision = detector.observe({ nowMs: nowMs(), rms: rms });
      if (decision.shouldStop) {
        stopRecording("silence");
      }
    } catch (error) {
      // An analyser that cannot be read must not leave the microphone
      // listening without a local stop path. Fall back to manual Detener.
      manualStopOnly = true;
      stopSilenceMonitor();
      setStatus(
        "Te escucho. Este navegador no puede detectar pausas; pulsa Detener cuando acabes."
      );
    }
  }

  function startSilenceMonitor() {
    if (!silenceApi || !mediaStream) {
      return false;
    }
    var AudioCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtor) {
      return false;
    }
    try {
      audioContext = new AudioCtor();
      mediaSource = audioContext.createMediaStreamSource(mediaStream);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = config.analyserFftSize;
      analyser.smoothingTimeConstant = 0.2;
      mediaSource.connect(analyser);
      detector = silenceApi.createSilenceDetector(config);
      detector.reset(nowMs());
      silenceTimer = window.setInterval(
        sampleSilence,
        config.pollIntervalMs
      );
      if (
        audioContext.state === "suspended" &&
        typeof audioContext.resume === "function"
      ) {
        var resumed = audioContext.resume();
        if (resumed && typeof resumed.catch === "function") {
          resumed.catch(function () {
            /* Manual Stop remains available if resume is blocked. */
          });
        }
      }
      return true;
    } catch (error) {
      stopSilenceMonitor();
      return false;
    }
  }

  function stopRecording(reason) {
    if (stopping) {
      return;
    }
    stopping = true;
    clearMaxTimer();
    stopSilenceMonitor();
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
    releaseStream();
    setPhase("thinking", "Procesando lo que has dicho…");
    if (reason === "max") {
      setStatus(
        "Se ha alcanzado el tiempo máximo de grabación. Procesando…"
      );
    } else if (reason === "silence" && !manualStopOnly) {
      setStatus("Te he escuchado. Procesando…");
    }
  }

  function onRecordingStopped() {
    var mimeType =
      (mediaRecorder && mediaRecorder.mimeType) ||
      (chunks.length > 0 ? chunks[0].type : "audio/webm");
    var blob = new Blob(chunks, { type: mimeType });
    chunks = [];
    mediaRecorder = null;
    stopping = false;
    updateStopButton();
    if (!blob.size) {
      setPhase("idle", "No se ha grabado audio. Inténtalo otra vez.");
      return;
    }
    sendAudio(blob);
  }

  async function startRecording() {
    if (busy) {
      return;
    }
    if (!consentGranted()) {
      openConsent();
      return;
    }
    if (!supportsRecording()) {
      showError(
        "Este navegador no puede grabar audio. Puedes escribir tu mensaje en la opción de abajo."
      );
      setPhase("idle", "Micrófono no disponible.");
      return;
    }

    clearError();
    stopPlayback();
    stopping = false;
    manualStopOnly = false;
    setStatus("Pidiendo permiso para el micrófono…");

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      var mimeType = preferredMimeType();
      mediaRecorder = mimeType
        ? new window.MediaRecorder(mediaStream, { mimeType: mimeType })
        : new window.MediaRecorder(mediaStream);

      chunks = [];
      mediaRecorder.addEventListener("dataavailable", function (event) {
        if (event.data && event.data.size > 0) {
          chunks.push(event.data);
        }
      });
      mediaRecorder.addEventListener("stop", onRecordingStopped, {
        once: true
      });
      mediaRecorder.start();

      var automatic = startSilenceMonitor();
      manualStopOnly = !automatic;
      setPhase("listening");
      if (automatic) {
        setStatus(
          "Te escucho. Habla con naturalidad; pararé cuando hagas una pausa."
        );
      } else {
        setStatus(
          "Te escucho. Este navegador no puede detectar pausas; pulsa Detener cuando acabes."
        );
      }
      maxTimer = window.setTimeout(function () {
        if (mediaRecorder && mediaRecorder.state === "recording") {
          stopRecording("max");
        }
      }, config.maxRecordingMs);
    } catch (error) {
      clearMaxTimer();
      stopSilenceMonitor();
      releaseStream();
      mediaRecorder = null;
      stopping = false;
      setPhase("idle", "Micrófono no disponible.");
      showError(
        "No he podido usar el micrófono. Revisa el permiso del navegador o usa la opción de escribir."
      );
    }
  }

  micButton.addEventListener("click", function () {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      stopRecording("manual");
    } else {
      startRecording();
    }
  });

  if (stopButton) {
    stopButton.addEventListener("click", function () {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        stopRecording("manual");
        return;
      }
      stopPlayback();
      setPhase("idle", "Listo para escucharte.");
    });
  }

  if (replayButton) {
    replayButton.addEventListener("click", function () {
      playAnswer();
    });
  }

  if (consentAccept) {
    consentAccept.addEventListener("click", function () {
      rememberConsent();
      closeConsent();
      startRecording();
    });
  }

  if (consentCancel) {
    consentCancel.addEventListener("click", function () {
      closeConsent();
      setStatus(
        "Puedes usar la opción de escribir si prefieres no activar la voz."
      );
    });
  }

  window.addEventListener("beforeunload", function () {
    clearMaxTimer();
    stopSilenceMonitor();
    releaseStream();
    stopPlayback();
    clearLastAudio();
  });
})();
