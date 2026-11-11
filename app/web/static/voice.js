/* Voice-first public chat interaction.
 *
 * Manual single-turn mode: one tap starts a MediaRecorder. A local Web Audio
 * AnalyserNode watches the microphone level; after speech has been detected and
 * a sustained pause follows, the client stops recording and uploads the turn
 * automatically. A visible Detener button remains available, and a hard
 * maximum recording time bounds every listening turn.
 *
 * Hands-free conversation mode: only an explicit "Empezar conversación" action
 * may start the loop. While it is active the client listens for one bounded
 * turn, uploads it once, plays the answer with the microphone released, waits
 * for playback to end plus a safe delay, then resumes listening. Pausing on a
 * hidden tab, page navigation, provider/microphone failure, missing TTS or the
 * session time limit stops the loop and requires another explicit start.
 *
 * There is no wake word, no audio streaming and no dependency: each recorded
 * blob is sent once to the same-origin voice endpoint. Raw audio is kept in
 * memory just long enough to upload it; the app never stores it.
 */
(function () {
  "use strict";

  var CONSENT_KEY = "memo_voice_consent_v1";
  var FALLBACK_CONFIG = {
    maxRecordingMs: 60000,
    maxSessionMs: 600000,
    resumeDelayMs: 800,
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
  var sessionApi =
    window.MemoVoiceSession &&
    typeof window.MemoVoiceSession.createConversationSession === "function"
      ? window.MemoVoiceSession
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
  var sessionStartButton = document.getElementById("voice-session-start");
  var sessionPauseButton = document.getElementById("voice-session-pause");
  var consentDialog = document.getElementById("voice-consent");
  var consentAccept = document.getElementById("voice-consent-accept");
  var consentCancel = document.getElementById("voice-consent-cancel");
  if (!endpoint || !conversation || !liveTranscript || !statusEl || !micButton) {
    return;
  }

  var config = voiceConfig();
  var session = sessionApi
    ? sessionApi.createConversationSession(config)
    : null;
  var mediaRecorder = null;
  var mediaStream = null;
  var audioContext = null;
  var mediaSource = null;
  var analyser = null;
  var analyserBuffer = null;
  var detector = null;
  var silenceTimer = null;
  var maxTimer = null;
  var sessionLimitTimer = null;
  var resumeTimer = null;
  var chunks = [];
  var stopping = false;
  var manualStopOnly = false;
  var currentPhase = "idle";
  var playback = null;
  var lastAudioUrl = null;
  var busy = false;
  var consentIntent = null;

  function voiceConfig() {
    var defaults = Object.assign({}, FALLBACK_CONFIG);
    if (sessionApi && sessionApi.DEFAULT_CONFIG) {
      Object.assign(defaults, sessionApi.DEFAULT_CONFIG);
    }
    var overrides =
      window.MEMO_VOICE_CONFIG && typeof window.MEMO_VOICE_CONFIG === "object"
        ? window.MEMO_VOICE_CONFIG
        : {};
    var merged = Object.assign({}, defaults, overrides);
    if (silenceApi && typeof silenceApi.normalizeConfig === "function") {
      Object.assign(merged, silenceApi.normalizeConfig(merged));
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

  function recordingLimitMs() {
    return session ? session.config.maxRecordingMs : config.maxRecordingMs;
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

  function openConsent(intent) {
    consentIntent = intent || "manual";
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
    root.classList.toggle("is-waiting", phase === "waiting");
    root.classList.toggle("is-paused", phase === "paused");
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

  function updateSessionControls() {
    if (!session || !sessionStartButton || !sessionPauseButton) {
      return;
    }
    var active = session.state().active;
    sessionStartButton.hidden = active;
    sessionPauseButton.hidden = !active;
    sessionStartButton.setAttribute("aria-pressed", active ? "true" : "false");
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
    var answered = new Audio(lastAudioUrl);
    playback = answered;
    answered.addEventListener("ended", function () {
      if (playback !== answered) {
        return;
      }
      playback = null;
      updateStopButton();
      if (
        session &&
        session.state().active &&
        session.state().status === "speaking"
      ) {
        var decision = session.playbackEnded(nowMs());
        applySessionEffects(decision.effects);
        return;
      }
      setPhase("idle", "Listo para escucharte.");
    });
    answered.addEventListener("error", function () {
      if (playback !== answered) {
        return;
      }
      playback = null;
      updateStopButton();
      if (
        session &&
        session.state().active &&
        session.state().status === "speaking"
      ) {
        pauseSession(
          "playback_error",
          "No se ha podido reproducir la voz. La conversación está en pausa; puedes leer la respuesta."
        );
        return;
      }
      setPhase(
        "idle",
        "La voz no está disponible. Puedes leer la respuesta."
      );
    });
    setPhase("speaking", "Memo está respondiendo…");
    var started = answered.play();
    if (started && typeof started.catch === "function") {
      started.catch(function () {
        if (playback !== answered) {
          return;
        }
        playback = null;
        updateStopButton();
        if (
          session &&
          session.state().active &&
          session.state().status === "speaking"
        ) {
          pauseSession(
            "playback_error",
            "El navegador no ha podido reproducir la voz. La conversación está en pausa; puedes leer la respuesta."
          );
          return;
        }
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
    return Boolean(payload.audio);
  }

  function showHttpError(status, sessionTurn) {
    var message = "No se ha podido completar la respuesta.";
    var detail = "Algo ha ido mal. Inténtalo de nuevo.";
    if (status === 400) {
      message =
        "No he podido entenderte. Prueba a hablar más cerca del micrófono.";
      detail = "No se ha entendido el audio.";
    } else if (status === 413) {
      message = "La grabación es demasiado larga.";
      detail =
        "La grabación es demasiado larga. Prueba con una frase más corta.";
    } else if (status === 415) {
      message = "Formato de audio no compatible.";
      detail =
        "Este navegador usa un formato de audio no compatible. Prueba con otro navegador.";
    } else if (status === 404) {
      message = "Conversación nueva.";
      detail =
        "No he podido recuperar esta conversación. Empezamos una conversación nueva.";
    }

    if (
      sessionTurn &&
      session &&
      session.state().active
    ) {
      pauseSession("provider_error", message);
    } else if (
      !(sessionTurn && session && session.state().status === "paused")
    ) {
      setPhase("idle", message);
    }
    showError(detail);
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

  async function sendAudio(blob, options) {
    var sessionTurn = Boolean(options && options.sessionTurn);
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
        showHttpError(response.status, sessionTurn);
        return;
      }
      var payload = await response.json();
      var hasAudio = renderResult(payload);
      if (sessionTurn) {
        if (!session || !session.state().active) {
          /* The user paused while the turn was in flight. Render the text but
           * never auto-play or return to the microphone. */
          setPhase("paused", "La conversación está en pausa.");
          updateSessionControls();
          return;
        }
        var decision = session.answerReady(nowMs(), { hasAudio: hasAudio });
        applySessionEffects(decision.effects);
        if (!hasAudio) {
          setPhase(
            "paused",
            "La voz no está disponible ahora mismo; puedes leer la respuesta. La conversación está en pausa."
          );
          updateSessionControls();
        }
        return;
      }
      if (hasAudio) {
        playAnswer();
      } else {
        setPhase(
          "idle",
          "Respuesta lista. La voz no está disponible ahora mismo; puedes leerla."
        );
      }
    } catch (error) {
      if (sessionTurn && session && session.state().active) {
        pauseSession(
          "provider_error",
          "No he podido conectar con Memo. La conversación está en pausa."
        );
        showError("No he podido conectar con Memo. Inténtalo otra vez.");
      } else if (
        !(sessionTurn && session && session.state().status === "paused")
      ) {
        setPhase("idle", "Sin conexión.");
        showError("No he podido conectar con Memo. Inténtalo otra vez.");
      }
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

  function clearSessionLimit() {
    if (sessionLimitTimer !== null) {
      window.clearTimeout(sessionLimitTimer);
      sessionLimitTimer = null;
    }
  }

  function clearResumeTimer() {
    if (resumeTimer !== null) {
      window.clearTimeout(resumeTimer);
      resumeTimer = null;
    }
  }

  function scheduleSessionLimit() {
    clearSessionLimit();
    if (!session) {
      return;
    }
    sessionLimitTimer = window.setTimeout(function () {
      sessionLimitTimer = null;
      if (session.state().active) {
        pauseSession(
          "session_limit",
          "La conversación se ha detenido al alcanzar el límite de tiempo. Pulsa Empezar conversación para continuar."
        );
      }
    }, session.config.maxSessionMs);
  }

  function scheduleResume(delayMs) {
    clearResumeTimer();
    setPhase("waiting", "Preparando el micrófono…");
    resumeTimer = window.setTimeout(function () {
      resumeTimer = null;
      if (
        !session ||
        !session.state().active ||
        session.state().status !== "waiting"
      ) {
        return;
      }
      var decision = session.resumeReady(nowMs());
      applySessionEffects(decision.effects);
    }, Math.max(0, delayMs));
  }

  function stopEverythingForPause() {
    clearSessionLimit();
    clearResumeTimer();
    clearMaxTimer();
    stopSilenceMonitor();
    var discarded = mediaRecorder;
    mediaRecorder = null;
    stopping = false;
    chunks = [];
    if (discarded) {
      /* Detach the turn handlers before stopping so a pause can never upload
       * the in-progress audio or leak its data chunks into the next turn. */
      discarded.removeEventListener("dataavailable", onRecordingData);
      discarded.removeEventListener("stop", onRecordingStopped);
      if (discarded.state === "recording") {
        try {
          discarded.stop();
        } catch (error) {
          /* Ignore a recorder that has already been released. */
        }
      }
    }
    releaseStream();
    stopPlayback();
    updateStopButton();
  }

  function applySessionEffects(effects) {
    (effects || []).forEach(function (effect) {
      if (!effect) {
        return;
      }
      if (effect.type === "listen") {
        startRecording({ sessionMode: true });
      } else if (effect.type === "stop_turn") {
        if (mediaRecorder && mediaRecorder.state === "recording") {
          stopRecording(effect.reason || "max");
        }
      } else if (effect.type === "stop_all") {
        stopEverythingForPause();
      } else if (effect.type === "schedule_resume") {
        scheduleResume(effect.delayMs);
      } else if (effect.type === "play_audio") {
        playAnswer();
      }
    });
  }

  function pauseSession(reason, message) {
    if (!session || !session.state().active) {
      updateSessionControls();
      return;
    }
    var decision = session.pause(nowMs(), reason);
    applySessionEffects(decision.effects);
    updateSessionControls();
    setPhase(
      "paused",
      message ||
        "Conversación en pausa. Pulsa Empezar conversación para volver a hablar."
    );
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

  function onRecordingData(event) {
    if (event.data && event.data.size > 0) {
      chunks.push(event.data);
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
      if (session && session.state().active) {
        pauseSession(
          "empty_audio",
          "No se ha grabado audio. La conversación está en pausa."
        );
      } else {
        setPhase("idle", "No se ha grabado audio. Inténtalo otra vez.");
      }
      return;
    }
    var sessionTurn = false;
    if (
      session &&
      session.state().active &&
      session.state().status === "listening"
    ) {
      var decision = session.turnStopped(nowMs());
      sessionTurn = Boolean(decision && decision.ok);
    } else if (session && session.state().active) {
      /* A microphone turn ended outside the listening phase; dropping it is
       * safer than uploading an unexpected extra turn. */
      return;
    }
    sendAudio(blob, { sessionTurn: sessionTurn });
  }

  async function startRecording(options) {
    var sessionMode = Boolean(options && options.sessionMode);
    if (busy || stopping) {
      return;
    }
    if (
      session &&
      session.state().active &&
      session.state().status !== "listening"
    ) {
      return;
    }
    if (!consentGranted()) {
      openConsent(sessionMode ? "session" : "manual");
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
      var stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      if (
        sessionMode &&
        (!session ||
          !session.state().active ||
          session.state().status !== "listening")
      ) {
        stream.getTracks().forEach(function (track) {
          track.stop();
        });
        return;
      }
      mediaStream = stream;
      var mimeType = preferredMimeType();
      mediaRecorder = mimeType
        ? new window.MediaRecorder(mediaStream, { mimeType: mimeType })
        : new window.MediaRecorder(mediaStream);

      chunks = [];
      mediaRecorder.addEventListener("dataavailable", onRecordingData);
      mediaRecorder.addEventListener("stop", onRecordingStopped, {
        once: true
      });
      mediaRecorder.start();

      if (sessionMode && session) {
        session.listeningStarted(nowMs());
      }
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
      }, recordingLimitMs());
    } catch (error) {
      clearMaxTimer();
      stopSilenceMonitor();
      releaseStream();
      mediaRecorder = null;
      stopping = false;
      if (sessionMode) {
        showError(
          "No he podido usar el micrófono. Revisa el permiso del navegador o usa la opción de escribir."
        );
        if (session && session.state().active) {
          pauseSession(
            "microphone_error",
            "No he podido usar el micrófono. La conversación está en pausa; revisa el permiso y vuelve a empezar."
          );
        } else if (session && session.state().status === "paused") {
          setPhase(
            "paused",
            "La conversación está en pausa. Revisa el permiso del micrófono y pulsa Empezar conversación."
          );
        } else {
          setPhase("idle", "Micrófono no disponible.");
        }
        return;
      }
      setPhase("idle", "Micrófono no disponible.");
      showError(
        "No he podido usar el micrófono. Revisa el permiso del navegador o usa la opción de escribir."
      );
    }
  }

  function beginSession() {
    if (!session) {
      showError(
        "Este navegador no puede mantener una conversación manos libres. Puedes usar Hablar."
      );
      setPhase("idle", "Modo conversación no disponible.");
      return;
    }
    if (session.state().active) {
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
    var decision = session.start(nowMs());
    if (!decision.ok) {
      return;
    }
    updateSessionControls();
    scheduleSessionLimit();
    applySessionEffects(decision.effects);
  }

  function requestSessionStart() {
    if (!session) {
      beginSession();
      return;
    }
    if (session.state().active) {
      return;
    }
    if (!consentGranted()) {
      openConsent("session");
      return;
    }
    beginSession();
  }

  micButton.addEventListener("click", function () {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      stopRecording("manual");
      return;
    }
    if (session && session.state().active) {
      /* An active session owns the microphone; use Pausar conversación. */
      return;
    }
    startRecording();
  });

  if (stopButton) {
    stopButton.addEventListener("click", function () {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        stopRecording("manual");
        return;
      }
      if (session && session.state().active) {
        pauseSession(
          "user",
          "Conversación en pausa. Pulsa Empezar conversación para volver a hablar."
        );
        return;
      }
      stopPlayback();
      setPhase("idle", "Listo para escucharte.");
    });
  }

  if (replayButton) {
    replayButton.addEventListener("click", function () {
      if (session && session.state().active) {
        /* Replaying during a session must not race the automatic resume
         * timer. Stop hands-free mode first, then treat the replay as manual
         * output with no microphone listening. */
        pauseSession(
          "user_replay",
          "Conversación en pausa para repetir la respuesta."
        );
      }
      playAnswer();
    });
  }

  if (sessionStartButton) {
    sessionStartButton.addEventListener("click", requestSessionStart);
  }

  if (sessionPauseButton) {
    sessionPauseButton.addEventListener("click", function () {
      pauseSession(
        "user",
        "Conversación en pausa. Pulsa Empezar conversación para volver a hablar."
      );
    });
  }

  if (!session && sessionStartButton) {
    sessionStartButton.hidden = true;
  }

  if (consentAccept) {
    consentAccept.addEventListener("click", function () {
      rememberConsent();
      closeConsent();
      var intent = consentIntent;
      consentIntent = null;
      if (intent === "session") {
        beginSession();
      } else {
        startRecording();
      }
    });
  }

  if (consentCancel) {
    consentCancel.addEventListener("click", function () {
      var intent = consentIntent;
      consentIntent = null;
      closeConsent();
      if (intent === "session") {
        setStatus(
          "Puedes usar Hablar para un turno suelto o escribir si prefieres no activar la voz."
        );
        return;
      }
      setStatus(
        "Puedes usar la opción de escribir si prefieres no activar la voz."
      );
    });
  }

  function pauseForHiddenPage() {
    if (session && session.state().active) {
      pauseSession(
        "hidden_tab",
        "Conversación en pausa porque has salido de la página. Pulsa Empezar conversación para volver."
      );
    }
  }

  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      pauseForHiddenPage();
    }
  });

  window.addEventListener("pagehide", pauseForHiddenPage);

  window.addEventListener("beforeunload", function () {
    clearMaxTimer();
    clearSessionLimit();
    clearResumeTimer();
    stopSilenceMonitor();
    releaseStream();
    stopPlayback();
    clearLastAudio();
  });

  updateSessionControls();
})();
