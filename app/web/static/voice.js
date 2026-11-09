/* Voice-first public chat interaction.
 *
 * Push-to-talk only: one microphone press starts a MediaRecorder, a second
 * press (or Detener) stops it, and the recorded blob is sent to the same-origin
 * voice endpoint. There is no wake word, VAD, or streaming. Raw audio is kept
 * in memory just long enough to upload it; the app never stores it.
 */
(function () {
  "use strict";

  var CONSENT_KEY = "memo_voice_consent_v1";
  var MAX_RECORDING_MS = 60000;
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

  var mediaRecorder = null;
  var mediaStream = null;
  var chunks = [];
  var autoStopTimer = null;
  var playback = null;
  var lastAudioUrl = null;
  var busy = false;

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
    var recording =
      mediaRecorder !== null && mediaRecorder.state === "recording";
    stopButton.hidden = !(recording || playback !== null);
  }

  function updateRecordingUi(active) {
    root.classList.toggle("is-recording", active);
    micButton.setAttribute("aria-pressed", active ? "true" : "false");
    var label = micButton.querySelector(".js-mic-label");
    if (label) {
      label.textContent = active ? "Terminar" : "Hablar";
    }
    updateStopButton();
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
      updateStopButton();
      setStatus("Listo para escucharte.");
    });
    playback.addEventListener("error", function () {
      playback = null;
      updateStopButton();
      setStatus("La voz no está disponible. Puedes leer la respuesta.");
    });
    setStatus("Memo está respondiendo…");
    var started = playback.play();
    if (started && typeof started.catch === "function") {
      started.catch(function () {
        setStatus("Pulsa Repetir para escuchar la respuesta.");
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
      setStatus(
        "Respuesta lista. La voz no está disponible ahora mismo; puedes leerla."
      );
    }
  }

  function showHttpError(status) {
    if (status === 400) {
      showError(
        "No he podido entenderte. Prueba a hablar más cerca del micrófono."
      );
      setStatus("No se ha entendido el audio.");
      return;
    }
    if (status === 413) {
      showError(
        "La grabación es demasiado larga. Prueba con una frase más corta."
      );
      setStatus("La grabación es demasiado larga.");
      return;
    }
    if (status === 415) {
      showError(
        "Este navegador usa un formato de audio no compatible. Prueba con otro navegador."
      );
      setStatus("Formato de audio no compatible.");
      return;
    }
    if (status === 404) {
      showError(
        "No he podido recuperar esta conversación. Empezamos una conversación nueva."
      );
      setStatus("Conversación nueva.");
      return;
    }
    showError("Algo ha ido mal. Inténtalo de nuevo.");
    setStatus("No se ha podido completar la respuesta.");
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
    root.classList.add("is-processing");
    clearError();
    setStatus("Procesando lo que has dicho…");
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
      showError("No he podido conectar con Memo. Inténtalo otra vez.");
      setStatus("Sin conexión.");
    } finally {
      busy = false;
      root.classList.remove("is-processing");
    }
  }

  function stopRecording() {
    if (autoStopTimer !== null) {
      window.clearTimeout(autoStopTimer);
      autoStopTimer = null;
    }
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
    releaseStream();
    updateRecordingUi(false);
    setStatus("Procesando lo que has dicho…");
  }

  function onRecordingStopped() {
    var mimeType =
      (mediaRecorder && mediaRecorder.mimeType) ||
      (chunks.length > 0 ? chunks[0].type : "audio/webm");
    var blob = new Blob(chunks, { type: mimeType });
    chunks = [];
    mediaRecorder = null;
    updateStopButton();
    if (!blob.size) {
      setStatus("No se ha grabado audio. Inténtalo otra vez.");
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
      setStatus("Micrófono no disponible.");
      return;
    }

    clearError();
    stopPlayback();
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

      updateRecordingUi(true);
      setStatus("Te escucho. Pulsa Terminar o Detener cuando acabes.");
      autoStopTimer = window.setTimeout(function () {
        if (mediaRecorder && mediaRecorder.state === "recording") {
          stopRecording();
          setStatus(
            "Se ha alcanzado el tiempo máximo de grabación. Procesando…"
          );
        }
      }, MAX_RECORDING_MS);
    } catch (error) {
      releaseStream();
      mediaRecorder = null;
      updateRecordingUi(false);
      showError(
        "No he podido usar el micrófono. Revisa el permiso del navegador o usa la opción de escribir."
      );
      setStatus("Micrófono no disponible.");
    }
  }

  micButton.addEventListener("click", function () {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      stopRecording();
    } else {
      startRecording();
    }
  });

  if (stopButton) {
    stopButton.addEventListener("click", function () {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        stopRecording();
        return;
      }
      stopPlayback();
      setStatus("Listo para escucharte.");
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
    releaseStream();
    stopPlayback();
    clearLastAudio();
  });
})();
