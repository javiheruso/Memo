/* Pure, dependency-free hands-free conversation state machine.
 *
 * The browser client owns the microphone, MediaRecorder, playback and network
 * calls. This module owns only the deterministic turn-taking policy so it can
 * be unit-tested in Node without a real microphone (see
 * tests/js/voice_session.test.js).
 *
 * A session starts only when the caller explicitly invokes ``start``. It never
 * starts because a response ended, because a page loaded, or because the
 * detector saw silence. Any pause, provider/microphone failure, hidden tab or
 * session limit ends the active session and requires another explicit start.
 *
 * Effects are plain objects. The caller is responsible for translating them
 * into browser actions:
 *   {type: "listen"}                      acquire the mic and record one turn
 *   {type: "send_turn", turn}             upload the bounded turn that just ended
 *   {type: "play_audio"}                  play the synthesized answer
 *   {type: "schedule_resume", delayMs}    wait, then call resumeReady
 *   {type: "stop_turn", reason}           hard-stop the current recording
 *   {type: "stop_all", reason}            release mic, stop playback and timers
 */
(function (root, factory) {
  "use strict";

  var api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.MemoVoiceSession = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : null, function () {
  "use strict";

  var DEFAULT_CONFIG = Object.freeze({
    // Absolute session ceiling. The microphone is never part of an active
    // session past this point, regardless of current phase.
    maxSessionMs: 600000,
    // Per-turn recording ceiling, matching the manual single-turn client.
    maxRecordingMs: 60000,
    // Safe delay after playback ends before the next listening turn starts.
    // It reduces the chance of recording the tail of Memo's own voice.
    resumeDelayMs: 800
  });

  var STATUSES = Object.freeze({
    idle: "idle",
    listening: "listening",
    thinking: "thinking",
    speaking: "speaking",
    waiting: "waiting",
    paused: "paused"
  });

  function isFiniteNumber(value) {
    return typeof value === "number" && isFinite(value);
  }

  function positiveNumber(config, key) {
    if (!config || !isFiniteNumber(config[key])) {
      return DEFAULT_CONFIG[key];
    }
    return config[key] > 0 ? config[key] : DEFAULT_CONFIG[key];
  }

  function normalizeConfig(config) {
    return Object.freeze({
      maxSessionMs: positiveNumber(config, "maxSessionMs"),
      maxRecordingMs: positiveNumber(config, "maxRecordingMs"),
      resumeDelayMs: positiveNumber(config, "resumeDelayMs")
    });
  }

  function createConversationSession(config) {
    var settings = normalizeConfig(config);
    var status = STATUSES.idle;
    var active = false;
    var startedAtMs = 0;
    var turnStartedAtMs = null;
    var turns = 0;
    var pauseReason = null;
    var resumeAtMs = null;
    var lastNowMs = 0;

    function nowFrom(value) {
      return isFiniteNumber(value) ? value : lastNowMs;
    }

    function state() {
      return {
        status: status,
        active: active,
        startedAtMs: startedAtMs,
        turnStartedAtMs: turnStartedAtMs,
        turns: turns,
        pauseReason: pauseReason,
        resumeAtMs: resumeAtMs,
        config: settings
      };
    }

    function rejected(reason) {
      return { ok: false, reason: reason, effects: [], state: state() };
    }

    function accepted(effects) {
      return { ok: true, reason: null, effects: effects, state: state() };
    }

    function start(nowMs) {
      if (active) {
        return rejected("already_active");
      }
      lastNowMs = nowFrom(nowMs);
      active = true;
      status = STATUSES.listening;
      startedAtMs = lastNowMs;
      turnStartedAtMs = lastNowMs;
      turns = 0;
      pauseReason = null;
      resumeAtMs = null;
      return accepted([{ type: "listen" }]);
    }

    function listeningStarted(nowMs) {
      if (!active || status !== STATUSES.listening) {
        return rejected("not_listening");
      }
      lastNowMs = nowFrom(nowMs);
      turnStartedAtMs = lastNowMs;
      return accepted([]);
    }

    function turnStopped(nowMs) {
      if (!active || status !== STATUSES.listening) {
        return rejected("not_listening");
      }
      lastNowMs = nowFrom(nowMs);
      status = STATUSES.thinking;
      turnStartedAtMs = null;
      turns += 1;
      return accepted([{ type: "send_turn", turn: turns }]);
    }

    function answerReady(nowMs, result) {
      if (!active || status !== STATUSES.thinking) {
        return rejected("not_thinking");
      }
      lastNowMs = nowFrom(nowMs);
      if (result && result.hasAudio) {
        status = STATUSES.speaking;
        return accepted([{ type: "play_audio" }]);
      }
      // TTS is optional. Without audio the text answer stays visible but the
      // session pauses: recording again could capture the user's reaction to
      // an answer they have not heard yet.
      active = false;
      status = STATUSES.paused;
      pauseReason = "tts_unavailable";
      return accepted([{ type: "stop_all", reason: pauseReason }]);
    }

    function playbackEnded(nowMs) {
      if (!active || status !== STATUSES.speaking) {
        return rejected("not_speaking");
      }
      lastNowMs = nowFrom(nowMs);
      status = STATUSES.waiting;
      resumeAtMs = lastNowMs + settings.resumeDelayMs;
      return accepted([
        { type: "schedule_resume", delayMs: settings.resumeDelayMs }
      ]);
    }

    function resumeReady(nowMs) {
      if (!active || status !== STATUSES.waiting) {
        return rejected("not_waiting");
      }
      lastNowMs = nowFrom(nowMs);
      if (resumeAtMs !== null && lastNowMs < resumeAtMs) {
        return {
          ok: false,
          reason: "resume_delay",
          effects: [
            {
              type: "schedule_resume",
              delayMs: Math.max(0, resumeAtMs - lastNowMs)
            }
          ],
          state: state()
        };
      }
      status = STATUSES.listening;
      turnStartedAtMs = lastNowMs;
      resumeAtMs = null;
      return accepted([{ type: "listen" }]);
    }

    function pause(nowMs, reason) {
      if (!active) {
        return rejected("inactive");
      }
      lastNowMs = nowFrom(nowMs);
      active = false;
      status = STATUSES.paused;
      pauseReason = reason || "paused";
      turnStartedAtMs = null;
      resumeAtMs = null;
      return accepted([{ type: "stop_all", reason: pauseReason }]);
    }

    function tick(nowMs) {
      if (!active) {
        return rejected("inactive");
      }
      lastNowMs = nowFrom(nowMs);
      if (lastNowMs - startedAtMs >= settings.maxSessionMs) {
        return pause(lastNowMs, "session_limit");
      }
      if (
        status === STATUSES.listening &&
        turnStartedAtMs !== null &&
        lastNowMs - turnStartedAtMs >= settings.maxRecordingMs
      ) {
        return accepted([{ type: "stop_turn", reason: "max_recording" }]);
      }
      return accepted([]);
    }

    return {
      config: settings,
      state: state,
      start: start,
      listeningStarted: listeningStarted,
      turnStopped: turnStopped,
      answerReady: answerReady,
      playbackEnded: playbackEnded,
      resumeReady: resumeReady,
      pause: pause,
      tick: tick
    };
  }

  return {
    DEFAULT_CONFIG: DEFAULT_CONFIG,
    STATUSES: STATUSES,
    createConversationSession: createConversationSession,
    normalizeConfig: normalizeConfig
  };
});
