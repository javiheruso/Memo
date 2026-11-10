/* Pure, dependency-free helpers for the public voice client.
 *
 * The silence detector is a small state machine kept separate from the DOM so
 * it can be unit-tested in Node (see tests/js/voice_silence.test.js). It never
 * touches the microphone, the network, or browser storage: the caller supplies
 * RMS samples from an AnalyserNode and a monotonic timestamp.
 *
 * Defaults are intentionally conservative. They are client-side constants, not
 * server configuration: a deployment may override them with
 * ``window.MEMO_VOICE_CONFIG`` before voice.js loads.
 */
(function (root, factory) {
  "use strict";

  var api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.MemoVoiceSilence = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : null, function () {
  "use strict";

  var DEFAULT_CONFIG = Object.freeze({
    // Hard upper bound; the microphone is never listened to indefinitely.
    maxRecordingMs: 60000,
    // Do not auto-stop before this much time has elapsed, so a slow speaker
    // can begin without being cut off.
    initialGraceMs: 2500,
    // Silence must persist this long after speech was detected.
    silenceDurationMs: 1600,
    // RMS of normalized [-1, 1] samples. Raise it in a noisy room.
    silenceRmsThreshold: 0.012,
    analyserFftSize: 2048,
    pollIntervalMs: 100
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
      maxRecordingMs: positiveNumber(config, "maxRecordingMs"),
      initialGraceMs: positiveNumber(config, "initialGraceMs"),
      silenceDurationMs: positiveNumber(config, "silenceDurationMs"),
      silenceRmsThreshold: positiveNumber(config, "silenceRmsThreshold"),
      analyserFftSize: positiveNumber(config, "analyserFftSize"),
      pollIntervalMs: positiveNumber(config, "pollIntervalMs")
    });
  }

  function rmsFromFloatSamples(samples) {
    if (!samples || !isFiniteNumber(samples.length) || samples.length === 0) {
      return 0;
    }
    var sum = 0;
    var count = 0;
    for (var index = 0; index < samples.length; index += 1) {
      var value = Number(samples[index]);
      if (!isFinite(value)) {
        continue;
      }
      sum += value * value;
      count += 1;
    }
    return count === 0 ? 0 : Math.sqrt(sum / count);
  }

  function rmsFromByteSamples(samples) {
    if (!samples || !isFiniteNumber(samples.length) || samples.length === 0) {
      return 0;
    }
    var sum = 0;
    var count = 0;
    for (var index = 0; index < samples.length; index += 1) {
      var value = Number(samples[index]);
      if (!isFinite(value)) {
        continue;
      }
      // getByteTimeDomainData is unsigned 8-bit centered at 128.
      var centered = (value - 128) / 128;
      sum += centered * centered;
      count += 1;
    }
    return count === 0 ? 0 : Math.sqrt(sum / count);
  }

  /* Create a detector. Call ``reset(startMs)`` when recording starts, then
   * ``observe({nowMs, rms})`` on every analyser poll. The result includes
   * ``shouldStop`` and a human-readable ``reason`` only when local silence
   * detection should stop recording.
   */
  function createSilenceDetector(config) {
    var settings = normalizeConfig(config);
    var startedAtMs = 0;
    var lastObservedAtMs = 0;
    var speechStarted = false;
    var silenceSinceMs = null;

    function reset(startMs) {
      startedAtMs = isFiniteNumber(startMs) ? startMs : 0;
      lastObservedAtMs = startedAtMs;
      speechStarted = false;
      silenceSinceMs = null;
    }

    function observe(sample) {
      var nowMs =
        sample && isFiniteNumber(sample.nowMs)
          ? sample.nowMs
          : lastObservedAtMs;
      var rms = sample && isFiniteNumber(sample.rms) ? sample.rms : 0;
      lastObservedAtMs = nowMs;

      var graceUntilMs = startedAtMs + settings.initialGraceMs;
      var loud = rms >= settings.silenceRmsThreshold;

      if (loud) {
        speechStarted = true;
        silenceSinceMs = null;
      } else if (speechStarted && silenceSinceMs === null) {
        // Silence only counts from the end of the initial grace window. This
        // guarantees the full grace period even when speech was detected very
        // early, then paused.
        silenceSinceMs = Math.max(nowMs, graceUntilMs);
      }

      var silenceForMs =
        silenceSinceMs === null ? 0 : Math.max(0, nowMs - silenceSinceMs);
      var shouldStop =
        nowMs >= graceUntilMs &&
        speechStarted &&
        silenceSinceMs !== null &&
        silenceForMs >= settings.silenceDurationMs;

      return {
        shouldStop: shouldStop,
        reason: shouldStop ? "silence" : null,
        speechStarted: speechStarted,
        silenceForMs: silenceForMs,
        graceActive: nowMs < graceUntilMs
      };
    }

    function state() {
      return {
        speechStarted: speechStarted,
        silenceForMs:
          silenceSinceMs === null
            ? 0
            : Math.max(0, lastObservedAtMs - silenceSinceMs),
        graceActive:
          lastObservedAtMs < startedAtMs + settings.initialGraceMs
      };
    }

    reset(0);

    return {
      config: settings,
      reset: reset,
      observe: observe,
      state: state
    };
  }

  return {
    DEFAULT_CONFIG: DEFAULT_CONFIG,
    createSilenceDetector: createSilenceDetector,
    normalizeConfig: normalizeConfig,
    rmsFromByteSamples: rmsFromByteSamples,
    rmsFromFloatSamples: rmsFromFloatSamples
  };
});
