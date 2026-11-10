"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const {
  DEFAULT_CONFIG,
  createSilenceDetector,
  normalizeConfig,
  rmsFromByteSamples,
  rmsFromFloatSamples
} = require("../../app/web/static/voice-silence.js");

test("defaults stay conservative and bounded", () => {
  assert.equal(DEFAULT_CONFIG.maxRecordingMs, 60000);
  assert.equal(DEFAULT_CONFIG.initialGraceMs, 2500);
  assert.equal(DEFAULT_CONFIG.silenceDurationMs, 1600);
  assert.ok(DEFAULT_CONFIG.silenceRmsThreshold > 0);
  assert.ok(DEFAULT_CONFIG.pollIntervalMs > 0);
});

test("normalizeConfig ignores non-positive and invalid overrides", () => {
  const config = normalizeConfig({
    initialGraceMs: -1,
    silenceDurationMs: 0,
    silenceRmsThreshold: "loud",
    pollIntervalMs: Infinity
  });
  assert.equal(config.initialGraceMs, DEFAULT_CONFIG.initialGraceMs);
  assert.equal(config.silenceDurationMs, DEFAULT_CONFIG.silenceDurationMs);
  assert.equal(config.silenceRmsThreshold, DEFAULT_CONFIG.silenceRmsThreshold);
  assert.equal(config.pollIntervalMs, DEFAULT_CONFIG.pollIntervalMs);
});

test("float RMS uses normalized time-domain samples", () => {
  assert.equal(rmsFromFloatSamples([]), 0);
  assert.equal(rmsFromFloatSamples(null), 0);
  assert.ok(Math.abs(rmsFromFloatSamples([0.5, -0.5]) - 0.5) < 1e-9);
  assert.ok(Math.abs(rmsFromFloatSamples([0, 0, 0]) - 0) < 1e-9);
});

test("byte RMS centers unsigned 8-bit samples at 128", () => {
  assert.equal(rmsFromByteSamples([128, 128, 128]), 0);
  const loud = rmsFromByteSamples([0, 255]);
  assert.ok(loud > 0.99 && loud <= 1);
});

test("silence alone never auto-stops before speech is detected", () => {
  const detector = createSilenceDetector({
    initialGraceMs: 100,
    silenceDurationMs: 100,
    silenceRmsThreshold: 0.01
  });
  detector.reset(0);

  for (let now = 0; now <= 5000; now += 100) {
    const decision = detector.observe({ nowMs: now, rms: 0 });
    assert.equal(decision.shouldStop, false);
  }
});

test("speech followed by sustained silence stops after the grace window", () => {
  const detector = createSilenceDetector({
    initialGraceMs: 100,
    silenceDurationMs: 100,
    silenceRmsThreshold: 0.01
  });
  detector.reset(0);

  assert.equal(
    detector.observe({ nowMs: 90, rms: 0.2 }).speechStarted,
    true
  );
  // Silence begins after the grace window, so the 100ms clock starts at 110ms.
  assert.equal(detector.observe({ nowMs: 110, rms: 0 }).shouldStop, false);
  assert.equal(detector.observe({ nowMs: 190, rms: 0 }).shouldStop, false);
  const stopped = detector.observe({ nowMs: 210, rms: 0 });
  assert.equal(stopped.shouldStop, true);
  assert.equal(stopped.reason, "silence");
});

test("a short pause during the grace period does not stop recording", () => {
  const detector = createSilenceDetector({
    initialGraceMs: 2500,
    silenceDurationMs: 1600,
    silenceRmsThreshold: 0.01
  });
  detector.reset(0);

  detector.observe({ nowMs: 200, rms: 0.2 });
  detector.observe({ nowMs: 300, rms: 0 });
  assert.equal(detector.observe({ nowMs: 2600, rms: 0 }).shouldStop, false);
  assert.equal(detector.observe({ nowMs: 3000, rms: 0 }).shouldStop, false);
});

test("new speech resets the silence timer", () => {
  const detector = createSilenceDetector({
    initialGraceMs: 1,
    silenceDurationMs: 100,
    silenceRmsThreshold: 0.01
  });
  detector.reset(0);

  detector.observe({ nowMs: 0, rms: 0.2 });
  detector.observe({ nowMs: 50, rms: 0 });
  detector.observe({ nowMs: 120, rms: 0.3 });
  detector.observe({ nowMs: 150, rms: 0 });
  assert.equal(detector.observe({ nowMs: 240, rms: 0 }).shouldStop, false);
  assert.equal(detector.observe({ nowMs: 260, rms: 0 }).shouldStop, true);
});

test("state reports grace and accumulated silence for diagnostics", () => {
  const detector = createSilenceDetector({
    initialGraceMs: 100,
    silenceDurationMs: 100,
    silenceRmsThreshold: 0.01
  });
  detector.reset(0);
  detector.observe({ nowMs: 20, rms: 0 });
  assert.equal(detector.state().graceActive, true);
  assert.equal(detector.state().speechStarted, false);
});
