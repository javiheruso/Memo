"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const {
  DEFAULT_CONFIG,
  STATUSES,
  createConversationSession,
  normalizeConfig
} = require("../../app/web/static/voice-session.js");

function effects(result) {
  return result.effects.map((effect) => effect.type);
}

test("session defaults are bounded and conservative", () => {
  assert.equal(DEFAULT_CONFIG.maxSessionMs, 600000);
  assert.equal(DEFAULT_CONFIG.maxRecordingMs, 60000);
  assert.equal(DEFAULT_CONFIG.resumeDelayMs, 800);
});

test("normalizeConfig replaces invalid bounds without changing defaults", () => {
  const config = normalizeConfig({
    maxSessionMs: 0,
    maxRecordingMs: -1,
    resumeDelayMs: "soon"
  });
  assert.equal(config.maxSessionMs, DEFAULT_CONFIG.maxSessionMs);
  assert.equal(config.maxRecordingMs, DEFAULT_CONFIG.maxRecordingMs);
  assert.equal(config.resumeDelayMs, DEFAULT_CONFIG.resumeDelayMs);
});

test("a new session is idle and never starts itself", () => {
  const session = createConversationSession();
  const state = session.state();
  assert.equal(state.status, STATUSES.idle);
  assert.equal(state.active, false);

  // No response, playback or silence event can start hands-free mode.
  assert.deepEqual(effects(session.playbackEnded(10)), []);
  assert.deepEqual(effects(session.resumeReady(20)), []);
  assert.deepEqual(effects(session.turnStopped(30)), []);
  assert.equal(session.state().active, false);
  assert.equal(session.state().status, STATUSES.idle);
});

test("explicit start begins listening and records exactly one turn per stop", () => {
  const session = createConversationSession({
    resumeDelayMs: 100,
    maxRecordingMs: 5000
  });

  const started = session.start(0);
  assert.equal(started.ok, true);
  assert.deepEqual(effects(started), ["listen"]);
  assert.equal(session.state().status, STATUSES.listening);
  assert.equal(session.state().active, true);

  // A duplicate start must not open a second microphone loop.
  assert.equal(session.start(10).ok, false);
  assert.deepEqual(effects(session.start(10)), []);

  const stopped = session.turnStopped(100);
  assert.equal(stopped.ok, true);
  assert.deepEqual(effects(stopped), ["send_turn"]);
  assert.equal(stopped.effects[0].turn, 1);
  assert.equal(session.state().status, STATUSES.thinking);

  // The same bounded turn is never submitted twice.
  assert.equal(session.turnStopped(120).ok, false);
  assert.deepEqual(effects(session.turnStopped(120)), []);
  assert.equal(session.state().turns, 1);
});

test("silence turn submission transitions through speaking and resumes after the safe delay", () => {
  const session = createConversationSession({ resumeDelayMs: 200 });
  session.start(0);
  session.turnStopped(1000);

  const answer = session.answerReady(1500, { hasAudio: true });
  assert.equal(answer.ok, true);
  assert.deepEqual(effects(answer), ["play_audio"]);
  assert.equal(session.state().status, STATUSES.speaking);

  const ended = session.playbackEnded(3000);
  assert.equal(ended.ok, true);
  assert.deepEqual(effects(ended), ["schedule_resume"]);
  assert.equal(ended.effects[0].delayMs, 200);
  assert.equal(session.state().status, STATUSES.waiting);

  // A timer that fires early must wait for the full safe delay.
  const early = session.resumeReady(3150);
  assert.equal(early.ok, false);
  assert.equal(early.reason, "resume_delay");
  assert.deepEqual(effects(early), ["schedule_resume"]);
  assert.equal(session.state().status, STATUSES.waiting);

  const resumed = session.resumeReady(3200);
  assert.equal(resumed.ok, true);
  assert.deepEqual(effects(resumed), ["listen"]);
  assert.equal(session.state().status, STATUSES.listening);

  const nextTurn = session.turnStopped(4000);
  assert.equal(nextTurn.effects[0].turn, 2);
  assert.equal(session.state().turns, 2);
});

test("missing or failed TTS pauses the session instead of listening again", () => {
  const session = createConversationSession();
  session.start(0);
  session.turnStopped(500);

  const answer = session.answerReady(700, { hasAudio: false });
  assert.equal(answer.ok, true);
  assert.deepEqual(effects(answer), ["stop_all"]);
  assert.equal(session.state().active, false);
  assert.equal(session.state().status, STATUSES.paused);
  assert.equal(session.state().pauseReason, "tts_unavailable");

  // Nothing can silently re-enter hands-free mode after the pause.
  assert.equal(session.resumeReady(1000).ok, false);
  assert.equal(session.playbackEnded(1000).ok, false);
  assert.deepEqual(effects(session.resumeReady(1000)), []);
  assert.equal(session.state().status, STATUSES.paused);
});

test("hidden tab, user pause and provider failure all stop without auto-resume", () => {
  const pauseReasons = ["hidden_tab", "user", "provider_error"];

  for (const reason of pauseReasons) {
    const session = createConversationSession();
    session.start(0);
    session.turnStopped(100);

    const paused = session.pause(200, reason);
    assert.equal(paused.ok, true);
    assert.deepEqual(effects(paused), ["stop_all"]);
    assert.equal(session.state().active, false);
    assert.equal(session.state().status, STATUSES.paused);
    assert.equal(session.state().pauseReason, reason);

    assert.equal(session.resumeReady(300).ok, false);
    assert.equal(session.tick(400).ok, false);
    assert.equal(session.state().status, STATUSES.paused);

    // Only an explicit start may begin a new listening loop.
    const restarted = session.start(500);
    assert.equal(restarted.ok, true);
    assert.deepEqual(effects(restarted), ["listen"]);
    assert.equal(session.state().status, STATUSES.listening);
  }
});

test("session duration and per-turn recording bounds are hard stops", () => {
  const session = createConversationSession({
    maxSessionMs: 1000,
    maxRecordingMs: 300,
    resumeDelayMs: 50
  });
  session.start(0);

  assert.equal(session.tick(100).ok, true);
  assert.deepEqual(effects(session.tick(250)), []);
  const recordingLimit = session.tick(300);
  assert.equal(recordingLimit.ok, true);
  assert.deepEqual(effects(recordingLimit), ["stop_turn"]);
  assert.equal(recordingLimit.effects[0].reason, "max_recording");
  assert.equal(session.state().status, STATUSES.listening);

  const sessionLimit = session.tick(1000);
  assert.equal(sessionLimit.ok, true);
  assert.deepEqual(effects(sessionLimit), ["stop_all"]);
  assert.equal(session.state().active, false);
  assert.equal(session.state().status, STATUSES.paused);
  assert.equal(session.state().pauseReason, "session_limit");
});

test("manual single-turn fallback is independent of the session machine", () => {
  const session = createConversationSession();

  // Without an explicit start the machine has no active session, so manual
  // recording can use the existing single-turn client path unchanged.
  assert.equal(session.state().active, false);
  assert.equal(session.pause(0, "user").ok, false);
  assert.equal(session.state().status, STATUSES.idle);

  // A completed manual turn does not activate hands-free listening.
  assert.equal(session.state().turns, 0);
  assert.equal(session.state().status, STATUSES.idle);
});
