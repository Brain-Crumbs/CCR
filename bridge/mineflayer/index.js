'use strict';

// Mineflayer bridge entrypoint: speaks the line-delimited JSON protocol from
// cognitive_runtime/programs/minecraft/remote.py on stdin/stdout, backed by a
// live Minecraft server through a WorldSession.  All logging goes to stderr so
// it never corrupts the protocol stream on stdout.
//
// Commands are processed strictly in order (step() is async), one JSON
// response line per command.

const readline = require('readline');
const { WorldSession } = require('./world');

const session = new WorldSession();

function send(response) {
  process.stdout.write(JSON.stringify(response) + '\n');
}

async function handle(message) {
  const cmd = message.cmd;
  try {
    if (cmd === 'reset') {
      return await session.reset(message.seed || 0, message.config || {}, message.connection || {});
    }
    if (cmd === 'step') {
      const spec = message.action || {};
      return await session.step({ name: spec.name || 'NULL', params: spec.params || {} });
    }
    if (cmd === 'observe') {
      return session.observe(message.timestamp || 0.0);
    }
    if (cmd === 'close') {
      session.close();
      return { ok: true, _close: true };
    }
    return { ok: false, error: `unknown command ${JSON.stringify(cmd)}` };
  } catch (err) {
    return { ok: false, error: err && err.message ? err.message : String(err) };
  }
}

// Serialize command processing: buffer incoming lines, drain one at a time.
const queue = [];
let draining = false;

async function drain() {
  if (draining) return;
  draining = true;
  while (queue.length > 0) {
    const line = queue.shift();
    let message;
    try {
      message = JSON.parse(line);
    } catch (e) {
      send({ ok: false, error: `bad json: ${e.message}` });
      continue;
    }
    const response = await handle(message);
    const closing = response._close;
    delete response._close;
    send(response);
    if (closing) {
      process.exit(0);
    }
  }
  draining = false;
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on('line', (line) => {
  const trimmed = line.trim();
  if (trimmed) {
    queue.push(trimmed);
    drain();
  }
});
rl.on('close', () => {
  session.close();
  process.exit(0);
});
