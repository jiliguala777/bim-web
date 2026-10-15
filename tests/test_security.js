const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

(async () => {
    const origin = 'https://bim.example.test';
    const sent = [];
    class BrowserRequest extends Request {
        constructor(input, options) {
            super(typeof input === 'string' ? new URL(input, origin) : input, options);
        }
    }
    const context = {
        Request: BrowserRequest, Headers, URL,
        window: { location: { origin, href: origin + '/' }, fetch: async request => { sent.push(request); } },
        document: { querySelector: () => ({ content: 'signed-csrf-token' }) },
    };
    vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../static/security.js'), 'utf8'), context);
    await context.window.fetch('/login', { method: 'POST', body: 'payload', headers: { 'Content-Type': 'application/json' } });
    assert.equal(sent[0].headers.get('X-CSRFToken'), 'signed-csrf-token');
    assert.equal(await sent[0].text(), 'payload');
    assert.equal(sent[0].headers.get('Content-Type'), 'application/json');
    await context.window.fetch('/health');
    assert.equal(sent[1].headers.has('X-CSRFToken'), false);
    await context.window.fetch('https://other.example.test/api', { method: 'POST' });
    assert.equal(sent[2].headers.has('X-CSRFToken'), false);
    console.log('security fetch tests passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
