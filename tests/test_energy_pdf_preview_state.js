"use strict";

const assert = require("assert");
const fs = require("fs");
const vm = require("vm");

const html = fs.readFileSync("templates/energy.html", "utf8");
const prepareFunctionMatch = html.match(
  /function pdfPrepareSessionMatchesCurrent\([\s\S]*?\n        \}/
);
const functionMatch = html.match(
  /function pdfPreviewRequestMatchesCurrent\([\s\S]*?\n        \}/
);

assert.ok(
  prepareFunctionMatch,
  "energy template must define pdfPrepareSessionMatchesCurrent"
);
assert.ok(
  functionMatch,
  "energy template must define pdfPreviewRequestMatchesCurrent"
);

const sandbox = {};
vm.runInNewContext(
  `${prepareFunctionMatch[0]};
  ${functionMatch[0]};
  prepareMatcher = pdfPrepareSessionMatchesCurrent;
  previewMatcher = pdfPreviewRequestMatchesCurrent;`,
  sandbox
);
const prepareMatchesCurrent = sandbox.prepareMatcher;
const previewMatchesCurrent = sandbox.previewMatcher;

assert.strictEqual(
  prepareMatchesCurrent(21, 22),
  false,
  "a prepare response from the previous upload session must not update state"
);
assert.strictEqual(
  prepareMatchesCurrent(22, 22),
  true,
  "the latest upload prepare response may update state"
);

const pageTwoRequest = {
  sessionId: 8,
  requestId: 14,
  uploadToken: "upload-a",
  pageNumber: "2",
};
const pageThreeRequest = {
  sessionId: 8,
  requestId: 15,
  uploadToken: "upload-a",
  pageNumber: "3",
};

assert.strictEqual(
  previewMatchesCurrent(pageTwoRequest, 8, 15, "upload-a", "3"),
  false,
  "an older page response must not replace the latest page"
);
assert.strictEqual(
  previewMatchesCurrent(pageThreeRequest, 8, 15, "upload-a", "3"),
  true,
  "the latest page response may update preview state"
);
assert.strictEqual(
  previewMatchesCurrent(pageThreeRequest, 9, 16, "upload-b", "1"),
  false,
  "a response from the previous upload session must not update state"
);
assert.strictEqual(
  previewMatchesCurrent(pageThreeRequest, 8, 16, "upload-a", "3"),
  false,
  "a stale image load callback must fail after a newer preview starts"
);

console.log("energy PDF preview state tests passed");
