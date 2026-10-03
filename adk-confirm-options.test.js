/*
 * Tests for adk-confirm-options.js. Run by text_summarizer/tests/
 * test_adk_devui_confirm_patch.py, which also hands it a patched bundle to
 * exercise the other half of the patch.
 *
 *   node adk-confirm-options.test.js [patched-bundle.js]
 *
 * node's built-in test runner only, so there is nothing to install.
 */
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const api = require(path.join(__dirname, 'adk-confirm-options.js'));

/* --- the event shape ADK actually sends ------------------------------------ */

/* Event #8 of session f0842db4-9d42-4911-8b34-255a3731021f, with the ids left as
 * they were. Everything this file asserts about card shape is read off a real
 * event rather than off the docs, because the docs do not describe it. */
/* Event #8 of session 40c7c0d6-71d0-4116-b13f-0648a5abf5db, dumped exactly as the
 * browser receives it: `Event.model_dump(by_alias=True)`, which camel-cases every
 * field. That is the shape this shim has to read, and getting it wrong is
 * invisible to a fixture written in the database's snake_case -- which is what
 * every earlier version of this file used. */
const REAL_CONFIRMATION_EVENT = {
  content: {
    parts: [
      {
        functionCall: {
          id: 'adk-5f9fc2b0-b53f-46b6-b2cd-b87dfe1186fb',
          name: 'adk_request_confirmation',
          args: {
            originalFunctionCall: {
              id: 'call_491797',
              args: {
                question: 'Para qual cidade você deseja ver os filmes disponíveis?',
                options: ['Osasco', 'Niterói', 'Campinas'],
                default: 'Osasco',
                consequence: 'O vault possui informações sobre sessões e filmes ...',
              },
              name: 'ask_user',
            },
            toolConfirmation: {
              hint: 'Para qual cidade você deseja ver os filmes disponíveis?',
              confirmed: false,
              payload: { options: ['Osasco', 'Niterói', 'Campinas'], default: 'Osasco' },
            },
          },
        },
      },
    ],
    role: 'model',
  },
  author: 'text_summarizer',
};

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

/* The pending-card map the DOM tests render from. */
function cards() {
  const card = api.cardFromEvent(clone(REAL_CONFIRMATION_EVENT));
  return { [card.id]: card };
}

/* --- pure ------------------------------------------------------------------ */

test('a real confirmation event yields a card with its options', () => {
  const card = api.cardFromEvent(clone(REAL_CONFIRMATION_EVENT));
  assert.ok(card, 'the real event shape must be recognised');
  assert.strictEqual(card.id, 'adk-5f9fc2b0-b53f-46b6-b2cd-b87dfe1186fb');
  assert.deepStrictEqual(card.options, ['Osasco', 'Niterói', 'Campinas']);
  assert.strictEqual(card.fallback, 'Osasco');
  assert.match(card.hint, /Para qual cidade/);
});

test('the wire shape is camelCase, and snake_case is not what arrives', () => {
  /** The defect this file's fixture was hiding.
   *
   *  ADK serialises events with `by_alias=True`, which camel-cases every field.
   *  The browser therefore receives `part.functionCall`. An earlier version of
   *  this shim read `part.function_call` — the shape the *database* stores — and
   *  every test in this file passed, because every fixture was hand-written in the
   *  same wrong shape.
   *
   *  Found live: session 40c7c0d6-71d0-4116-b13f-0648a5abf5db, a turn that paused
   *  correctly with three options in the event, and a card rendering only the
   *  stock checkbox and Submit.
   *
   *  So the fixture is asserted to BE camelCase, which fails if someone "tidies"
   *  it back to snake_case, and a snake_case event is asserted to be understood
   *  too — that is the shape a future reader finds in the database and in any
   *  Python-side dump, and it should not render nothing.
   */
  const part = REAL_CONFIRMATION_EVENT.content.parts[0];
  assert.ok(part.functionCall, 'the fixture must be in the shape the browser receives');
  assert.strictEqual(part.function_call, undefined,
    'the fixture is in the database shape again; the shim reads the wire shape');

  const card = api.cardFromEvent(clone(REAL_CONFIRMATION_EVENT));
  assert.ok(card, 'the camelCase wire shape must be recognised');

  const snake = { content: { parts: [{ function_call: part.functionCall }] } };
  const alsoCard = api.cardFromEvent(snake);
  assert.ok(alsoCard, 'the database shape must be understood as well');
  assert.deepStrictEqual(alsoCard.options, card.options);
});

test('options come from the tool arguments, not the round-tripped payload', () => {
  /* If these were read from toolConfirmation.payload, a card whose payload the
   * user has already edited would render what is in the textarea instead of what
   * the agent offered -- and the buttons would no longer be the options. */
  const event = clone(REAL_CONFIRMATION_EVENT);
  event.content.parts[0].functionCall.args.toolConfirmation.payload = {
    options: ['Something', 'Else', 'Entirely'],
  };
  const card = api.cardFromEvent(event);
  assert.deepStrictEqual(card.options, ['Osasco', 'Niterói', 'Campinas']);
});

test('one option is not a menu', () => {
  const event = clone(REAL_CONFIRMATION_EVENT);
  event.content.parts[0].functionCall.args.originalFunctionCall.args.options = ['Osasco'];
  assert.strictEqual(api.cardFromEvent(event), null);
});

test('every option is drawable, however many there are', () => {
  /** The cap is gone on both sides, and this is what holds them there.
   *
   *  Both halves used to carry the same hardcoded 5 -- `ask_user.MAX_OPTIONS`
   *  and a private copy in this file. That looked like a guard and was a trap:
   *  if either moved alone, the tool would accept a question the shim refused to
   *  draw, and the user would see the stock checkbox-and-Submit form and conclude
   *  the agent had never asked. Nothing failed; the feature was just quietly
   *  absent, which is the shape of the bug this file's own camelCase test is
   *  about.
   *
   *  A long list costs a taller card. The cap it replaced cost an answer: the
   *  tool refused "all the films showing tonight" and picked one itself.
   */
  for (const n of [2, 3, 5, 6, 9, 11, 20, 47]) {
    const event = clone(REAL_CONFIRMATION_EVENT);
    event.content.parts[0].functionCall.args.originalFunctionCall.args.options =
      Array.from({ length: n }, (_, i) => `Option ${i + 1}`);

    const card = api.cardFromEvent(event);
    assert.ok(card, `${n} options must still be drawable`);
    assert.strictEqual(card.options.length, n);

    // ...and the buttons really are rendered, not just described.
    const { doc, row } = makeConfirmationDoc(card.id);
    assert.strictEqual(api.render(doc, { document: doc }, { [card.id]: card }), 1);
    const bar = doc.getElementById(api.buttonBarId(card.id));
    assert.strictEqual(bar.children.length, n, `${n} buttons expected`);
    assert.deepStrictEqual(
      bar.children.map((b) => b.attrs['data-choice']),
      Array.from({ length: n }, (_, i) => `Option ${i + 1}`)
    );
    assert.strictEqual(bar.parentNode, row);
  }
});

test('a non-string option is not renderable', () => {
  const event = clone(REAL_CONFIRMATION_EVENT);
  event.content.parts[0].functionCall.args.originalFunctionCall.args.options =
    ['Osasco', { nested: true }];
  assert.strictEqual(api.cardFromEvent(event), null);
});

test('a card that has already been answered is a record, not a menu', () => {
  /* The bundle's own three values are "pending", "sending" and "sent", and it
   * gates on `!== "sent" && !== "sending"`. Re-rendering buttons onto a submitted
   * card would let a user answer a question that has already been answered. */
  for (const status of ['sent', 'sending']) {
    const event = clone(REAL_CONFIRMATION_EVENT);
    event.content.parts[0].functionCall.responseStatus = status;
    assert.strictEqual(api.cardFromEvent(event), null, status + ' must not be drawable');
  }
  const pending = clone(REAL_CONFIRMATION_EVENT);
  pending.content.parts[0].functionCall.responseStatus = 'pending';
  assert.ok(api.cardFromEvent(pending), 'pending is the one drawable state');
});

test('an ordinary tool call is not a confirmation', () => {
  assert.strictEqual(
    api.cardFromEvent({ content: { parts: [{ function_call: { id: 'x', name: 'note_read' } }] } }),
    null
  );
  assert.strictEqual(api.cardFromEvent({}), null);
  assert.strictEqual(api.cardFromEvent(null), null);
});

test('a /run array of events yields every pending card', () => {
  const body = [
    { content: { parts: [{ text: 'hello' }] } },
    clone(REAL_CONFIRMATION_EVENT),
    {
      content: {
        parts: [
          {
            function_call: {
              id: 'adk-second',
              name: 'adk_request_confirmation',
              args: {
                originalFunctionCall: {
                  args: { question: 'q', options: ['a', 'b'], default: 'a' },
                },
              },
            },
          },
        ],
      },
    },
  ];
  const cards = api.cardsFromBody(body);
  assert.deepStrictEqual(Object.keys(cards).sort(), ['adk-second', 'adk-5f9fc2b0-b53f-46b6-b2cd-b87dfe1186fb'].sort());
});

test('an SSE stream yields the same card as the plain array', () => {
  const frame = 'data: ' + JSON.stringify(REAL_CONFIRMATION_EVENT) + '\n\n';
  const cards = api.cardsFromSse(frame);
  assert.ok(cards['adk-5f9fc2b0-b53f-46b6-b2cd-b87dfe1186fb']);
});

test('an SSE keepalive is not an error', () => {
  /* The stream carries non-JSON lines. A throw here would break rendering for
   * every confirmation in the session. */
  const text = ': keepalive\n\ndata: [DONE]\n\ndata: {not json\n\n';
  assert.deepStrictEqual(api.cardsFromResponseText(text), {});
});

test('an empty or non-JSON body is not an error', () => {
  assert.deepStrictEqual(api.cardsFromResponseText(''), {});
  assert.deepStrictEqual(api.cardsFromResponseText('   '), {});
  assert.deepStrictEqual(api.cardsFromResponseText('<html>500</html>'), {});
});

/* --- a DOM, small enough to fit here -------------------------------------- */

/* Not a general DOM implementation: exactly the surface the shim touches, and no
 * more, so that a change to the shim which needs something else fails loudly
 * rather than against a mock that quietly pretends. */
function findById(node, id) {
  if (node.id === id) return node;
  for (const child of node.children) {
    const hit = findById(child, id);
    if (hit) return hit;
  }
  return null;
}

function makeDoc(ids) {
  const byId = new Map(Object.entries(ids || {}));

  function makeEl(tag) {
    return {
      tagName: tag,
      id: '',
      className: '',
      textContent: '',
      attrs: {},
      children: [],
      parentNode: null,
      listeners: {},
      setAttribute(name, value) { this.attrs[name] = value; },
      get firstElementChild() { return this.children[0] || null; },
      appendChild(child) {
        child.parentNode = this;
        this.children.push(child);
        return child;
      },
      insertBefore(child, reference) {
        child.parentNode = this;
        /* A faithful insertBefore: an unshift-on-append "implementation" would
         * place the node correctly by accident and the placement assertion below
         * would prove nothing about the reference node being passed right. */
        const at = reference ? this.children.indexOf(reference) : -1;
        if (at === -1) throw new Error('insertBefore: reference node is not a child');
        this.children.splice(at, 0, child);
        return child;
      },
      addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); },
      closest(selector) {
        assert.strictEqual(selector, 'label', 'the shim only ever calls closest("label")');
        for (let n = this; n; n = n.parentNode) {
          if (n.tagName === selector) return n;
        }
        return null;
      },
      click() {
        (this.listeners.click || []).forEach((fn) => fn({ stopPropagation() {} }));
      },
      descendants() {
        return this.children.flatMap((c) => [c, ...c.descendants()]);
      },
      querySelectorAll(selector) {
        assert.strictEqual(selector, 'button', 'the shim only ever selects buttons');
        return this.descendants().filter((c) => c.tagName === 'button');
      },
    };
  }

  const doc = {
    root: makeEl('body'),
    createElement: makeEl,
    /* Walks the tree, as the real one does -- and it has to. The shim inserts
     * the button row and then looks it up by id to decide it is already done;
     * a registry-only stub would never find it and would insert a second copy on
     * every re-render, which is a property of the stub rather than of the code. */
    getElementById: (id) => findById(doc.root, id) || byId.get(id) || null,
    register: (el) => { byId.set(el.id, el); return el; },
  };
  return doc;
}

/* The card as the bundle's `iRe` renders it: hint, a read-only JSON viewer, then
 * a row holding the Confirm checkbox, then Submit. Only the parts the shim looks
 * for are modelled -- that is the point of the guard on the checkbox id.
 *
 * The card is appended to a root element because it lives in the document in
 * production, and the shim inserts the button row as its *sibling*. A detached
 * card would make every insert fail, which would be a property of the stub rather
 * than of the code under test. */
function makeConfirmationDoc(functionCallId) {
  const doc = makeDoc();
  const card = doc.root.appendChild(doc.createElement('div'));
  card.className = 'confirmation-card';

  const viewer = card.appendChild(doc.createElement('div'));
  viewer.className = 'json-viewer';

  const row = card.appendChild(doc.createElement('div'));
  const label = row.appendChild(doc.createElement('label'));
  const box = label.appendChild(doc.createElement('input'));
  box.id = api.checkboxId(functionCallId);
  doc.register(box);
  label.appendChild(doc.createElement('span')).textContent = 'Confirmed';

  const submit = card.appendChild(doc.createElement('button'));
  submit.textContent = ' Submit ';

  return { doc, card, row, label, box, submit };
}

test('the checkbox id follows the bundle scheme', () => {
  /* If ADK renames this, the shim finds nothing and renders nothing -- silently,
   * because there is no error path. Asserting the exact scheme makes that a test
   * failure instead. */
  assert.strictEqual(api.checkboxId('adk-x'), 'confirmed-checkbox-standalone-adk-x');
});

test('options render as one button each, in order, above the checkbox', () => {
  const id = 'adk-5f9fc2b0-b53f-46b6-b2cd-b87dfe1186fb';
  const { doc, row, label, box } = makeConfirmationDoc(id);
  const drawn = api.render(doc, { document: doc }, cards());

  assert.strictEqual(drawn, 1);
  const bar = doc.getElementById(api.buttonBarId(id));
  assert.ok(bar, 'the button row must exist');
  assert.strictEqual(bar.attrs['data-adk-confirm-options'], '3');
  assert.deepStrictEqual(bar.children.map((b) => b.textContent), ['Osasco', 'Niterói', 'Campinas']);
  assert.deepStrictEqual(bar.children.map((b) => b.attrs['data-choice']), ['Osasco', 'Niterói', 'Campinas']);
  /* Inside the card, immediately above the Confirm checkbox -- so the buttons sit
   * under the question rather than above it. Asserted against the real parents:
   * an element in the wrong tree compares as index -1 and would pass any ordering
   * assertion, which is how a placement bug hides. */
  assert.strictEqual(bar.parentNode, row, 'the button row belongs to the checkbox row');
  assert.deepStrictEqual(row.children, [bar, label], 'buttons, then the Confirm row');
  assert.strictEqual(label.children[0], box);
});

test('rendering twice does not double the buttons', () => {
  /* The shim re-renders on every DOM mutation, so this is the normal case rather
   * than an edge one. */
  const id = 'adk-5f9fc2b0-b53f-46b6-b2cd-b87dfe1186fb';
  const { doc, row } = makeConfirmationDoc(id);
  const pending = cards();
  assert.strictEqual(api.render(doc, { document: doc }, pending), 1);
  assert.strictEqual(api.render(doc, { document: doc }, pending), 0);
  assert.strictEqual(row.children.filter((c) => c.className === api.MARKER).length, 1);
});

test('a card that is not on screen yet is skipped, not an error', () => {
  const doc = makeDoc();
  assert.strictEqual(api.render(doc, { document: doc }, cards()), 0);
});

test('clicking an option records that choice and presses Submit', () => {
  const id = 'adk-5f9fc2b0-b53f-46b6-b2cd-b87dfe1186fb';
  const { doc, submit } = makeConfirmationDoc(id);
  const win = { document: doc };
  api.render(doc, win, cards());

  let submitted = 0;
  submit.addEventListener('click', () => { submitted++; });

  const bar = doc.getElementById(api.buttonBarId(id));
  bar.children[2].click(); // "Campinas" -- the last of three

  assert.deepStrictEqual(win[api.CHOICE_GLOBAL][id], { choice: 'Campinas' });
  assert.strictEqual(submitted, 1, 'the click must actually press Submit');
});

/* --- the other half: what the patched bundle then sends -------------------- */

const bundlePath = process.argv[2];

test('the patched bundle sends the clicked option as the confirmation payload', (t) => {
  if (!bundlePath) {
    t.skip('no patched bundle supplied');
    return;
  }
  const source = fs.readFileSync(bundlePath, 'utf8');

  /* Take the real patched prefix of `onSend`, up to and including the `a` it
   * builds, and run it. That is the byte sequence the Docker build installs --
   * not a re-typed copy of it, which would test the test.
   *
   * The slice starts at the `{` opening the confirmation branch, because
   * `onSend(){...}` on its own is a method body and not a valid function body: the
   * outer braces would have to be closed, and nothing here should be guessing at
   * where they end. */
  const start = source.indexOf('onSend(){');
  assert.ok(start !== -1, 'could not find onSend in the bundle');
  const end = source.indexOf('this.functionCall.responseStatus', start);
  assert.ok(end !== -1, 'could not find the end of the confirmation branch');
  const head = source.slice(start, end);
  const branch = head.indexOf('isConfirmationRequest)');
  assert.ok(branch !== -1, 'could not find the isConfirmationRequest branch');
  const prefix = head.slice(head.indexOf('{', branch) + 1);

  function responseFor(choice) {
    const win = {};
    if (choice !== undefined) win[api.CHOICE_GLOBAL] = { 'adk-5f9fc2b0-b53f-46b6-b2cd-b87dfe1186fb': choice };
    const self = {
      confirmationModel: {
        confirmed: false, // unticked, exactly as the stock UI leaves it
        payload: JSON.stringify(
          REAL_CONFIRMATION_EVENT.content.parts[0].functionCall.args.originalFunctionCall.args
        ),
      },
      functionCall: { id: 'adk-5f9fc2b0-b53f-46b6-b2cd-b87dfe1186fb', name: 'adk_request_confirmation' },
    };
    return new Function('window', prefix + '\nreturn a;').call(self, win);
  }

  /* With a button clicked: the choice travels, and it travels *confirmed* -- a
   * button click is a confirmation, while a bare Submit is not, and ask_user
   * reports those two cases differently. */
  const clicked = responseFor({ choice: 'Campinas' });
  assert.strictEqual(clicked.confirmed, true);
  assert.deepStrictEqual(clicked.payload, { choice: 'Campinas' });

  /* Without one: byte-for-byte the stock behaviour, because `o` is left as the
   * fallback. A card the shim did not draw still behaves as before. */
  const untouched = responseFor(undefined);
  assert.strictEqual(untouched.confirmed, false);
  assert.deepStrictEqual(untouched.payload.options, ['Osasco', 'Niterói', 'Campinas']);
  assert.deepStrictEqual(untouched.payload.default, 'Osasco');
});