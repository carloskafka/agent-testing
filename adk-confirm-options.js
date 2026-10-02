/*
 * Clickable options for ADK tool confirmations. See patch-adk-devui-confirm.py.
 *
 * The problem
 * -----------
 * ADK's bundled dev UI renders a tool confirmation as a checkbox, a read-only
 * JSON dump of the tool's arguments, and a Submit button. `ToolConfirmation.payload`
 * is typed `Optional[Any]` -- arbitrary JSON, not a schema -- and nothing in the
 * bundle reads `options` from it. The whole of `initForm` for a confirmation is:
 *
 *     this.confirmationModel.confirmed = ...args.toolConfirmation.confirmed || false
 *     this.confirmationModel.payload    = JSON.stringify(...args.originalFunctionCall.args)
 *
 * So `ask_user`'s options arrive as JSON text inside a code viewer. Measured on
 * session f0842db4-9d42-4911-8b34-255a3731021f: asked "Para qual cidade...", the
 * user was shown
 *
 *     {"question":"...","options":["Osasco","Campinas"],"default":"Osasco",...}
 *
 * and had to work out on their own that the first array was meant to be a menu.
 *
 * What this does
 * --------------
 * Two halves, and they are in two different files on purpose:
 *
 *   1. This file renders `options` as real buttons. It learns what to render by
 *      observing the `/run` responses the app already receives, so the data is
 *      the payload ADK actually sent rather than the DOM Angular happened to
 *      render -- which is the fragile version of this.
 *
 *   2. patch-adk-devui-confirm.py rewrites the single expression in the bundle
 *      that builds the response, so a button click sends `{choice: <label>}`.
 *      One string, asserted unique, refuses to patch a bundle that does not have
 *      it. DOM code cannot do this part: `confirmationModel` is an Angular
 *      component property with no DOM handle, and there is no payload textarea on
 *      the confirmation view to type into.
 *
 * Splitting it this way means each half can fail loudly on its own. A future ADK
 * that renames the anchor fails the Docker build; a future ADK that reshapes the
 * confirmation card leaves this file rendering nothing visible, which the
 * `data-adk-confirm-options` attribute and the node tests are there to catch.
 */
(function (global) {
  'use strict';

  var MARKER = 'adk-confirm-options';

  /* Where patch-adk-devui-confirm.py has the bundle read an explicit choice from.
   * Kept here so the two halves name the same global; asserted in the tests. */
  var CHOICE_GLOBAL = '__adkConfirmChoice';

  /* The bundle's own id scheme for the Confirm checkbox, from `iRe` in main-*.js:
   *   H("id", mQ("confirmed-checkbox-standalone-", e.functionCall.id))
   * so the function call's id is recoverable from the rendered card. */
  function checkboxId(functionCallId) {
    return 'confirmed-checkbox-standalone-' + functionCallId;
  }

  /* How many options are worth a button row. Matches ask_user.MAX_OPTIONS: below
   * two there is no fork (ask_user refuses those), and above five it is a form
   * rather than a decision. A confirmation from some other tool carrying six is
   * left as the checkbox-and-JSON the stock UI gives it. */
  var MIN_OPTIONS = 2;
  var MAX_OPTIONS = 5;

  /* --- pure: reading ADK's own payloads ---------------------------------- */

  /* An event array as `/run` returns it, or the `{event: ...}` envelope shape. */
  function eventsFrom(body) {
    if (Array.isArray(body)) return body;
    if (body && Array.isArray(body.events)) return body.events;
    if (body && typeof body === 'object' && body.content) return [body];
    return [];
  }

  /* One confirmation card's worth of renderable data, or null if this event is
   * not a confirmation we should draw buttons for. */
  function cardFromEvent(event) {
    if (!event || !event.content || !Array.isArray(event.content.parts)) return null;

    var call = null;
    for (var i = 0; i < event.content.parts.length; i++) {
      var part = event.content.parts[i];
      /* camelCase, not snake_case. The wire shape is `Event.model_dump(
       * by_alias=True)`, which camel-cases every field, so what the browser sees
       * is `part.functionCall` -- `part.function_call` is the shape the *database*
       * stores and the shape every test fixture in this repo was written in.
       *
       * Both are accepted, and that is not belt-and-braces: this shim reads what
       * came off the network, so camelCase is the case that actually occurs, and
       * the snake_case branch is here for the same reason the `events` table is
       * worth reading when something is wrong -- a shape arriving from somewhere
       * else should not silently render nothing.
       *
       * Found live on session 40c7c0d6-71d0-4116-b13f-0648a5abf5db: a paused turn
       * with three perfectly good options in the event, and the card showing only
       * the stock checkbox and Submit. Every fixture passed, because every fixture
       * was the shape this code was written against. See the test that pins both.
       */
      var fc = part && (part.functionCall || part.function_call);
      if (fc && fc.name === 'adk_request_confirmation') {
        call = fc;
        break;
      }
    }
    if (!call || !call.id) return null;

    var args = call.args || {};
    var original = args.originalFunctionCall || {};
    var toolArgs = original.args || {};

    /* The options live in the *tool's* arguments, because that is what ask_user
     * passes them as. They are deliberately NOT read from
     * `toolConfirmation.payload`: that is the round-tripped copy, and a card
     * whose payload has already been edited by the user must render what the
     * agent originally offered, not what is sitting in the textarea. */
    var options = toolArgs.options;
    if (!Array.isArray(options) || options.length < MIN_OPTIONS) return null;
    if (options.length > MAX_OPTIONS) return null;
    if (options.some(function (o) { return typeof o !== 'string' || !o; })) return null;

    /* Once submitted the card is a transcript entry, not a menu. The bundle's own
     * vocabulary for this is `responseStatus`, and its three values are "pending",
     * "sending" and "sent" -- it gates on `!== "sent" && !== "sending"`, so
     * "pending" is the only state in which offering buttons means anything. */
    if ((call.responseStatus || 'pending') !== 'pending') return null;

    return {
      id: call.id,
      options: options.slice(),
      fallback: typeof toolArgs.default === 'string' ? toolArgs.default : null,
      question: typeof toolArgs.question === 'string' ? toolArgs.question : '',
      hint: (args.toolConfirmation && args.toolConfirmation.hint) || '',
    };
  }

  function cardsFromBody(body) {
    var out = {};
    eventsFrom(body).forEach(function (event) {
      var card = cardFromEvent(event);
      if (card && !card.answered) out[card.id] = card;
    });
    return out;
  }

  /* SSE framing: `data: {...}` per line. Only the confirmation-shaped events are
   * worth pulling out, and JSON.parse of a non-JSON line is guarded rather than
   * assumed, because the stream also carries plain-text keepalives. */
  function cardsFromSse(text) {
    var out = {};
    String(text).split('\n').forEach(function (line) {
      if (line.indexOf('data:') !== 0) return;
      var chunk = line.slice(5).trim();
      if (!chunk || chunk === '[DONE]') return;
      var parsed;
      try { parsed = JSON.parse(chunk); } catch (e) { return; }
      Object.assign(out, cardsFromBody(parsed));
    });
    return out;
  }

  function cardsFromResponseText(text) {
    var trimmed = String(text).trim();
    if (!trimmed) return {};
    if (trimmed[0] === '{' || trimmed[0] === '[') {
      try { return cardsFromBody(JSON.parse(trimmed)); } catch (e) { /* fall through */ }
    }
    return cardsFromSse(text);
  }

  /* --- DOM: rendering ----------------------------------------------------- */

  function buttonBarId(functionCallId) {
    return MARKER + '-' + functionCallId;
  }

  /* The card is the confirmation panel the checkbox lives in. Walking up to the
   * element that also holds Submit is what puts the buttons and the button they
   * drive on screen together. */
  function cardRootOf(box) {
    var node = box;
    while (node && node.parentNode && node.parentNode !== undefined) {
      if (findSubmitIn(node)) return node;
      node = node.parentNode;
    }
    return box.parentNode || box;
  }

  function findSubmitIn(root) {
    var all = root.querySelectorAll ? root.querySelectorAll('button') : [];
    for (var i = 0; i < all.length; i++) {
      if (/^\s*submit\s*$/i.test(all[i].textContent || '')) return all[i];
    }
    return null;
  }

  function renderOne(doc, win, card) {
    if (doc.getElementById(buttonBarId(card.id))) return false; // already done

    var box = doc.getElementById(checkboxId(card.id));
    if (!box) return false; // the card has not been rendered yet

    var root = cardRootOf(box);

    var bar = doc.createElement('div');
    bar.id = buttonBarId(card.id);
    bar.className = MARKER;
    /* Counted, so a test (or a human with devtools open) can tell "no options"
     * from "the shim never ran". */
    bar.setAttribute('data-adk-confirm-options', String(card.options.length));
    bar.setAttribute('data-adk-confirm-question', card.question || card.hint || '');

    card.options.forEach(function (label) {
      var button = doc.createElement('button');
      button.type = 'button';
      button.className = MARKER + '-option';
      button.textContent = label;
      button.setAttribute('data-choice', label);
      button.addEventListener('click', function () {
        choose(win, card.id, label, doc);
      });
      bar.appendChild(button);
    });

    /* Placement: immediately above the Confirm checkbox's row, inside the card.
     *
     * The obvious choice -- insert as a sibling of the whole card -- puts the
     * buttons *above the question*, because the card opens with the hint. The row
     * is found by walking up from the checkbox to its <label>'s parent, which is
     * the one structural fact the bundle states outright in its template:
     *     label > input + span("Confirmed")
     * Falls back to appending inside the card if that shape is ever not what is
     * rendered, so a reshuffle degrades to "below the question" rather than to
     * "no buttons". */
    var label = typeof box.closest === 'function' ? box.closest('label') : null;
    var row = label && label.parentNode ? label.parentNode : null;
    var anchor = row || root;
    if (!anchor || typeof anchor.insertBefore !== 'function') return false;
    /* The reference node is a child of the anchor. When the row was found it is
     * that row's first element; when it was not, appending to the card is the
     * fallback. Passing `row` here would be the anchor as its own reference node,
     * which is a DOMException rather than a placement. */
    anchor.insertBefore(bar, row ? row.firstElementChild || null : null);
    return true;
  }

  function render(doc, win, cards) {
    var drawn = 0;
    Object.keys(cards).forEach(function (id) {
      if (renderOne(doc, win, cards[id])) drawn++;
    });
    return drawn;
  }

  /* The one thing this file is for: make the click mean something. The patched
   * bundle prefers `CHOICE_GLOBAL[id]` over the prefilled payload and forces
   * `confirmed` true, because a button click *is* a confirmation -- unlike a bare
   * Submit, which the stock UI offers and which means "I did not choose". */
  function choose(win, functionCallId, label, doc) {
    win[CHOICE_GLOBAL] = win[CHOICE_GLOBAL] || {};
    win[CHOICE_GLOBAL][functionCallId] = { choice: label };

    var box = doc.getElementById(checkboxId(functionCallId));
    if (!box) return false;
    var submit = findSubmitIn(cardRootOf(box));
    if (!submit) return false;
    submit.click();
    return true;
  }

  /* --- wiring: learning what to render ------------------------------------ */

  function merge(into, extra) {
    Object.keys(extra || {}).forEach(function (id) {
      if (!into[id]) into[id] = extra[id];
    });
    return into;
  }

  function hookFetch(win, state) {
    var original = win.fetch;
    if (typeof original !== 'function' || original[MARKER]) return false;

    function patched(input, init) {
      var response = original.call(this, input, init);
      return response.then(function (res) {
        var url = String((res && res.url) || (typeof input === 'string' ? input : ''));
        if (url.indexOf('/run') === -1) return res;
        /* clone(): the app still has to read this body itself. */
        try {
          res.clone().text().then(function (text) {
            merge(state.cards, cardsFromResponseText(text));
            render(win.document, win, state.cards);
          });
        } catch (e) { /* a response with no readable body is not our problem */ }
        return res;
      });
    }

    patched[MARKER] = true;
    win.fetch = patched;
    return true;
  }

  function start(win) {
    if (!win || win[MARKER]) return null;
    win[MARKER] = true;

    var state = { cards: {} };
    hookFetch(win, state);

    /* The card appears after the response resolves, so a single pass would race
     * it. Cheap enough to re-run on every mutation; each pass is a getElementById
     * per pending card and stops once the bar exists. */
    var tick = function () {
      try { render(win.document, win, state.cards); } catch (e) { /* keep going */ }
    };
    if (win.document && win.MutationObserver) {
      new win.MutationObserver(tick).observe(win.document.documentElement,
        { childList: true, subtree: true });
    }
    if (win.document && win.document.addEventListener) {
      win.document.addEventListener('DOMContentLoaded', tick);
    }
    tick();
    return state;
  }

  var api = {
    MARKER: MARKER,
    CHOICE_GLOBAL: CHOICE_GLOBAL,
    MIN_OPTIONS: MIN_OPTIONS,
    MAX_OPTIONS: MAX_OPTIONS,
    checkboxId: checkboxId,
    buttonBarId: buttonBarId,
    cardFromEvent: cardFromEvent,
    cardsFromBody: cardsFromBody,
    cardsFromSse: cardsFromSse,
    cardsFromResponseText: cardsFromResponseText,
    render: render,
    choose: choose,
    start: start,
  };

  /* Exported for node; in the browser this is the only global it adds besides
   * the patched bundle's read of CHOICE_GLOBAL. */
  if (typeof module === 'object' && module.exports) module.exports = api;
  global.__adkConfirmOptions = api;

  if (typeof window !== 'undefined' && window.document) start(window);
})(typeof window !== 'undefined' ? window : globalThis);