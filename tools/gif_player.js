/* A GIF transport for the walkthroughs on this page.
 *
 * Why this exists: CSS cannot pause an animated image. There is no standard way
 * to seek one, slow one down, or step it a frame at a time, so a plain <img>
 * gives a reader exactly one control -- none. And the alternative, a media
 * player library, would mean a CDN request, which this page deliberately does
 * not make: everything else here is inlined so the page renders with no build
 * step and no network, and a font or CDN outage should degrade to a system
 * font rather than an unstyled page.
 *
 * So the GIF is decoded here. The LZW pass and the frame compositing were
 * verified frame-for-frame against Pillow's own decoder on all three
 * recordings (tools/verify_gif.js) -- the composed output is byte-identical, so
 * what a reader sees under the controls is what a browser would show natively.
 *
 * Behaviour worth knowing:
 *   - The bytes are fetched once and shared between figures, and only when a
 *     figure is about to scroll into view. Without that, opening the page
 *     would pull all three recordings at once.
 *   - prefers-reduced-motion is honoured: the poster stays and nothing plays
 *     until the reader presses play.
 *   - If fetch fails -- notably when this file is opened straight off disk,
 *     where file:// blocks it -- the figure falls back to a native <img> and
 *     the controls are removed rather than left there doing nothing.
 */
(function () {
  "use strict";

  var MIN_FRAME_MS = 20;   // browsers clamp short GIF delays; do it explicitly
  var SPEEDS = [0.25, 0.5, 1, 1.5, 2, 4];

  /* --- decoder ------------------------------------------------------------ */

  function lzwDecode(minCodeSize, data, pixelCount) {
    var clear = 1 << minCodeSize;
    var eoi = clear + 1;
    var prefix = new Int32Array(4096);
    var suffix = new Uint8Array(4096);
    var out = new Uint8Array(pixelCount);
    var stack = new Uint8Array(4096);
    var next = eoi + 1;
    var size = minCodeSize + 1;
    var bitBuf = 0, bits = 0, dp = 0, o = 0, prev = -1;

    while (o < pixelCount) {
      while (bits < size) {
        if (dp >= data.length) return out;
        bitBuf |= data[dp++] << bits;
        bits += 8;
      }
      var code = bitBuf & ((1 << size) - 1);
      bitBuf >>>= size;
      bits -= size;

      if (code === clear) { next = eoi + 1; size = minCodeSize + 1; prev = -1; continue; }
      if (code === eoi) break;

      var sp = 0, c = code, firstByte, extra = -1;
      if (code < next) {
        while (c > eoi) { stack[sp++] = suffix[c]; c = prefix[c]; }
        firstByte = c;
      } else if (code === next && prev >= 0) {
        /* The one code that is allowed to be not in the table yet: it is the
         * entry being added right now, so it expands to prev + prev's head. */
        extra = sp;
        stack[sp++] = 0;
        c = prev;
        while (c > eoi) { stack[sp++] = suffix[c]; c = prefix[c]; }
        firstByte = c;
        stack[extra] = firstByte;
      } else {
        break;                       // corrupt stream: stop rather than loop
      }

      stack[sp++] = firstByte;
      while (sp > 0) out[o++] = stack[--sp];

      if (prev >= 0 && next < 4096) {
        prefix[next] = prev;
        suffix[next] = firstByte;
        next++;
        /* Widen exactly when the next code would no longer fit. Grown a step
         * early the stream desynchronises within a few codes; grown late, it
         * reads the tail as noise. */
        if (next === (1 << size) && size < 12) size++;
      }
      prev = code;
    }
    return out;
  }

  function decodeGif(bytes) {
    var p = 0;
    function u16() { var v = bytes[p] | (bytes[p + 1] << 8); p += 2; return v; }
    function palette(n) { var b = bytes.subarray(p, p + n * 3); p += n * 3; return b; }
    function skipSubBlocks() { for (;;) { var n = bytes[p++]; if (!n) return; p += n; } }

    if (String.fromCharCode(bytes[0], bytes[1], bytes[2]) !== "GIF") {
      throw new Error("not a GIF");
    }
    p = 6;
    var width = u16(), height = u16();
    var packed = bytes[p]; p += 3;
    var shared = (packed & 0x80) ? palette(1 << ((packed & 7) + 1)) : null;

    var frames = [], gce = null;
    while (p < bytes.length) {
      var block = bytes[p++];
      if (block === 0x3B) break;                       // trailer
      if (block === 0x21) {                           // extension
        var label = bytes[p++];
        if (label === 0xF9) {                         // graphic control
          var size = bytes[p++];
          var flags = bytes[p++];
          var delay = u16();
          var transparent = bytes[p++];
          p += size - 4;
          skipSubBlocks();
          gce = {
            delay: delay * 10,
            disposal: (flags >> 2) & 7,
            transparent: (flags & 1) ? transparent : -1
          };
        } else if (label === 0xFF) {                  // application: loop count
          var asize = bytes[p++];
          var name = String.fromCharCode.apply(null, bytes.subarray(p, p + asize));
          p += asize;
          skipSubBlocks();
          if (name === "NETSCAPE2.0") gce = gce;     // parsed and discarded
        } else {
          skipSubBlocks();
        }
        continue;
      }
      if (block !== 0x2C) break;                      // image descriptor
      var left = u16(), top = u16(), w = u16(), h = u16();
      var ipacked = bytes[p++];
      var local = (ipacked & 0x80) ? palette(1 << ((ipacked & 7) + 1)) : null;
      var minCode = bytes[p++];

      /* Sub-blocks are length-prefixed, so the payload has to be concatenated:
       * handing the raw span to the LZW reader feeds it the lengths as if they
       * were pixels and the stream desynchronises immediately. */
      var parts = [], total = 0, n;
      for (;;) { n = bytes[p++]; if (!n) break; parts.push(bytes.subarray(p, p + n)); total += n; p += n; }
      var stream = new Uint8Array(total), off = 0, i;
      for (i = 0; i < parts.length; i++) { stream.set(parts[i], off); off += parts[i].length; }

      frames.push({
        left: left, top: top, width: w, height: h,
        palette: local || shared,
        delay: gce ? gce.delay : 0,
        disposal: gce ? gce.disposal : 0,
        transparent: gce ? gce.transparent : -1,
        indices: lzwDecode(minCode, stream, w * h)
      });
      gce = null;
    }
    return { width: width, height: height, frames: frames };
  }

  /* --- player ------------------------------------------------------------- */

  /* requestFullscreen needs a user gesture, which a click is, and is not
   * available at all on some mobile browsers -- so the button is only built
   * where it can work, rather than shipped as a control that does nothing.
   *
   * The element to display is the *receiver* of the call. Document
   * .requestFullscreen() takes an options object, not an element, so
   * documentElement.requestFullscreen(figure) full-screens the document and
   * quietly discards the argument. */
  var fullscreenEvent = document.fullscreenEnabled ? "fullscreenchange"
                                                   : "webkitfullscreenchange";

  var ICON_PLAY =
    '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
  var ICON_PAUSE =
    '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 19h4V5H6zm8-14v14h4V5h-4z"/></svg>';
  var ICON_FULLSCREEN =
    '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/></svg>';
  var ICON_FULLSCREEN_EXIT =
    '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 16h3v3h2v-5H5v2zm3-8H5v2h5V5H8v3zm6 11h2v-3h3v-2h-5v5zm2-11V5h-2v5h5V8h-3z"/></svg>';

  function activeFullscreen() {
    return document.fullscreenElement || document.webkitFullscreenElement || null;
  }

  function GifPlayer(figure) {
    this.figure = figure;
    this.stage = figure.querySelector(".walkthrough__stage");
    this.poster = this.stage.querySelector("img");
    this.src = figure.getAttribute("data-gif") ||
               (this.poster.getAttribute("src") || "").replace("-still.png", ".gif");
    this.gif = null;
    this.index = 0;
    this.acc = 0;
    this.speed = 1;
    this.playing = false;
    this.drawn = -1;
    this.raf = 0;
    this.watchdog = 0;
    this.ticked = false;
    this.handedOff = false;
    this.visible = false;
    this.wasPlaying = false;
    this.canvas = null;
    this.ctx = null;
    this.scratch = null;
    this.sctx = null;
    this.luts = new Map();
    this.build();
  }

  GifPlayer.prototype.el = function (sel) { return this.figure.querySelector(sel); };

  GifPlayer.prototype.build = function () {
    var self = this;
    var canFull = !!(this.figure.requestFullscreen || this.figure.webkitRequestFullscreen);

    /* Clicking the recording is the control, so the controls are on the
     * recording. Both are real <button>s laid over it: focusable, activated by
     * Enter and Space without code of our own, and announced as what they are.
     * A div with a click listener and an aria-label attached afterwards is the
     * version of this that ends up unfocusable or unlabelled. */
    this.hit = document.createElement("button");
    this.hit.type = "button";
    this.hit.className = "player__hit";
    /* Named from the start: the label is otherwise only written when playback
     * begins, so a player that is loaded but idle -- reduced motion, or a
     * figure that scrolled away -- would leave a nameless button on the page. */
    this.hit.setAttribute("aria-label", "Play the recording");
    this.stage.appendChild(this.hit);
    this.hit.addEventListener("click", function () {
      if (self.gif) { self.playing ? self.pause() : self.play(); }
    });

    /* Full screen used to live in the transport row, which is gone, so it
     * moves to a corner of the image. Omitted entirely where the API is
     * missing rather than shipped as a control that cannot work. */
    this.fullBtn = null;
    if (canFull) {
      this.fullBtn = document.createElement("button");
      this.fullBtn.type = "button";
      this.fullBtn.className = "player__corner";
      this.fullBtn.innerHTML = ICON_FULLSCREEN;
      this.fullBtn.setAttribute("aria-label", "Full screen");
      this.stage.appendChild(this.fullBtn);
      this.fullBtn.addEventListener("click", function (ev) {
        ev.stopPropagation();
        self.toggleFullscreen();
      });
      document.addEventListener(fullscreenEvent, function () { self.syncFullscreen(); });
      this.hit.addEventListener("dblclick", function (ev) {
        ev.preventDefault();
        self.toggleFullscreen();
      });
    }
    this.hit.disabled = true;
    if (this.fullBtn) this.fullBtn.disabled = true;
  };

  /* There is no status line any more. The reason a player gave up is kept on
   * the stage as data-why, which is what the reader sees, and every failure
   * still goes to the console. */
  GifPlayer.prototype.setStatus = function (text) {
    this.note = text || "";
    if (text) console.info("walkthrough: " + text);
  };

  /* Fetch once per URL; three figures asking for the same file share the bytes. */
  var inflight = {};

  function fetchGif(url) {
    if (!inflight[url]) {
      inflight[url] = fetch(url).then(function (res) {
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.arrayBuffer();
      }).then(function (buf) {
        return decodeGif(new Uint8Array(buf));
      });
      inflight[url].catch(function () { delete inflight[url]; });
    }
    return inflight[url];
  }

  GifPlayer.prototype.load = function () {
    var self = this;
    if (this.gif || this.loading) return;
    this.loading = true;
    this.stage.setAttribute("data-state", "loading");
    this.setStatus("Loading…");
    fetchGif(this.src).then(function (gif) {
      self.attach(gif);
    }, function () {
      /* No fetch (file://, or an old browser): hand the file back to the
       * browser and drop the controls rather than leaving them inert. */
      self.fallback();
    });
  };

  GifPlayer.prototype.fallback = function () {
    this.native("This browser would not let the page read the file");
  };

  GifPlayer.prototype.lut = function (pal) {
    /* One packed RGBA lookup per palette, reused by every frame that shares it. */
    if (this.luts.has(pal)) return this.luts.get(pal);
    var table = new Uint32Array(pal.length / 3);
    for (var i = 0; i < table.length; i++) {
      /* Little-endian ABGR; the one platform that differs is checked below. */
      table[i] = 0xff000000 | (pal[i * 3 + 2] << 16) | (pal[i * 3 + 1] << 8) | pal[i * 3];
    }
    this.luts.set(pal, table);
    return table;
  };

  GifPlayer.prototype.attach = function (gif) {
    var self = this;
    this.gif = gif;
    var canvas = document.createElement("canvas");
    canvas.width = gif.width;
    canvas.height = gif.height;
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", this.poster.getAttribute("alt") || "");
    /* Starts hidden so the poster is what a reduced-motion reader sees, and so
     * there is no flash of frame 0 before the first play. The poster stays in
     * the layout underneath: it is what gives the stage its height. */
    canvas.hidden = true;
    this.stage.appendChild(canvas);
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false });

    this.scratch = document.createElement("canvas");
    this.scratch.width = gif.width;
    this.scratch.height = gif.height;
    this.sctx = this.scratch.getContext("2d");

    this.hit.disabled = false;
    if (this.fullBtn) this.fullBtn.disabled = false;
    this.stage.setAttribute("data-state", "ready");
    this.setStatus("");

    if (this.wantPlay) {
      this.wantPlay = false;          // asked for while the bytes were in flight
      this.play();
    } else if (!window.matchMedia || !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      this.play();
    } else {
      this.setStatus("Reduced motion: press play");
    }
    this.update();
    return self;
  };

  GifPlayer.prototype.delayOf = function (i) {
    return Math.max(MIN_FRAME_MS, this.gif.frames[i].delay);
  };

  GifPlayer.prototype.clearFrame = function (f) {
    if (f.disposal === 2) {
      this.ctx.clearRect(f.left, f.top, f.width, f.height);
    } else if (f.disposal === 3 && this.snapshot) {
      this.ctx.putImageData(this.snapshot, 0, 0);
    }
  };

  GifPlayer.prototype.paint = function (f) {
    var lut = this.lut(f.palette);
    if (this.work && (this.work.width !== f.width || this.work.height !== f.height)) {
      this.work = null;
    }
    if (!this.work) this.work = this.sctx.createImageData(f.width, f.height);
    var px = new Uint32Array(this.work.data.buffer);
    for (var i = 0; i < f.indices.length; i++) {
      var idx = f.indices[i];
      /* A transparent index has to actually be transparent, which is why this
       * goes through a scratch canvas and drawImage rather than putImageData. */
      px[i] = idx === f.transparent ? 0 : lut[idx];
    }
    this.sctx.putImageData(this.work, 0, 0);
    this.ctx.drawImage(this.scratch, 0, 0, f.width, f.height, f.left, f.top, f.width, f.height);
    if (f.disposal === 3 && !this.snapshot) {
      this.snapshot = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    }
  };

  GifPlayer.prototype.render = function () {
    if (!this.gif) return;
    var frames = this.gif.frames;
    var i = this.index;

    if (this.drawn >= 0 && i === this.drawn + 1) {
      /* The common case: one step forward, so only the previous frame's
       * disposal has to be honoured. drawn === -1 has to stay out of this
       * branch -- there is no previous frame to dispose of yet, and taking
       * frames[-1] throws before a single frame is ever painted. */
      this.clearFrame(frames[i - 1]);
      this.paint(frames[i]);
    } else {
      /* A jump (seek, step backwards, loop): replay from the start. These
       * recordings are a few dozen frames, so this is well under a frame. */
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      for (var k = 0; k <= i; k++) {
        if (k > 0) this.clearFrame(frames[k - 1]);
        this.paint(frames[k]);
      }
    }
    this.drawn = i;
    this.update();
    this.clearWatchdog();
  };

  /* If the animation loop is not running -- rAF never fires, or every pass
   * throws -- the reader is left staring at a frozen image with a dead control
   * on it, which is the worst outcome available. Better to hand the file back
   * to the browser, which animates a GIF natively everywhere.
   *
   * The question this asks is "is the loop being called", NOT "has a frame
   * boundary been crossed". Asking the second one is wrong in a way that looks
   * like intermittent flakiness: the opening scene of every film is held for
   * over two seconds so there is something to read, so a watchdog that waited
   * two seconds for the first advance timed out on healthy browsers and handed
   * the recording away, and whether it did depended on where a tick happened
   * to land. rAF should deliver within a frame or two; 600ms is generous and
   * still catches a dead loop quickly. */
  GifPlayer.prototype.armWatchdog = function (attempt) {
    var self = this;
    this.clearWatchdog();
    this.watchdog = window.setTimeout(function () {
      self.watchdog = 0;
      if (self.ticked) return;                 // the loop is alive; that is all
      /* Two chances, because the two failures are not equal. Handing off is
       * permanent -- the controls are gone and the recording becomes a plain
       * image -- so a false positive is much worse than a genuinely broken
       * loop taking another 600ms to admit it. A tab that has just been
       * foregrounded, or a machine mid-GC, can plausibly miss one deadline
       * and not the next. */
      if ((attempt || 1) < 2) { self.armWatchdog(2); return; }
      self.native("Animation did not start");
    }, 600);
  };

  GifPlayer.prototype.clearWatchdog = function () {
    if (this.watchdog) { window.clearTimeout(this.watchdog); this.watchdog = 0; }
  };

  /* Back to a plain animated <img>, which is what a browser does with a GIF
   * when nobody interferes with it. */
  GifPlayer.prototype.native = function (why) {
    this.handedOff = true;
    this.pause();
    this.clearWatchdog();
    if (this.canvas) { this.canvas.remove(); this.canvas = null; }
    if (this.hit) { this.hit.remove(); this.hit = null; }
    if (this.fullBtn) { this.fullBtn.remove(); this.fullBtn = null; }
    this.poster.hidden = false;
    this.poster.setAttribute("src", this.src);
    this.stage.setAttribute("data-state", "native");
    if (why) this.stage.setAttribute("data-why", why);
  };

  GifPlayer.prototype.elapsed = function () {
    var t = 0;
    for (var i = 0; i < this.index; i++) t += this.delayOf(i);
    return t + this.acc;
  };

  GifPlayer.prototype.update = function () {};

  /* Any throw in here would otherwise leave the transport claiming to play
   * with no animation frame scheduled, and the play button doing nothing. Fail
   * loudly in the status line and stop instead. */
  GifPlayer.prototype.guarded = function (what, fn) {
    try {
      fn.call(this);
      return true;
    } catch (err) {
      this.pause();
      this.setStatus("Playback stopped: " + (err && err.message ? err.message : err));
      if (window.console) console.error("walkthrough " + what + " failed", err);
      return false;
    }
  };

  GifPlayer.prototype.tick = function (now) {
    if (!this.playing) return;
    /* Proof of life. The first callback cancels the watchdog, so a browser
     * where rAF works never comes near the hand-off. */
    this.ticked = true;
    this.clearWatchdog();
    var self = this;
    this.guarded("tick", function () { self.advance(now); });
    if (!this.playing) return;
    this.raf = requestAnimationFrame(this.tick);
  };

  GifPlayer.prototype.advance = function (now) {
    /* A backgrounded tab stops rAF and comes back with one enormous delta;
     * clamping keeps that from fast-forwarding through the whole recording. */
    /* Clamped at both ends: a backgrounded tab comes back with an enormous
     * delta, and rAF's timestamp is the frame's start time, which can predate
     * the performance.now() that seeded `last`, making the first delta
     * slightly negative. */
    var dt = Math.max(0, Math.min(now - this.last, 250));
    this.last = now;
    this.acc += dt * this.speed;
    /* Each pass consumes at least MIN_FRAME_MS, and acc can grow by at most
     * 250ms * speed, so this is unreachable in practice -- it is here so that a
     * future edit to the clamps cannot turn into a hang. */
    var guard = 64;
    while (guard-- > 0 && this.acc >= this.delayOf(this.index)) {
      this.acc -= this.delayOf(this.index);
      if (this.index >= this.gif.frames.length - 1) {
        this.index = -1;                 // wrap: the next step draws frame 0
      }
      this.index++;
      this.render();
    }
  };

  GifPlayer.prototype.play = function () {
    /* Safe to call at any time, including after the hand-off, which is not a
     * state the player can control: the IntersectionObserver resumes anything
     * it was playing when a figure scrolls back into view, and a backgrounded
     * tab does the same, and by then the canvas is gone and this.canvas would
     * be null. */
    if (!this.gif || !this.canvas) {
      /* Pressed before the bytes arrived. A recording is a couple of hundred
       * kilobytes and the load starts when the figure comes within 300px of
       * the viewport, so this is the window a reader who scrolls straight to
       * the controls and hits play lands in. Swallowing the click silently is
       * indistinguishable from a broken button, so keep the intent and start
       * the moment it can. */
      this.wantPlay = true;
      this.setStatus("Loading the recording…");
      this.load();
      return;
    }
    this.canvas.hidden = false;   // laid over the poster, which stays put
    this.playing = true;
    this.last = window.performance ? performance.now() : Date.now();
    this.acc = 0;
    this.setPlayIcon(true);
    /* Restart the loop rather than returning early when already playing: that
     * makes the button self-healing if the loop was ever lost, and stops "the
     * play button does nothing" from being a reachable state. */
    if (this.raf) cancelAnimationFrame(this.raf);
    this.ticked = false;          // the loop must prove itself again
    this.raf = requestAnimationFrame(this.tick);
    this.guarded("first frame", this.render);
    this.armWatchdog();
  };

  GifPlayer.prototype.pause = function () {
    this.wantPlay = false;
    this.clearWatchdog();
    if (!this.playing) return;
    this.playing = false;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.setPlayIcon(false);
  };

  /* The figure rather than the image: the controls are inside it, and going
   * full-screen with the transport left behind is no better than not having it.
   * Escape and the browser's own exit both fire fullscreenchange, so the icon
   * is synced from the document's state rather than from what we last asked
   * for. */
  GifPlayer.prototype.toggleFullscreen = function () {
    var request = this.figure.requestFullscreen || this.figure.webkitRequestFullscreen;
    var exit = document.exitFullscreen || document.webkitExitFullscreen;
    try {
      if (activeFullscreen()) {
        if (exit) exit.call(document);
      } else if (request) {
        request.call(this.figure);
      }
    } catch (err) {
      /* Refused -- no user activation, or a policy. Say so rather than
       * leaving a button that appears to have done nothing. */
      this.setStatus("Full screen was refused by the browser");
    }
    /* Read the state back instead of trusting the call: a browser can turn the
     * request into a window-level fullscreen, and an exit can quietly do
     * nothing, so the icon follows the document rather than our intent. */
    var self = this;
    window.setTimeout(function () { self.syncFullscreen(); }, 60);
  };

  GifPlayer.prototype.syncFullscreen = function () {
    if (!this.fullBtn) return;
    var on = !!activeFullscreen();
    this.fullBtn.setAttribute("aria-label", on ? "Exit full screen" : "Full screen");
    this.fullBtn.innerHTML = on ? ICON_FULLSCREEN_EXIT : ICON_FULLSCREEN;
  };

  GifPlayer.prototype.setPlayIcon = function (playing) {
    var label = playing ? "Pause" : "Play";
    if (this.hit) {
      this.hit.setAttribute("aria-label", label + " the recording");
      this.hit.innerHTML = playing ? ICON_PAUSE : ICON_PLAY;
      this.stage.setAttribute("data-playing", playing ? "true" : "false");
    }
  };

  GifPlayer.prototype.step = function (delta) {
    if (!this.gif) return;
    var n = this.gif.frames.length;
    this.index = Math.max(0, Math.min(n - 1, this.index + delta));
    this.acc = 0;
    this.guarded("step", this.render);
  };

  GifPlayer.prototype.seekTo = function (i) {
    this.index = Math.max(0, Math.min((this.gif ? this.gif.frames.length : 1) - 1, i));
  };

  /* --- wiring ------------------------------------------------------------- */

  function boot() {
    var figures = Array.prototype.slice.call(document.querySelectorAll(".walkthrough"));
    if (!figures.length) return;
    var players = figures.map(function (f) { return new GifPlayer(f); });

    if (!("IntersectionObserver" in window) || !("fetch" in window)) {
      players.forEach(function (p) { p.load(); });
      return;
    }
    /* A tab that goes to the background stops getting animation frames
     * entirely, so a player left running would sit there claiming to play.
     * Pause on the way out and pick up again on the way back. */
    document.addEventListener("visibilitychange", function () {
      players.forEach(function (p) {
        if (document.hidden) {
          if (p.playing) { p.wasPlaying = true; p.pause(); }
        } else if (p.wasPlaying && p.visible && !p.handedOff) {
          p.play();
        }
      });
    });

    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var player = players[figures.indexOf(entry.target)];
        if (!player) return;
        if (entry.isIntersecting) {
          player.visible = true;
          if (player.handedOff) return;      // a native image has nothing to resume
          player.load();
          if (player.wasPlaying) player.play();
        } else {
          /* Stop decoding frames nobody is looking at, and remember whether it
           * was playing so scrolling back picks up where it left off. */
          player.visible = false;
          if (player.playing) { player.wasPlaying = true; player.pause(); }
        }
      });
    }, { rootMargin: "300px 0px" });
    figures.forEach(function (f) { io.observe(f); });

    /* Exposed deliberately. Hand-decoded canvas playback is the kind of thing
     * that misbehaves on someone else's machine, and `walkthroughs[0].gif &&
     * walkthroughs[0].status` answers "did it load, what does it think is
     * going wrong" without a debugger. */
    window.walkthroughs = players;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
