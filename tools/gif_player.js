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
    this.loop = true;
    this.playing = false;
    this.drawn = -1;
    this.raf = 0;
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
    var tpl = document.getElementById("gifPlayerControls");
    if (!tpl) return;
    var bar = tpl.content.cloneNode(true);
    /* Directly under the image, not at the end of the figure: the controls
     * belong to the recording, and the caption reads better after them. */
    this.stage.parentNode.insertBefore(bar, this.stage.nextSibling);
    this.seek = this.el("[data-role=seek], .player__seek");
    this.posOut = this.el("[data-role=pos]");
    this.durOut = this.el("[data-role=dur]");
    this.frameOut = this.el("[data-role=frame]");
    this.countOut = this.el("[data-role=count]");
    this.status = this.el("[data-role=status]");
    this.playBtn = this.el('[data-act="play"]');
    this.buttons = Array.prototype.slice.call(this.figure.querySelectorAll(".player__btn"));

    var self = this;
    this.figure.addEventListener("click", function (ev) {
      var btn = ev.target.closest ? ev.target.closest("[data-act]") : null;
      if (!btn) return;
      var act = btn.getAttribute("data-act");
      if (act === "play") { self.playing ? self.pause() : self.play(); }
      else if (act === "prev") { self.pause(); self.step(-1); }
      else if (act === "next") { self.pause(); self.step(1); }
      else if (act === "stop") { self.pause(); self.seekTo(0); self.render(); }
    });
    this.seek.addEventListener("input", function () {
      self.pause();
      self.seekTo(parseInt(self.seek.value, 10) || 0);
      self.render();
    });
    this.el("[data-role=speed]").addEventListener("change", function (ev) {
      self.speed = parseFloat(ev.target.value) || 1;
    });
    this.el("[data-role=loop]").addEventListener("change", function (ev) {
      self.loop = ev.target.checked;
    });

    this.buttons.forEach(function (b) { b.disabled = true; });
    this.seek.disabled = true;
    this.setStatus("Loads when it scrolls into view");
  };

  GifPlayer.prototype.setStatus = function (text) {
    if (this.status) this.status.textContent = text;
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
    this.stage.setAttribute("data-state", "native");
    this.poster.setAttribute("src", this.src);
    var bar = this.el(".player");
    if (bar) bar.remove();
    this.setStatus("Playing natively (controls unavailable)");
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

    this.total = gif.frames.reduce(function (sum, f) {
      return sum + Math.max(MIN_FRAME_MS, f.delay);
    }, 0);
    this.durOut.textContent = (this.total / 1000).toFixed(1) + "s";
    this.countOut.textContent = gif.frames.length;
    this.seek.max = gif.frames.length - 1;

    this.buttons.forEach(function (b) { b.disabled = false; });
    this.seek.disabled = false;
    this.stage.setAttribute("data-state", "ready");
    this.setStatus("");

    if (!window.matchMedia || !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
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

    if (i === this.drawn + 1) {
      /* The common case: one step forward, so only the previous frame's
       * disposal has to be honoured. */
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
  };

  GifPlayer.prototype.elapsed = function () {
    var t = 0;
    for (var i = 0; i < this.index; i++) t += this.delayOf(i);
    return t + this.acc;
  };

  GifPlayer.prototype.update = function () {
    if (!this.gif) return;
    if (document.activeElement !== this.seek) this.seek.value = this.index;
    this.posOut.textContent = (this.elapsed() / 1000).toFixed(1) + "s";
    this.frameOut.textContent = this.index + 1;
    this.seek.setAttribute("aria-valuetext",
      "frame " + (this.index + 1) + " of " + this.gif.frames.length);
  };

  GifPlayer.prototype.tick = function (now) {
    if (!this.playing) return;
    /* A backgrounded tab stops rAF and comes back with one enormous delta;
     * clamping keeps that from fast-forwarding through the whole recording. */
    var dt = Math.min(now - this.last, 250);
    this.last = now;
    this.acc += dt * this.speed;
    var guard = this.gif.frames.length * 2;
    while (guard-- > 0 && this.acc >= this.delayOf(this.index)) {
      this.acc -= this.delayOf(this.index);
      if (this.index >= this.gif.frames.length - 1) {
        if (!this.loop) {
          this.acc = 0;
          this.pause();
          this.update();
          return;
        }
        this.index = -1;                 // wrap: the next step draws frame 0
      }
      this.index++;
      this.render();
    }
    this.raf = requestAnimationFrame(this.tick);
  };

  GifPlayer.prototype.play = function () {
    if (!this.gif || this.playing) return;
    this.canvas.hidden = false;   // laid over the poster, which stays put
    this.playing = true;
    this.last = window.performance ? performance.now() : Date.now();
    this.acc = 0;
    this.render();
    this.setPlayIcon(true);
    this.raf = requestAnimationFrame(this.tick);
  };

  GifPlayer.prototype.pause = function () {
    if (!this.playing) return;
    this.playing = false;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.setPlayIcon(false);
  };

  GifPlayer.prototype.setPlayIcon = function (playing) {
    if (!this.playBtn) return;
    this.playBtn.setAttribute("aria-label", playing ? "Pause" : "Play");
    this.playBtn.innerHTML = playing
      ? '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 19h4V5H6zm8-14v14h4V5h-4z"/></svg>'
      : '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
  };

  GifPlayer.prototype.step = function (delta) {
    if (!this.gif) return;
    var n = this.gif.frames.length;
    this.index = Math.max(0, Math.min(n - 1, this.index + delta));
    this.acc = 0;
    this.render();
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
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var player = players[figures.indexOf(entry.target)];
        if (!player) return;
        if (entry.isIntersecting) {
          player.visible = true;
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
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
