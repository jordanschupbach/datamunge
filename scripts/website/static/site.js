// The documentation website's page behaviour: theme, search, contents, navigation.
(function () {
  "use strict";
  var base = window.SITE_BASE || "";
  var root = document.documentElement;

  // -- Theme ------------------------------------------------------------------
  var toggle = document.querySelector(".theme-toggle");
  if (toggle) toggle.addEventListener("click", function () {
    var next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("site-theme", next); } catch (e) {}
  });

  // -- Mobile navigation --------------------------------------------------------
  var navBtn = document.querySelector(".nav-toggle");
  var scrim = document.querySelector(".scrim");
  function setNav(open) {
    var wasOpen = document.body.classList.contains("nav-open");
    document.body.classList.toggle("nav-open", open);
    if (navBtn) navBtn.setAttribute("aria-expanded", String(open));
    if (scrim) scrim.hidden = !open;
    if (open || wasOpen) {
      var region = open ? document.querySelector(".sidebar") : content;
      if (region) {
        setRegion(region);
        region.focus({ preventScroll: true });
      }
    }
  }
  if (navBtn) navBtn.addEventListener("click", function () { setNav(!document.body.classList.contains("nav-open")); });
  if (scrim) scrim.addEventListener("click", function () { setNav(false); });

  // Keep the current page visible in the sidebar.
  var current = document.querySelector(".sidebar a.current");
  if (current) current.scrollIntoView({ block: "center" });

  // -- On this page -------------------------------------------------------------
  var tocBox = document.querySelector(".pagetoc-inner");
  var content = document.querySelector(".content");
  if (tocBox && content) {
    var isReport = !!content.querySelector(".report");
    var tocSel = isReport ? ".report > h1[id]:not(.title), .report > h2[id], .content > h2[id]"
      : content.querySelector(".prose") ? ".prose h2[id], .prose h3[id]" : ".content h2[id]";
    var heads = Array.prototype.slice.call(content.querySelectorAll(tocSel));
    // Members: link their first section.
    heads = heads.filter(function (h) {
      if (h.id) return true;
      var next = h.nextElementSibling;
      if (next && next.id) { h.dataset.target = next.id; return true; }
      return false;
    });
    if (heads.length > 1) {
      var top = isReport ? "H1" : "H2";
      var html = '<div class="pagetoc-title">On this page</div><ul>';
      heads.forEach(function (h) {
        var id = h.dataset.target || h.id;
        var lvl = h.tagName === top || (isReport && h.parentNode === content) ? "l1" : "l2";
        html += '<li><a class="' + lvl + '" href="#' + encodeURIComponent(id) + '">' + escapeHtml(h.textContent) + "</a></li>";
      });
      tocBox.innerHTML = html + "</ul>";
      var links = tocBox.querySelectorAll("a");
      var spy = function () {
        var y = window.scrollY + 120, idx = 0;
        for (var i = 0; i < heads.length; i++) if (heads[i].getBoundingClientRect().top + window.scrollY <= y) idx = i;
        links.forEach(function (a, i) { a.classList.toggle("active", i === idx); });
      };
      window.addEventListener("scroll", throttle(spy, 80), { passive: true });
      spy();
    }
  }

  // -- Copy buttons ---------------------------------------------------------------
  document.querySelectorAll(".content pre").forEach(function (pre) {
    if (pre.classList.contains("output")) return;
    var b = document.createElement("button");
    b.type = "button"; b.className = "copy-btn"; b.textContent = "Copy";
    b.addEventListener("click", function () {
      var text = (pre.querySelector("code") || pre).innerText;
      var done = function () { b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1400); };
      if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, function () {});
    });
    pre.appendChild(b);
  });

  // -- Search -----------------------------------------------------------------------
  var input = document.getElementById("search");
  var list = document.getElementById("search-results");
  var index = null, loading = false, results = [], sel = -1;
  function load(then) {
    if (index) return then();
    if (loading) return;
    loading = true;
    var s = document.createElement("script");
    s.src = base + "search-index.js";
    s.onload = function () {
      index = (window.SEARCH_INDEX || []).map(function (e) {
        e.l = normalize(e.t);
        e.last = e.l.split("::").pop();
        var context = normalize(e.d || "");
        var slug = e.k === "report" ? e.u.split("/").pop().replace(/\.html$/, "") : "";
        e.words = new Set(tokens(e.l + " " + context + " " + normalize(slug)));
        e.hay = e.l + " " + context + " " + normalize(e.s || "");
        return e;
      });
      then();
    };
    document.head.appendChild(s);
  }
  var KIND_RANK = { namespace: 0, "class": 1, struct: 1, concept: 1, "function": 2, report: 2, guide: 1 };
  function normalize(text) {
    return text.toLowerCase().normalize("NFKD").replace(/\p{M}/gu, "");
  }
  function tokens(text) {
    return text.split(/[^\p{L}\p{N}]+/u).filter(Boolean);
  }
  // Bounded Damerau edit distance. A typo may insert, delete, replace or swap adjacent letters.
  function distance(a, b, limit) {
    if (Math.abs(a.length - b.length) > limit) return limit + 1;
    var older, previous = Array.from({ length: b.length + 1 }, function (_, i) { return i; });
    for (var i = 1; i <= a.length; i++) {
      var row = [i], best = i;
      for (var j = 1; j <= b.length; j++) {
        row[j] = Math.min(row[j - 1] + 1, previous[j] + 1,
          previous[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
        if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1])
          row[j] = Math.min(row[j], older[j - 2] + 1);
        best = Math.min(best, row[j]);
      }
      if (best > limit) return limit + 1;
      older = previous;
      previous = row;
    }
    return previous[b.length];
  }
  function score(e, q, words) {
    var s;
    if (e.last === q || e.l === q) s = 0;
    else if (e.last.startsWith(q)) s = 1;
    else if (e.l.startsWith(q)) s = 2;
    else if (e.last.includes(q)) s = 3;
    else if (e.l.includes(q)) s = 4;
    else if (words.length && words.every(function (word) { return e.words.has(word); })) s = 5;
    else if (words.length && words.every(function (word) { return e.hay.includes(word); })) s = 6;
    else {
      var edits = 0;
      if (!words.length) return -1;
      for (var word of words) {
        if (e.words.has(word)) continue;
        var limit = word.length >= 7 ? 2 : word.length >= 4 ? 1 : 0;
        var best = limit + 1;
        for (var candidate of e.words) {
          if (candidate.startsWith(word)) { best = 0; break; }
          if (limit) best = Math.min(best, distance(word, candidate, limit));
        }
        if (best > limit) return -1;
        edits += best;
      }
      s = 7 + edits;
    }
    var k = KIND_RANK[e.k]; if (k === undefined) k = 3;
    return s * 10 + k + e.l.length / 1000;
  }
  function run() {
    var q = normalize(input.value.trim());
    if (!q) { list.hidden = true; return; }
    var words = tokens(q);
    var found = [];
    for (var i = 0; i < index.length; i++) {
      var sc = score(index[i], q, words);
      if (sc >= 0) found.push([sc, index[i]]);
    }
    found.sort(function (a, b) { return a[0] - b[0]; });
    results = found.slice(0, 40).map(function (x) { return x[1]; });
    sel = results.length ? 0 : -1;
    render();
  }
  function render() {
    if (!results.length) {
      list.innerHTML = '<li class="sr-empty">Nothing matches.</li>';
    } else {
      list.innerHTML = results.map(function (e, i) {
        return '<li role="option" aria-selected="' + (i === sel) + '"><a href="' + base + e.u + '">' +
          '<span class="sr-kind">' + escapeHtml(e.k) + '</span><span class="sr-title">' + escapeHtml(e.t) + "</span>" +
          (e.k === "report" && e.d ? '<span class="sr-sum">' + escapeHtml(e.d) + "</span>" :
            e.s ? '<span class="sr-sum">' + escapeHtml(e.s) + "</span>" : "") + "</a></li>";
      }).join("");
    }
    list.hidden = false;
  }
  function move(d) {
    if (!results.length) return;
    sel = (sel + d + results.length) % results.length;
    render();
    var li = list.children[sel]; if (li) li.scrollIntoView({ block: "nearest" });
  }
  if (input && list) {
    input.addEventListener("focus", function () { load(function () { if (input.value) run(); }); });
    input.addEventListener("input", function () { load(run); });
    input.addEventListener("keydown", function (ev) {
      if (ev.key === "ArrowDown") { ev.preventDefault(); move(1); }
      else if (ev.key === "ArrowUp") { ev.preventDefault(); move(-1); }
      else if (ev.key === "Enter" && sel >= 0 && results[sel]) { location.href = base + results[sel].u; }
      else if (ev.key === "Escape") { input.value = ""; list.hidden = true; input.blur(); }
    });
    document.addEventListener("click", function (ev) {
      if (!ev.target.closest || !ev.target.closest(".search")) list.hidden = true;
    });
    document.addEventListener("keydown", function (ev) {
      var t = ev.target, typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
      if (ev.key === "/" && !typing && !ev.ctrlKey && !ev.metaKey) { ev.preventDefault(); input.focus(); }
      if (ev.key === "Escape") setNav(false);
    });
  }

  // -- Region focus and scrolling ---------------------------------------------------
  var regionSelector = ".topbar, .sidebar, main.content, .pagetoc";
  var regions = Array.from(document.querySelectorAll(regionSelector));
  var topbar = document.querySelector(".topbar");
  var activeRegion;
  regions.forEach(function (region) { region.tabIndex = -1; });

  function setRegion(region) {
    if (activeRegion === region) return;
    if (activeRegion) activeRegion.classList.remove("keyboard-region-active");
    activeRegion = region;
    if (region) region.classList.add("keyboard-region-active");
  }

  function regionRect(region) {
    if (region.matches(".pagetoc") && !region.querySelector("a")) return null;
    var rect = region.getBoundingClientRect();
    var ceiling = topbar && region !== topbar ? topbar.getBoundingClientRect().bottom : 0;
    var left = Math.max(0, rect.left), right = Math.min(root.clientWidth, rect.right);
    var top = Math.max(0, ceiling, rect.top), bottom = Math.min(root.clientHeight, rect.bottom);
    if (right <= left || bottom <= top) return null;
    return { left: left, right: right, top: top, bottom: bottom,
      x: (left + right) / 2, y: (top + bottom) / 2 };
  }

  function moveRegion(key) {
    var origin = activeRegion && regionRect(activeRegion);
    if (!origin && content) {
      setRegion(content);
      origin = regionRect(content);
    }
    if (!origin) return false;

    var horizontal = key === "h" || key === "l";
    var direction = key === "h" || key === "k" ? -1 : 1;
    var best, bestDistance = Infinity;
    regions.forEach(function (region) {
      var point = regionRect(region);
      if (!point || region === activeRegion) return;
      var separated = horizontal ?
        (direction < 0 ? point.right <= origin.left + 1 : point.left >= origin.right - 1) :
        (direction < 0 ? point.bottom <= origin.top + 1 : point.top >= origin.bottom - 1);
      var drawer = horizontal && navBtn && navBtn.getClientRects().length &&
        document.body.classList.contains("nav-open") &&
        (region.matches(".sidebar") || activeRegion.matches(".sidebar"));
      if (!separated && !drawer) return;
      var forward = (horizontal ? point.x - origin.x : point.y - origin.y) * direction;
      var sideways = horizontal ? point.y - origin.y : point.x - origin.x;
      if (forward <= 1) return;
      var distance = Math.hypot(forward, 2 * sideways);
      if (distance < bestDistance) { best = region; bestDistance = distance; }
    });
    if (!best) return false;
    closeHints();
    if (activeRegion.matches(".sidebar") && !best.matches(".sidebar")) setNav(false);
    setRegion(best);
    best.focus({ preventScroll: true });
    return true;
  }

  function trackRegion(ev) {
    var region = ev.target.closest && ev.target.closest(regionSelector);
    if (region) setRegion(region);
  }

  setRegion(content);
  document.addEventListener("focusin", trackRegion);
  document.addEventListener("pointerdown", trackRegion);

  function scrollRegion(key) {
    // Directional focus uses viewport geometry. Scrolling follows the selected
    // pane even while its drawer is sliding into view or it reaches a scroll limit.
    var closedDrawer = activeRegion && activeRegion.matches(".sidebar") &&
      navBtn && navBtn.getClientRects().length && !document.body.classList.contains("nav-open");
    if (!activeRegion || !activeRegion.getClientRects().length || closedDrawer) setRegion(content);
    if (!activeRegion) return;
    var scroller = activeRegion === content ? document.scrollingElement :
      activeRegion.querySelector(":scope > .sidebar-inner, :scope > .pagetoc-inner");
    if (scroller) scroller.scrollBy({ top: key === "j" ? 60 : -60, behavior: "instant" });
  }

  // -- Follow links and controls ----------------------------------------------------
  var hintKeys = "asdfghjkl";
  var hintLayer, hints = [], hintPrefix = "";
  var keyboardTarget = "a[href], area[href], button, summary, [role=button], [role=link], " +
    "input[type=button], input[type=submit], input[type=reset], input[type=image]";

  function hintPosition(target) {
    if (target.closest('[inert], [aria-disabled="true"], [aria-hidden="true"]') ||
        target.matches(":disabled")) return null;
    for (var rect of target.getClientRects()) {
      var left = Math.max(0, rect.left), right = Math.min(root.clientWidth, rect.right);
      var top = Math.max(0, rect.top), bottom = Math.min(root.clientHeight, rect.bottom);
      if (left >= right || top >= bottom) continue;
      var y = (top + bottom) / 2;
      for (var x of [left + (right - left) * .1, (left + right) / 2, right - (right - left) * .1]) {
        var hit = document.elementFromPoint(x, y);
        if (hit && target.contains(hit)) return { left: x, top: top };
      }
    }
    return null;
  }

  function closeHints() {
    if (!hintLayer) return;
    hintLayer.remove();
    hintLayer = null;
    hints = [];
    hintPrefix = "";
  }

  function openHints() {
    var targets = [];
    document.querySelectorAll(keyboardTarget).forEach(function (target) {
      var point = hintPosition(target);
      if (point) targets.push({ target: target, point: point });
    });
    if (!targets.length) return;
    var width = 1;
    while (Math.pow(hintKeys.length, width) < targets.length) width++;
    hintLayer = document.createElement("div");
    hintLayer.className = "follow-hints";
    hintLayer.setAttribute("aria-hidden", "true");
    hints = targets.map(function (item, index) {
      var keys = "";
      for (var i = 0; i < width; i++) {
        keys = hintKeys[index % hintKeys.length] + keys;
        index = Math.floor(index / hintKeys.length);
      }
      var marker = document.createElement("span");
      marker.textContent = keys;
      marker.style.left = item.point.left + "px";
      marker.style.top = item.point.top + "px";
      hintLayer.appendChild(marker);
      return { keys: keys, marker: marker, target: item.target };
    });
    document.body.appendChild(hintLayer);
    hints.forEach(function (hint) {
      var right = hint.marker.getBoundingClientRect().right;
      if (right > root.clientWidth) hint.marker.style.left = (root.clientWidth - hint.marker.offsetWidth) + "px";
    });
  }

  document.addEventListener("keydown", function (ev) {
    if (ev.defaultPrevented || ev.isComposing || ev.ctrlKey || ev.metaKey) return;
    if (ev.altKey) {
      var direction = /^[hjkl]$/i.test(ev.key) ? ev.key.toLowerCase()
        : /^Key[HJKL]$/.test(ev.code) ? ev.code.slice(-1).toLowerCase() : "";
      if (direction && !ev.shiftKey && moveRegion(direction)) ev.preventDefault();
      return;
    }
    var target = ev.target;
    if (target && (target.isContentEditable || (target.closest &&
        target.closest("input, textarea, select, [role=textbox], [role=combobox]")))) return;
    if (!hintLayer) {
      if (ev.key === "f" && !ev.repeat) {
        openHints();
        if (hintLayer) ev.preventDefault();
      } else if (ev.key === "j" || ev.key === "k") {
        ev.preventDefault();
        scrollRegion(ev.key);
      }
      return;
    }
    if (ev.key === "Tab") { closeHints(); return; }
    if (ev.key !== "Escape" && ev.key !== "Backspace" && ev.key.length !== 1) return;
    ev.preventDefault();
    ev.stopPropagation();
    if (ev.repeat) return;
    if (ev.key === "Escape") { closeHints(); return; }
    hintPrefix = ev.key === "Backspace" ? hintPrefix.slice(0, -1) : hintPrefix + ev.key.toLowerCase();
    var matches = hints.filter(function (hint) {
      var match = hint.keys.startsWith(hintPrefix);
      hint.marker.hidden = !match;
      return match;
    });
    if (!matches.length) closeHints();
    else if (matches[0].keys === hintPrefix) {
      var chosen = matches[0].target;
      closeHints();
      if (chosen.isConnected && hintPosition(chosen)) chosen.click();
    }
  }, true);
  document.addEventListener("pointerdown", closeHints, true);
  document.addEventListener("scroll", closeHints, { capture: true, passive: true });
  window.addEventListener("resize", closeHints);
  window.addEventListener("blur", closeHints);

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function throttle(fn, ms) {
    var t = 0;
    return function () { var now = Date.now(); if (now - t > ms) { t = now; fn(); } };
  }
})();
