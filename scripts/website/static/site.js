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
    document.body.classList.toggle("nav-open", open);
    if (navBtn) navBtn.setAttribute("aria-expanded", String(open));
    if (scrim) scrim.hidden = !open;
  }
  if (navBtn) navBtn.addEventListener("click", function () { setNav(!document.body.classList.contains("nav-open")); });
  if (scrim) scrim.addEventListener("click", function () { setNav(false); });

  // Keep the current page visible in the sidebar.
  var current = document.querySelector(".sidebar a.current");
  if (current) current.scrollIntoView({ block: "center" });

  // Sidebar filter (the reports list).
  var filter = document.querySelector(".side-filter");
  if (filter) filter.addEventListener("input", function () {
    var q = filter.value.trim().toLowerCase();
    document.querySelectorAll(".sidebar .side-group").forEach(function (g) {
      var any = false;
      g.querySelectorAll("li").forEach(function (li) {
        var show = !q || li.textContent.toLowerCase().indexOf(q) >= 0;
        li.hidden = !show;
        any = any || show;
      });
      g.hidden = !any;
    });
  });

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
        e.l = e.t.toLowerCase(); e.last = e.l.split("::").pop(); return e;
      });
      then();
    };
    document.head.appendChild(s);
  }
  var KIND_RANK = { namespace: 0, "class": 1, struct: 1, concept: 1, "function": 2, report: 2, guide: 1 };
  function score(e, q, words) {
    var s;
    if (e.last === q || e.l === q) s = 0;
    else if (e.last.indexOf(q) === 0) s = 1;
    else if (e.l.indexOf(q) === 0) s = 2;
    else if (e.last.indexOf(q) >= 0) s = 3;
    else if (e.l.indexOf(q) >= 0) s = 4;
    else {
      var hay = e.l + " " + (e.s || "").toLowerCase() + " " + (e.d || "").toLowerCase();
      for (var i = 0; i < words.length; i++) if (hay.indexOf(words[i]) < 0) return -1;
      s = 6;
    }
    var k = KIND_RANK[e.k]; if (k === undefined) k = 3;
    return s * 10 + k + e.l.length / 1000;
  }
  function run() {
    var q = input.value.trim().toLowerCase();
    if (!q) { list.hidden = true; return; }
    var words = q.split(/\s+/);
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
          (e.s ? '<span class="sr-sum">' + escapeHtml(e.s) + "</span>" : "") + "</a></li>";
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
