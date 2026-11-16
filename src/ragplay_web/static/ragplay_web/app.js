// Small behaviours. No inline scripts (the CSP only allows files from this site).
(function () {
  "use strict";

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const money = (v) => "$" + Math.round(v).toLocaleString();

  function monthLabel(ym, short) {
    const [y, m] = ym.split("-").map(Number);
    return new Date(y, m - 1, 1).toLocaleString(undefined, short ? { month: "short" } : { month: "long", year: "numeric" });
  }

  // ---- dashboard trend chart
  function trendChart() {
    const el = document.getElementById("trend-chart");
    const data = document.getElementById("trend-data");
    if (!el || !data || !window.Chart) return;
    const d = JSON.parse(data.textContent);
    const accent = css("--accent"), bar = css("--bar"), muted = css("--muted"), border = css("--border");
    new Chart(el, {
      type: "bar",
      data: {
        labels: d.labels.map((m) => monthLabel(m, true)),
        datasets: [{
          data: d.values,
          backgroundColor: d.labels.map((m) => (m === d.month ? accent : bar)),
          borderRadius: 6, maxBarThickness: 42,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { title: (items) => monthLabel(d.labels[items[0].dataIndex]), label: (c) => money(c.raw) } },
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: muted }, border: { display: false } },
          y: { grid: { color: border }, ticks: { color: muted, callback: (v) => money(v) }, border: { display: false } },
        },
        onClick: (_, els) => { if (els.length) location.search = "?month=" + d.labels[els[0].index]; },
        onHover: (e, els) => { e.native.target.style.cursor = els.length ? "pointer" : "default"; },
      },
    });
  }

  // ---- month picker: submit on change
  document.addEventListener("change", (e) => {
    if (e.target.matches("[data-autosubmit]")) e.target.form.submit();
  });

  // ---- chat
  function chatForm() { return document.getElementById("chat-form"); }

  document.addEventListener("click", (e) => {
    const chip = e.target.closest("[data-question]");
    const form = chatForm();
    if (!chip || !form) return;
    form.q.value = chip.dataset.question;
    form.requestSubmit();
  });

  document.addEventListener("keydown", (e) => {
    // Enter sends, Shift+Enter adds a line
    if (e.target.matches("#chat-form textarea") && e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (e.target.value.trim()) chatForm().requestSubmit();
    }
  });

  document.addEventListener("input", (e) => {
    if (e.target.matches("#chat-form textarea")) {
      e.target.style.height = "auto";
      e.target.style.height = e.target.scrollHeight + "px";
    }
  });

  document.addEventListener("htmx:beforeRequest", (e) => {
    const form = chatForm();
    if (e.detail.elt !== form) return;
    // show the question straight away; the server's reply replaces it
    const q = form.q.value.trim();
    const pending = document.createElement("div");
    pending.className = "turn";
    pending.id = "pending-turn";
    const msg = document.createElement("div");
    msg.className = "msg msg-user msg-pending";
    msg.textContent = q;
    pending.appendChild(msg);
    document.getElementById("conversation").appendChild(pending);
    document.getElementById("starters")?.remove();
    pending.scrollIntoView({ behavior: "smooth", block: "end" });
  });

  document.addEventListener("htmx:beforeSwap", (e) => {
    if (e.detail.requestConfig.elt === chatForm()) document.getElementById("pending-turn")?.remove();
  });

  document.addEventListener("htmx:afterRequest", (e) => {
    const form = chatForm();
    if (e.detail.elt !== form) return;
    if (e.detail.successful) {
      form.q.value = "";
      form.q.style.height = "auto";
      document.querySelector("#conversation .turn:last-child")?.scrollIntoView({ behavior: "smooth", block: "start" });
    } else {
      document.getElementById("pending-turn")?.remove();
      alert("Sorry, that request failed. Is the database reachable and Ollama running?");
    }
    form.q.focus();
  });

  document.addEventListener("DOMContentLoaded", () => {
    trendChart();
    document.querySelector("#conversation .turn:last-child")?.scrollIntoView({ block: "end" });
  });
})();
