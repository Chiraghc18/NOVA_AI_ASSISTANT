const $ = (id) => document.getElementById(id);

const LABELS = {
  idle: "Idle",
  listening: "Listening...",
  transcribing: "Transcribing...",
  thinking: "Thinking...",
  speaking: "Speaking...",
};

const DOT_COLORS = {
  idle: "#8091b8",
  listening: "#3dffb0",
  transcribing: "#ffc24f",
  thinking: "#ffc24f",
  speaking: "#b277ff",
};

let lastCount = -1;
let lastStatus = "";

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

function setBusy(busy) {
  $("micBtn").disabled = busy;
  $("composer").querySelector("button").disabled = busy;
}

function renderMessages(messages) {
  const box = $("messages");
  box.innerHTML = "";
  if (!messages.length) {
    box.innerHTML = '<div class="empty">Press the mic or type a message to talk to NOVA.</div>';
    return;
  }
  for (const m of messages) {
    const div = document.createElement("div");
    div.className = "msg " + m.role;
    div.textContent = m.content;
    box.appendChild(div);
  }
  box.scrollTop = box.scrollHeight;
}

async function poll() {
  try {
    const s = await api("/api/state");

    if (s.status !== lastStatus) {
      $("orb").dataset.state = s.status;
      $("statusText").textContent = LABELS[s.status] || s.status;
      document.querySelector(".status-pill .dot").style.background = DOT_COLORS[s.status] || "#8091b8";
      setBusy(s.status !== "idle");
      if (lastStatus && s.status === "idle") loadMemory();
      lastStatus = s.status;
    }

    $("note").textContent = s.note || "";

    if (s.messages.length !== lastCount) {
      renderMessages(s.messages);
      lastCount = s.messages.length;
    }
  } catch (e) {
    $("statusText").textContent = "Server offline";
  }
}

async function loadMemory() {
  try {
    const m = await api("/api/memory");
    $("memCount").textContent = "(" + m.count + ")";
    const list = $("memList");
    list.innerHTML = "";
    if (!m.recent.length) {
      list.innerHTML = '<div class="empty">No memories yet.</div>';
      return;
    }
    for (const text of m.recent) {
      const div = document.createElement("div");
      div.className = "mem";
      div.textContent = text;
      list.appendChild(div);
    }
  } catch (_) {}
}

async function loadSettings() {
  try {
    const s = await api("/api/settings");
    $("kv").innerHTML = "";
    const rows = [
      ["User", s.user],
      ["Whisper", s.whisper],
    ];
    for (const [k, v] of rows) {
      const dt = document.createElement("dt");
      dt.textContent = k;
      const dd = document.createElement("dd");
      dd.textContent = v;
      $("kv").append(dt, dd);
    }
    const sel = $("modelSelect");
    sel.innerHTML = "";
    for (const [key, info] of Object.entries(s.models)) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = info.name + " (" + info.storage + ")";
      if (info.model === s.model) opt.selected = true;
      sel.appendChild(opt);
    }
  } catch (_) {}
}

function fillSelect(sel, items, current) {
  sel.innerHTML = "";
  for (const d of items) {
    const opt = document.createElement("option");
    opt.value = d.index;
    opt.textContent = d.label;
    if (d.index === current) opt.selected = true;
    sel.appendChild(opt);
  }
}

async function loadAudio() {
  try {
    const a = await api("/api/audio");
    fillSelect($("micSelect"), a.inputs, a.input);
    fillSelect($("spkSelect"), a.outputs, a.output);
  } catch (_) {}
}

async function saveAudio() {
  try {
    await api("/api/audio", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        input: Number($("micSelect").value),
        output: Number($("spkSelect").value),
      }),
    });
  } catch (err) {
    $("note").textContent = err.message;
    loadAudio();
  }
}

$("micSelect").addEventListener("change", saveAudio);
$("spkSelect").addEventListener("change", saveAudio);

$("micBtn").addEventListener("click", async () => {
  try { await api("/api/talk", { method: "POST" }); }
  catch (e) { $("note").textContent = e.message; }
});

$("stopBtn").addEventListener("click", () => api("/api/stop", { method: "POST" }).catch(() => {}));

$("composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("textInput");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  try {
    await api("/api/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
  } catch (err) {
    $("note").textContent = err.message;
  }
});

$("modelSelect").addEventListener("change", (e) => {
  api("/api/model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key: e.target.value }),
  }).catch((err) => ($("note").textContent = err.message));
});

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    $("panel-memory").classList.toggle("hidden", btn.dataset.tab !== "memory");
    $("panel-settings").classList.toggle("hidden", btn.dataset.tab !== "settings");
  });
});

loadMemory();
loadSettings();
loadAudio();
poll();
setInterval(poll, 400);