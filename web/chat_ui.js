import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "MiniMaxH3MultimodalChat";
const MIN_HEIGHT = 300;
const CHROME = 120;
const PADDING = 10;
const MAX_CONVERSATIONS = 20;

function injectStyles() {
  if (document.getElementById("mmx-chat-styles")) return;
  const style = document.createElement("style");
  style.id = "mmx-chat-styles";
  style.textContent = `
    .mmx-chat { box-sizing:border-box; display:flex; flex-direction:row; gap:0; width:100%; height:100%;
      min-height:300px; color:var(--input-text,#e5e7eb); background:var(--comfy-menu-bg,#202124);
      border:1px solid var(--border-color,#444); border-radius:6px; font:13px/1.45 Arial,sans-serif;
      position:relative; overflow:visible; }
    .mmx-chat__sidebar { box-sizing:border-box; display:flex; flex-direction:column; flex:0 0 150px;
      min-width:120px; max-width:190px; padding:6px; gap:4px; overflow-y:auto;
      background:#1a1c1f; border-right:1px solid #33363b; }
    .mmx-chat__sidebar-title { color:#9ca3af; font-size:11px; font-weight:600; padding:2px 4px 4px; }
    .mmx-chat__conv { box-sizing:border-box; width:100%; padding:6px 8px; text-align:left; cursor:pointer;
      color:#cfd6de; background:transparent; border:1px solid transparent; border-radius:4px;
      font:12px/1.3 Arial,sans-serif; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .mmx-chat__conv:hover { background:#2a2d31; }
    .mmx-chat__conv--active { background:#263c33; border-color:#416957; color:#d9ede4; }
    .mmx-chat__conv-title { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .mmx-chat__conv-remove { float:right; margin:-1px -4px 0 6px; width:16px; height:16px; padding:0;
      color:#8f9aa6; background:transparent; border:0; cursor:pointer; font-size:14px; line-height:14px; }
    .mmx-chat__conv-remove:hover { color:#ef8b8b; }
    .mmx-chat__conv-edit { box-sizing:border-box; width:100%; padding:3px 5px; color:#e5edf6;
      background:#17181a; border:1px solid #4b657d; border-radius:4px; font:12px/1.3 Arial,sans-serif; outline:none; }
    .mmx-chat__sidebar-actions { display:flex; flex-direction:column; gap:4px; margin-top:auto; padding-top:6px; }
    .mmx-chat__side-button { width:100%; padding:5px 8px; color:#e5edf6; background:#303d4b;
      border:1px solid #4b657d; border-radius:4px; cursor:pointer; font:12px/1.3 Arial,sans-serif; }
    .mmx-chat__side-button:hover { background:#3c5268; }
    .mmx-chat__side-button--danger { background:#3b2d2d; border-color:#7d4b4b; }
    .mmx-chat__side-button--danger:hover { background:#523636; }
    .mmx-chat__main { box-sizing:border-box; display:flex; flex-direction:column; gap:8px; flex:1 1 auto;
      min-width:0; padding:8px; }
    .mmx-chat__flow { display:flex; align-items:center; gap:7px; min-height:22px; color:#b8c0ca; font-size:11px; }
    .mmx-chat__stage { padding:2px 7px; color:#f4c982; border:1px solid #765d32; border-radius:4px; background:#332b1d; }
    .mmx-chat__skill { overflow:hidden; color:#9daab8; text-overflow:ellipsis; white-space:nowrap; }
    .mmx-chat__messages { flex:1 1 auto; min-height:0; overflow-y:auto; padding:2px; }
    .mmx-chat__empty { display:grid; height:100%; place-items:center; color:#9ca3af; }
    .mmx-chat__message { margin:0 0 8px; padding:7px 9px; white-space:pre-wrap; overflow-wrap:anywhere;
      border:1px solid #414348; border-radius:5px; background:#292b2f; }
    .mmx-chat__message--user { margin-left:24px; border-color:#3f6858; background:#253b33; }
    .mmx-chat__role { display:block; margin-bottom:3px; color:#aeb4bd; font-size:11px; font-weight:600; }
    .mmx-chat__composer { display:grid; grid-template-columns:1fr auto; gap:6px; flex:0 0 auto; }
    .mmx-chat__chips { display:flex; flex-wrap:wrap; gap:5px; flex:0 0 auto; min-height:0; }
    .mmx-chat__chips:empty { display:none; }
    .mmx-chat__chip { display:inline-flex; align-items:center; max-width:100%; height:24px; padding:0 8px;
      color:#d9ede4; background:#263c33; border:1px solid #416957; border-radius:4px; cursor:pointer;
      font-size:11px; }
    .mmx-chat__chip:hover { background:#315b49; }
    .mmx-chat__input { box-sizing:border-box; width:100%; height:88px; min-height:88px; max-height:120px; resize:vertical;
      padding:7px 8px; color:var(--input-text,#f3f4f6); background:var(--comfy-input-bg,#17181a);
      border:1px solid var(--border-color,#4b4d52); border-radius:4px; outline:none; font:inherit; }
    .mmx-chat__input:focus { border-color:#55a07e; }
    .mmx-chat__actions { display:flex; flex-direction:column; gap:6px; }
    .mmx-chat__button { min-width:58px; height:28px; padding:0 10px; color:#f4f4f5; background:#3b3e43;
      border:1px solid #565a60; border-radius:4px; cursor:pointer; font:inherit; }
    .mmx-chat__button:hover:not(:disabled) { background:#494d53; }
    .mmx-chat__button--send { background:#347257; border-color:#438e6c; }
    .mmx-chat__button--send:hover:not(:disabled) { background:#3d8264; }
    .mmx-chat__button:disabled { cursor:default; opacity:0.55; }
    .mmx-chat__status { flex:0 0 auto; min-height:18px; color:#9ca3af; font-size:11px; }
    .mmx-chat__status[data-state="busy"] { color:#72c69e; }
    .mmx-chat__status[data-state="error"] { color:#ef8b8b; }
    .mmx-chat__menu { position:absolute; z-index:1000; min-width:180px; max-height:220px; overflow-y:auto;
      background:#2a2d31; border:1px solid #565a60; border-radius:4px; box-shadow:0 4px 12px rgba(0,0,0,.45); }
    .mmx-chat__menu-item { padding:6px 10px; cursor:pointer; color:#e5edf6; font-size:12px; }
    .mmx-chat__menu-item:hover { background:#3c5268; }
    .mmx-chat__menu-item--off { color:#8f9aa6; }
    .mmx-chat__menu-empty { padding:6px 10px; color:#8f9aa6; font-size:12px; }
  `;
  document.head.appendChild(style);
}

function el(tag, className, text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function parseHistory(raw) {
  try {
    const value = JSON.parse(raw || "[]");
    return Array.isArray(value) ? value.filter((item) => item && typeof item.content === "string") : [];
  } catch (_) {
    return [];
  }
}

function firstValue(value) {
  return Array.isArray(value) ? value[0] : value;
}

function collectPromptLinks(value, output, result = new Set()) {
  if (Array.isArray(value) && value.length === 2 && output?.[String(value[0])]) {
    result.add(String(value[0]));
    return result;
  }
  if (Array.isArray(value)) {
    for (const item of value) collectPromptLinks(item, output, result);
  } else if (value && typeof value === "object") {
    for (const item of Object.values(value)) collectPromptLinks(item, output, result);
  }
  return result;
}

async function buildChatOnlyPrompt(node) {
  const prompt = await app.graphToPrompt();
  const output = prompt?.output;
  const targetId = String(node.id);
  if (!output?.[targetId]) throw new Error("Chat node is not in the executable prompt.");
  const keep = new Set();
  const addWithAncestors = (nodeId) => {
    const id = String(nodeId);
    if (keep.has(id)) return;
    const apiNode = output[id];
    if (!apiNode) return;
    keep.add(id);
    for (const sourceId of collectPromptLinks(apiNode.inputs || {}, output)) addWithAncestors(sourceId);
  };
  addWithAncestors(targetId);
  const scoped = {};
  for (const [id, apiNode] of Object.entries(output)) if (keep.has(String(id))) scoped[id] = apiNode;
  prompt.output = scoped;
  return prompt;
}

function canonicalMediaTokens() {
  const tokens = ["@first_frame", "@last_frame"];
  for (let i = 0; i <= 8; i++) tokens.push(`@ref_image_${i}`);
  for (let i = 0; i <= 2; i++) {
    tokens.push(`@ref_video_${i}`, `@ref_video_audio_${i}`, `@ref_audio_${i}`);
  }
  return tokens;
}

function hideWidget(widget) {
  if (!widget) return;
  widget.origType = widget.type;
  widget.origComputeSize = widget.computeSize;
  widget.computeSize = () => [0, -4];
  widget.type = `converted-widget:mmx-chat-${widget.name}`;
  if (widget.inputEl) widget.inputEl.style.display = "none";
  if (widget.element) widget.element.style.display = "none";
  widget.hidden = true;
  if (!widget.options) widget.options = {};
  widget.options.hidden = true;
}

function loadConversations(raw) {
  let data = {};
  try {
    const parsed = JSON.parse(raw || "{}");
    if (parsed && typeof parsed === "object") data = parsed;
  } catch (_) {
    data = {};
  }
  const items = Array.isArray(data.items)
    ? data.items.slice(0, MAX_CONVERSATIONS).map((item) => ({
        title: typeof item?.title === "string" && item.title ? item.title : "新对话",
        history: typeof item?.history === "string" ? item.history : "[]",
      }))
    : [];
  if (!items.length) items.push({ title: "新对话", history: "[]" });
  let active = Number.isInteger(data.active) ? data.active : 0;
  if (active < 0 || active >= items.length) active = 0;
  return { active, items };
}

function setupChatNode(node) {
  injectStyles();
  const promptWidget = node.widgets?.find((w) => w.name === "prompt");
  const historyWidget = node.widgets?.find((w) => w.name === "chat_history");
  const conversationsWidget = node.widgets?.find((w) => w.name === "conversations");
  const requestWidget = node.widgets?.find((w) => w.name === "request_id");
  if (!promptWidget || !historyWidget || !requestWidget || !conversationsWidget || typeof node.addDOMWidget !== "function") return;
  hideWidget(promptWidget);
  hideWidget(historyWidget);
  hideWidget(conversationsWidget);
  hideWidget(requestWidget);

  let conversations = loadConversations(conversationsWidget.value);
  historyWidget.value = conversations.items[conversations.active]?.history || "[]";

  const root = el("div", "mmx-chat");
  const sidebar = el("div", "mmx-chat__sidebar");
  const main = el("div", "mmx-chat__main");
  const flow = el("div", "mmx-chat__flow");
  const stage = el("span", "mmx-chat__stage", "未开始");
  const skillLabel = el("span", "mmx-chat__skill", "Skill: auto");
  const messages = el("div", "mmx-chat__messages");
  const composer = el("div", "mmx-chat__composer");
  const input = el("textarea", "mmx-chat__input");
  const actions = el("div", "mmx-chat__actions");
  const sendButton = el("button", "mmx-chat__button mmx-chat__button--send", "发送");
  const clearButton = el("button", "mmx-chat__button", "清空");
  const status = el("div", "mmx-chat__status", "准备就绪");
  const chips = el("div", "mmx-chat__chips");
  const menu = el("div", "mmx-chat__menu");
  menu.style.display = "none";
  input.placeholder = "输入消息，点击媒体名或输入 @ 插入引用，Enter 发送，Shift+Enter 换行";
  sendButton.type = "button";
  clearButton.type = "button";
  actions.append(sendButton, clearButton);
  composer.append(input, actions);
  flow.append(el("span", "", "流程"), stage, skillLabel);
  main.append(flow, messages, chips, composer, status);
  root.append(sidebar, main, menu);

  for (const eventName of ["pointerdown", "mousedown", "mouseup", "click", "dblclick", "wheel"]) {
    root.addEventListener(eventName, (event) => event.stopPropagation());
  }

  let mediaItems = canonicalMediaTokens().map((token) => ({ token, connected: false }));

  const refreshMediaTokens = () => {
    const connected = new Set();
    const names = [];
    for (const input of node.inputs || []) {
      if (input.link == null) continue;
      if (input.name) names.push(input.name);
      if (input.widget?.name) names.push(input.widget.name);
    }
    for (const widget of node.widgets || []) {
      if (widget.name) names.push(widget.name);
    }
    for (const token of canonicalMediaTokens()) {
      const bare = token.slice(1);
      const pattern = new RegExp(`(?:^|\\.)${bare}$`);
      if (names.some((name) => pattern.test(name))) connected.add(token);
    }
    mediaItems = canonicalMediaTokens().map((token) => ({ token, connected: connected.has(token) }));
    renderChips();
  };

  const renderChips = () => {
    const connected = mediaItems.filter((item) => item.connected);
    const signature = connected.map((item) => item.token).join(",");
    if (chips.dataset.signature === signature) return;
    chips.dataset.signature = signature;
    chips.replaceChildren();
    if (!connected.length) return;
    const label = el("span", "", "媒体：");
    label.style.cssText = "color:#8f9aa6;font-size:11px;line-height:24px;";
    chips.append(label);
    for (const item of connected) {
      const chip = el("button", "mmx-chat__chip", item.token);
      chip.type = "button";
      chip.title = "点击在光标处插入引用";
      chip.addEventListener("click", () => {
        if (input.readOnly) {
          status.textContent = "已连接外部文本输入，引用请在外部文本节点中添加";
          status.dataset.state = "idle";
          return;
        }
        insertReference(item.token);
        input.focus();
      });
      chips.append(chip);
    }
  };
  refreshMediaTokens();

  let promptLinked = false;

  const findLinkedTextInput = () =>
    node.inputs?.find((i) => (i.name === "input_string" || i.name === "prompt") && i.link != null) || null;

  const getLinkedPromptValue = () => {
    const promptInput = findLinkedTextInput();
    if (!promptInput) return null;
    const links = node.graph?.links;
    const link = typeof links?.get === "function" ? links.get(promptInput.link) : links?.[promptInput.link];
    const origin = link ? node.graph?.getNodeById(link.origin_id) : null;
    if (!origin) return null;
    const widget = origin.widgets?.find((w) => w.name === "value") ?? origin.widgets?.[0];
    if (widget && widget.value !== undefined) return String(widget.value ?? "");
    return null;
  };

  const refreshPromptLink = () => {
    promptLinked = !!findLinkedTextInput();
    if (promptLinked) {
      const value = getLinkedPromptValue();
      if (value !== null) {
        input.value = value;
        input.readOnly = true;
        input.title = "已连接外部文本输入（发送时以外部文本为准）";
        status.textContent = "已连接外部文本输入";
        status.dataset.state = "idle";
      }
    } else {
      input.readOnly = false;
      input.title = "";
    }
  };
  refreshPromptLink();

  const saveConversations = () => {
    conversationsWidget.value = JSON.stringify(conversations);
    node.graph?.setDirtyCanvas?.(true, true);
  };

  const updateSkillBar = () => {
    let skillId = "";
    try {
      const data = JSON.parse(historyWidget.value || "[]");
      const meta = Array.isArray(data) ? data.find((item) => item?.role === "_meta") : null;
      if (meta && typeof meta.skill === "string" && meta.skill) skillId = meta.skill;
    } catch (_) {
      skillId = "";
    }
    const combo = node.widgets?.find((w) => w.name === "skill");
    if (skillId) {
      skillLabel.textContent = `Skill: ${skillId}（已加载）`;
      skillLabel.title = skillId;
    } else if (combo && combo.value && combo.value !== "auto") {
      skillLabel.textContent = `Skill: ${combo.value}（待首轮加载）`;
      skillLabel.title = combo.value;
    } else {
      skillLabel.textContent = "Skill: auto（未加载）";
      skillLabel.title = "";
    }
  };

  const render = () => {
    const history = parseHistory(historyWidget.value);
    messages.replaceChildren();
    if (!history.length) {
      messages.append(el("div", "mmx-chat__empty", "暂无对话"));
      return;
    }
    for (const item of history) {
      const block = el("div", `mmx-chat__message mmx-chat__message--${item.role === "user" ? "user" : "assistant"}`);
      block.append(el("span", "mmx-chat__role", item.role === "user" ? "用户" : "助手"), el("div", "", item.content));
      messages.append(block);
    }
    messages.scrollTop = messages.scrollHeight;
  };

  const renderSidebar = () => {
    sidebar.replaceChildren();
    sidebar.append(el("div", "mmx-chat__sidebar-title", "最近聊天"));
    conversations.items.forEach((item, index) => {
      const button = el("button", `mmx-chat__conv${index === conversations.active ? " mmx-chat__conv--active" : ""}`);
      button.type = "button";
      const remove = el("button", "mmx-chat__conv-remove", "×");
      remove.type = "button";
      remove.title = "删除该对话";
      remove.addEventListener("click", (event) => {
        event.stopPropagation();
        removeConversation(index);
      });
      const title = el("span", "mmx-chat__conv-title", item.title);
      button.append(remove, title);
      button.addEventListener("click", () => switchConversation(index));
      button.addEventListener("dblclick", (event) => {
        event.stopPropagation();
        const editor = document.createElement("input");
        editor.className = "mmx-chat__conv-edit";
        editor.value = item.title;
        button.replaceChildren(editor);
        editor.focus();
        editor.select();
        const commit = () => {
          const name = editor.value.trim();
          if (name) {
            item.title = name;
            saveConversations();
          }
          renderSidebar();
        };
        editor.addEventListener("keydown", (e) => {
          if (e.key === "Enter") commit();
          else if (e.key === "Escape") renderSidebar();
        });
        editor.addEventListener("blur", commit);
      });
      sidebar.append(button);
    });
    const actionsBox = el("div", "mmx-chat__sidebar-actions");
    const newButton = el("button", "mmx-chat__side-button", "＋ 新建对话");
    newButton.type = "button";
    newButton.addEventListener("click", newConversation);
    const clearAllButton = el("button", "mmx-chat__side-button mmx-chat__side-button--danger", "清空全部");
    clearAllButton.type = "button";
    clearAllButton.addEventListener("click", clearAll);
    actionsBox.append(newButton, clearAllButton);
    sidebar.append(actionsBox);
  };

  const switchConversation = (index) => {
    if (index === conversations.active) return;
    conversations.items[conversations.active].history = historyWidget.value;
    conversations.active = index;
    historyWidget.value = conversations.items[index].history;
    promptWidget.value = "";
    input.value = "";
    saveConversations();
    render();
    renderSidebar();
    updateSkillBar();
    node.graph?.setDirtyCanvas?.(true, true);
  };

  const newConversation = () => {
    conversations.items[conversations.active].history = historyWidget.value;
    conversations.items.push({ title: "新对话", history: "[]" });
    if (conversations.items.length > MAX_CONVERSATIONS) {
      conversations.items.splice(0, 1);
    }
    conversations.active = conversations.items.length - 1;
    historyWidget.value = "[]";
    promptWidget.value = "";
    input.value = "";
    saveConversations();
    render();
    renderSidebar();
    updateSkillBar();
    node.graph?.setDirtyCanvas?.(true, true);
  };

  const removeConversation = (index) => {
    if (conversations.items.length <= 1) {
      clearAll();
      return;
    }
    conversations.items.splice(index, 1);
    if (conversations.active >= conversations.items.length) conversations.active = conversations.items.length - 1;
    if (conversations.active >= index) conversations.active = Math.max(0, conversations.active - 1);
    historyWidget.value = conversations.items[conversations.active].history;
    promptWidget.value = "";
    input.value = "";
    saveConversations();
    render();
    renderSidebar();
    updateSkillBar();
    node.graph?.setDirtyCanvas?.(true, true);
  };

  const clearAll = () => {
    conversations = { active: 0, items: [{ title: "新对话", history: "[]" }] };
    historyWidget.value = "[]";
    promptWidget.value = "";
    requestWidget.value = `${Date.now()}-clear`;
    input.value = "";
    saveConversations();
    render();
    renderSidebar();
    updateSkillBar();
    setBusy(false, "对话已清空");
    node.graph?.setDirtyCanvas?.(true, true);
  };

  const setBusy = (busy, message = busy ? "正在生成..." : "准备就绪", state = busy ? "busy" : "idle") => {
    node.__mmxChatBusy = busy;
    sendButton.disabled = busy;
    clearButton.disabled = busy;
    input.disabled = busy;
    status.textContent = message;
    status.dataset.state = state;
  };

  const send = async () => {
    const text = input.value.trim();
    if ((!text && !promptLinked) || node.__mmxChatBusy) return;
    const item = conversations.items[conversations.active];
    if ((!item || item.title === "新对话") && text) {
      item.title = text.slice(0, 12);
      renderSidebar();
    }
    if (!promptLinked) {
      promptWidget.value = text;
    }
    requestWidget.value = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setBusy(true);
    node.graph?.setDirtyCanvas?.(true, true);
    try {
      const prompt = await buildChatOnlyPrompt(node);
      await api.queuePrompt(0, prompt);
      status.textContent = "已加入队列...";
    } catch (error) {
      setBusy(false, `加入队列失败：${error?.message || error}`, "error");
    }
  };

  const showMenu = (filter) => {
    menu.replaceChildren();
    const matches = mediaItems.filter((item) => item.token.includes(filter));
    if (!matches.length) {
      menu.append(el("div", "mmx-chat__menu-empty", "无可用媒体引用"));
    } else {
      for (const item of matches) {
        const row = el("div", `mmx-chat__menu-item${item.connected ? "" : " mmx-chat__menu-item--off"}`);
        row.textContent = `${item.token}${item.connected ? "" : "（未连接）"}`;
        row.addEventListener("mousedown", (event) => event.preventDefault());
        row.addEventListener("click", () => insertReference(item.token));
        menu.append(row);
      }
    }
    const rect = input.getBoundingClientRect();
    const rootRect = root.getBoundingClientRect();
    menu.style.left = `${Math.max(rootRect.left, Math.min(rect.left, rect.right - 200))}px`;
    const spaceAbove = rect.top - rootRect.top;
    const spaceBelow = rootRect.bottom - rect.bottom;
    if (spaceAbove >= 220) {
      menu.style.top = `${rect.top - menu.offsetHeight - 4}px`;
    } else {
      menu.style.top = `${rect.bottom + 4}px`;
    }
    menu.style.display = "block";
  };

  const findPendingMention = () => {
    const start = input.selectionStart;
    const text = input.value;
    const before = text.slice(0, start);
    const lastSpace = Math.max(before.lastIndexOf(" "), before.lastIndexOf("\n"), before.lastIndexOf("\t"));
    const at = before.lastIndexOf("@");
    if (at === -1 || at <= lastSpace) return -1;
    const typed = text.slice(at, start);
    if (canonicalMediaTokens().includes(typed)) return -1;
    return at;
  };

  const insertReference = (token) => {
    const start = input.selectionStart;
    const text = input.value;
    const at = findPendingMention();
    if (at !== -1) {
      input.value = text.slice(0, at) + token + " " + text.slice(start);
      input.selectionStart = input.selectionEnd = at + token.length + 1;
    } else {
      const prefix = text.slice(0, start);
      const spacer = prefix && !/\s$/.test(prefix) ? " " : "";
      input.value = text.slice(0, start) + spacer + token + " " + text.slice(start);
      input.selectionStart = input.selectionEnd = start + spacer.length + token.length + 1;
    }
    menu.style.display = "none";
    input.focus();
  };

  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      menu.style.display = "none";
      return;
    }
    if (menu.style.display !== "none" && (event.key === "Enter" || event.key === "Tab")) {
      const first = menu.querySelector(".mmx-chat__menu-item");
      if (first) {
        event.preventDefault();
        insertReference(first.textContent.split("（")[0]);
      }
      return;
    }
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      send();
    }
  });
  input.addEventListener("input", () => {
    refreshMediaTokens();
    const at = findPendingMention();
    if (at !== -1) {
      showMenu(input.value.slice(at + 1, input.selectionStart));
    } else {
      menu.style.display = "none";
    }
  });
  input.addEventListener("blur", () => {
    window.setTimeout(() => {
      menu.style.display = "none";
    }, 150);
  });

  sendButton.addEventListener("click", send);
  clearButton.addEventListener("click", clearAll);

  const domWidget = node.addDOMWidget("mmx_chat", "mmx_chat", root, {
    getMinHeight: () => MIN_HEIGHT,
    getMaxHeight: () => undefined,
    getHeight: () => Math.max(MIN_HEIGHT + PADDING, (node.size?.[1] || 470) - CHROME + PADDING),
    hideOnZoom: false,
    serialize: false,
  });

  domWidget.computeLayoutSize = () => ({
    minHeight: MIN_HEIGHT,
    minWidth: 280,
    maxHeight: 100000,
    maxWidth: 100000,
  });

  const updateChatLayout = (size = node.size) => {
    void size;
    root.style.minHeight = `${MIN_HEIGHT}px`;
    node.graph?.setDirtyCanvas?.(true, true);
  };
  domWidget.afterResize = () => updateChatLayout();
  const originalOnResize = node.onResize;
  node.onResize = function (size) {
    const result = originalOnResize?.apply(this, arguments);
    updateChatLayout(size || this.size);
    return result;
  };

  const originalOnExecuted = node.onExecuted;
  node.onExecuted = function (output) {
    originalOnExecuted?.apply(this, arguments);
    const rawHistory = firstValue(output?.chat_history);
    if (typeof rawHistory === "string") historyWidget.value = rawHistory;
    const reply = firstValue(output?.reply);
    const promptText = firstValue(output?.prompt_text);
    const report = firstValue(output?.report);
    if (typeof report === "string") {
      for (const line of report.split("\n")) {
        if (line.startsWith("skill=")) {
          const id = line.split("=")[1] || "";
          skillLabel.textContent = id ? `Skill: ${id}（已加载）` : "Skill: 未加载";
          skillLabel.title = id;
        } else if (line.startsWith("stage=")) {
          stage.textContent = line.split("=")[1] || "未开始";
        }
      }
    }
    if (typeof promptText === "string" && promptText) {
      status.textContent = reply ? "已回复" : `已生成提示词：${promptText.slice(0, 120)}${promptText.length > 120 ? "…" : ""}`;
    }
    if (typeof reply === "string" && reply) {
      promptWidget.value = "";
      input.value = "";
    }
    if (conversations.items[conversations.active]) {
      conversations.items[conversations.active].history = historyWidget.value;
      saveConversations();
    }
    render();
    renderSidebar();
    updateSkillBar();
    setBusy(false);
    this.graph?.setDirtyCanvas?.(true, true);
  };

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function () {
    const result = originalOnConfigure?.apply(this, arguments);
    conversations = loadConversations(conversationsWidget.value);
    historyWidget.value = conversations.items[conversations.active]?.history || "[]";
    window.setTimeout(() => {
      render();
      renderSidebar();
      updateSkillBar();
      refreshPromptLink();
    }, 0);
    return result;
  };

  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (type, index, connected, linkInfo) {
    const result = originalOnConnectionsChange?.apply(this, arguments);
    refreshMediaTokens();
    refreshPromptLink();
    return result;
  };

  const refreshTimer = window.setInterval(() => {
    refreshMediaTokens();
    refreshPromptLink();
  }, 1000);
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function () {
    window.clearInterval(refreshTimer);
    return originalOnRemoved?.apply(this, arguments);
  };

  node.setSize([Math.max(node.size?.[0] || 0, 480), Math.max(node.size?.[1] || 0, 470)]);
  window.setTimeout(() => {
    updateChatLayout();
    render();
    renderSidebar();
    updateSkillBar();
    refreshPromptLink();
  }, 0);
}

app.registerExtension({
  name: "ComfyUI.MiniMaxH3MultimodalChat.UI",
  nodeCreated(node) {
    if (node.constructor?.comfyClass === NODE_ID) setupChatNode(node);
  },
});
