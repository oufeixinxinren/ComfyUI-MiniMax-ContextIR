import { app } from "../../scripts/app.js";

const NODE_ID = "MiniMaxH3UnifiedContextIR";
const CONTEXT_IR_WIDGETS = ["ratio", "api_key", "base_url", "callback_url"];

function setWidgetHidden(widget, hidden) {
  if (!widget) return;
  // Official ComfyUI mechanism (same as the core CreateBoundingBoxes node):
  // the Vue widget renderer reads options.hidden, and legacy canvas uses widget.hidden.
  widget.hidden = hidden;
  if (!widget.options) widget.options = {};
  widget.options.hidden = hidden;
}

function applyVisibility(node) {
  const toggle = node.widgets?.find((w) => w.name === "use_context_ir");
  if (!toggle) return;
  if (!toggle._mmxHooked) {
    toggle._mmxHooked = true;
    toggle.callback = () => applyVisibility(node);
  }
  const show = !!toggle.value;
  for (const w of node.widgets || []) {
    if (CONTEXT_IR_WIDGETS.includes(w.name)) {
      setWidgetHidden(w, !show);
    }
  }
  if (node.graph && typeof node.graph.setDirtyCanvas === "function") {
    node.graph.setDirtyCanvas(true, true);
  }
}

app.registerExtension({
  name: "ComfyUI.MiniMaxH3UnifiedContextIR.UI",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_ID) return;

    const onAddWidget = nodeType.prototype.addWidget;
    nodeType.prototype.addWidget = function (...args) {
      const widget = onAddWidget?.apply(this, args);
      applyVisibility(this);
      return widget;
    };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      applyVisibility(this);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      applyVisibility(this);
      return r;
    };
  },
});
