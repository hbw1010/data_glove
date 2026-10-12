import './frame-adjustments.css';

const copySegment = (segment) => ({ start: segment.start, stop: segment.stop, zero_adjustments_deg: { ...segment.zero_adjustments_deg } });
const sameValues = (a, b) => Object.keys(a).every((key) => a[key] === b[key]) && Object.keys(a).length === Object.keys(b).length;

// Later intervals replace, rather than accumulate on, existing local adjustments.
export function replaceFrameAdjustment(segments, replacement) {
  const next = [];
  for (const segment of segments) {
    if (segment.stop <= replacement.start || segment.start >= replacement.stop) next.push(copySegment(segment));
    else {
      if (segment.start < replacement.start) next.push({ ...copySegment(segment), stop: replacement.start });
      if (segment.stop > replacement.stop) next.push({ ...copySegment(segment), start: replacement.stop });
    }
  }
  next.push(copySegment(replacement));
  next.sort((a, b) => a.start - b.start);
  return next.reduce((merged, segment) => {
    const previous = merged.at(-1);
    if (previous?.stop === segment.start && sameValues(previous.zero_adjustments_deg, segment.zero_adjustments_deg)) previous.stop = segment.stop;
    else merged.push(segment);
    return merged;
  }, []);
}

// Resolve against this hand's real timestamps, never the merged timeline's index.
export function frameAtRelativeTime(source, time) {
  const values = source.time_s;
  if (!values.length || time < 0 || time > values.at(-1) - values[0]) return -1;
  let low = 0, high = values.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (values[middle] - values[0] <= time) low = middle + 1;
    else high = middle;
  }
  return low - 1;
}

export function createFrameAdjustments({ slider, getState, joints, handLabel, pause, seek, onDraftChange, onApplied, onModeChange, fail }) {
  const root = document.createElement('section'); root.id = 'correction-segments'; root.className = 'correction-segments';
  root.setAttribute('aria-label', '当前手按帧区间修正');
  const toolbar = document.createElement('div'); toolbar.className = 'row';
  const button = (id, text, action) => {
    const element = document.createElement('button'); element.id = `correction-segment-${id}`; element.type = 'button'; element.textContent = text;
    element.addEventListener('click', action); return element;
  };
  const startButton = button('start', '选择新的修正区间', () => begin());
  const applyButton = button('apply', '应用区间并返回回放', apply);
  const cancelButton = button('cancel', '取消未应用区间', cancel);
  const hint = document.createElement('p'); hint.id = 'correction-segment-hint'; hint.className = 'notice'; hint.setAttribute('role', 'status');
  const bounds = document.createElement('div'); bounds.className = 'row';
  const inputs = {};
  for (const [name, title] of [['start', '起帧（含，0 起始）'], ['end', '终帧（含，可与起帧相同）']]) {
    const label = document.createElement('label'); label.textContent = title;
    const input = document.createElement('input'); input.type = 'number'; input.step = '1'; input.min = '0'; input.id = `correction-segment-${name}-frame`;
    input.addEventListener('input', () => {
      const s = getState(); if (!s?.segmentDraft) return;
      s.segmentDraft[name] = input.value === '' ? null : input.valueAsNumber;
      draftChanged(false);
      const index = s.segmentDraft[name];
      if (Number.isInteger(index) && index >= 0 && index < s.source.count) seek(s.source.time_s[index] - s.source.time_s[0]);
    });
    inputs[name] = input; label.append(input); bounds.append(label);
  }
  const markers = document.createElement('div'); markers.id = 'correction-segment-markers'; markers.className = 'correction-segment-markers'; markers.setAttribute('aria-label', '已应用区间时间标记');
  const list = document.createElement('ul'); list.id = 'correction-segment-list';
  toolbar.append(startButton, applyButton, cancelButton); root.append(toolbar, hint, bounds, markers, list);
  slider.after(root);

  function validDraft(s = getState()) {
    const draft = s?.segmentDraft;
    return !!draft && Number.isInteger(draft.start) && Number.isInteger(draft.end) && draft.start >= 0 && draft.end >= draft.start && draft.end < s.source.count && joints.every((name) => Number.isFinite(draft.delta[name]));
  }
  function segments(s, includeDraft = true) {
    if (!includeDraft || !validDraft(s)) return s.frameAdjustments;
    return replaceFrameAdjustment(s.frameAdjustments, { start: s.segmentDraft.start, stop: s.segmentDraft.end + 1, zero_adjustments_deg: s.segmentDraft.delta });
  }
  function localValues(s, index = s.index) {
    const draft = s.segmentDraft;
    if (validDraft(s) && draft.start <= index && index <= draft.end) return draft.delta;
    return s.frameAdjustments.find((segment) => segment.start <= index && index < segment.stop)?.zero_adjustments_deg;
  }
  function canLeave() {
    if (!getState()?.segmentDraft) return true;
    fail(new Error('当前区间尚未应用。请先点击“应用区间并返回回放”或“取消未应用区间”，再切换、保存或导出。'));
    return false;
  }
  function begin(segment) {
    const s = getState();
    if (!s?.session || s.busy || !canLeave()) return;
    if (!joints.every((name) => Number.isFinite(s.delta[name]))) { fail(new Error('请先修正无效的全局增量，再选择区间。')); return; }
    pause();
    s.segmentDraft = { start: segment?.start ?? null, end: segment ? segment.stop - 1 : null, delta: Object.fromEntries(joints.map((name) => [name, segment?.zero_adjustments_deg[name] ?? 0])) };
    draftChanged(); onModeChange();
    if (segment) seek(s.source.time_s[segment.start] - s.source.time_s[0]);
  }
  function draftChanged(syncInputs = true) {
    const s = getState(); if (!s?.segmentDraft) return;
    onDraftChange(s); render(syncInputs);
  }
  function apply() {
    const s = getState(); if (!validDraft(s)) return;
    s.frameAdjustments = segments(s); s.segmentDraft = null;
    onApplied(s); render(); onModeChange();
  }
  function cancel() {
    const s = getState(); if (!s?.segmentDraft) return;
    s.segmentDraft = null; onDraftChange(s); render(); onModeChange();
  }
  slider.addEventListener('pointerdown', (event) => {
    const s = getState(); if (!s?.segmentDraft || event.button !== 0) return;
    event.preventDefault(); event.stopImmediatePropagation(); pause();
    // Native range thumbs reserve half a thumb at each edge.
    const rect = slider.getBoundingClientRect(), inset = Math.min(8, rect.width / 2);
    const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left - inset) / Math.max(1, rect.width - inset * 2)));
    const relative = Number(slider.min) + fraction * (Number(slider.max) - Number(slider.min));
    const index = frameAtRelativeTime(s.source, relative);
    if (index < 0) { fail(new Error('此公共时间超出当前手的真实帧范围，请在本手时间范围内选点。')); return; }
    if (s.segmentDraft.start === null) s.segmentDraft.start = index;
    else {
      const first = s.segmentDraft.start;
      s.segmentDraft.start = Math.min(first, index); s.segmentDraft.end = Math.max(first, index);
    }
    draftChanged(); seek(s.source.time_s[index] - s.source.time_s[0]);
  }, true);

  function render(syncInputs = true) {
    const s = getState(), draft = s?.segmentDraft;
    startButton.disabled = !s?.session || s.busy || !!draft;
    applyButton.hidden = cancelButton.hidden = bounds.hidden = !draft;
    applyButton.disabled = !validDraft(s);
    slider.classList.toggle('correction-selecting-segment', !!draft);
    slider.setAttribute('aria-label', draft ? '区间选择：鼠标两次点击选择当前手起终帧；也可使用下方帧数输入' : '公共相对时间（秒）');
    for (const name of ['start', 'end']) {
      inputs[name].max = Math.max(0, (s?.source.count ?? 1) - 1);
      if (syncInputs) inputs[name].value = draft?.[name] ?? '';
      const value = draft?.[name];
      inputs[name].setAttribute('aria-invalid', String(!!draft && value !== null && (!Number.isInteger(value) || value < 0 || value >= s.source.count || (name === 'end' && value < draft.start))));
    }
    hint.textContent = !s?.session ? '先建立本侧草稿，再选择局部修正区间。' : draft
      ? `${handLabel(s.source.hand)}区间编辑：${draft.start === null ? '请在上方公共进度条点击起点，再点击终点。' : draft.end === null ? `已选起帧 ${draft.start}，请再点击终点（同一点可选单帧）。` : `起帧 ${draft.start} ～ 终帧 ${draft.end}（含）。可继续点击调整终点，或输入帧号。`} 下方 21 个增量只编辑本段局部值；预览仅在选段内使用未应用值。应用或取消后恢复正常回放。`
      : `${handLabel(s.source.hand)}正常回放 · 下方滑条编辑全局增量；已应用 ${s.frameAdjustments.length} 个局部区间，选段之外保持全局零位。点击标记或“重新编辑”可调整已应用区间。`;
    markers.replaceChildren(); list.replaceChildren();
    if (!s?.session) return;
    const duration = Number(slider.max);
    s.frameAdjustments.forEach((segment, index) => {
      const title = `${handLabel(s.source.hand)}帧 ${segment.start} ～ ${segment.stop - 1}（含）`;
      const marker = button(`marker-${index}`, String(index + 1), () => begin(segment)); marker.title = title; marker.setAttribute('aria-label', `重新编辑${title}`); marker.disabled = !!draft || s.busy;
      const start = s.source.time_s[segment.start] - s.source.time_s[0];
      const end = s.source.time_s[Math.min(segment.stop, s.source.count - 1)] - s.source.time_s[0];
      marker.style.left = `${duration ? 100 * start / duration : 0}%`;
      marker.style.width = `${duration ? Math.max(0, 100 * (end - start) / duration) : 100}%`;
      markers.append(marker);
      const item = document.createElement('li'), text = document.createElement('span'); text.textContent = `${index + 1}. ${title} `;
      const edit = button(`edit-${index}`, '重新编辑', () => begin(segment));
      const remove = button(`remove-${index}`, '移除此段', () => {
        if (!canLeave()) return;
        s.frameAdjustments = s.frameAdjustments.filter((entry) => entry !== segment); onApplied(s); render(); onModeChange();
      });
      edit.disabled = remove.disabled = !!draft || s.busy; item.append(text, edit, remove); list.append(item);
    });
  }
  return { render, canLeave, validDraft, segments, localValues, draftChanged };
}
