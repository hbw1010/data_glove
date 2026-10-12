import './style.css';
import { ModelView } from './scene.js';
import { createWristReplay } from './wrist-replay.js';
import { createCorrectionEditor } from './correction-editor.js';
import { createExportQueue } from './export-queue.js';

const $ = (id) => document.getElementById(id);
const moduleTabs = Array.from($('module-tabs').querySelectorAll('[role="tab"]'), (tab) => ({
  tab, panel: $(tab.getAttribute('aria-controls')),
}));
function selectModule(targetId, { scroll = true, focus = false } = {}) {
  const target = $(targetId);
  const panel = target?.closest('.module-panel');
  if (!panel) return false;
  const previousPanel = document.activeElement?.closest('.module-panel');
  for (const entry of moduleTabs) {
    const selected = entry.panel === panel;
    entry.panel.hidden = !selected;
    entry.tab.setAttribute('aria-selected', String(selected));
    entry.tab.tabIndex = selected ? 0 : -1;
    if (selected) {
      if (focus || previousPanel && previousPanel !== panel) entry.tab.focus({ preventScroll: true });
      entry.tab.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }
  }
  if (scroll) target.scrollIntoView({ block: 'start' });
  return true;
}
for (const { tab, panel } of moduleTabs) {
  tab.addEventListener('click', () => selectModule(panel.id));
}
$('module-tabs').addEventListener('keydown', (event) => {
  const index = moduleTabs.findIndex(({ tab }) => tab === event.target);
  if (index < 0) return;
  let next;
  switch (event.key) {
    case 'ArrowLeft': case 'ArrowUp': next = (index + moduleTabs.length - 1) % moduleTabs.length; break;
    case 'ArrowRight': case 'ArrowDown': next = (index + 1) % moduleTabs.length; break;
    case 'Home': next = 0; break;
    case 'End': next = moduleTabs.length - 1; break;
    default: return;
  }
  event.preventDefault();
  selectModule(moduleTabs[next].panel.id, { focus: true });
});
document.querySelector('main').addEventListener('click', (event) => {
  if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  const link = event.target.closest('a[href^="#"]');
  if (link && selectModule(link.getAttribute('href').slice(1), { focus: true })) event.preventDefault();
});
const state = {
  info: null, recording: null, calibration: null, raw: null, rawIndex: 0,
  group: 0, importVersion: 0, calVersion: 0, rawVersion: 0, previewVersion: 0,
  calibrationBusy: false, result: null, resultVersion: 0,
  resultTime: 0, resultTimes: [], playing: false, playOrigin: 0, playTime: 0, playRate: 1,
  jobSubmitting: false,
  mode: 'offline', live: null, liveBusy: false, liveRevision: 0, livePollPending: false,
  liveReceivedAt: 0, liveError: '', livePreviewPending: false, livePreviewAt: 0, selectedHand: null, liveTransport: 'dgst',
  calibrationSession: null, calibrationDirty: false, recordingCalibration: '',
  taskListBusy: false, taskSelection: '', importBusy: false,
};
let rawTarget = null, rawRequest = null, rawDrain = null;
let previewTarget = null, previewRequest = null, previewDrain = null;

async function api(path, body, method = body === undefined ? 'GET' : 'POST', options = {}) {
  let response;
  try {
    response = await fetch(path, {
      signal: options.signal,
      method, ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
    });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error(`本地服务连接失败：${error.message}`);
  }
  const data = await response.json().catch((error) => {
    if (error.name === 'AbortError') throw error;
    return null;
  });
  if (!response.ok) {
    const detail = data?.detail;
    const error = new Error(typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `请求失败（HTTP ${response.status}）`);
    error.status = response.status;
    throw error;
  }
  if (data === null) throw new Error('本地服务返回了无效 JSON');
  return data;
}

function errorMessage(error) {
  $('global-error').textContent = error.message ?? String(error);
  $('global-error').hidden = false;
}
function clearError() { $('global-error').hidden = true; }
function bind(id, event, action) {
  $(id).addEventListener(event, async (e) => {
    e.preventDefault();
    clearError();
    try { await action(e); } catch (error) { errorMessage(error); }
  });
}
function handLabel(hand) { return hand === 'left' ? '左手' : hand === 'right' ? '右手' : `未知手侧 ${hand}`; }
function identity(item) { return `${handLabel(item.hand)} · device_id=${item.device_id}`; }
function format(number, digits = 4) { return Number.isFinite(number) ? number.toFixed(digits) : '—'; }
function clampIndex(value, count) { return Math.max(0, Math.min(count - 1, Math.trunc(Number(value) || 0))); }
function upperBound(times, value) {
  let low = 0, high = times.length;
  while (low < high) { const mid = (low + high) >>> 1; if (times[mid] <= value) low = mid + 1; else high = mid; }
  return low;
}
function lowerBound(times, value) {
  let low = 0, high = times.length;
  while (low < high) { const mid = (low + high) >>> 1; if (times[mid] < value) low = mid + 1; else high = mid; }
  return low;
}
function checkTimeline(data) {
  if (!['left', 'right'].includes(data.hand)) throw new Error('文件手侧必须明确为 left 或 right');
  if (!Number.isInteger(data.count) || data.count < 1 || !Array.isArray(data.time_s) || data.time_s.length !== data.count) throw new Error('文件帧数与时间轴不一致');
  if (data.time_s.some((time, index) => !Number.isFinite(time) || (index > 0 && time < data.time_s[index - 1]))) throw new Error('文件时间轴非有限或倒退');
}

function createView(viewport, status) {
  try { return new ModelView($(viewport), $(status)); }
  catch (error) {
    $(status).textContent = `无法创建三维视图，请启用 WebGL：${error.message}`;
    $(status).classList.add('invalid');
    return { clear() {}, hide() {}, reset() {}, pose() {}, skeleton() {}, load: async () => false };
  }
}
const views = { preview: createView('preview-viewport', 'preview-status') };
const resultStages = [
  ['exoskeleton', 'exo', 'exoskeleton_names'],
  ['hand', 'hand', 'keypoint_names'],
  ['wujihand2', 'wuji', 'wuji_names'],
];
const resultSides = ['left', 'right'].map((hand) => ({
  hand, prefix: `result-${hand}`, result: null, times: [], index: -1, version: 0,
  target: null, request: null, drain: null,
  views: Object.fromEntries(resultStages.map(([key, name]) => [
    key, createView(`result-${hand}-${name}-viewport`, `result-${hand}-${name}-status`),
  ])),
  camera: createWristReplay({
    api, upperBound, handLabel, format, prefix: `result-${hand}-glove`,
    source: 'glove-video', namespace: 'results', externalTimeline: true,
  }),
}));

const exportQueue = createExportQueue({ api, handLabel, openResult, refreshResults, selectModule });
const correctionEditor = createCorrectionEditor({ api, upperBound, handLabel, format, refreshInfo, submitJob: exportQueue.submit });

async function refreshInfo() {
  const info = await api('/api/info');
  state.info = info;
  $('calibration-dir').textContent = info.calibration_dir;
  $('output-dir').textContent = info.output_dir;
  renderCalibrationList();
  correctionEditor.refreshCalibrations(info.calibrations ?? []);
  if (state.calibration) renderGroups();
  renderLive();
  updateActions();
}
function renderCalibrationList(selected = $('calibration-select').value, liveSelected = $('live-calibration-select').value) {
  for (const [id, source, choice] of [
    ['calibration-select', state.recording, selected],
    ['live-calibration-select', state.live?.state === 'streaming' ? state.live : null, liveSelected],
  ]) {
    const matching = (state.info?.calibrations ?? []).filter((item) => source && item.hand === source.hand && item.device_id === source.device_id);
    $(id).replaceChildren(new Option(source ? '请选择已保存匹配标定' : id === 'calibration-select' ? '请先导入录制' : '请先连接手套', ''));
    for (const item of matching) $(id).add(new Option(`${item.name} · ${handLabel(item.hand)}`, item.name));
    if (matching.some((item) => item.name === choice)) $(id).value = choice;
    $(id).disabled = !source;
  }
  updateActions();
}
function liveFresh() {
  return state.live?.state === 'streaming' && !state.liveError && state.live.frame
    && Number.isFinite(state.live.frame_age_s)
    && state.live.frame_age_s + (performance.now() - state.liveReceivedAt) / 1000 < 1;
}
function liveSessionActive() {
  return ['connecting', 'streaming', 'stopping'].includes(state.live?.state)
    || Boolean(state.live?.session_id && state.live.state !== 'disconnected');
}
function calibrationSource() { return state.mode === 'live' ? state.live : state.recording; }
function sourceKey() {
  return state.mode === 'live' ? `live:${state.live?.session_id}:${state.live?.hand}:${state.live?.device_id}` : `offline:${state.recording?.id}`;
}
function selectedCalibration() { return $(state.mode === 'live' ? 'live-calibration-select' : 'calibration-select').value; }
function updateActions() {
  const live = state.mode === 'live', fresh = Boolean(liveFresh());
  const recording = state.live?.recording?.state === 'recording';
  const blocked = state.calibrationBusy || state.liveBusy || state.importBusy || (live && recording);
  const sourceReady = live ? fresh : Boolean(state.recording);
  $('new-calibration').disabled = !state.info || !sourceReady || blocked;
  $('load-calibration').disabled = !state.info || !sourceReady || !selectedCalibration() || blocked;
  $('start-job').disabled = !state.recording || !$('calibration-select').value || state.jobSubmitting || state.importBusy;
  $('capture').disabled = !state.calibration || !sourceReady || blocked || !$('pose-confirm').checked;
  $('save-calibration').disabled = !state.calibration || !sourceReady || blocked;
  $('calibration-mode').disabled = state.calibrationBusy || state.liveBusy || state.importBusy || recording;
  $('pose-confirm').disabled = !sourceReady || blocked;
  for (const element of $('calibration-editor').querySelectorAll('select, .steps button, #offline-window input, #offline-window button, #offline-timeline input, #offline-timeline button')) element.disabled = blocked || element.dataset.mappingMissing === 'true';
  const connected = liveSessionActive();
  const connectionBlocked = connected || state.liveBusy || state.calibrationBusy || state.importBusy || recording;
  for (const id of ['live-device', 'live-hand-left', 'live-hand-right', 'live-refresh', 'live-transport']) $(id).disabled = connectionBlocked;
  const encoderOnly = state.liveTransport === 'adb-ma782-v1';
  $('live-transport').value = state.liveTransport;
  $('live-transport-hint').textContent = encoderOnly
    ? '仅编码器兼容模式：无需相机，不启动 DGST。可直接选择 ADB debug 设备和手侧后连接，无需启用板端采集；读取全部 21 路真实 raw counts，角度 = raw × 360 / 65536，不应用标定、不生成假数据。设备 ID 与采集元数据由本机软件生成。'
    : 'DGST 完整采集使用板端流与设备元数据；不会自动切换到兼容模式。';
  $('live-prepare-hint').hidden = encoderOnly;
  for (const hand of ['left', 'right']) $(`live-hand-${hand}`).setAttribute('aria-pressed', String(state.selectedHand === hand));
  $('live-hand-hint').textContent = connected
    ? `当前选择：${handLabel(state.selectedHand)}。请先断开，才能切换手侧；不改板端配置。`
    : state.selectedHand
      ? `已选择${handLabel(state.selectedHand)}，请确认与实际佩戴一致后连接；不改板端配置。`
      : '请先按实际佩戴点击左手或右手，再连接；不改板端配置。';
  $('live-prepare').disabled = encoderOnly || connectionBlocked || !$('live-device').value;
  $('live-connect').disabled = connectionBlocked || !$('live-device').value || !state.selectedHand;
  $('live-disconnect').disabled = (!state.live?.session_id && !connected && state.live?.state !== 'error') || state.liveBusy || state.calibrationBusy || state.importBusy;
  $('live-calibration-select').disabled = !fresh || state.liveBusy || state.calibrationBusy || state.importBusy || recording;
  $('live-record-start').disabled = !fresh || !$('live-calibration-select').value || state.liveBusy || state.calibrationBusy || state.importBusy || recording || (live && state.calibrationDirty);
  $('live-record-stop').disabled = !recording || state.liveBusy || state.calibrationBusy;
  const importBlocked = state.calibrationBusy || state.liveBusy || state.importBusy || recording;
  for (const form of ['import-form', 'aligned-import-form']) {
    for (const element of $(form).querySelectorAll('input, textarea, select, button')) element.disabled = importBlocked;
  }
  $('task-select').disabled = importBlocked || state.taskListBusy;
  $('import-task').disabled = importBlocked || state.taskListBusy || !$('task-select').value;
  $('recording-candidates').disabled = $('import-candidate').disabled = importBlocked;
}
function invalidateCalibration() {
  ++state.calVersion; ++state.rawVersion; ++state.previewVersion;
  rawTarget = previewTarget = null;
  rawRequest?.abort(); previewRequest?.abort();
  state.calibration = null; state.raw = null; state.calibrationBusy = false;
  state.calibrationSession = null; state.calibrationDirty = false;
  $('pose-confirm').checked = false;
  $('capture-status').textContent = '';
  $('calibration-editor').hidden = true;
  $('calibration-status').textContent = '尚未新建标定';
  $('save-status').textContent = '';
  views.preview.clear('等待新建或载入标定');
}
function setMode(mode) {
  if (state.mode !== mode) invalidateCalibration();
  state.mode = mode;
  $('calibration-mode').value = mode;
  const live = mode === 'live';
  $('offline-window').hidden = $('offline-timeline').hidden = live;
  $('live-capture-hint').hidden = !live;
  $('mode-hint').textContent = live ? '使用“实时手套采集”模块的“当前手套已保存匹配标定”选择后载入，或新建六步标定。只捕获点击后 1.5 秒样本；切换设备或断连清空草稿。' : '使用“单手处理与导出”模块的“已有匹配标定”选择后载入，或从录制中选择真实零位区间新建；不访问硬件。';
  $('pose-confirm-label').textContent = live ? '我已实际摆出本步骤零位姿态，并将保持静止直到采集结束' : '我确认所选区间内保持本步骤要求的零位姿态';
  updateActions();
}
bind('calibration-mode', 'change', (event) => setMode(event.target.value));

function acceptRecording(recording, calibrationName = '', updateCorrection = true) {
  if (!correctionEditor.canLeave()) return false;
  checkTimeline(recording);
  state.recording = recording; state.rawIndex = 0;
  $('recording-path').value = recording.path;
  $('recording-info').textContent = `${identity(recording)} · ${recording.count} 帧 · ${format(recording.duration_s, 3)} 秒\n${recording.path}`;
  $('recording-warnings').replaceChildren();
  for (const warning of recording.warnings ?? []) { const li = document.createElement('li'); li.textContent = warning; $('recording-warnings').append(li); }
  $('raw-slider').max = $('raw-index').max = recording.count - 1;
  $('window-start').max = recording.count - 1; $('window-stop').max = recording.count;
  renderCalibrationList(calibrationName);
  if (updateCorrection) correctionEditor.setRecording(recording, calibrationName);
  return true;
}

function updateRawChannels() {
  for (const cell of $('channel-values').querySelectorAll('[data-raw-index]')) cell.textContent = format(state.raw?.angles_deg?.[Number(cell.dataset.rawIndex)]);
}
const dialTicks = Array.from({ length: 36 }, (_, index) =>
  `<line x1="0" y1="-48" x2="0" y2="${index % 9 === 0 ? -40 : -44}" transform="rotate(${index * 10})"/>`).join('');
const liveAngleDials = Array.from({ length: 21 }, (_, index) => {
  const card = document.createElement('article');
  card.className = 'encoder-dial stale';
  card.setAttribute('aria-labelledby', `live-angle-label-${index}`);
  card.innerHTML = `<h4 id="live-angle-label-${index}"><span>J${index + 1}</span><span class="dial-joint-name">等待统一关节定义</span></h4>
    <span class="dial-source">映射不可用</span>
    <svg class="dial-face" viewBox="-60 -60 120 120" aria-hidden="true">
      <circle class="dial-ring" r="52"/>
      <g class="dial-ticks">${dialTicks}</g>
      <g class="dial-labels"><text x="0" y="-30">0</text><text x="34" y="3">90</text><text x="0" y="38">180</text><text x="-34" y="3">270</text></g>
      <g class="dial-needle"><path d="M -2.5 8 L 0 -35 L 2.5 8 Z"/></g>
      <circle class="dial-hub" r="4"/>
    </svg>
    <code class="dial-value">—</code><span class="dial-state">无实时数据</span>`;
  $('live-angles').append(card);
  return { card, label: card.querySelector('h4'), jointName: card.querySelector('.dial-joint-name'), source: card.querySelector('.dial-source'), needle: card.querySelector('.dial-needle'), value: card.querySelector('.dial-value'), status: card.querySelector('.dial-state'), angle: null };
});
function renderLive() {
  const live = state.live, fresh = Boolean(liveFresh());
  const labels = { disconnected: '未连接', connecting: '连接中', streaming: '正在采样', stopping: '正在停止', error: '采样失败' };
  const age = live?.frame_age_s == null ? null : live.frame_age_s + (performance.now() - state.liveReceivedAt) / 1000;
  const encoderOnly = live?.transport === 'adb-ma782-v1';
  const transportLabel = encoderOnly ? '仅编码器兼容 · 无相机 · 真实 raw' : 'DGST 完整采集';
  const liveIdentity = live?.identity_source === 'host_generated'
    ? `${handLabel(live.hand)} · 软件生成 device_id=${live.device_id}`
    : live ? identity(live) : '';
  $('live-status').textContent = live && live.state !== 'disconnected' ? `${labels[live.state] ?? live.state}${live.transport ? ` · ${transportLabel}` : ''} · serial=${live.serial ?? '—'} · ${liveIdentity}\n${format(live.rate_hz, 1)} Hz · 已接收 ${live.received_frames} 帧 · 最新帧距今 ${format(age, 2)} 秒${live.state === 'streaming' && !fresh ? ' · 无新鲜数据，已隐藏姿态' : ''}` : '未连接；不会自动发现或连接设备';
  $('live-status').classList.toggle('stale', Boolean(state.liveError || live?.state === 'error' || live?.state === 'streaming' && !fresh));
  const hasIdentity = liveSessionActive() && live?.device_id;
  const deviceHand = ['left', 'right'].includes(live?.device_hand) ? handLabel(live.device_hand) : '未知';
  $('live-device-hand').textContent = liveSessionActive() && live?.hand
    ? encoderOnly
      ? `用户选择：${handLabel(live.hand)}；${live.mapping_source === 'board_factory' ? `出厂记录手侧：${deviceHand}` : '无可用出厂手侧声明，用户选侧不代表已核验的设备事实'}。不改板端配置。`
      : `用户选择：${handLabel(live.hand)}；设备声明：${deviceHand}。${live.device_hand && live.device_hand !== live.hand ? '设备声明与实际选择不一致，以你的选择为准。' : '以你的选择为准，不改板端配置。'}`
    : '';
  const mappingLabel = live?.mapping_source === 'board_factory' ? '已校验的板端出厂映射'
    : live?.mapping_source === 'explicit_hand_profile' ? '用户明确选侧的配置映射（未核验，不是设备出厂事实）'
      : live?.mapping_source === 'device_metadata' ? '板端设备元数据' : '未知';
  $('live-identity').textContent = hasIdentity
    ? `身份来源：${live.identity_source === 'host_generated' ? '本机软件自建 ID（基于真实板端 SoC serial，不是板端返回的 device_id）' : '板端设备元数据'}；真实 board_device_id=${live.board_device_id ?? '未提供'}；采集元数据来源：${live.metadata_source === 'host_generated' ? '本机软件生成，不是固件返回的 metadata.tar' : '板端设备'}；原始采流映射来源：${mappingLabel}。${live.cs_by_joint ? ` 原始列 J1～J21 对应 CS：${live.cs_by_joint.join(', ')}（采流证据，不是下方统一 J 关节定义）。` : ''}`
    : '';
  const problem = state.liveError || live?.error;
  if (problem) { $('live-error').textContent = problem; $('live-error').hidden = false; }
  const frame = fresh ? live.frame : null;
  liveAngleDials.forEach((dial, index) => {
    const channel = `J${index + 1}`;
    const entry = live?.joint_layout?.[index];
    const definition = state.info?.joint_definitions?.find((item) => item.channel === channel);
    const jointName = entry?.channel === channel ? entry.joint_name : definition?.joint_name;
    const mapped = entry?.channel === channel && Number.isInteger(entry.cs)
      && Number.isInteger(entry.source_index) && entry.source_index >= 0 && entry.source_index < 21;
    dial.jointName.textContent = (entry?.channel === channel ? entry.label : definition?.label) ?? jointName ?? '等待统一关节定义';
    dial.label.title = `${channel} · ${jointName ?? '关节定义不可用'}（左右统一解剖定义）`;
    dial.source.textContent = mapped ? `CS${entry.cs} · 原始J${entry.source_index + 1}` : '映射不可用';
    const value = mapped ? frame?.angles_deg?.[entry.source_index] : null;
    const angle = Number.isFinite(value) && value >= 0 && value < 360 ? value : null;
    dial.status.textContent = !mapped ? '等待通道映射' : angle === null ? '无实时数据' : '原始角';
    if (dial.angle === angle) return;
    dial.angle = angle;
    dial.card.classList.toggle('stale', angle === null);
    dial.value.textContent = angle === null ? '—' : `${format(angle, 2)}°`;
    if (angle !== null) dial.needle.setAttribute('transform', `rotate(${angle})`);
  });
  $('live-frame-info').textContent = frame ? `${encoderOnly ? '主机交付序号 ' : ''}sequence=${frame.sequence} · ${encoderOnly ? '板端扫描中点 CLOCK_MONOTONIC ' : ''}timestamp_ns=${frame.timestamp_ns} · 原始绝对角 [0,360)` : '没有新鲜编码器帧；不显示历史角度';
  if (state.mode === 'live') {
    state.raw = frame;
    updateRawChannels();
    if (!fresh) {
      $('pose-confirm').checked = false;
      ++state.previewVersion;
      views.preview.hide(problem || '连接中断或帧已过期，等待新鲜编码器数据');
    }
  }
  const recording = live?.recording;
  if (recording) {
    const names = { recording: '正在录制', complete: '录制已完整发布', error: '录制失败，未发布为完整文件' };
    $('live-record-status').textContent = `${names[recording.state] ?? recording.state} · ${recording.count} 帧 · ${format(recording.duration_s, 2)} 秒${recording.error ? `\n${recording.error}` : ''}\n${recording.path}`;
    $('live-record-status').classList.toggle('error', recording.state === 'error');
  } else {
    $('live-record-status').textContent = `先完成六步标定并保存，再开始动作录制。${state.liveTransport === 'adb-ma782-v1' ? '兼容模式仅保存真实编码器，不采集相机视频。' : '仅保存真实编码器，视频只排空丢弃。'}`;
    $('live-record-status').classList.remove('error');
  }
  updateActions();
}
function applyLiveStatus(status) {
  const old = state.live;
  const changed = old?.session_id !== status.session_id || old?.serial !== status.serial || old?.hand !== status.hand || old?.device_id !== status.device_id || old?.transport !== status.transport;
  const lost = status.state !== 'streaming';
  state.live = status; state.liveReceivedAt = performance.now(); state.liveError = '';
  if (liveSessionActive() && status.transport) state.liveTransport = status.transport === 'adb-ma782-v1' ? 'adb-ma782-v1' : 'dgst';
  if (liveSessionActive() && ['left', 'right'].includes(status.hand)) state.selectedHand = status.hand;
  if (state.mode === 'live' && (changed || lost) && (state.calibration || state.calibrationBusy)) {
    invalidateCalibration();
    $('calibration-status').textContent = '会话或设备身份已变化，旧草稿已清空；请重新新建或载入匹配标定';
  }
  if (changed || old?.state !== status.state) renderCalibrationList(undefined, changed ? '' : undefined);
  if (!status.error) $('live-error').hidden = true;
  renderLive();
}
async function pollLive() {
  if (!state.livePollPending && !state.liveBusy) {
    state.livePollPending = true;
    const revision = state.liveRevision;
    try {
      const status = await api('/api/live/status');
      if (revision === state.liveRevision) applyLiveStatus(status);
    } catch (error) {
      if (revision === state.liveRevision) {
        state.liveError = error.message;
        if (state.mode === 'live' && (state.calibration || state.calibrationBusy)) invalidateCalibration();
        renderLive();
      }
    } finally { state.livePollPending = false; }
  }
  setTimeout(pollLive, 250);
}
async function liveAction(action) {
  if (state.liveBusy || state.calibrationBusy) return;
  state.liveBusy = true; ++state.liveRevision;
  $('live-error').hidden = true; updateActions();
  try { await action(); }
  catch (error) {
    $('live-error').textContent = error.message; $('live-error').hidden = false;
    throw error;
  } finally { ++state.liveRevision; state.liveBusy = false; updateActions(); }
}
bind('live-refresh', 'click', () => liveAction(async () => {
  const result = await api('/api/live/devices');
  $('live-device').replaceChildren(new Option('请明确选择 USB serial', ''));
  for (const device of result.devices) $('live-device').add(new Option(`${device.serial} · ${device.mode}`, device.serial));
  $('live-warning').textContent = (result.warnings ?? []).join('\n') || (result.devices.length ? '发现不会连接；请选择要操作的设备。' : '未发现 USB 手套。请检查连接后刷新。');
  state.selectedHand = null;
}));
bind('live-device', 'change', () => {
  state.selectedHand = null;
  if (state.mode === 'live') invalidateCalibration();
  updateActions();
});
bind('live-transport', 'change', (event) => {
  if (liveSessionActive() || state.liveBusy || state.calibrationBusy || state.live?.recording?.state === 'recording') {
    updateActions();
    return;
  }
  state.liveTransport = event.target.value;
  if (state.mode === 'live') invalidateCalibration();
  state.recordingCalibration = '';
  $('live-calibration-select').value = '';
  $('live-warning').textContent = '';
  $('live-error').hidden = true;
  renderLive();
});
for (const hand of ['left', 'right']) bind(`live-hand-${hand}`, 'click', () => {
  if (liveSessionActive() || state.liveBusy || state.calibrationBusy || state.live?.recording?.state === 'recording') return;
  state.selectedHand = hand;
  updateActions();
});
bind('live-prepare', 'click', () => liveAction(async () => {
  if (state.liveTransport === 'adb-ma782-v1') return;
  const serial = $('live-device').value;
  if (!serial || !confirm(`准备 ${serial}？\n将启动已有 DGST + ACM + ADB、暂时移除 ECM，停止 RTSP 并让 DGST 接管相机，USB 会短暂重连。\n不刷固件、不改持久配置或板端标定，视频不保存。`)) return;
  $('live-warning').textContent = `正在为 ${serial} 启用板端采集，请等待 USB 重枚举…`;
  await api('/api/live/prepare', { serial });
  $('live-device').replaceChildren(new Option('准备完成，请刷新并重新确认 serial', ''));
  state.selectedHand = null;
  $('live-warning').textContent = '准备完成；等待 USB 重新出现后，点击“刷新发现”、重新选择 serial，再连接。';
}));
bind('live-connect', 'click', () => liveAction(async () => {
  const serial = $('live-device').value, hand = state.selectedHand;
  if (!serial || !['left', 'right'].includes(hand) || liveSessionActive() || state.live?.recording?.state === 'recording') return;
  const encoder_only = state.liveTransport === 'adb-ma782-v1';
  $('live-warning').textContent = `正在以${encoder_only ? '仅编码器兼容模式（无相机）' : 'DGST 完整采集模式'}连接 ${serial}，使用你选择的${handLabel(hand)}…`;
  try {
    const status = await api('/api/live/connect', { serial, hand, encoder_only });
    applyLiveStatus(status);
    setMode('live');
    $('live-warning').textContent = `${encoder_only ? '仅编码器兼容会话已建立：无相机、真实 raw；自建 ID、真实板 ID 和映射来源见上方。' : '实时会话已建立。'}请新建或载入匹配标定，再逐组摆姿态确认；不会自动捕获。`;
    await refreshInfo();
  } catch (error) {
    try { applyLiveStatus(await api('/api/live/status')); } catch { /* 保留连接请求的原始错误。 */ }
    throw error;
  }
}));
bind('live-disconnect', 'click', () => liveAction(async () => {
  if (state.live?.recording?.state === 'recording' && !confirm('当前仍在录制。直接断开会使未完成数据标记为失败；若需完整文件请取消并先停止录制。仍要断开？')) return;
  if (state.mode === 'live') invalidateCalibration();
  state.liveError = '正在断开，已隐藏旧姿态'; renderLive();
  applyLiveStatus(await api('/api/live/disconnect', {}));
}));
bind('live-calibration-select', 'change', updateActions);
bind('live-record-start', 'click', () => liveAction(async () => {
  if (!liveFresh() || !$('live-calibration-select').value || state.mode === 'live' && state.calibrationDirty) return;
  const session = state.live.session_id, name = $('live-calibration-select').value;
  const status = await api('/api/live/recording/start', { session_id: session, calibration_name: name });
  if (session !== state.live?.session_id) return;
  state.recordingCalibration = name;
  applyLiveStatus(status);
}));
bind('live-record-stop', 'click', () => liveAction(async () => {
  const session = state.live?.session_id;
  if (!session || state.live?.recording?.state !== 'recording') return;
  if (!correctionEditor.canLeave()) return;
  const result = await api('/api/live/recording/stop', { session_id: session });
  if (session !== state.live?.session_id) return;
  applyLiveStatus(result.live);
  if (!correctionEditor.canLeave()) {
    $('job-status').textContent = `原始录制已停止并保存：${result.recording.path}。请先应用或取消微调区间，再按此路径导入。`;
    return;
  }
  ++state.importVersion;
  setMode('offline'); invalidateCalibration();
  if (!acceptRecording(result.recording, state.recordingCalibration || $('live-calibration-select').value)) return;
  selectModule('correction-section');
  $('job-status').textContent = '实时原始录制已导入，匹配标定已保留。在“单手处理与导出”模块点击“开始离线处理”，完成后在“左右手只读回放”模块打开结果。';
}));
async function captureLiveGroup() {
  if (!liveFresh() || !state.calibration || state.calibrationSession !== state.live.session_id || state.calibrationBusy || state.liveBusy || !$('pose-confirm').checked) return;
  const session = state.live.session_id, version = state.calVersion, group = state.info.groups[state.group];
  state.calibrationBusy = true; updateActions();
  const started = performance.now();
  const countdown = () => {
    if (session !== state.live?.session_id || version !== state.calVersion) return;
    const remaining = Math.max(0, 1.5 - (performance.now() - started) / 1000);
    $('capture-status').textContent = remaining > 0 ? `请保持 ${group.label} 零位姿态：正在采集点击后的新样本，约剩 ${remaining.toFixed(1)} 秒…` : '请继续保持姿态，等待采样窗口完成及稳定性检查…';
  };
  countdown();
  const timer = setInterval(countdown, 100);
  try {
    const next = await api('/api/live/calibration/capture', { session_id: session, calibration: state.calibration, group: group.name, pose_confirmed: true, duration_s: 1.5 });
    if (session !== state.live?.session_id || version !== state.calVersion || state.mode !== 'live') return;
    state.calibration = next; ++state.calVersion; state.calibrationDirty = true;
    $('capture-status').textContent = `${group.label}采集成功。切换下一组并重新摆姿态确认；六组完成后保存。`;
    $('save-status').textContent = '零位已修改，尚未保存';
    renderGroups(); renderChannels();
  } catch (error) {
    if (session === state.live?.session_id && version === state.calVersion) $('capture-status').textContent = `本组未完成：${error.message}。请重新摆姿态并确认后再采集。`;
    throw error;
  } finally {
    clearInterval(timer);
    if (session === state.live?.session_id) { state.calibrationBusy = false; $('pose-confirm').checked = false; updateActions(); }
  }
}
async function previewLiveFrame() {
  if (state.mode !== 'live' || !liveFresh() || !state.calibration || state.calibrationSession !== state.live.session_id || state.livePreviewPending) return;
  const session = state.live.session_id, calibrationVersion = state.calVersion, version = ++state.previewVersion;
  state.livePreviewPending = true; state.livePreviewAt = performance.now();
  try {
    const [pose, loaded] = await Promise.all([
      api('/api/live/calibration/preview', { session_id: session, calibration: state.calibration }),
      views.preview.load(state.live.hand, 'exoskeleton'),
    ]);
    if (state.mode !== 'live' || session !== state.live?.session_id || calibrationVersion !== state.calVersion || version !== state.previewVersion || !liveFresh()) return;
    if (loaded) views.preview.pose(pose.joint_names, pose.qpos_rad);
  } catch (error) {
    if (version === state.previewVersion) views.preview.hide(`实时预览失败：${error.message}`);
  } finally { state.livePreviewPending = false; }
}
// 独立新鲜度看门狗：即使 HTTP 请求挂起，也不保留最后一帧伪装实时。
setInterval(() => {
  renderLive();
  if (!state.calibrationBusy && !state.liveBusy && performance.now() - state.livePreviewAt >= 500) void previewLiveFrame();
}, 250);
void pollLive();

async function refreshTasks() {
  if (state.taskListBusy) return;
  const path = $('task-root').value.trim();
  if (!path) throw new Error('请填写任务大目录');
  state.taskListBusy = true;
  $('refresh-tasks').disabled = $('task-select').disabled = $('import-task').disabled = true;
  $('task-list-status').classList.remove('error');
  $('task-list-status').textContent = '正在发现双手子目录及采集 HDF5 文件…';
  try {
    const data = await api(`/api/tasks?path=${encodeURIComponent(path)}`);
    $('task-select').replaceChildren(new Option('请选择要微调的双手子任务', ''));
    for (const task of data.tasks) $('task-select').add(new Option(task.name, task.path));
    if (data.tasks.some((task) => task.path === state.taskSelection)) $('task-select').value = state.taskSelection;
    else state.taskSelection = '';
    if ($('task-root').value.trim() === path) $('task-root').value = data.path;
    $('task-list-status').textContent = `${data.path}\n发现 ${data.tasks.length} 个可配对的双手子任务。选择后点击“导入所选子任务”；未自动导入。`;
  } catch (error) {
    $('task-list-status').classList.add('error');
    $('task-list-status').textContent = `任务列表读取失败：${error.message}。保留上次列表，可重试或使用高级路径导入。`;
  } finally {
    state.taskListBusy = false;
    $('refresh-tasks').disabled = false;
    updateActions();
  }
}
bind('task-root-form', 'submit', refreshTasks);
bind('task-select', 'change', () => {
  if (!correctionEditor.canLeave()) { $('task-select').value = state.taskSelection; return; }
  state.taskSelection = $('task-select').value;
  $('import-task').disabled = !state.taskSelection || state.taskListBusy;
});
bind('import-task', 'click', () => {
  if ($('task-select').value) return importRecording($('task-select').value, true, false);
});

function acceptRecordingPair(pair, path) {
  checkTimeline(pair.recordings.left); checkTimeline(pair.recordings.right);
  if (!acceptRecording(pair.recordings.left, '', false)) return;
  correctionEditor.setRecordings(pair.recordings);
  $('recording-path').value = path;
  $('recording-info').textContent = `双手已联合导入 · recording_uuid=${pair.recording_uuid}\n${Object.values(pair.recordings).map((item) => `${identity(item)} · ${item.count} 帧 · ${format(item.duration_s, 3)} 秒`).join('\n')}`;
  $('recording-warnings').replaceChildren();
  for (const item of Object.values(pair.recordings)) for (const warning of item.warnings ?? []) {
    const li = document.createElement('li'); li.textContent = `${handLabel(item.hand)}：${warning}`; $('recording-warnings').append(li);
  }
  $('recording-candidates-row').hidden = true;
}

async function readAlignedParameters(path) {
  if (!path) throw new Error('请填写对齐 HDF5 文件或所在目录路径');
  const selections = ['left', 'right'].map((hand) => ({
    hand, file: $(`aligned-calibration-${hand}`).files[0], convention: $(`aligned-convention-${hand}`).value,
  }));
  for (const { hand, file, convention } of selections) {
    if (!file) throw new Error(`请上传${handLabel(hand)}完整项目标定 JSON 文件`);
    if (!convention) throw new Error(`请选择${handLabel(hand)}角度约定，不能根据文件数值猜测`);
  }
  $('aligned-import-status').textContent = '正在读取左右手参数；读取与导入期间已锁定路径、文件与角度约定。';
  const calibrations = {}, angle_conventions = {};
  for (const { hand, file, convention } of selections) {
    let text;
    try { text = await file.text(); } catch { throw new Error(`${handLabel(hand)}标定文件读取失败，请重新选择文件`); }
    let calibration;
    try { calibration = JSON.parse(text); } catch { throw new Error(`${handLabel(hand)}标定文件不是有效 JSON`); }
    if (!calibration || typeof calibration !== 'object' || Array.isArray(calibration)) throw new Error(`${handLabel(hand)}标定必须是完整项目标定 JSON 对象`);
    calibrations[hand] = calibration;
    angle_conventions[hand] = convention;
  }
  if ($('recording-path').value.trim() !== path || selections.some(({ hand, file, convention }) =>
    $(`aligned-calibration-${hand}`).files[0] !== file || $(`aligned-convention-${hand}`).value !== convention)) {
    throw new Error('参数读取期间路径、文件或角度约定已改变，请重新导入');
  }
  return { calibrations, angle_conventions };
}

async function importRecording(path, discover = true, useMetadata = true, hand = null, aligned = false) {
  if (state.importBusy || state.calibrationBusy || state.liveBusy || state.live?.recording?.state === 'recording') return;
  if (!correctionEditor.canLeave()) return;
  state.importBusy = true;
  updateActions();
  let version;
  try {
    let metadata;
    const text = !aligned && useMetadata ? $('metadata').value.trim() : '';
    if (text) {
      try { metadata = JSON.parse(text); } catch { throw new Error('补充元数据不是有效 JSON'); }
      if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) throw new Error('补充元数据必须是 JSON 对象');
    }
    const alignedParameters = aligned ? await readAlignedParameters(path) : null;
    if (!correctionEditor.canLeave()) return;
    setMode('offline');
    version = ++state.importVersion;
    state.recording = null;
    correctionEditor.beginLoading();
    selectModule('correction-section');
    invalidateCalibration(); renderCalibrationList(); updateActions();
    $('recording-info').textContent = '正在只读发现 / 导入…';
    $('recording-warnings').replaceChildren();
    if (discover || aligned) $('recording-candidates-row').hidden = true;
    await refreshInfo();
    if (version !== state.importVersion) return;
    if (aligned) {
      const pair = await api('/api/recordings/open-aligned', { path, ...alignedParameters });
      if (version !== state.importVersion) return;
      acceptRecordingPair(pair, path);
      $('aligned-import-status').textContent = '双手参数已手动导入，仅用于当前录制；未写默认标定或源文件，未提交处理或导出。';
      return;
    }
    if (discover && !metadata) {
      const found = await api('/api/recordings/discover', { path });
      if (version !== state.importVersion) return;
      if (!found.recordings.length) throw new Error('未发现可导入的原始录制');
      if (found.recordings.length > 1) {
        $('recording-candidates').replaceChildren(new Option('请选择单手原始录制', ''));
        for (const candidate of found.recordings) $('recording-candidates').add(new Option(`${identity(candidate)} · ${candidate.path}`, JSON.stringify({ path: candidate.path, hand: candidate.hand })));
        $('recording-candidates-row').hidden = false;
        if (found.recordings.length === 2 && found.recordings.some((item) => item.hand === 'left') && found.recordings.some((item) => item.hand === 'right')) {
          const pair = await api('/api/recordings/open-pair', { path });
          if (version !== state.importVersion) return;
          acceptRecordingPair(pair, path);
          return;
        }
        $('recording-info').textContent = `发现 ${found.recordings.length} 个候选，不能唯一配对。请选择单手录制导入。`;
        correctionEditor.loadingFailed(new Error('候选录制不能唯一配对，请在“选择原始子任务”模块选择单手录制后重新导入。'));
        selectModule('recording-candidates-row');
        return;
      }
      path = found.recordings[0].path;
      hand = found.recordings[0].hand;
    }
    const recording = await api('/api/recordings/open', { path, hand, ...(metadata ? { metadata } : {}) });
    if (version !== state.importVersion) return;
    if (!acceptRecording(recording)) return;
  } catch (error) {
    if (version === undefined || version === state.importVersion) {
      if (version !== undefined) {
        correctionEditor.loadingFailed(error);
        $('recording-info').textContent = `导入失败：${error.message}`;
      }
      if (aligned) $('aligned-import-status').textContent = `手动导入失败：${error.message}`;
      $('advanced-import').open = $('aligned-import').open = true;
      selectModule('recording-section');
      throw error;
    }
  } finally {
    state.importBusy = false;
    updateActions();
  }
}
bind('import-form', 'submit', () => importRecording($('recording-path').value.trim()));
bind('aligned-import-form', 'submit', () => importRecording($('recording-path').value.trim(), false, false, null, true));
bind('import-candidate', 'click', () => {
  if (!$('recording-candidates').value) throw new Error('请先明确选择左 / 右手候选录制');
  const candidate = JSON.parse($('recording-candidates').value);
  return importRecording(candidate.path, false, true, candidate.hand);
});

const poseGuides = {
  thumb: '在录制中找到手掌展开、拇指伸直并外展约 60° 的静止区间。',
  index: '在录制中找到食指伸直、并拢且与掌面保持零位的静止区间。',
  middle: '在录制中找到中指伸直、并拢且与掌面保持零位的静止区间。',
  ring: '在录制中找到无名指伸直、并拢且与掌面保持零位的静止区间。',
  pinky: '在录制中找到小拇指伸直、并拢且与掌面保持零位的静止区间。',
  pinky_cmc: '在录制中找到小拇指根部贴平掌面的静止区间。',
};
function renderGroups() {
  const groups = state.info?.groups ?? [];
  $('groups').replaceChildren();
  groups.forEach((group, index) => {
    const li = document.createElement('li');
    const button = document.createElement('button');
    const captured = state.calibration.captured_groups.includes(group.name);
    button.textContent = `${index + 1}. ${group.label}${captured ? ' · 已采集' : ''}`;
    button.className = `${index === state.group ? 'selected' : ''} ${captured ? 'complete' : ''}`;
    button.type = 'button';
    button.addEventListener('click', () => { state.group = index; $('pose-confirm').checked = false; renderGroups(); updateActions(); });
    li.append(button); $('groups').append(li);
  });
  const group = groups[state.group];
  $('group-title').textContent = group ? `${group.label} · ${group.channels.join('、')}` : '未收到标定分组';
  const guide = poseGuides[group?.name] ?? '请对照外骨骼零位模型确认本组姿态。';
  $('pose-guide').textContent = state.mode === 'live' ? guide.replace('在录制中找到', '请现在摆出').replace('的静止区间。', '的姿态，并保持静止。') : guide;
  $('calibration-status').textContent = `${identity(state.calibration)} · 已完成 ${state.calibration.captured_groups.length}/${groups.length} 组`;
}
function renderChannels() {
  $('channel-values').replaceChildren();
  const calibration = state.calibration;
  if (!calibration) return;
  for (let index = 0; index < 21; index++) {
    const name = `J${index + 1}`;
    const cs = calibration.capture_cs_by_joint?.[index];
    const rawIndex = calibration.expected_cs_by_joint?.indexOf(cs) ?? -1;
    const rawName = rawIndex < 0 ? null : `J${rawIndex + 1}`;
    const tr = document.createElement('tr');
    const values = [name, rawName === null ? '映射不可用' : `${rawName} / CS${cs}`, format(state.raw?.angles_deg?.[rawIndex]), format(rawName === null ? null : calibration.zero_offsets_deg[rawName])];
    for (const value of values) { const td = document.createElement('td'); td.textContent = value; tr.append(td); }
    const jointName = state.info?.joint_definitions?.find((item) => item.channel === name)?.joint_name;
    if (jointName) tr.children[0].title = jointName;
    tr.children[2].dataset.rawIndex = rawIndex;
    const td = document.createElement('td');
    const select = document.createElement('select');
    select.setAttribute('aria-label', `${name}${rawName === null ? ' 映射不可用' : `（原始${rawName} / CS${cs}）`} 编码器方向`);
    select.add(new Option('+1 正向', '1')); select.add(new Option('−1 反向', '-1'));
    if (rawName === null) select.add(new Option('映射不可用', ''));
    select.value = rawName === null ? '' : calibration.encoder_directions[rawName];
    select.dataset.mappingMissing = String(rawName === null);
    select.disabled = state.calibrationBusy || rawName === null;
    select.addEventListener('change', () => {
      if (rawName === null) return;
      state.calibration.encoder_directions[rawName] = Number(select.value);
      ++state.calVersion;
      state.calibrationDirty = true;
      $('save-status').textContent = '方向已修改，尚未保存';
      updateActions();
      void previewFrame();
    });
    td.append(select); tr.append(td); $('channel-values').append(tr);
  }
}
function renderWindow() {
  $('pose-confirm').checked = false; updateActions();
  const recording = state.recording;
  if (!recording) return;
  const start = Number($('window-start').value), stop = Number($('window-stop').value);
  const valid = Number.isInteger(start) && Number.isInteger(stop) && start >= 0 && start < stop && stop <= recording.count;
  $('window-summary').textContent = valid ? `[${start}, ${stop}) · ${stop - start} 帧 · ${format(recording.time_s[start], 3)} → ${format(recording.time_s[stop - 1], 3)} 秒 · 实际跨度 ${format(recording.time_s[stop - 1] - recording.time_s[start], 3)} 秒` : '区间无效：必须满足 0 ≤ start < stop ≤ 总帧数';
}
function quickWindow() {
  if (!state.recording) return;
  const { time_s: times, count } = state.recording;
  const start = state.rawIndex;
  const end = lowerBound(times, times[start] + 1.5);
  $('window-start').value = start;
  $('window-stop').value = Math.min(count, end + 1);
  renderWindow();
}
async function activateCalibration(calibration) {
  const source = calibrationSource();
  if (!source || calibration.hand !== source.hand || calibration.device_id !== source.device_id) throw new Error('标定身份与当前数据源不匹配，拒绝载入');
  state.calibration = calibration; ++state.calVersion;
  state.calibrationSession = state.mode === 'live' ? source.session_id : null;
  state.group = Math.max(0, state.info.groups.findIndex((group) => !calibration.captured_groups.includes(group.name)));
  $('calibration-editor').hidden = false;
  $('save-status').textContent = ''; $('pose-confirm').checked = false;
  renderGroups(); renderChannels(); updateActions();
  if (state.mode === 'live') { state.raw = liveFresh() ? state.live.frame : null; updateRawChannels(); await previewFrame(); }
  else { quickWindow(); await showRawFrame(state.rawIndex); }
}
async function obtainCalibration(path, body, dirty = false) {
  if (!calibrationSource() || state.calibrationBusy || state.liveBusy) return;
  const key = sourceKey(), version = state.calVersion;
  state.calibrationBusy = true; updateActions();
  try {
    const calibration = await api(path, body);
    if (key !== sourceKey() || version !== state.calVersion) return;
    if (state.mode === 'live') {
      // 载入文件也必须由当前实时会话校验接线，不能仅凭手侧/设备号认定可用。
      await api('/api/live/calibration/preview', { session_id: state.live.session_id, calibration });
      if (key !== sourceKey() || version !== state.calVersion || !liveFresh()) return;
    }
    state.calibrationDirty = dirty;
    await activateCalibration(calibration);
  } finally { if (key === sourceKey()) { state.calibrationBusy = false; updateActions(); } }
}
bind('new-calibration', 'click', async () => {
  if (state.calibration && !confirm('新建会丢弃当前尚未保存的标定编辑，是否继续？')) return;
  if (state.mode === 'live') await obtainCalibration('/api/live/calibration/new', {}, true);
  else if (state.recording) await obtainCalibration('/api/calibration/new', { recording_id: state.recording.id }, true);
});
bind('load-calibration', 'click', async () => {
  if (state.calibration && !confirm('载入将替换当前标定编辑，是否继续？')) return;
  if (selectedCalibration()) await obtainCalibration(`/api/calibrations/${encodeURIComponent(selectedCalibration())}`);
});
bind('window-current', 'click', quickWindow);
bind('window-start', 'input', renderWindow);
bind('window-stop', 'input', renderWindow);
bind('window-goto', 'click', () => showRawFrame(Number($('window-start').value)));
bind('pose-confirm', 'change', updateActions);
bind('capture', 'click', async () => {
  if (state.mode === 'live') { await captureLiveGroup(); return; }
  const recording = state.recording, calibration = state.calibration;
  if (!recording || !calibration || !$('pose-confirm').checked || state.calibrationBusy) return;
  const start = Number($('window-start').value), stop = Number($('window-stop').value);
  if (!Number.isInteger(start) || !Number.isInteger(stop) || start < 0 || stop <= start || stop > recording.count) throw new Error('零位区间必须为合法整数 [start, stop)，结束帧不包含');
  const group = state.info.groups[state.group];
  const version = state.calVersion;
  state.calibrationBusy = true; updateActions();
  try {
    const next = await api('/api/calibration/capture', { recording_id: recording.id, calibration, group: group.name, start, stop });
    if (recording !== state.recording || version !== state.calVersion) return;
    state.calibration = next; ++state.calVersion;
    state.calibrationDirty = true;
    $('pose-confirm').checked = false;
    $('save-status').textContent = `${group.label}已采集，尚未保存到文件`;
    renderGroups(); renderChannels(); await previewFrame();
  } finally { if (recording === state.recording) { state.calibrationBusy = false; updateActions(); } }
});

function showRawFrame(index) {
  const recording = state.recording;
  if (state.mode !== 'offline' || !recording || !state.calibration) return;
  index = clampIndex(index, recording.count);
  state.rawIndex = index;
  const version = ++state.rawVersion;
  ++state.previewVersion;
  previewTarget = null;
  rawRequest?.abort(); previewRequest?.abort();
  state.raw = null;
  updateRawChannels();
  $('raw-slider').value = $('raw-index').value = index;
  $('raw-time').textContent = `正在读取第 ${index} 帧…`;
  rawTarget = { recording, index, version, calVersion: state.calVersion };
  if (!rawDrain) rawDrain = drainRawFrames().finally(() => { rawDrain = null; });
  return rawDrain;
}
async function drainRawFrames() {
  while (rawTarget) {
    const { recording, index, version, calVersion } = rawTarget;
    rawTarget = null;
    if (state.mode !== 'offline' || recording !== state.recording || calVersion !== state.calVersion) continue;
    const current = () => version === state.rawVersion && recording === state.recording && calVersion === state.calVersion && state.mode === 'offline';
    rawRequest = new AbortController();
    try {
      const frame = await api(`/api/recordings/${encodeURIComponent(recording.id)}/frame?index=${index}`, undefined, 'GET', { signal: rawRequest.signal });
      if (!current()) continue;
      state.raw = frame;
      $('raw-time').textContent = `帧 ${index} / ${recording.count - 1} · 相对时间 ${format(frame.time_s, 6)} s · timestamp_ns=${frame.timestamp_ns}`;
      updateRawChannels(); await previewFrame();
    } catch (error) {
      if (current() && error.name !== 'AbortError') {
        views.preview.hide(`原始帧读取失败：${error.message}`);
        $('raw-time').textContent = `第 ${index} 帧读取失败`; errorMessage(error);
      }
    } finally { rawRequest = null; }
  }
}
function previewFrame() {
  if (state.mode === 'live') return previewLiveFrame();
  if (!state.recording || !state.calibration || !state.raw) return;
  previewTarget = {
    recording: state.recording, calibration: state.calibration, index: state.rawIndex,
    version: ++state.previewVersion, calVersion: state.calVersion,
  };
  if (!previewDrain) previewDrain = drainPreviews().finally(() => { previewDrain = null; });
  return previewDrain;
}
async function drainPreviews() {
  while (previewTarget) {
    const { recording, calibration, index, version, calVersion } = previewTarget;
    previewTarget = null;
    if (state.mode !== 'offline' || recording !== state.recording || calVersion !== state.calVersion) continue;
    const current = () => version === state.previewVersion && recording === state.recording && calVersion === state.calVersion && state.mode === 'offline';
    previewRequest = new AbortController();
    try {
      const [pose, loaded] = await Promise.all([
        api('/api/calibration/preview', { recording_id: recording.id, calibration, index }, 'POST', { signal: previewRequest.signal }),
        views.preview.load(recording.hand, 'exoskeleton'),
      ]);
      if (current() && loaded) views.preview.pose(pose.joint_names, pose.qpos_rad);
    } catch (error) {
      if (current() && error.name !== 'AbortError') views.preview.hide(`预览失败：${error.message}`);
    } finally { previewRequest = null; }
  }
}
bind('raw-slider', 'input', (event) => showRawFrame(event.target.value));
bind('raw-index', 'change', (event) => showRawFrame(event.target.value));
bind('raw-prev', 'click', () => showRawFrame(state.rawIndex - 1));
bind('raw-next', 'click', () => showRawFrame(state.rawIndex + 1));
bind('reset-preview', 'click', () => views.preview.reset());
bind('refresh-calibrations', 'click', refreshInfo);
bind('calibration-select', 'change', updateActions);
bind('save-calibration', 'click', async () => {
  if (!state.calibration || state.calibrationBusy) return;
  const calibration = structuredClone(state.calibration);
  const key = sourceKey(), version = state.calVersion;
  state.calibrationBusy = true; updateActions();
  try {
    let saved;
    try { saved = await api('/api/calibration/save', { calibration, overwrite: false }); }
    catch (error) {
      if (error.status !== 409) throw error;
      if (!confirm(`${error.message}\n将覆盖 ${identity(calibration)} 的已有标定。确认覆盖？`)) return;
      saved = await api('/api/calibration/save', { calibration, overwrite: true });
    }
    if (key !== sourceKey() || version !== state.calVersion) return;
    state.calibrationDirty = false;
    $('save-status').textContent = `已保存：${saved.path}`;
    await refreshInfo();
    if (key === sourceKey()) renderCalibrationList(state.mode === 'offline' ? saved.name : undefined, state.mode === 'live' ? saved.name : undefined);
  } finally { if (key === sourceKey()) { state.calibrationBusy = false; updateActions(); } }
});

bind('start-job', 'click', async () => {
  if (state.jobSubmitting || !state.recording || !$('calibration-select').value) return;
  const recording = state.recording;
  const calibrationName = $('calibration-select').value;
  const selfCollision = $('job-self-collision').checked;
  state.jobSubmitting = true; updateActions();
  $('job-status').textContent = `正在提交 ${identity(recording)} · ${calibrationName}…`;
  try {
    const job = await exportQueue.submit('/api/jobs', { recording_id: recording.id, calibration_name: calibrationName, self_collision: selfCollision });
    $('job-status').textContent = `已加入全局任务队列：${job.id}。可继续编辑或导入其他录制，进度见“全局导出队列”模块。`;
  } catch (error) { $('job-status').textContent = `提交失败：${error.message}`; throw error; }
  finally { state.jobSubmitting = false; updateActions(); }
});

async function refreshResults() {
  const data = await api('/api/results');
  const selected = $('result-select').value;
  $('result-select').replaceChildren(new Option('请选择已有处理结果', ''));
  for (const result of data.results) $('result-select').add(new Option(result.name, result.path));
  if (data.results.some((result) => result.path === selected)) $('result-select').value = selected;
}
function pause(syncFrame = true) {
  const wasPlaying = state.playing;
  state.playing = false; $('play').textContent = '播放';
  for (const side of resultSides) side.camera.pause();
  if (wasPlaying && syncFrame && state.result) void showResultTime(state.resultTime);
}
function clearStages(side, reason) {
  for (const [key, name] of resultStages) {
    side.views[key].hide(reason);
    $(`${side.prefix}-${name}-values`).textContent = reason;
  }
}
function enablePlayback(enabled) {
  for (const id of ['prev-frame', 'play', 'next-frame', 'result-seconds', 'result-slider']) $(id).disabled = !enabled;
}
async function openResult(path, hand = null) {
  if (!path?.trim()) throw new Error('请填写或选择 processed.h5 路径');
  selectModule('result-replay');
  pause(false);
  const version = ++state.resultVersion;
  state.result = null; state.resultTime = 0; state.resultTimes = [];
  enablePlayback(false); $('download').hidden = true;
  $('result-slider').value = $('result-seconds').value = 0;
  $('result-time').textContent = '时间轴未加载';
  for (const side of resultSides) {
    ++side.version;
    side.target = null; side.request?.abort();
    side.result = null; side.times = []; side.index = -1;
    side.camera.clear();
    $(side.prefix).hidden = true;
    $(`${side.prefix}-info`).textContent = '';
    $(`${side.prefix}-time`).textContent = '';
    for (const view of Object.values(side.views)) view.clear('正在打开结果…');
    clearStages(side, '正在打开结果…');
  }
  $('result-info').textContent = '正在只读打开结果…';
  try {
    const first = await api('/api/results/open', { path: path.trim(), ...(hand ? { hand } : {}) });
    if (version !== state.resultVersion) return;
    checkTimeline(first);
    const hands = [...new Set([first.hand, ...(first.available_hands ?? [])])];
    const opened = await Promise.allSettled(hands.map(async (side) => {
      const result = side === first.hand ? first : await api('/api/results/open', { path: first.path, hand: side });
      checkTimeline(result);
      if (result.hand !== side) throw new Error('结果返回手侧与请求不一致');
      return result;
    }));
    if (version !== state.resultVersion) return;
    const loads = [];
    for (const side of resultSides) {
      const entry = opened[hands.indexOf(side.hand)];
      if (!entry) continue;
      $(side.prefix).hidden = false;
      if (entry.status === 'rejected') {
        const reason = `${handLabel(side.hand)}打开失败：${entry.reason.message}`;
        $(`${side.prefix}-info`).textContent = reason;
        clearStages(side, reason);
        continue;
      }
      const result = entry.value;
      side.result = result;
      side.times = result.time_s.map((time) => time - result.time_s[0]);
      const collisionNote = result.wujihand2_self_collision?.enabled === true
        ? 'Hand2：已启用逐帧自碰撞约束'
        : result.wujihand2_self_collision?.enabled === false
          ? 'Hand2：未启用自碰撞优化（快速模式，结果可能穿模）'
          : 'Hand2：此结果未记录自碰撞配置';
      $(`${side.prefix}-info`).textContent = `${identity(result)} · ${result.count} 帧 · ${format(side.times.at(-1), 6)} 秒\n${collisionNote}`;
      void side.camera.open(result);
      loads.push(side.views.exoskeleton.load(side.hand, 'exoskeleton'), side.views.wujihand2.load(side.hand, 'wujihand2'));
    }
    state.resultTimes = [...new Set(resultSides.flatMap((side) => side.times))].sort((a, b) => a - b);
    $('result-path').value = first.path;
    $('result-info').textContent = `${first.path}\n${hands.map(handLabel).join(' + ')} · 各侧独立时间戳与帧号；公共相对秒轴 ${format(state.resultTimes.at(-1), 6)} 秒。\n各设备编码器首帧相对对齐，不代表跨设备硬件同步；下载保留整份 HDF5。`;
    $('result-slider').max = $('result-seconds').max = state.resultTimes.at(-1);
    $('download').href = `/api/results/${encodeURIComponent(first.id)}/download`;
    $('download').hidden = false;
    $('download').setAttribute('download', 'processed.h5');
    await Promise.all(loads);
    if (version !== state.resultVersion) return;
    state.result = first;
    enablePlayback(true);
    await showResultTime(0);
  } catch (error) {
    if (version !== state.resultVersion) return;
    state.result = null;
    enablePlayback(false);
    $('result-info').textContent = `打开结果失败：${error.message}`;
    for (const side of resultSides) clearStages(side, error.message);
    throw error;
  }
}
function renderValues(id, names, values) {
  const fragment = document.createDocumentFragment();
  names.forEach((name, index) => {
    const row = document.createElement('div');
    const label = document.createElement('span'); label.textContent = name;
    const value = document.createElement('code'); value.textContent = Array.isArray(values[index]) ? values[index].map((item) => format(item, 5)).join(', ') : format(values[index], 5);
    row.append(label, value); fragment.append(row);
  });
  $(id).replaceChildren(fragment);
}
function showResultTime(value) {
  if (!state.result) return;
  const time = Math.max(0, Math.min(state.resultTimes.at(-1), Number(value) || 0));
  state.resultTime = time;
  $('result-slider').value = time;
  $('result-seconds').value = format(time, 6);
  $('result-time').textContent = `公共相对时间 ${format(time, 6)} / ${format(state.resultTimes.at(-1), 6)} s · 各侧目标帧与实际显示帧见下方`;
  const drains = [];
  for (const side of resultSides) {
    const result = side.result;
    if (!result) continue;
    side.camera.follow(time, state.playing, Number($('speed').value));
    const index = time < side.times[0] || time > side.times.at(-1) ? -1 : upperBound(side.times, time) - 1;
    if (state.playing && index === side.index) continue;
    // 连续播放保留请求合并；主动定位和越界废弃此前响应。
    if (!state.playing || index < 0) {
      ++side.version;
      side.request?.abort();
    }
    side.index = index;
    if (index < 0) {
      side.target = null;
      const reason = `当前时间超出${handLabel(side.hand)}范围（0 ～ ${format(side.times.at(-1), 6)} s），不显示末帧`;
      clearStages(side, reason);
      $(`${side.prefix}-time`).textContent = reason;
      continue;
    }
    $(`${side.prefix}-time`).textContent = `目标帧 ${index} / ${result.count - 1} · ${format(side.times[index], 6)} s · 正在读取，模型可能仍显示上一有效帧`;
    for (const [key, name] of resultStages) {
      if (result.valid?.[key]?.[index] !== true) {
        const reason = '无效 · 本阶段 valid=false；详细原因见处理文件';
        side.views[key].hide(reason); $(`${side.prefix}-${name}-values`).textContent = reason;
      }
    }
    side.target = { result, index, version: side.version };
    if (!side.drain) side.drain = drainResultFrames(side).finally(() => { side.drain = null; });
    if (!state.playing) drains.push(side.drain);
  }
  if (drains.length) return Promise.all(drains);
}
async function drainResultFrames(side) {
  while (side.target) {
    const { result, index, version } = side.target;
    side.target = null;
    const current = () => version === side.version && result === side.result && side.index >= 0;
    if (!current()) continue;
    side.request = new AbortController();
    try {
      const frame = await api(`/api/results/${encodeURIComponent(result.id)}/frame?index=${index}`, undefined, 'GET', { signal: side.request.signal });
      if (!current()) continue;
      $(`${side.prefix}-time`).textContent = `目标帧 ${side.index} · 已读取帧 ${index} / ${result.count - 1} · ${format(side.times[index], 6)} s\n文件 time_s=${format(frame.time_s, 6)} · timestamp_ns=${frame.timestamp_ns}`;
      for (const [key, name, namesKey] of resultStages) {
        // 旧的连续播放采样不能覆盖当前目标的无效状态。
        if (index !== side.index && result.valid?.[key]?.[side.index] !== true) continue;
        const values = frame[key], id = `${side.prefix}-${name}-values`;
        if (result.valid?.[key]?.[index] !== true || values === null || values === undefined) {
          const reason = `无效 · ${frame.errors?.[key] || '本阶段 valid=false 或数据缺失；详细原因见处理文件'}`;
          side.views[key].hide(reason); $(id).textContent = reason;
        } else {
          const names = result[namesKey];
          if (key === 'hand') side.views[key].skeleton(values);
          else side.views[key].pose(names, values);
          renderValues(id, names, values);
        }
      }
    } catch (error) {
      if (current() && error.name !== 'AbortError') {
        clearStages(side, `本帧读取失败：${error.message}`);
        $(`${side.prefix}-time`).textContent = `${handLabel(side.hand)}第 ${index} 帧读取失败：${error.message}`;
      }
    } finally { side.request = null; }
  }
}
function seek(time) { pause(false); return showResultTime(time); }
function stepResult(direction) {
  const times = state.resultTimes;
  if (!times.length) return;
  const index = direction < 0 ? lowerBound(times, state.resultTime) - 1 : upperBound(times, state.resultTime);
  return seek(times[clampIndex(index, times.length)]);
}
bind('result-form', 'submit', () => openResult($('result-path').value));
bind('refresh-results', 'click', refreshResults);
bind('open-selected-result', 'click', () => openResult($('result-select').value));
bind('prev-frame', 'click', () => stepResult(-1));
bind('next-frame', 'click', () => stepResult(1));
bind('result-slider', 'input', (event) => seek(event.target.value));
bind('result-seconds', 'change', (event) => seek(event.target.value));
for (const side of resultSides) bind(`${side.prefix}-reset`, 'click', () => {
  for (const view of Object.values(side.views)) view.reset();
});
bind('reset-all', 'click', () => {
  for (const side of resultSides) for (const view of Object.values(side.views)) view.reset();
});
bind('play', 'click', () => {
  if (state.playing) { pause(); return; }
  if (!state.result) return;
  if (state.resultTime >= state.resultTimes.at(-1)) void showResultTime(0);
  state.playing = true;
  state.playOrigin = performance.now();
  state.playTime = state.resultTime;
  state.playRate = Number($('speed').value);
  for (const side of resultSides) side.camera.follow(state.playTime, true, state.playRate);
  $('play').textContent = '暂停';
});
bind('speed', 'change', () => {
  const now = performance.now();
  if (state.playing) {
    state.playTime = Math.min(state.resultTimes.at(-1), state.playTime + (now - state.playOrigin) / 1000 * state.playRate);
    state.playOrigin = now;
    state.playRate = Number($('speed').value);
    void showResultTime(state.playTime);
  }
});
function playbackTick(now) {
  correctionEditor.tick(now);
  if (state.playing && state.result) {
    const end = state.resultTimes.at(-1);
    const targetTime = Math.min(end, state.playTime + (now - state.playOrigin) / 1000 * state.playRate);
    void showResultTime(targetTime);
    if (targetTime >= end) pause();
  }
  requestAnimationFrame(playbackTick);
}
requestAnimationFrame(playbackTick);
Promise.allSettled([refreshInfo(), refreshResults()]).then((results) => {
  const errors = results.filter((result) => result.status === 'rejected').map((result) => result.reason.message);
  if (errors.length) errorMessage(new Error(errors.join('\n')));
});

const initialParams = new URLSearchParams(window.location.search);
const initialResult = initialParams.get('result');
const initialRecording = initialParams.get('recording');
if (initialResult) {
  $('result-path').value = initialResult;
  void openResult(initialResult).catch(errorMessage);
} else if (initialRecording) {
  $('advanced-import').open = true;
  $('recording-path').value = initialRecording;
  void importRecording(initialRecording).catch(errorMessage);
}
if (!initialRecording || initialResult) void refreshTasks().catch(errorMessage);
