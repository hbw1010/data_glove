import { ModelView } from './scene.js';
import { createWristReplay } from './wrist-replay.js';
import { createFrameAdjustments, frameAtRelativeTime } from './frame-adjustments.js';

export function createCorrectionEditor({ api, upperBound, handLabel, format, refreshInfo, submitJob }) {
  const $ = (id) => document.getElementById(`correction-${id}`);
  const joints = Array.from({ length: 21 }, (_, index) => `J${index + 1}`);
  const cameras = [['glove', 'glove-video'], ['extra', 'extra-cameras']].map(([prefix, source]) =>
    createWristReplay({ api, upperBound, handLabel, format, prefix: `correction-${prefix}`, namespace: 'recordings', source, externalTimeline: true, gated: true }));
  const views = new Map();
  let states = {}, activeHand = '', calibrations = [], generation = 0;
  let times = [], time = 0, playing = false, playOrigin = 0, playTime = 0, submitting = false;
  let preparation = 'idle', initialAligned = false, preparationError = '', progressKey = '', holding = false;
  let bufferingSince = null, recoveredSince = null, bufferingNoticeVisible = false;
  const rows = new Map();
  const all = () => Object.values(states);
  const alive = (s) => states[s.source.hand] === s;
  const valid = (s) => !!s?.session && joints.every((name) => Number.isFinite(s.delta[name]) && (s.directions[name] === 1 || s.directions[name] === -1));
  const exportable = (s) => valid(s) && !s.busy && !s.segmentDraft;
  const segments = createFrameAdjustments({
    slider: $('slider'), getState: () => states[activeHand], joints, handLabel, pause, seek, fail,
    onDraftChange: (s) => { ++s.revision; s.controller?.abort(); for (const row of rows.values()) row.refreshEffective(); queuePreview(s); },
    onApplied: changed,
    onModeChange: () => { $('error').hidden = true; renderJoints(); status(); },
  });

  function fail(error) { $('error').textContent = error.message ?? String(error); $('error').hidden = false; }
  function bind(id, event, action) {
    $(id).addEventListener(event, async (e) => {
      e.preventDefault(); $('error').hidden = true;
      const version = generation;
      try { await action(e); } catch (error) { if (version === generation) fail(error); }
    });
  }
  function on(s, element, event, action) {
    element.addEventListener(event, async () => {
      if (!alive(s)) return;
      $('error').hidden = true;
      try { await action(); } catch (error) { if (alive(s)) fail(error); }
    });
  }
  function updateBufferingNotice(now) {
    let visible = false;
    if (!playing) {
      bufferingSince = recoveredSince = null;
    } else if (holding) {
      recoveredSince = null;
      bufferingSince ??= now;
      visible = bufferingNoticeVisible || now - bufferingSince >= 350;
    } else {
      bufferingSince = null;
      recoveredSince ??= now;
      visible = bufferingNoticeVisible && now - recoveredSince < 500;
    }
    if (visible === bufferingNoticeVisible) return;
    bufferingNoticeVisible = visible;
    $('buffering').dataset.visible = String(visible);
    $('buffering').setAttribute('aria-hidden', String(!visible));
  }

  function pause() {
    playing = false; holding = false; $('play').textContent = '播放';
    updateBufferingNotice(performance.now());
    for (const camera of cameras) camera.pause();
  }
  function status() {
    const list = all(), pair = list.length === 2;
    for (const s of list) {
      s.ui.calibration.disabled = s.busy;
      s.ui.new.disabled = s.busy || !s.ui.calibration.value;
      s.ui.status.textContent = !s.session ? '尚未建立草稿'
        : !valid(s) ? '零位增量或编码器方向无效，姿态已隐藏，不能保存或导出'
          : s.dirty ? '本录制草稿尚未保存（已加载所选基线）' : '当前参数已保存';
      if (s.busy) s.ui.status.textContent += ' · 正在处理';
      if (s.segmentDraft) s.ui.status.textContent += ' · 区间尚未应用';
      s.ui.tab.textContent = `${handLabel(s.source.hand)}${s.dirty ? ' · 未保存' : ''}`;
      s.ui.tab.setAttribute('aria-selected', String(activeHand === s.source.hand));
    }
    $('new').textContent = pair ? '建立双手草稿' : '建立草稿';
    $('save').textContent = pair ? '保存并覆盖两侧原标定' : '保存并覆盖原标定';
    $('export').textContent = submitting ? '正在保存并提交…' : pair ? '一键正式导出双手全部帧' : '正式导出全部帧';
    $('new').disabled = submitting || !list.length || list.some((s) => s.busy || !s.ui.calibration.value);
    $('save').disabled = preparation !== 'ready' || submitting || !list.length || !list.every(exportable);
    $('export').disabled = preparation !== 'ready' || submitting || !list.length || !list.every(exportable);
    $('reset-all').disabled = !states[activeHand]?.session;
    $('reset-all').textContent = states[activeHand]?.segmentDraft ? '当前区间局部增量复位为 0°' : '当前手全部全局增量复位为 0°';
    for (const id of ['play', 'prev', 'next', 'slider']) $(id).disabled = preparation !== 'ready' || !times.length;
    $('play').disabled ||= !!states[activeHand]?.segmentDraft;
    $('status').textContent = list.length ? '“保存”覆盖设备全局零位与方向，并保存本录制的完整修正；重新导入或建立草稿可恢复最近保存的零位、方向与区间。直接导出仅保存快照，不改原标定。当前草稿基线已冻结，刷新列表不会重新载入磁盘标定。' : '请先导入录制。';
    $('editor').hidden = !activeHand || preparation !== 'ready';
    $('editor').inert = preparation !== 'ready';
    segments.render();
  }
  function clearDraft(s, message) {
    ++s.revision; ++s.editRevision; ++s.activation;
    cancelAnimationFrame(s.timer); s.timer = null; s.requested = false; s.controller?.abort();
    s.session = null; s.delta = {}; s.directions = {}; s.frameAdjustments = []; s.segmentDraft = null; s.savedName = ''; s.dirty = false;
    s.modelReady = false; s.poseIndex = -1; s.poseRevision = -1; s.prepareError = '';
    s.view?.clear(message); $(`${s.source.hand}-time`).textContent = '尚无有效预览帧';
    if (activeHand === s.source.hand) { rows.clear(); $('joints').replaceChildren(); }
  }
  function refreshBaseline(s, calibrationName = '') {
    const matching = calibrations.filter((item) => item.hand === s.source.hand && item.device_id === s.source.device_id);
    s.ui.calibration.replaceChildren(new Option('请选择草稿基线', ''));
    if (s.source.saved_correction) {
      s.ui.calibration.add(new Option('最近保存的录制修正（零位/方向/区间）', '@saved'));
    }
    for (const item of matching) s.ui.calibration.add(new Option(item.name, item.name));
    if (s.source.recording_calibration_available) {
      s.ui.calibration.add(new Option(s.source.recording_calibration_label, '@recording'));
    }
    if (s.selectedCalibration !== null) s.ui.calibration.value = s.selectedCalibration;
    else if (s.source.saved_correction) s.ui.calibration.value = '@saved';
    else if (matching.some((item) => item.name === calibrationName)) s.ui.calibration.value = calibrationName;
    else if (matching.length === 1) s.ui.calibration.value = matching[0].name;
    else if (s.source.recording_calibration_available) s.ui.calibration.value = '@recording';
    if (s.ui.calibration.value || s.selectedCalibration !== null) s.selectedCalibration = s.ui.calibration.value;
  }
  function refreshCalibrations(items, calibrationName = '') {
    calibrations = items;
    for (const s of all()) refreshBaseline(s, calibrationName);
    if (preparation === 'loading') prepareDrafts();
    status();
  }
  function createSide(source) {
    const hand = source.hand;
    const card = document.createElement('article');
    card.innerHTML = `<h3>${handLabel(hand)} · 原标定与微调</h3><p class="mono"></p><label>建立草稿时使用的基线<select data-role="calibration"></select></label><div class="row"><button data-role="new">建立本侧草稿</button></div><p data-role="status" class="notice" role="status"></p>`;
    card.querySelector('p').textContent = `device_id=${source.device_id} · ${source.count} 帧`;
    const ui = Object.fromEntries([...card.querySelectorAll('[data-role]')].map((element) => [element.dataset.role, element]));
    const tab = document.createElement('button'); tab.type = 'button'; tab.setAttribute('role', 'tab'); ui.tab = tab;
    $('sources').append(card); $('hand-tabs').append(tab);
    const s = { source, ui, session: null, delta: {}, directions: {}, frameAdjustments: [], segmentDraft: null, selectedCalibration: null, savedName: '', dirty: false, busy: false, revision: 0, editRevision: 0, activation: 0, index: -1, pending: false, requested: false, timer: null, controller: null, view: views.get(hand), modelReady: false, poseIndex: -1, poseRevision: -1, poseTime: 0, prepareError: '', autoStarted: false };
    on(s, tab, 'click', () => { if (activeHand !== hand && !segments.canLeave()) return; activeHand = hand; renderJoints(); status(); });
    on(s, ui.calibration, 'change', () => {
      if (!segments.canLeave()) { ui.calibration.value = s.selectedCalibration; return; }
      pause(); preparation = 'loading'; preparationError = '';
      s.selectedCalibration = ui.calibration.value; clearDraft(s, '基线已改变，请重新建立草稿'); status();
    });
    on(s, ui.new, 'click', () => activate(s));
    return s;
  }
  function setRecordings(recordings, calibrationName = '') {
    if (!segments.canLeave()) return false;
    ++generation; pause();
    preparation = recordings ? 'loading' : 'idle'; initialAligned = false; preparationError = ''; progressKey = '';
    for (const s of all()) { cancelAnimationFrame(s.timer); s.controller?.abort(); }
    for (const view of views.values()) view.clear('录制已切换，等待新草稿');
    states = {}; rows.clear(); $('joints').replaceChildren();
    for (const id of ['sources', 'hand-tabs']) $(id).replaceChildren();
    for (const camera of cameras) camera.clear();
    $('error').hidden = true;
    $('submission').hidden = true;
    for (const hand of ['left', 'right']) {
      const source = recordings?.[hand]; $(`${hand}-model`).hidden = !source;
      $(`${hand}-time`).textContent = '尚无有效预览帧';
      if (source) states[hand] = createSide(source);
    }
    activeHand = states.left ? 'left' : states.right ? 'right' : '';
    times = [...new Set(all().flatMap((s) => s.source.time_s.map((value) => value - s.source.time_s[0])))].sort((a, b) => a - b);
    time = 0; $('slider').max = times.at(-1) ?? 0; $('slider').value = 0;
    $('editor').hidden = true;
    $('source').textContent = all().map((s) => `${handLabel(s.source.hand)} · ${s.source.count} 帧 · ${s.source.path}`).join('\n') || '请先在“选择原始子任务”模块导入原始录制';
    refreshCalibrations(calibrations, calibrationName);
    const source = states.left?.source ?? states.right?.source;
    if (source) for (const camera of cameras) void camera.open(source);
    seek(0); status();
    prepareDrafts();
    return true;
  }
  function setRecording(source, calibrationName = '') {
    return setRecordings(source ? { [source.hand]: source } : {}, calibrationName);
  }

  function prepareDrafts() {
    for (const s of all()) if (!s.autoStarted && !s.busy && !s.session && s.ui.calibration.value) {
      s.autoStarted = true;
      void activate(s);
    }
  }

  function beginLoading() {
    if (!setRecordings(null)) return;
    preparation = 'importing'; renderProgress([]);
  }

  function loadingFailed(error) {
    preparationError = error.message ?? String(error); preparation = 'error';
    pause(); status(); renderProgress([]);
  }

  function renderProgress(media) {
    const list = all(), streams = media.flatMap((item) => item.cameras);
    const indexed = media.length === cameras.length && media.every((item) => !item.error && (item.phase === 'ready' || item.cameras.length > 0));
    const decoded = indexed && media.every((item) => item.ready);
    const modeled = list.length > 0 && list.every((s) => s.modelReady && !!s.session);
    const complete = preparation === 'ready';
    const steps = [
      [list.length > 0, '读取录制', list.length ? `${list.length} 侧真实时间轴已读取` : preparationError || '正在发现并读取原始数据'],
      [indexed, '校验视频索引', media.map((item, index) => `${index ? '其他相机' : '本侧手套'}：${item.error || (item.phase === 'ready' || item.cameras.length ? '索引已读取' : '处理中')}`).join('；') || '等待读取录制'],
      [decoded, '缓冲并验证视频', streams.map((item) => `${item.label}：${item.error || (item.ready ? '已解码' : '等待可解码画面')}`).join('；') || (indexed ? '此录制无关联视频' : '等待视频索引')],
      [modeled, '加载双手模型', list.map((s) => `${handLabel(s.source.hand)}：${s.prepareError || (s.modelReady ? '模型已就绪' : s.busy ? '载入标定与 URDF' : '请选择原标定并建立草稿')}`).join('；') || '等待读取录制'],
      [complete, '对齐公共时间', complete ? `就绪 · ${format(time, 6)} s` : preparationError || (initialAligned ? '等待视频定位及双方模型当前帧' : '等待视频和模型就绪')],
    ];
    const key = `${preparation}|${steps.map(([done, name, detail]) => `${done}:${name}:${detail}`).join('|')}`;
    if (key === progressKey) return;
    progressKey = key;
    $('loading').hidden = preparation === 'idle';
    $('loading').dataset.state = preparation;
    $('loading-title').textContent = complete ? '加载完成，可以开始微调' : preparation === 'error' ? '准备失败，尚未开放微调' : '正在准备共同回放';
    $('loading-progress').value = steps.filter(([done]) => done).length;
    $('loading-steps').replaceChildren(...steps.map(([done, name, detail]) => {
      const item = document.createElement('li'); item.dataset.done = String(done);
      const title = document.createElement('strong'); title.textContent = `${done ? '已完成' : '待完成'} · ${name}`;
      const text = document.createElement('span'); text.textContent = detail;
      item.append(title, text); return item;
    }));
    $('loading-retry').hidden = preparation !== 'error' || !list.length;
  }

  function updatePreparation(media) {
    if (preparation === 'idle' || preparation === 'importing') { renderProgress(media); return; }
    const list = all();
    const error = preparationError || media.find((item) => item.error)?.error || list.find((s) => s.prepareError)?.prepareError;
    if (error) {
      if (preparation !== 'error') { preparationError = error; preparation = 'error'; pause(); status(); }
      renderProgress(media); return;
    }
    if (list.length && media.every((item) => item.ready) && list.every((s) => s.modelReady && valid(s))) {
      if (!initialAligned) {
        const start = Math.max(0, ...media.map((item) => item.start ?? 0));
        const end = Math.min(...list.map((s) => s.source.time_s.at(-1) - s.source.time_s[0]), ...media.map((item) => item.end ?? Infinity));
        if (start > end) { loadingFailed(new Error('视频与双手录制没有共同有效时间范围，无法开始共同回放。')); return; }
        initialAligned = true; seek(start);
      } else if (media.every((item) => item.synchronized) && list.every((s) => s.poseIndex === s.index && s.poseRevision === s.revision)) {
        preparation = 'ready'; status();
      }
    }
    renderProgress(media);
  }
  function changed(s) {
    if (s.segmentDraft) { segments.draftChanged(); return; }
    ++s.revision; ++s.editRevision; s.controller?.abort(); s.dirty = true; s.savedName = '';
    status(); queuePreview(s);
  }
  function renderJoints() {
    rows.clear(); $('joints').replaceChildren();
    const s = states[activeHand];
    if (!s?.session) return;
    const values = s.segmentDraft?.delta ?? s.delta;
    const mode = s.segmentDraft ? '本段局部增量' : '全局增量';
    $('direction-hint').textContent = s.segmentDraft
      ? '区间内只修正零位，方向按钮已禁用；全局编码器方向请应用或取消区间后，回到普通模式编辑。归零不改变方向。'
      : '全局编码器方向：+1 正向，−1 反向，点击切换；作用于当前手全部帧，归零不改变方向。';
    const layout = new Map(s.session.joint_layout.map((item) => [item.channel, item]));
    for (const name of joints) {
      const item = layout.get(name);
      const baseIndex = s.session.base_calibration.expected_cs_by_joint.indexOf(item?.cs);
      const base = s.session.base_calibration.zero_offsets_deg[`J${baseIndex + 1}`];
      if (!item || baseIndex < 0 || !Number.isFinite(base) || !Number.isInteger(item.source_index)) throw new Error(`${name} 缺少完整 CS / 零位映射，不能微调`);
      if (s.directions[name] !== 1 && s.directions[name] !== -1) throw new Error(`${name} 缺少有效的标准关节编码器方向，不能微调`);
      const tr = document.createElement('tr'); tr.dataset.joint = name;
      const identity = document.createElement('td'), joint = document.createElement('strong'), mapping = document.createElement('small');
      joint.textContent = `${name} · ${item.label}`;
      mapping.textContent = `CS${item.cs} · 原始 J${item.source_index + 1}（列 ${item.source_index}） · 标定 J${baseIndex + 1}`;
      identity.append(joint, mapping);
      const control = document.createElement('td'); control.className = 'correction-adjustment';
      const slider = document.createElement('input'); slider.type = 'range'; slider.min = -180; slider.max = 180; slider.step = 0.1;
      slider.setAttribute('aria-label', `${handLabel(s.source.hand)} ${name} ${item.label} ${mode}滑条`); slider.dataset.joint = name;
      const input = document.createElement('input'); input.type = 'number'; input.step = 0.1;
      input.setAttribute('aria-label', `${handLabel(s.source.hand)} ${name} ${item.label} ${mode}数值`); input.dataset.joint = name;
      const calibration = document.createElement('td'); calibration.className = 'mono';
      const baseValue = document.createElement('small');
      const effective = document.createElement('span'); calibration.append(baseValue, effective);
      const resetCell = document.createElement('td'), reset = document.createElement('button'); reset.type = 'button'; reset.textContent = '归零';
      resetCell.className = 'correction-actions';
      const direction = document.createElement('button'); direction.type = 'button'; direction.className = 'correction-direction'; direction.dataset.joint = name;
      direction.disabled = !!s.segmentDraft;
      direction.setAttribute('aria-describedby', 'correction-direction-hint');
      const refreshDirection = () => {
        direction.textContent = s.directions[name] === 1 ? '+1' : '−1';
        direction.setAttribute('aria-label', `${handLabel(s.source.hand)} ${name} ${item.label} 全局编码器方向 ${direction.textContent}，点击反向`);
        direction.title = s.segmentDraft ? '全局编码器方向；请应用或取消区间后回到普通模式编辑' : '切换全局编码器方向，作用于当前手全部帧，不改变零位';
      };
      on(s, direction, 'click', () => {
        if (s.segmentDraft) return;
        s.directions[name] = -s.directions[name]; refreshDirection(); changed(s);
      });
      refreshDirection();
      const refreshEffective = () => {
        const local = segments.localValues(s)?.[name] ?? 0, total = base + s.delta[name] + local;
        baseValue.textContent = `基础 ${format(base, 3)}° + 全局 ${format(s.delta[name], 3)}° + 局部 ${format(local, 3)}° →`;
        effective.textContent = s.index < 0 ? '当前时间无本手帧' : Number.isFinite(total) ? `${format(((total % 360) + 360) % 360, 3)}°（当前帧）` : '无效';
      };
      const update = (value, keepInput = false) => {
        values[name] = value;
        if (!keepInput) input.value = Number.isFinite(value) ? value : '';
        slider.value = Number.isFinite(value) ? Math.min(180, Math.max(-180, value)) : 0;
        input.setAttribute('aria-invalid', String(!Number.isFinite(value)));
        slider.title = Number.isFinite(value) && Math.abs(value) > 180 ? '数值超出滑条范围；数值输入仍保留并生效。' : '滑条范围 −180° ～ +180°';
        refreshEffective();
      };
      input.addEventListener('input', () => { update(input.valueAsNumber, true); changed(s); });
      slider.addEventListener('input', () => { update(slider.valueAsNumber); changed(s); });
      reset.addEventListener('click', () => { update(0); changed(s); });
      update(values[name]); rows.set(name, { update, refreshEffective });
      control.append(slider, input); resetCell.append(direction, reset); tr.append(identity, control, calibration, resetCell); $('joints').append(tr);
    }
  }
  function queuePreview(s) {
    if (!alive(s)) return;
    s.requested = valid(s) && s.index >= 0;
    if (!s.requested) {
      s.view?.hide(s.index < 0 ? '当前公共时间超出本手范围，无帧' : '等待有效微调草稿');
      if (s.index < 0) $(`${s.source.hand}-time`).textContent = `当前 ${format(time, 6)} s 无帧 · 本手终点 ${format(s.source.time_s.at(-1) - s.source.time_s[0], 6)} s`;
      return;
    }
    if (!s.view?.root?.visible) s.view?.hide(`正在计算目标帧 ${s.index}…`);
    if (!s.pending && s.timer === null) {
      if (playing) void preview(s);
      else s.timer = requestAnimationFrame(() => { s.timer = null; void preview(s); });
    }
  }
  async function preview(s) {
    if (!alive(s) || !s.requested || !valid(s) || !s.modelReady || s.pending || s.index < 0) return;
    s.requested = false; s.pending = true;
    const revision = s.revision, session = s.session, index = s.index;
    const controller = new AbortController(); s.controller = controller;
    try {
      const frame = await api('/api/corrections/preview', { correction_id: session.id, index, zero_adjustments_deg: { ...s.delta }, encoder_directions: { ...s.directions }, frame_adjustments: segments.segments(s) }, 'POST', { signal: controller.signal });
      if (controller.signal.aborted || !alive(s) || revision !== s.revision || session !== s.session || s.index < 0) return;
      s.view?.pose(frame.joint_names, frame.qpos_rad);
      if (!s.view?.root?.visible) throw new Error('URDF 当前帧未能正确显示');
      s.poseIndex = frame.index; s.poseRevision = revision; s.poseTime = frame.time_s; s.prepareError = '';
      $(`${s.source.hand}-time`).textContent = `实际帧 ${frame.index} / ${s.source.count - 1} · ${format(frame.time_s - s.source.time_s[0], 6)} s\ntimestamp_ns=${frame.timestamp_ns}`;
    } catch (error) {
      if (!controller.signal.aborted && alive(s) && revision === s.revision && session === s.session) { s.prepareError = `预览失败：${error.message}`; s.view?.hide(s.prepareError); fail(error); }
    } finally {
      s.pending = false;
      if (s.controller === controller) s.controller = null;
      if (alive(s) && s.requested && s.timer === null) {
        if (playing) void preview(s);
        else s.timer = requestAnimationFrame(() => { s.timer = null; void preview(s); });
      }
    }
  }
  function seek(value, stop = true) {
    if (stop) pause();
    time = Math.max(0, Math.min(times.at(-1) ?? 0, Number(value) || 0));
    $('slider').value = time;
    $('time').textContent = `${format(time, 6)} / ${format(times.at(-1) ?? 0, 6)} s`;
    for (const camera of cameras) camera.follow(time, playing);
    for (const s of all()) {
      const index = frameAtRelativeTime(s.source, time);
      if (index !== s.index || stop) {
        if (stop || index < 0) { ++s.revision; s.controller?.abort(); }
        s.index = index; queuePreview(s);
      }
    }
    for (const row of rows.values()) row.refreshEffective();
  }
  async function activate(s) {
    if (s.busy || !alive(s) || !s.ui.calibration.value || !segments.canLeave()) return;
    const savedName = s.ui.calibration.value === '@saved' ? s.source.saved_correction.name : '';
    const path = savedName ? '/api/corrections/load' : '/api/corrections/new';
    const body = savedName ? { recording_id: s.source.id, name: savedName }
      : { recording_id: s.source.id, calibration_name: s.ui.calibration.value === '@recording' ? null : s.ui.calibration.value };
    pause(); preparation = 'loading'; preparationError = '';
    clearDraft(s, '正在建立微调草稿…'); s.busy = true; status();
    const activation = s.activation;
    try {
      const data = await api(path, body);
      if (!alive(s) || activation !== s.activation) return;
      s.session = data; s.delta = { ...data.zero_adjustments_deg }; s.directions = { ...data.encoder_directions }; s.savedName = savedName; s.dirty = !savedName;
      s.frameAdjustments = (data.frame_adjustments ?? []).map((segment) => ({ ...segment, zero_adjustments_deg: { ...segment.zero_adjustments_deg } }));
      if (activeHand === s.source.hand) renderJoints();
      if (!s.view) { s.view = new ModelView($(`${s.source.hand}-viewport`), $(`${s.source.hand}-preview-status`)); views.set(s.source.hand, s.view); }
      s.modelReady = await s.view.load(s.source.hand, 'exoskeleton');
      if (!s.modelReady && alive(s) && activation === s.activation) throw new Error(s.view.modelError || '模型未能加载');
      if (!alive(s) || activation !== s.activation) return;
      queuePreview(s);
    } catch (error) {
      if (alive(s) && activation === s.activation) { clearDraft(s, '草稿加载失败'); s.prepareError = error.message; fail(error); }
    } finally { if (alive(s)) { s.busy = false; status(); } }
  }
  async function save(s, overwriteCalibration = false) {
    if (!segments.canLeave() || !valid(s) || s.busy || !alive(s)) return;
    if (!overwriteCalibration && !s.dirty && s.savedName) return s.savedName;
    const session = s.session, revision = s.editRevision;
    s.busy = true; status();
    try {
      const saved = await api('/api/corrections/save', { correction_id: session.id, zero_adjustments_deg: { ...s.delta }, encoder_directions: { ...s.directions }, frame_adjustments: segments.segments(s, false), overwrite_calibration: overwriteCalibration });
      if (!alive(s) || session !== s.session) return;
      if (revision === s.editRevision) { s.savedName = saved.name; s.dirty = false; }
      s.source.saved_correction = { name: saved.name, path: saved.path, created_at: saved.created_at };
      s.selectedCalibration = '@saved'; refreshBaseline(s);
      if (overwriteCalibration) {
        $('submission').textContent += `${handLabel(s.source.hand)}原标定已覆盖：${saved.calibration_path}；本次零位、方向与区间可从“最近保存的录制修正”恢复。\n`;
        $('submission').hidden = false;
      }
      return saved.name;
    } catch (error) { throw new Error(`${handLabel(s.source.hand)}保存失败：${error.message}`); }
    finally { if (alive(s) && session === s.session) { s.busy = false; status(); } }
  }
  bind('refresh', 'click', refreshInfo);
  bind('loading-retry', 'click', () => {
    pause(); preparation = 'loading'; preparationError = ''; initialAligned = false;
    const source = states.left?.source ?? states.right?.source;
    if (source) for (const camera of cameras) void camera.open(source);
    for (const s of all()) {
      s.prepareError = '';
      if (s.session && s.modelReady) queuePreview(s);
      else void activate(s);
    }
    status();
  });
  bind('new', 'click', () => Promise.all(all().map((s) => activate(s))));
  bind('save', 'click', () => {
    $('submission').textContent = ''; $('submission').hidden = true;
    return Promise.all(all().map((s) => save(s, true)));
  });
  bind('reset-all', 'click', () => { const s = states[activeHand]; if (!s?.session) return; for (const row of rows.values()) row.update(0); changed(s); });
  bind('reset-view', 'click', () => { for (const s of all()) s.view?.reset(); });
  bind('prev', 'click', () => { const index = upperBound(times, time) - 1; seek(times[Math.max(0, times[index] === time ? index - 1 : index)]); });
  bind('next', 'click', () => seek(times[Math.min(times.length - 1, upperBound(times, time))]));
  bind('slider', 'input', (event) => seek(event.target.value));
  bind('play', 'click', () => {
    if (preparation !== 'ready' || !times.length || states[activeHand]?.segmentDraft) return;
    if (playing) { pause(); return; }
    if (time >= times.at(-1)) seek(0);
    playing = true; playOrigin = performance.now(); playTime = time; $('play').textContent = '暂停';
  });
  bind('export', 'click', async () => {
    if (!segments.canLeave()) return;
    const list = all(), version = generation;
    if (submitting || !list.length || !list.every(exportable)) return;
    const sessions = list.map((s) => s.session);
    const selfCollision = $('self-collision').checked;
    submitting = true; status();
    try {
      // Each save captures its side synchronously before the first await.
      // Export uses those returned names even if editing continues meanwhile.
      const names = await Promise.all(list.map((s) => save(s)));
      if (version !== generation) return;
      if (names.some((name) => !name) || list.some((s, index) => s.session !== sessions[index])) {
        throw new Error('录制或基础标定已改变，未提交导出；请确认当前草稿后重试。');
      }
      const snapshots = Object.fromEntries(list.map((s, index) => [s.source.hand, { recording_id: s.source.id, correction_name: names[index] }]));
      const payload = { ...(list.length === 2 ? snapshots : snapshots[list[0].source.hand]), self_collision: selfCollision };
      const job = await submitJob(list.length === 2 ? '/api/jobs/pair' : '/api/jobs', payload);
      if (version === generation) {
        $('submission').textContent = `已加入全局队列：${job.id}；可继续编辑或切换录制。\n本次使用的修正快照：${list.map((s) => `${handLabel(s.source.hand)} ${snapshots[s.source.hand].correction_name}`).join('；')}`;
        $('submission').hidden = false;
      }
    } finally { submitting = false; status(); }
  });
  function tick(now) {
    const media = cameras.map((camera) => camera.getReadiness(time, playing && !holding));
    if (preparation !== 'ready') { updatePreparation(media); return; }
    const error = media.find((item) => item.error)?.error || all().find((s) => s.prepareError)?.prepareError;
    if (error) { loadingFailed(new Error(error)); return; }
    if (!playing) return;
    const waiting = media.some((item) => !item.ready || item.buffering || !item.synchronized)
      || all().some((s) => s.index >= 0 && (s.poseRevision !== s.revision || time - s.poseTime > 0.12));
    if (waiting || holding) {
      if (!holding) {
        holding = true;
        for (const camera of cameras) camera.follow(time, false);
      }
      playTime = time; playOrigin = now;
      const aligned = cameras.every((camera) => {
        const paused = camera.getReadiness(time, false), continuous = camera.getReadiness(time, true);
        return paused.ready && paused.synchronized && !continuous.buffering;
      }) && all().every((s) => s.index < 0 || (s.poseIndex === s.index && s.poseRevision === s.revision));
      if (aligned) {
        holding = false;
        for (const camera of cameras) camera.follow(time, true);
      }
      updateBufferingNotice(now);
      return;
    }
    updateBufferingNotice(now);
    const target = playTime + (now - playOrigin) / 1000;
    seek(target, false);
    if (target >= times.at(-1)) pause();
  }
  status();
  return { setRecording, setRecordings, beginLoading, loadingFailed, refreshCalibrations, tick, canLeave: segments.canLeave };
}
