export function createExportQueue({ api, handLabel, openResult, refreshResults, selectModule }) {
  const $ = (id) => document.getElementById(`export-queue-${id}`);
  const labels = { queued: '排队中', running: '运行中', cancelling: '正在停止并移除', complete: '已完成', error: '失败' };
  const rows = new Map();
  let timer = null, pending = null, refreshAgain = false;

  function element(tag, className = '') {
    const node = document.createElement(tag);
    node.className = className;
    return node;
  }
  function announce(message, error = false) {
    $('action').textContent = message;
    $('action').hidden = false;
    $('action').classList.toggle('error', error);
  }
  function action(row, label, name, callback) {
    const button = element('button');
    button.type = 'button'; button.textContent = label; button.dataset.action = name;
    button.addEventListener('click', async () => {
      if (row.busy) return;
      row.busy = true;
      for (const control of row.actions.children) control.disabled = true;
      try { await callback(row.job); }
      catch (error) { announce(`${label}失败：${error.message}`, true); }
      finally {
        row.busy = false;
        for (const control of row.actions.children) control.disabled = row.job.state === 'cancelling';
      }
    });
    row.actions.append(button);
  }
  async function remove(job) {
    if (job.state === 'running' && !window.confirm(`停止并移除任务「${job.task_name}」？\n将终止当前计算并让下一个排队任务继续。只移除任务，不删除已生成的 HDF5 文件。`)) return;
    await api(`/api/jobs/${encodeURIComponent(job.id)}`, undefined, 'DELETE');
    announce(`已请求移除任务「${job.task_name}」。已生成的 HDF5 文件不会删除。`);
    await refresh();
  }
  async function download(job) {
    const result = await api('/api/results/open', { path: job.path });
    const link = element('a');
    link.href = `/api/results/${encodeURIComponent(result.id)}/download`;
    link.download = 'processed.h5';
    document.body.append(link); link.click(); link.remove();
  }
  function render(jobs) {
    const present = new Set(jobs.map((job) => job.id));
    for (const [id, row] of rows) if (!present.has(id)) { row.root.remove(); rows.delete(id); }
    let newlyComplete = false;
    for (const job of jobs) {
      let row = rows.get(job.id);
      if (!row) {
        row = { root: element('article', 'export-task'), title: element('h3'), source: element('p', 'mono'), summary: element('p'), sides: element('p'), snapshot: element('p'), progress: element('progress'), path: element('p', 'mono'), error: element('p', 'notice error'), actions: element('div', 'row'), busy: false, actionKey: '' };
        row.root.dataset.jobId = job.id;
        row.progress.setAttribute('aria-label', '全部原始帧处理进度');
        row.root.append(row.title, row.source, row.summary, row.progress, row.sides, row.snapshot, row.path, row.error, row.actions);
        rows.set(job.id, row); $('list').append(row.root);
      }
      if (job.state === 'complete' && row.job?.state !== 'complete') newlyComplete = true;
      row.job = job; row.root.dataset.state = job.state;
      const hands = ['left', 'right'].filter((hand) => job.sides?.[hand]);
      row.title.textContent = `${job.task_name} · ${hands.length === 2 ? '双手同文件' : '单手'} · ${labels[job.state] ?? job.state}`;
      row.source.textContent = `来源：${job.source_root}\n提交：${job.created_at} · 任务 ${job.id}`;
      row.summary.textContent = `${job.done}/${job.total} 帧${job.state === 'queued' ? ` · 排队位置 ${job.queue_position}` : ''}${job.active_hand ? ` · 当前处理${handLabel(job.active_hand)}` : ''} · Wuji Hand2 自碰撞：${job.self_collision ? '已启用（较慢）' : '未启用（快速模式，可能穿模）'}`;
      row.progress.max = Math.max(job.total, 1); row.progress.value = job.done;
      row.sides.textContent = Object.entries(job.sides ?? {}).map(([hand, side]) => `${handLabel(hand)}：${labels[side.state] ?? side.state} ${side.done}/${side.total}`).join('；');
      const snapshots = Object.entries(job.snapshots ?? {}).filter(([, value]) => value);
      row.snapshot.textContent = snapshots.length ? `本次使用的修正快照：${snapshots.map(([hand, value]) => `${handLabel(hand)} ${value}`).join('；')}` : '';
      row.snapshot.hidden = !snapshots.length;
      row.path.textContent = `${job.state === 'complete' ? '结果' : '预期输出'}：${job.path ?? job.output_path}`;
      row.error.textContent = job.error ?? ''; row.error.hidden = !job.error;
      const actionKey = `${job.state}:${job.path}:${hands.join(',')}:${job.self_collision}`;
      if (row.actionKey !== actionKey) {
        row.actionKey = actionKey; row.actions.replaceChildren();
        if (job.state === 'complete' && job.path) {
          for (const hand of hands) action(row, `打开${handLabel(hand)}结果`, `open-${hand}`, async (current) => {
            await openResult(current.path, hand);
          });
          action(row, '下载完整 HDF5', 'download', download);
          if (job.self_collision !== true) action(row, '补做自碰撞（另存）', 'collision', (current) =>
            submit('/api/jobs/wujihand2-collision', { path: current.path }));
        }
        action(row, job.state === 'running' ? '停止并移除' : job.state === 'cancelling' ? '正在停止…' : '移除任务', 'remove', remove);
      }
      for (const control of row.actions.children) control.disabled = row.busy || job.state === 'cancelling';
    }
    $('empty').hidden = jobs.length > 0;
    if (newlyComplete) void refreshResults().catch((error) => announce(`任务已完成，但结果列表刷新失败：${error.message}；仍可从任务行打开文件。`, true));
  }
  function refresh() {
    clearTimeout(timer);
    if (pending) { refreshAgain = true; return pending; }
    pending = (async () => {
      do {
        refreshAgain = false;
        try {
          const data = await api('/api/jobs');
          render(data.jobs);
          $('status').classList.remove('error');
          $('status').textContent = `任务队列已同步 · ${data.jobs.length} 项 · 每秒自动刷新；不影响录制编辑。`;
        } catch (error) {
          $('status').classList.add('error');
          $('status').textContent = `任务队列读取失败：${error.message}。保留上次列表，状态可能已过期；后台任务仍可能运行，将自动重试。可继续编辑。`;
        }
      } while (refreshAgain);
    })().finally(() => { pending = null; timer = setTimeout(refresh, 1000); });
    return pending;
  }
  async function submit(path, body) {
    try {
      const job = await api(path, body);
      announce(`已加入全局任务队列：${job.id}。可继续编辑或切换录制；处理使用本次提交的快照。`);
      void refresh();
      selectModule('export-queue');
      return job;
    } catch (error) {
      announce(`提交任务失败：${error.message}。如连接中断，任务仍可能已入队，请查看刷新后的队列再决定是否重新提交。`, true);
      void refresh();
      throw error;
    }
  }
  $('collision-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const button = $('collision-submit');
    if (button.disabled) return;
    const path = $('collision-path').value.trim();
    button.disabled = true;
    try { await submit('/api/jobs/wujihand2-collision', { path }); }
    catch { /* submit 已显示后端错误与刷新队列的重试提示。 */ }
    finally { button.disabled = false; }
  });
  $('refresh').addEventListener('click', () => void refresh());
  void refresh();
  return { submit };
}
