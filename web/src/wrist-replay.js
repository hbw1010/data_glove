const sources = {
  wrist: {
    label: '腕相机', title: '外置双目腕相机（辅助参考）', linkedByDefault: false,
    missing: '此录制没有可用的同侧腕相机关联；仍可进行模型预览。',
    warning: '手套与外置腕相机时钟不同，无可用精确时钟映射；不得将首帧相对联动视为精确同步。',
    linked: '首帧相对联动：手套首帧对齐双目共同起点；视频原生连续播放并校正漂移，不是硬件精确同步。越界不显示画面。',
    independent: '独立相机时间轴（默认）：按墙钟 1× 连续播放，不等待慢路解码；不影响模型，各路保留真实首帧时间差。',
    timeline: '相机共同时间轴',
  },
  'glove-video': {
    label: '手套视频', title: 'Data glove 自身视频（IMX415）', linkedByDefault: true,
    missing: '此数据没有可用的同侧手套自身视频；支持 dataglove/video.h264 及采集 HDF5 内嵌 H264，仍可正常回放模型。',
    warning: '按来源声明的设备时钟或录制器主机映射时间戳选帧；保留视频相对编码器的真实起止偏移，不代表曝光级硬同步。',
    linked: '默认跟随编码器原始时间轴，不把视频首帧归零。视频原生连续播放并校正漂移；下方区分目标与实际呈现帧，越界不显示画面。',
    independent: '独立视频时间轴：可拖动或按墙钟 1× 连续播放完整视频，包括编码器范围之外的首尾；不等待解码，不影响 URDF。重新勾选即可跟随原始时间轴。',
    timeline: '编码器原点视频时间轴',
  },
  'extra-cameras': {
    label: '其余手套与头部相机', title: '其余手套与头部相机（辅助参考）', linkedByDefault: false,
    missing: '此录制没有其余可用的手套或头部相机。',
    warning: '包含其他手套与头部相机；各自编码器首帧的相对联动只作动作参考，不代表精确同步。',
    linked: '首帧相对联动：各来源起点对齐原始录制起点，保留各来源内部帧间隔；不是硬件精确同步。',
    independent: '独立时间轴（默认）：按墙钟 1× 连续播放，不等待各路解码；可独立拖动，不影响 URDF 或当前手套视频。',
    timeline: '各来源首帧相对时间轴',
  },
};

export function createWristReplay({ api, upperBound, handLabel, format, prefix = 'wrist', namespace = 'results', source = 'wrist', externalTimeline = false, gated = false }) {
  const profile = sources[source];
  const $ = (id) => document.getElementById(`${prefix}${id.slice('wrist'.length)}`);
  let revision = 0, cameras = [], available = false, playing = false;
  let time = 0, start = 0, end = 0, modelTime = 0, lastTick = 0;
  let modelContinuous = false, modelRate = 1;
  let modelRange = null;
  let metadataPhase = 'idle', metadataError = null;

  function pause() {
    playing = modelContinuous = false;
    for (const camera of cameras) camera.show(time, false, modelRate);
    if (!externalTimeline) $('wrist-play').textContent = `播放${profile.label}`;
  }

  function clear() {
    ++revision;
    playing = modelContinuous = false;
    modelRate = 1;
    if (!externalTimeline) $('wrist-play').textContent = `播放${profile.label}`;
    available = false;
    time = start = end = modelTime = 0;
    modelRange = null;
    metadataPhase = 'idle';
    metadataError = null;
    for (const camera of cameras) {
      camera.dispose();
    }
    cameras = [];
    $('wrist-cameras').replaceChildren();
    $('wrist-replay').hidden = true;
    if (!externalTimeline) {
      $('wrist-linked').checked = profile.linkedByDefault;
      $('wrist-linked').disabled = true;
      $('wrist-play').disabled = $('wrist-slider').disabled = true;
      $('wrist-warning').textContent = '';
      $('wrist-time').textContent = '';
      $('wrist-mode').textContent = '';
    }
    $('wrist-info').textContent = '';
  }

  function createCamera(data, version) {
    const article = document.createElement('article');
    const heading = document.createElement('h3');
    heading.textContent = `${data.label ?? data.id} · ${data.width} × ${data.height} · ${data.count} 帧`;
    heading.title = heading.textContent;
    const surface = document.createElement('div');
    surface.className = 'wrist-surface';
    surface.style.aspectRatio = `${data.width} / ${data.height}`;
    const video = document.createElement('video');
    video.id = `${prefix}-${data.id}`;
    video.muted = true;
    video.playsInline = true;
    video.preload = gated ? 'auto' : 'metadata';
    video.setAttribute('aria-label', `${data.label ?? data.id} ${profile.title}画面`);
    video.hidden = true;
    const message = document.createElement('span');
    message.className = 'wrist-placeholder';
    message.textContent = '正在加载视频…';
    const status = document.createElement('p');
    status.id = `${prefix}-${data.id}-time`;
    status.className = 'mono wrist-frame-time';
    surface.append(video, message);
    article.append(heading, surface, status);
    $('wrist-cameras').append(article);
    const camera = {
      data, video, message, status, desired: -1, displayed: -1,
      failed: false, decoded: false, error: null, continuous: false, target: 0, rate: 1,
    };
    const active = () => revision === version && !camera.failed;
    const hasFrameCallback = typeof video.requestVideoFrameCallback === 'function';
    let frameCallback = null, seekTimer = null, pending = false, seekStarted = 0, sought = -1, manualFrame = null;
    let lastCorrection = -Infinity, playPromise = null, retryPlayAt = 0;
    let presentedTime = null;
    let primed = false;

    function fail(text) {
      camera.failed = true;
      camera.error = text;
      stopVideo();
      clearTimeout(seekTimer);
      if (frameCallback !== null) video.cancelVideoFrameCallback(frameCallback);
      hide(text);
      status.textContent = '不显示旧画面；可重新导入录制或打开结果重试。';
    }

    function verifyDecoded() {
      if (!active() || video.seeking || pending || video.readyState < 2) return;
      const frame = upperBound(data.video_time_s, video.currentTime + 1e-7) - 1;
      if (frame >= 0 && frame < data.count) camera.decoded = true;
    }

    function outsideModelRange(target) {
      return modelRange && (target < modelRange[0] || target > modelRange[1]);
    }

    function frameAt(target) {
      return outsideModelRange(target) || target < data.time_s[0] || target > data.time_s.at(-1)
        ? -1 : upperBound(data.time_s, target) - 1;
    }

    function timingAt(index, target) {
      const pts = data.video_time_s[index];
      const next = data.video_time_s[index + 1];
      const segment = Math.min(index, data.count - 2);
      const span = segment < 0 ? 0 : data.time_s[segment + 1] - data.time_s[segment];
      const scale = span > 0 ? (data.video_time_s[segment + 1] - data.video_time_s[segment]) / span : 1;
      return { pts, next, scale, targetPts: next === undefined ? pts : pts + (target - data.time_s[index]) * scale };
    }

    function primeDecode() {
      if (!gated || camera.decoded || primed || video.readyState < 1) return;
      // A linked clock may start before this camera. Decode its first frame
      // without changing the public target or revealing an out-of-range image.
      primed = true;
      const first = data.video_time_s[0], next = data.video_time_s[1];
      seek(next === undefined ? first : first + (next - first) / 2);
    }

    function hide(text) {
      video.hidden = true;
      message.hidden = false;
      message.textContent = text;
      camera.displayed = -1;
    }

    function updateStatus() {
      const frame = camera.displayed;
      const target = camera.desired < 0 ? '' : `目标帧 ${camera.desired}；`;
      const evidence = hasFrameCallback ? '实际呈现帧' : '媒体时间估计帧（浏览器不支持呈现回调）';
      const text = frame < 0
        ? `${target}尚无已确认画面`
        : `${target}${evidence} ${frame} / ${data.count - 1} · 相机帧时间 ${format(data.time_s[frame], 6)} s\ntimestamp_ns=${data.timestamp_ns[frame]} · 视频帧 PTS ${format(data.video_time_s[frame], 6)} s`;
      if (status.textContent !== text) status.textContent = status.title = text;
    }

    function recordFrame(mediaTime) {
      if (!active() || camera.desired < 0 || video.readyState < 2) return;
      if (!hasFrameCallback && (video.seeking || pending)) return;
      const frame = Math.max(0, Math.min(data.count - 1, upperBound(data.video_time_s, mediaTime + 1e-7) - 1));
      // A late seek/frame callback must not replace the newest paused target.
      if (!camera.continuous && frame !== camera.desired) return;
      camera.displayed = frame;
      video.hidden = false;
      message.hidden = true;
      updateStatus();
    }

    function watchFrame() {
      if (!active() || !hasFrameCallback || frameCallback !== null) return;
      frameCallback = video.requestVideoFrameCallback((_, metadata) => {
        frameCallback = null;
        presentedTime = metadata.mediaTime;
        recordFrame(metadata.mediaTime);
        watchFrame();
      });
    }

    function stopVideo() {
      video.pause();
    }

    function startVideo() {
      if (!active() || !camera.continuous || camera.desired < 0 || pending || video.seeking
        || video.ended || !video.paused || playPromise || performance.now() < retryPlayAt) return;
      const promise = video.play();
      if (!promise) return;
      playPromise = promise;
      promise.then(() => {
        if (playPromise === promise) playPromise = null;
        if (!active() || !camera.continuous || camera.desired < 0) stopVideo();
        else sync();
      }, (error) => {
        if (playPromise === promise) playPromise = null;
        if (!active()) return;
        if (gated && error.name !== 'AbortError') {
          fail(`连续播放失败：${error.message}`);
          return;
        }
        // pause()/load() can legitimately abort an in-flight play request.
        retryPlayAt = performance.now() + (error.name === 'AbortError' ? 0 : 1000);
        if (error.name === 'AbortError') sync();
        if (error.name !== 'AbortError' && camera.continuous && camera.desired >= 0) {
          message.hidden = false;
          message.textContent = `连续播放暂不可用：${error.message}；可暂停后拖动定位。`;
        }
      });
    }

    function seek(pts) {
      clearTimeout(seekTimer);
      pending = true;
      sought = camera.desired;
      seekStarted = lastCorrection = performance.now();
      try {
        // Exact selection uses the presentation interval, not a rounded boundary.
        video.currentTime = pts;
      } catch (error) {
        pending = false;
        fail(`视频定位失败：${error.message}`);
        return;
      }
      // Some media implementations omit seeked for a no-op seek. Also allow a
      // newer target to supersede a stalled seek instead of holding it forever.
      seekTimer = setTimeout(() => {
        seekTimer = null;
        if (!active()) return;
        pending = false;
        sync();
      }, 1500);
    }

    function sync() {
      if (!active()) return;
      verifyDecoded();
      if (camera.desired < 0) {
        primeDecode();
        return;
      }
      if (video.readyState < 1) return;
      const now = performance.now();
      if ((pending || video.seeking) && now - seekStarted < 1500 && (camera.continuous || sought === camera.desired)) return;
      if (gated && video.seeking && sought === camera.desired) return;
      const index = camera.desired;
      const { pts, next, scale, targetPts } = timingAt(index, camera.target);
      if (!camera.continuous) {
        stopVideo();
        const current = upperBound(data.video_time_s, video.currentTime + 1e-7) - 1;
        if (video.seeking || current !== camera.desired || (!gated && video.readyState < 2)) {
          seek(next === undefined ? pts : pts + (next - pts) / 2);
        } else if (video.readyState < 2) {
          // preload=auto is already fetching this frame; re-seeking would
          // restart work while a slow response is still making progress.
          return;
        } else {
          pending = false;
          verifyDecoded();
          if (!hasFrameCallback) recordFrame(video.currentTime);
          else if (presentedTime !== null) recordFrame(presentedTime);
          // Show the decoded surface while waiting for evidence of presentation;
          // never label the requested frame as though it had been presented.
          video.hidden = false;
          message.hidden = camera.displayed >= 0;
        }
        return;
      }
      const baseRate = Math.max(0.0625, Math.min(16, camera.rate * (scale > 0 ? scale : 1)));
      const drift = targetPts - video.currentTime;
      const hardDrift = Math.max(0.12, baseRate * 0.15);
      if (video.seeking || (video.ended && index < data.count - 1)
        || (Math.abs(drift) > hardDrift && now - lastCorrection >= 750)) {
        seek(targetPts);
        return;
      }
      pending = false;
      // Small drift is corrected without interrupting the decoder. Large
      // discontinuities seek at most once per 750 ms, never once per render.
      const rate = Math.max(0.0625, Math.min(16, baseRate * (1 + Math.max(-0.08, Math.min(0.08, drift * 0.5)))));
      if (Math.abs(video.playbackRate - rate) > 0.001) video.playbackRate = rate;
      if (video.readyState >= 2) {
        video.hidden = false;
        message.hidden = true;
      }
      startVideo();
    }

    camera.show = (target, continuous = false, rate = 1) => {
      if (!active()) return;
      camera.target = target;
      camera.continuous = continuous;
      camera.rate = Number.isFinite(rate) && rate > 0 ? rate : 1;
      const outsideModel = outsideModelRange(target);
      camera.desired = frameAt(target);
      if (!continuous || camera.desired < 0) stopVideo();
      if (camera.desired < 0) {
        hide(outsideModel ? '当前时间超出该侧编码器范围，不显示末帧' : '当前相机时间无帧（超出本路录制范围）');
        status.textContent = outsideModel
          ? `该侧编码器范围 ${format(modelRange[0], 6)} ～ ${format(modelRange[1], 6)} s`
          : `本路有效范围 ${format(data.time_s[0], 6)} ～ ${format(data.time_s.at(-1), 6)} s`;
        primeDecode();
        return;
      }
      updateStatus();
      if (continuous) {
        if (manualFrame !== null) cancelAnimationFrame(manualFrame);
        manualFrame = null;
        sync();
      } else if (manualFrame === null) {
        manualFrame = requestAnimationFrame(() => { manualFrame = null; sync(); });
      }
    };
    camera.getReadiness = (target, continuous) => {
      verifyDecoded();
      const index = frameAt(target);
      const inRange = index >= 0;
      const buffered = video.readyState >= (continuous ? 3 : 2);
      const settled = !pending && !video.seeking;
      const current = upperBound(data.video_time_s, video.currentTime + 1e-7) - 1;
      const timing = inRange && continuous ? timingAt(index, target) : null;
      const aligned = continuous
        ? buffered && !video.ended && Math.abs(timing ? (timing.targetPts - video.currentTime) / (timing.scale > 0 ? timing.scale : 1) : 0) <= 0.15
        : buffered && video.paused && current === index;
      return {
        id: data.id,
        label: data.label ?? data.id,
        ready: camera.decoded && !camera.failed,
        stage: camera.failed ? 'error' : camera.decoded ? 'ready' : video.readyState < 1 ? 'metadata' : 'decoding',
        error: camera.error,
        synchronized: !camera.failed && camera.decoded && (!inRange || (settled && aligned)),
        buffering: !camera.failed && inRange && (!buffered || !settled),
      };
    };
    camera.dispose = () => {
      camera.failed = true;
      stopVideo();
      clearTimeout(seekTimer);
      if (manualFrame !== null) cancelAnimationFrame(manualFrame);
      if (frameCallback !== null) video.cancelVideoFrameCallback(frameCallback);
      video.removeAttribute('src');
      video.load();
    };
    video.addEventListener('loadedmetadata', sync);
    video.addEventListener('loadeddata', () => {
      sync();
      if (!hasFrameCallback) recordFrame(video.currentTime);
    });
    video.addEventListener('seeked', () => {
      if (!active() || video.seeking) return;
      pending = false;
      clearTimeout(seekTimer);
      seekTimer = null;
      sync();
      if (!hasFrameCallback) recordFrame(video.currentTime);
    });
    video.addEventListener('timeupdate', () => {
      if (!hasFrameCallback) recordFrame(video.currentTime);
    });
    video.addEventListener('canplay', sync);
    video.addEventListener('error', () => {
      if (!active()) return;
      fail(`视频无法读取或解码（媒体错误 ${video.error?.code ?? '未知'}）`);
    });
    watchFrame();
    video.src = data.url;
    return camera;
  }

  function show(target, continuous = false, rate = 1) {
    time = target;
    if (!externalTimeline) {
      $('wrist-slider').value = Math.max(start, Math.min(end, time));
      $('wrist-time').textContent = `${profile.timeline}目标 ${format(time, 6)} s · 范围 ${format(start, 6)} ～ ${format(end, 6)} s；实际画面帧见下方`;
    }
    for (const camera of cameras) camera.show(time, continuous, rate);
  }

  function updateMode() {
    playing = false;
    $('wrist-play').textContent = `播放${profile.label}`;
    const linked = $('wrist-linked').checked;
    $('wrist-play').disabled = $('wrist-slider').disabled = !available || linked;
    $('wrist-mode').textContent = linked ? profile.linked : profile.independent;
    if (available) show(linked ? modelTime : Math.max(start, Math.min(end, time)), linked && modelContinuous, modelRate);
  }

  async function open(result) {
    clear();
    metadataPhase = 'loading';
    if (externalTimeline && namespace === 'results') modelRange = [0, result.time_s.at(-1) - result.time_s[0]];
    const version = revision;
    $('wrist-replay').hidden = false;
    $('wrist-info').textContent = `正在加载${profile.title}与逐帧时间戳… ${gated ? '视频与模型就绪后才开放共同回放。' : '不影响 URDF 预览。'}`;
    try {
      const data = await api(`/api/${namespace}/${encodeURIComponent(result.id)}/${source}`);
      if (version !== revision) return;
      metadataError = data.errors?.length ? data.errors.join('\n') : null;
      metadataPhase = metadataError ? 'error' : 'ready';
      if (!data.available) {
        $('wrist-replay').hidden = false;
        $('wrist-info').textContent = data.errors?.length ? data.errors.join('\n') : profile.missing;
        return;
      }
      const selectedCameras = source === 'extra-cameras'
        ? data.cameras.filter((camera) => !camera.id.startsWith('head-') || /-RGB_(left|right)$/.test(camera.id))
        : data.cameras;
      if (!selectedCameras.length) {
        $('wrist-info').textContent = '没有可用的其他手套或头部 RGB 相机。';
        return;
      }
      available = true;
      $('wrist-replay').hidden = false;
      $('wrist-info').textContent = externalTimeline
        ? [namespace === 'results' ? data.clock_warning || profile.warning : '', ...(data.errors ?? [])].filter(Boolean).join('\n')
        : `${handLabel(data.hand)} · ${profile.title}\n${data.source_path}`;
      start = profile.linkedByDefault ? Math.min(...selectedCameras.map((camera) => camera.time_s[0])) : 0;
      end = Math.max(...selectedCameras.map((camera) => camera.time_s.at(-1)));
      if (!externalTimeline) {
        $('wrist-warning').textContent = [data.clock_warning, profile.warning, ...(data.errors ?? [])].filter(Boolean).join('\n');
        $('wrist-slider').min = start;
        $('wrist-slider').max = end;
        $('wrist-linked').disabled = false;
      }
      cameras = selectedCameras.map((camera) => createCamera({
        ...camera, label: camera.label ?? `${handLabel(data.hand)} · ${source === 'glove-video' ? 'Data glove' : camera.id}`,
      }, version));
      if (externalTimeline) show(modelTime, modelContinuous, modelRate);
      else updateMode();
    } catch (error) {
      if (version !== revision) return;
      available = false;
      metadataError = error.message;
      metadataPhase = 'error';
      $('wrist-replay').hidden = false;
      $('wrist-info').textContent = `${profile.label}加载失败：${error.message}`;
      if (!externalTimeline) $('wrist-play').disabled = $('wrist-slider').disabled = $('wrist-linked').disabled = true;
    }
  }

  if (!externalTimeline) {
  $('wrist-linked').addEventListener('change', updateMode);
  $('wrist-play').addEventListener('click', () => {
    if (!available || $('wrist-linked').checked) return;
    if (playing) { pause(); return; }
    if (time >= end) show(start);
    playing = true;
    lastTick = performance.now();
    show(time, true);
    $('wrist-play').textContent = `暂停${profile.label}`;
  });
  $('wrist-slider').addEventListener('input', (event) => {
    if (!available || $('wrist-linked').checked) return;
    pause();
    show(Number(event.target.value));
  });
  }

  return {
    clear, open, pause,
    getReadiness(target = modelTime, continuous = false) {
      const states = cameras.map((camera) => camera.getReadiness(target, continuous));
      const error = [metadataError, ...states.map((camera) => camera.error)].filter(Boolean).join('\n') || null;
      const ready = metadataPhase === 'ready' && !error && states.every((camera) => camera.ready);
      return {
        ready,
        synchronized: ready && states.every((camera) => camera.synchronized),
        buffering: metadataPhase === 'loading' || states.some((camera) => camera.buffering || !camera.ready && !camera.error),
        error,
        phase: error ? 'error' : ready ? 'ready' : metadataPhase === 'idle' ? 'idle' : 'loading',
        start: cameras.length ? Math.max(modelRange?.[0] ?? -Infinity, ...cameras.map((camera) => camera.data.time_s[0])) : null,
        end: cameras.length ? Math.min(modelRange?.[1] ?? Infinity, ...cameras.map((camera) => camera.data.time_s.at(-1))) : null,
        cameras: states.map(({ id, label, ready, stage, error }) => ({ id, label, ready, stage, error })),
      };
    },
    follow(target, continuous = false, rate = 1) {
      modelTime = target;
      modelContinuous = continuous;
      modelRate = Number.isFinite(rate) && rate > 0 ? rate : 1;
      if (available && (externalTimeline || $('wrist-linked').checked)) show(target, continuous, modelRate);
    },
    tick(now) {
      if (!playing) return;
      const elapsed = Math.max(0, (now - lastTick) / 1000);
      lastTick = now;
      show(Math.min(end, time + elapsed), true);
      if (time >= end) pause();
    },
  };
}
