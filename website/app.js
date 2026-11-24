/* E-Textile Validation — project page interactions. No dependencies. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const el = (tag, attrs = {}, html = '') => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (html) e.innerHTML = html;
    return e;
  };
  const redrawers = [];            // canvases that repaint when the theme changes

  /* ---------------------------------------------------------------- shell */
  const root = document.documentElement;
  $('#themeBtn').addEventListener('click', () => {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark'
      : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('etv-theme', root.dataset.theme); } catch (e) {}
    requestAnimationFrame(() => redrawers.forEach(f => f()));
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => redrawers.forEach(f => f()));

  const nav = $('#nav');
  addEventListener('scroll', () => nav.classList.toggle('scrolled', scrollY > 8), { passive: true });

  const links = $$('.links a');
  const secObs = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) links.forEach(a => a.classList.toggle('on', a.hash === '#' + e.target.id));
  }), { rootMargin: '-45% 0px -50% 0px' });
  $$('section[id]').forEach(s => secObs.observe(s));

  function countUp(b) {
    const to = parseFloat(b.dataset.count), dec = +(b.dataset.dec || 0);
    if (reduced || !isFinite(to)) return;
    const t0 = performance.now(), dur = 1100;
    const step = t => {
      const k = Math.min(1, (t - t0) / dur), v = to * (1 - Math.pow(1 - k, 3));
      b.textContent = v.toFixed(dec);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.classList.add('in');
    $$('[data-count]', e.target).forEach(countUp);
    if (e.target.matches('[data-count]')) countUp(e.target);
    io.unobserve(e.target);
  }), { threshold: 0.12 });
  $$('.reveal').forEach((e, i) => { e.style.transitionDelay = (i % 4) * 60 + 'ms'; io.observe(e); });
  io.observe($('#studyBars'));

  // videos play only while visible
  const vio = new IntersectionObserver(es => es.forEach(e => {
    const v = e.target;
    if (v.dataset.manual) return;
    e.isIntersecting ? v.play().catch(() => {}) : v.pause();
  }), { threshold: 0.25 });
  vio.observe($('#heroV'));

  /* ------------------------------------------------------------ QR code */
  // For talks: the hero shows a QR code; click it (or press Q) to fill the screen.
  const qrFull = $('#qrFull');
  const qr = show => { qrFull.hidden = !show; };
  $('#qrBtn').addEventListener('click', () => qr(true));
  qrFull.addEventListener('click', () => qr(false));
  addEventListener('keydown', e => {
    if (e.target.closest && e.target.closest('input, textarea')) return;
    if (e.key === 'Escape') qr(false);
    else if ((e.key === 'q' || e.key === 'Q') && !e.metaKey && !e.ctrlKey && !e.altKey) qr(qrFull.hidden);
  });

  /* ------------------------------------------------------------ why: loop */
  const LOOPS = {
    before: {
      title: 'Current practice: designer and engineer',
      steps: [
        { img: 'goal.png', t: 'Set the goal', s: 'e.g. sense the arm angle', e: 0,
          d: 'The designer decides what the garment should sense, for example the angle of the arm.' },
        { img: 'garment.png', t: 'Make sensors + garment', s: 'four sensors on a sleeve', e: 0,
          d: 'She makes the sensors and the garment, say four sensors on a sleeve.' },
        { img: 'signals.png', t: 'Collect data', s: 'sensor log + motion', e: 0,
          d: 'Somebody moves while the sensors are logged and the motion is recorded.' },
        { img: 'code.png', t: 'Write code, test it', s: 'new script for each garment', e: 1,
          d: 'An engineer writes code that converts the signals into motion and measures the error. This is repeated for every new garment.' },
        { img: 'change.png', t: 'Change the garment', s: 'move a sensor, start again', e: 0,
          d: 'Based on the result, the designer moves a sensor and the next iteration starts.' },
      ] },
    after: {
      title: 'With the toolchain: designer only',
      steps: [
        { img: 'change.png', t: 'Make or change the garment', s: 'sensors where you think they work', e: 0,
          d: 'The designer makes the garment, or moves a sensor after the last round.' },
        { img: 'record.svg', t: 'Record', s: 'webcam video + sensor log', e: 0,
          d: 'She records a webcam video and the sensor log at the same time. The video provides the ground truth.' },
        { img: 'look.svg', t: 'Look', s: 'sensors next to the motion', e: 0,
          d: 'The app plots the sensor signals next to the motion from the video, which shows whether the sensors respond to the movement.' },
        { img: 'train.svg', t: 'Train', s: 'built-in model, one click', e: 0,
          d: 'She trains the built-in model on the recordings with one click, on a laptop CPU.' },
        { img: 'compare.svg', t: 'Compare', s: 'predicted vs. real arm', e: 0,
          d: 'The app draws the arm predicted from the sensors next to the tracked arm and reports the error in degrees. If the error is too large, she changes the garment and starts the next iteration.' },
      ] },
  };
  const CMP = [
    ['People involved', '2: designer and engineer', '1: designer'],
    ['Code to write', 'new code for each garment', 'none'],
    ['Between iterations', 'hand-over to the engineer', 'none'],
  ];
  const ring = $('#ring'), nodesBox = $('#ringNodes'), center = $('#ringCenter'), runner = $('#runner');
  let mode = 'before', step = 0, angle = -90, target = -90, autoT = null, hold = 0, ringVisible = false;
  const angleOf = k => -90 + 72 * k;
  function drawLoop() {
    const L = LOOPS[mode];
    $$('#loopSwitch button').forEach(b => b.classList.toggle('on', b.dataset.mode === mode));
    nodesBox.innerHTML = '';
    L.steps.forEach((st, k) => {
      const a = angleOf(k) * Math.PI / 180;
      const n = el('button', { type: 'button', class: 'rnode' + (st.e ? ' eng' : ''),
        style: `left:${50 + 38 * Math.cos(a)}%;top:${50 + 38 * Math.sin(a)}%`, 'aria-label': st.t },
        `<span class="pic"><img src="media/loop/${st.img}" alt=""></span><b>${st.t}</b>`);
      n.addEventListener('click', () => { hold = performance.now() + 9000; setStep(k); });
      nodesBox.append(n);
    });
    $('#loopTitle').textContent = L.title;
    $('#loopCmp').innerHTML = `<div class="row"><span></span><span class="head">Current practice</span><span class="head">With the toolchain</span></div>` +
      CMP.map(([k, a, b]) => `<div class="row"><span>${k}</span><span class="cell bad ${mode === 'before' ? 'on' : ''}">${a}</span><span class="cell ${mode === 'after' ? 'on' : ''}">${b}</span></div>`).join('');
    setStep(0, true);
  }
  function setStep(k, jump) {
    const L = LOOPS[mode], st = L.steps[k];
    // always move forward round the ring, as the loop does
    let t = angleOf(k);
    while (t <= target - 1e-6 && !jump) t += 360;
    if (jump) { t = angleOf(k); angle = t; }
    target = t; step = k;
    $$('.rnode', nodesBox).forEach((n, i) => n.classList.toggle('on', i === k));
    const col = st.e ? '#d97a1a' : '#0066cc';
    ring.style.setProperty('--d', mode === 'before' && st.e ? '#d97a1a' : '#0066cc');
    $('#stepCard').innerHTML = `<i style="background:${col}">${k + 1}</i><b>${st.t}</b><p>${st.d}</p>
      <span class="who" style="color:${col}">${st.e ? 'Engineer' : 'Designer'}${mode === 'after' ? ', alone, no code' : ''}</span>`;
    if (mode === 'before') {
      center.innerHTML = `<div class="who"><img src="media/loop/designer.png" alt="Designer" class="${st.e ? 'dim' : ''}">
        <span class="swap ${st.e ? 'hot' : ''}">⇄</span><img src="media/loop/engineer.png" alt="Engineer" class="${st.e ? '' : 'dim'}"></div>
        <small>Designer ⇄ Engineer</small>`;
    } else {
      center.innerHTML = `<div class="who"><img src="media/loop/designer.png" alt="Designer" style="width:46%"></div><small>Designer, alone, no code</small>`;
    }
  }
  function animate() {
    angle += (target - angle) * (reduced ? 1 : 0.08);
    const a = angle * Math.PI / 180;
    runner.setAttribute('cx', 50 + 38 * Math.cos(a)); runner.setAttribute('cy', 50 + 38 * Math.sin(a));
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
  function tickLoop() {
    if (!ringVisible || performance.now() < hold) return;
    if (step === LOOPS[mode].steps.length - 1) {     // a round done: show the other loop
      mode = mode === 'before' ? 'after' : 'before'; drawLoop();
    } else setStep(step + 1);
  }
  $$('#loopSwitch button').forEach(b => b.addEventListener('click', () => {
    hold = performance.now() + 9000; mode = b.dataset.mode; drawLoop();
  }));
  new IntersectionObserver(es => es.forEach(e => { ringVisible = e.isIntersecting; }), { threshold: 0.35 }).observe(ring);
  if (!reduced) autoT = setInterval(tickLoop, 2600);
  drawLoop();

  /* ---------------------------------------------------------------- setup */
  const photo = $('#setupPhoto');
  const focusPart = k => {
    photo.classList.toggle('focus', k !== null);
    $$('.ring', photo).forEach(r => r.classList.toggle('on', r.dataset.part === String(k)));
    $$('#setupParts li').forEach(li => li.classList.toggle('on', li.dataset.part === String(k)));
  };
  $$('#setupParts li').forEach(li => {
    li.addEventListener('mouseenter', () => focusPart(li.dataset.part));
    li.addEventListener('mouseleave', () => focusPart(null));
    li.addEventListener('click', () => focusPart(li.dataset.part));
  });

  const MOTIONS = ['90° flexion', '90° abduction', 'Rotation', 'Arm circling', 'Figure of eight', '90° elbow flexion'];
  MOTIONS.forEach((m, k) => $('#motions').append(el('figure', {},
    `<img src="media/paper/motion_${k + 1}.jpg" alt="${m}" loading="lazy"><figcaption>${m}</figcaption>`)));

  /* ------------------------------------------------------------- pipeline */
  const PIPE = [
    { n: 'Record', tab: 'before', s: 'video + sensor CSV + frame times',
      d: 'A webcam films the person while the garment\'s sensors are logged. Both files carry timestamps from the same clock, so the streams can be lined up later.',
      li: ['Any webcam; the recording scripts are in the repository', 'Sync the two computers\' clocks first (a one-line check is included)'],
      img: 'media/paper/setup.jpg' },
    { n: 'Track', tab: 'Tab 1–2', s: 'pose from the video',
      d: 'MediaPipe finds 33 body landmarks in every frame. They become a 22-joint skeleton, and each joint keeps the tracker\'s own confidence.',
      li: ['About 0.08 s per frame on a laptop CPU', 'Cached next to the video: done once per recording'],
      img: 'media/tabs/tab2.jpg' },
    { n: 'Targets', tab: 'Tab 3', s: 'what to predict',
      d: 'For every frame, the skeleton gives the regression targets: limb directions, how far each limb is raised, and the elbow and knee angles.',
      li: ['Directions are unit vectors in the body\'s frame', 'All targets are continuous in every pose'],
      img: 'media/tabs/tab3.jpg' },
    { n: 'Align', tab: 'Tab 4', s: 'one clock for both',
      d: 'The sensor stream is interpolated onto the time of each video frame, over the window both recordings cover. Each session is windowed on its own, so no sample spans two recordings.',
      li: ['Works with any sensor rate', 'Sessions that do not overlap are reported, not dropped silently'],
      img: 'media/tabs/tab4.jpg' },
    { n: 'Train', tab: 'Tab 4', s: 'sensors → motion',
      d: 'A CNN-LSTM is trained to predict the selected targets from a short window of sensor readings. Training runs on the CPU and takes a few minutes.',
      li: ['Default settings are provided; each setting has a help text', 'Retraining reuses the cached skeletons'],
      img: 'media/tabs/tab4.jpg' },
    { n: 'Report', tab: 'Tab 5–6', s: 'errors and plots',
      d: 'On the test recordings: the error of every target, predicted and true values over time, the predicted limb drawn into the skeleton, and the importance of each sensor.',
      li: ['Metrics, a text report and all plots can be exported', 'Sensor importance indicates which channels the model uses'],
      img: 'media/tabs/tab5.jpg' },
  ];
  const pipe = $('#pipe'), pcard = $('#pipeCard');
  function showPipe(k) {
    $$('.pnode', pipe).forEach((b, i) => b.classList.toggle('on', i === k));
    const p = PIPE[k];
    pcard.innerHTML = `<div><h4>${k + 1}. ${p.n}: ${p.s}</h4><p>${p.d}</p><ul>${p.li.map(x => `<li>${x}</li>`).join('')}</ul></div>
      <div class="vis"><img src="${p.img}" alt="" loading="lazy"></div>`;
  }
  PIPE.forEach((p, k) => {
    const b = el('button', { class: 'pnode', type: 'button' },
      `<span class="tabno">${p.tab}</span><i>0${k + 1}</i><b>${p.n}</b><span>${p.s}</span>`);
    b.addEventListener('click', () => showPipe(k));
    b.addEventListener('mouseenter', () => showPipe(k));
    pipe.append(b);
  });
  showPipe(0);

  /* ------------------------------------------------------------- tutorial */
  const HOT = {"tab1":{"train_table":[1.88,17.69,96.25,15.46],"test_table":[1.88,49.54,96.25,15.28],"extract":[0.62,72.87,96.93,4.44]},"tab2":{"skeleton":[0.62,14.35,98.75,61.76],"tq":[85.47,9.26,12.08,4.35]},"tab3":{"angles":[2.5,33.33,19.84,23.24],"sensors":[2.5,72.96,19.84,14.81],"plots":[25.26,53.98,73.49,38.24],"info":[25.26,92.96,73.49,2.13]},"tab4":{"summary":[0.62,9.26,96.93,6.3],"hyper":[0.62,16.3,98.75,36.3],"start":[0.62,53.33,49.17,4.44],"loss":[0.62,61.39,98.75,28.15]},"tab5":{"pred_ax":[66.56,17.13,32.81,71.11],"ghost":[21.51,10.09,18.91,2.59],"values":[0.62,94.07,98.75,2.13],"scrub":[6.61,90.28,84.32,1.85]},"tab6":{"headline":[1.25,10.37,45.16,3.24],"table":[2.5,22.69,94.22,18.52],"pred":[2.5,52.59,94.22,45]}};
  const TABS = [
    { t: 'Data Import', s: 'add your sessions', vid: null,
      todo: ['Add Subject under Training, pick the three files of a session', 'Add at least one Test session', 'Extract Skeleton & Compute Angles'],
      h: { train_table: ['Training sessions', 'Click Add Subject and choose a session\'s video, sensor CSV and frame-time CSV. The dialog checks the files at once and explains any problem, for example two CSVs swapped.'],
           test_table: ['Test sessions', 'Held back from training and used only to score the model. A recording of another person tells you most about the garment.'],
           extract: ['Extract Skeleton & Compute Angles', 'Tracks the pose in every new video, with a progress window you can cancel. Results are cached next to each video, so this runs once per recording.'] } },
    { t: 'Skeleton Viewer', s: 'check the tracking', vid: 'media/loop_tab2.mp4',
      todo: ['Scrub through each recording', 'Look for limbs turning red', 'Open Tracking Quality for a summary'],
      h: { skeleton: ['Video and skeleton together', 'The skeleton follows the slider. Each joint is coloured by how confident the tracker was: green is reliable, red means that part of the body was probably hidden.'],
           tq: ['Tracking Quality', 'Lists the least reliable landmarks and targets of the recording, so you know which results to read with care.'] } },
    { t: 'Angles & Sensors', s: 'targets and inputs', vid: 'media/loop_tab3.mp4',
      todo: ['Click the bones to predict', 'Keep the sensor channels you want to test', 'Play the video to see if a sensor follows the movement'],
      h: { angles: ['Targets', 'Click a bone on the body outline: the upper arm adds its direction and the shoulder elevation, the forearm adds its direction and the elbow angle. The List view shows every target by name.'],
           sensors: ['Sensor channels', 'The garment channels the model may use. Only channels present in every session are offered.'],
           plots: ['One clock for everything', 'Targets above, sensors below, on a shared time axis. The red line follows the video, and the values at that moment are listed above it.'],
           info: ['The usable window', 'Only the time both recordings cover is used for training.'] } },
    { t: 'Training', s: 'train a model', vid: null,
      todo: ['Check the summary', 'Keep the default settings', 'Press Start Training'],
      h: { summary: ['What will be trained', 'Sessions, targets and inputs, as chosen in Tabs 1 and 3.'],
           hyper: ['Settings', 'All have working defaults. The question marks explain the less obvious ones, symptom first.'],
           start: ['Start Training', 'Trains on the CPU with a progress window, then scores the model on the test sessions.'],
           loss: ['Loss curve', 'Should fall and flatten. If it is still falling at the end, raise the epochs.'] } },
    { t: 'Pose', s: 'predicted poses', vid: 'media/loop_tab5.mp4',
      todo: ['Play the test recording', 'Compare the red bones with the tracked ones', 'Drag a skeleton to view it from the side'],
      h: { pred_ax: ['Predicted bones, in red', 'Each predicted bone starts at the tracked joint, has the tracked length and points in the predicted direction. The number at its end is the error in degrees; the grey dashed line shows the tracked position.'],
           ghost: ['True position', 'Shows or hides the grey ghost of each predicted bone.'],
           scrub: ['One slider', 'Moves the video and both skeletons. Dragging either skeleton turns both.'],
           values: ['This moment, in numbers', 'For a direction, how far off it is; for an angle, the predicted and the true value.'] } },
    { t: 'Results', s: 'metrics and export', vid: null,
      todo: ['Read the per-subject table first', 'Check the tracking-confidence column', 'Export the report'],
      h: { headline: ['Four numbers', 'Mean error and RMSE in degrees, error as a share of the range of motion, and the correlation with the truth.'],
           table: ['Per target', 'The same numbers for each target, plus how well its ground truth was tracked: a large error on a poorly tracked target says more about the video than the garment.'],
           pred: ['Prediction and ground truth', 'Predicted and true values over the whole test recording. Further down: sensor importance and the export buttons.'] } },
  ];
  const tabList = $('#tabList'), img = $('#tabImg'), vid = $('#tabVid'), dots = $('#tabDots'), tip = $('#tabTip');
  vid.dataset.manual = '1';
  let curTab = 0;
  function showSpot(k, name) {
    const [title, text] = TABS[k].h[name];
    $$('.dot', dots).forEach(d => d.classList.toggle('on', d.dataset.n === name));
    $$('.box', dots).forEach(b => b.classList.toggle('on', b.dataset.n === name));
    tip.innerHTML = `<h4>${title}</h4><p>${text}</p>`;
  }
  function showTab(k) {
    curTab = k;
    $$('button', tabList).forEach((b, i) => { b.classList.toggle('on', i === k); b.setAttribute('aria-selected', i === k); });
    const T = TABS[k];
    img.src = `media/tabs/tab${k + 1}.jpg`;
    img.alt = `Tab ${k + 1}, ${T.t}`;
    if (T.vid) {
      vid.src = T.vid; vid.poster = T.vid.replace('.mp4', '.jpg'); vid.hidden = false; img.hidden = true;
      vid.play().catch(() => {});
    } else { vid.pause(); vid.hidden = true; img.hidden = false; }
    dots.innerHTML = '';
    Object.entries(HOT[`tab${k + 1}`]).forEach(([name, r], i) => {
      if (!T.h[name]) return;
      const box = el('div', { class: 'box', 'data-n': name,
        style: `left:${r[0]}%;top:${r[1]}%;width:${r[2]}%;height:${Math.min(r[3], 100 - r[1])}%` });
      const d = el('button', { class: 'dot', 'data-n': name, 'aria-label': T.h[name][0],
        style: `left:${r[0] + Math.min(r[2], 6) / 2 + 1.5}%;top:${r[1] + Math.min(r[3], 100 - r[1]) / 2}%` }, String(i + 1));
      d.addEventListener('click', () => showSpot(k, name));
      d.addEventListener('mouseenter', () => showSpot(k, name));
      dots.append(box, d);
    });
    tip.innerHTML = `<h4>Tab ${k + 1} · ${T.t}</h4><ol class="todo">${T.todo.map(x => `<li>${x}</li>`).join('')}</ol>`;
  }
  TABS.forEach((T, k) => {
    const b = el('button', { role: 'tab', type: 'button' }, `<i>${k + 1}</i><b>${T.t}</b><span>${T.s}</span>`);
    b.addEventListener('click', () => showTab(k));
    tabList.append(b);
  });
  showTab(0);
  new IntersectionObserver(es => es.forEach(e => {
    if (!TABS[curTab].vid) return;
    e.isIntersecting ? vid.play().catch(() => {}) : vid.pause();
  }), { threshold: 0.2 }).observe($('#tabStage'));

  /* -------------------------------------------------------------- targets */
  // template joints, front view: x = the person's left, y = up (as in the app's picker)
  const P = { 0: [0, .95], 1: [.10, .90], 2: [-.10, .90], 4: [.12, .50], 5: [-.12, .50], 7: [.13, .11], 8: [-.13, .11],
    10: [.21, .05], 11: [-.21, .05], 12: [0, 1.47], 15: [0, 1.62], 16: [.19, 1.43], 17: [-.19, 1.43],
    18: [.30, 1.17], 19: [-.30, 1.17], 20: [.40, .94], 21: [-.40, .94] };
  const T_INFO = {
    upper_arm_direction: 'where the upper arm points: a unit vector forward, up, outward in the chest\'s frame',
    shoulder_elevation: 'how far the arm is raised from hanging, 0 to 180°, in any direction',
    forearm_direction: 'where the forearm points; with the upper arm it fixes the elbow bend and the shoulder\'s rotation',
    elbow_flexion: 'how far the elbow is bent, 0° straight',
    thigh_direction: 'where the thigh points, in the pelvis\'s frame',
    hip_elevation: 'how far the thigh is raised from hanging',
    shank_direction: 'where the lower leg points, in the pelvis\'s frame',
    knee_flexion: 'how far the knee is bent, 0° straight',
    ankle_flexion: 'the angle between shin and foot',
    lumbar_bending: 'the trunk bending sideways, relative to the hips',
    lumbar_rotation: 'the shoulders turned relative to the hips',
    neck_flexion: 'the head bending forward, measured to the nose',
    neck_bending: 'the head tilting towards a shoulder',
  };
  const BONES = {};
  for (const [side, s] of [['right', 1], ['left', 0]]) {
    const j = s ? { sh: 17, el: 19, wr: 21, hip: 2, kn: 5, an: 8, ft: 11 } : { sh: 16, el: 18, wr: 20, hip: 1, kn: 4, an: 7, ft: 10 };
    const S = side[0].toUpperCase() + side.slice(1);
    Object.assign(BONES, {
      [`${side}_upper_arm`]: [`${S} upper arm`, j.sh, j.el, ['upper_arm_direction', 'shoulder_elevation']],
      [`${side}_forearm`]: [`${S} forearm`, j.el, j.wr, ['forearm_direction', 'elbow_flexion']],
      [`${side}_thigh`]: [`${S} thigh`, j.hip, j.kn, ['thigh_direction', 'hip_elevation']],
      [`${side}_shank`]: [`${S} shank`, j.kn, j.an, ['shank_direction', 'knee_flexion']],
      [`${side}_foot`]: [`${S} foot`, j.an, j.ft, ['ankle_flexion']],
    });
  }
  BONES.trunk = ['Trunk', 0, 12, ['lumbar_bending', 'lumbar_rotation']];
  BONES.head = ['Head and neck', 12, 15, ['neck_flexion', 'neck_bending']];
  const svg = $('#bodySvg'), NS = 'http://www.w3.org/2000/svg';
  const sv = (tag, a) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); return e; };
  const Y = y => 1.74 - y;        // svg y grows downwards
  const picked = new Set(['right_upper_arm']);
  for (const [a, b] of [[12, 16], [12, 17], [0, 1], [0, 2]])
    svg.append(sv('line', { x1: P[a][0], y1: Y(P[a][1]), x2: P[b][0], y2: Y(P[b][1]), class: 'conn', 'stroke-width': .025 }));
  svg.append(sv('circle', { cx: 0, cy: Y(1.66), r: .07, fill: 'none', class: 'conn', 'stroke-width': .02 }));
  const boneEls = {};
  for (const [name, [, a, b]] of Object.entries(BONES)) {
    const pts = { x1: P[a][0], y1: Y(P[a][1]), x2: P[b][0], y2: Y(P[b][1]) };
    const halo = sv('line', { ...pts, class: 'halo', 'stroke-width': .095 });
    const line = sv('line', { ...pts, class: 'bone', 'stroke-width': .042, tabindex: 0, role: 'button' });
    const tt = sv('title', {}); tt.textContent = BONES[name][0]; line.append(tt);
    line.addEventListener('click', () => pickBone(name));
    line.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pickBone(name); } });
    svg.append(halo, line);
    boneEls[name] = { halo, line };
  }
  const tR = sv('text', { x: -.44, y: Y(1.62) }); tR.textContent = 'R';
  const tL = sv('text', { x: .41, y: Y(1.62) }); tL.textContent = 'L';
  svg.append(tR, tL);
  function paintBones() {
    for (const [name, { halo, line }] of Object.entries(boneEls)) {
      const on = picked.has(name);
      halo.classList.toggle('on', on);
      line.style.stroke = on ? css('--acc') : css('--mute');
    }
  }
  let showLimbFor = () => {};
  function pickBone(name) {
    picked.has(name) ? picked.delete(name) : picked.add(name);
    paintBones(); showBoneCard(name);
    if (picked.has(name)) showLimbFor(name);
  }
  function showBoneCard(last) {
    const card = $('#boneCard');
    const names = Object.keys(BONES).filter(b => picked.has(b));
    if (!names.length) {
      card.innerHTML = '<h4>Click a bone</h4><p class="hint">Each bone adds the targets that describe it. Click it again to remove it.</p>';
      return;
    }
    const side = b => b.startsWith('left_') ? 'left_' : b.startsWith('right_') ? 'right_' : '';
    const rows = names.flatMap(b => BONES[b][3].map(t => {
      const full = side(b) + t;
      return `<li><code>${full}</code><span>${T_INFO[t]}</span></li>`;
    }));
    const n = rows.length;
    card.innerHTML = `<h4>${names.map(b => BONES[b][0]).join(' + ')}</h4>
      <p class="hint">${n} target${n > 1 ? 's' : ''}. Tab 5 draws the directions into the skeleton and lists the angles as numbers.</p>
      <ul class="tlist">${rows.join('')}</ul>`;
  }
  paintBones(); showBoneCard();
  redrawers.push(paintBones);

  /* ---- the limb demo: move one bone of the right side, watch its targets */
  const armC = $('#armCanvas'), plotC = $('#armPlot');
  let yaw = -0.5, pitch = 0.18;
  // the posed right side; everything else stands still
  const pose = { uaRaise: 70, uaAz: 60, elBend: 60, elTwist: 0, thRaise: 30, thAz: 0, knBend: 40 };
  const DEG = Math.PI / 180;
  const add = (a, b, k = 1) => a.map((v, i) => v + k * b[i]);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = a => { const n = Math.hypot(...a) || 1; return a.map(v => v / n); };
  const rotAbout = (v, k, t) => add(add(v.map(x => x * Math.cos(t)), cross(k, v).map(x => x * Math.sin(t))), k, dot(k, v) * (1 - Math.cos(t)));
  // body frame of the limb's parent: (forward, up, outward)
  const FWD = [1, 0, 0], UP = [0, 1, 0], OUT = [0, 0, 1];
  function dirOf(r, a) { return [Math.sin(r * DEG) * Math.cos(a * DEG), -Math.cos(r * DEG), Math.sin(r * DEG) * Math.sin(a * DEG)]; }
  function bent(u, bend, twist, forward) {
    // a hinge bending `bend` degrees away from u: the elbow folds the forearm
    // forward, the knee folds the shank back; twist turns the bending plane
    let p = cross(OUT, u);
    if (Math.hypot(...p) < 1e-3) p = FWD;
    p = norm(p).map(v => forward ? v : -v);
    p = rotAbout(p, u, twist * DEG);
    return norm(add(u.map(v => v * Math.cos(bend * DEG)), p, Math.sin(bend * DEG)));
  }
  function limbs(ps) {
    const ua = dirOf(ps.uaRaise, ps.uaAz), th = dirOf(ps.thRaise, ps.thAz);
    return { ua, fa: bent(ua, ps.elBend, ps.elTwist, true), th, sh: bent(th, ps.knBend, 0, false) };
  }
  const LIMB = {
    upper_arm: { label: 'Upper arm', joint: 'shoulder', frame: 'chest',
      angle: ['shoulder elevation', ps => ps.uaRaise], vec: 'ua', sweep: 'uaRaise', range: [0, 180],
      ctl: [['Raise', 'uaRaise', 0, 180], ['Towards', 'uaAz', 0, 90, { 0: 'front', 90: 'side' }]],
      presets: [['Forward', { uaAz: 0 }], ['To the side', { uaAz: 90 }], ['In between', { uaAz: 45 }]],
      hint: 'Elevation is how far the arm is raised from hanging, in any direction. The direction is a unit vector in the chest frame (forward, up, outward). Both are continuous in every pose.' },
    forearm: { label: 'Forearm', joint: 'elbow', frame: 'chest',
      angle: ['elbow flexion', ps => ps.elBend], vec: 'fa', sweep: 'elBend', range: [0, 150],
      ctl: [['Bend', 'elBend', 0, 150], ['Twist', 'elTwist', -90, 90, { 0: 'none' }]],
      presets: [['Arm down', { uaRaise: 5, uaAz: 60 }], ['Arm forward', { uaRaise: 90, uaAz: 0 }], ['Arm to the side', { uaRaise: 90, uaAz: 90 }]],
      hint: 'Elbow flexion is how far the elbow is bent. The forearm direction, also in the chest frame, changes when the upper arm twists, while the elbow angle stays the same.' },
    thigh: { label: 'Thigh', joint: 'hip', frame: 'pelvis',
      angle: ['hip elevation', ps => ps.thRaise], vec: 'th', sweep: 'thRaise', range: [0, 120],
      ctl: [['Raise', 'thRaise', 0, 120], ['Towards', 'thAz', 0, 60, { 0: 'front' }]],
      presets: [['Forward', { thAz: 0 }], ['Out to the side', { thAz: 60 }]],
      hint: 'Hip elevation is how far the thigh is raised from hanging. The direction is a unit vector in the pelvis frame, so it does not depend on which way the person faces.' },
    shank: { label: 'Shank', joint: 'knee', frame: 'pelvis',
      angle: ['knee flexion', ps => ps.knBend], vec: 'sh', sweep: 'knBend', range: [0, 150],
      ctl: [['Bend', 'knBend', 0, 150]],
      presets: [['Standing', { thRaise: 0 }], ['Sitting', { thRaise: 90, thAz: 0 }]],
      hint: 'Knee flexion is how far the knee is bent. The shank direction in the pelvis frame differs between sitting and standing at the same knee angle.' },
  };
  let cur = 'upper_arm';
  const armBones = $('#armBones'), armCtl = $('#armCtl');
  Object.entries(LIMB).forEach(([k, L]) => {
    const b = el('button', { type: 'button', role: 'tab' }, `${L.label} <small>(${L.joint})</small>`);
    b.addEventListener('click', () => setLimb(k));
    armBones.append(b);
  });
  function setLimb(k) {
    cur = k; const L = LIMB[k];
    $$('button', armBones).forEach((b, i) => b.classList.toggle('on', Object.keys(LIMB)[i] === k));
    armCtl.innerHTML = '';
    L.ctl.forEach(([name, key, lo, hi, words]) => {
      const lab = el('label', {}, `${name} <input type="range" min="${lo}" max="${hi}" value="${pose[key]}"><output></output>`);
      const inp = $('input', lab), out = $('output', lab);
      const show = () => { out.textContent = (words && words[inp.value]) || inp.value + '°'; };
      inp.addEventListener('input', () => { pose[key] = +inp.value; show(); limbUpdate(); });
      show(); armCtl.append(lab);
    });
    const pr = el('div', { class: 'presets' });
    L.presets.forEach(([name, set]) => {
      const b = el('button', { type: 'button' }, name);
      b.addEventListener('click', () => { Object.assign(pose, set); setLimb(cur); });
      pr.append(b);
    });
    armCtl.append(pr);
    $('#armTitle').textContent = `Targets of the right ${L.label.toLowerCase()}`;
    $('#armHint').textContent = L.hint;
    limbUpdate();
  }
  function fit(c) {
    const dpr = devicePixelRatio || 1, w = c.clientWidth, h = Math.round(w * c._ratio);
    if (c._w !== w) { c._w = w; c._h = h; c.width = w * dpr; c.height = h * dpr; c.style.height = h + 'px'; }
    const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [g, c._w, c._h];
  }
  armC._ratio = 440 / 520; plotC._ratio = 300 / 560;
  function project(p, cx, cy, s, yw, pt) {
    // humanml3d (x the person's left, y up, z forward), camera in front: screen x = x
    let [x, y, z] = p;
    const c = Math.cos(yw), sn = Math.sin(yw);
    [x, z] = [x * c + z * sn, -x * sn + z * c];
    const cp = Math.cos(pt), sp = Math.sin(pt);
    [y, z] = [y * cp - z * sp, y * sp + z * cp];
    return [cx + x * s, cy - y * s, z];
  }
  const world = d => [-d[2], d[1], d[0]];        // right side: outward = -x
  function drawFigure() {
    const [g, w, h] = fit(armC);
    g.clearRect(0, 0, w, h);
    const L = limbs(pose);
    const J = { pel: [0, .95, 0], lhip: [.09, .87, 0], rhip: [-.09, .87, 0], neck: [0, 1.47, 0], head: [0, 1.66, .02],
      lsh: [.18, 1.42, 0], rsh: [-.18, 1.42, 0], lel: [.21, 1.14, 0], lwr: [.22, .88, .03],
      lkn: [.10, .48, .02], lan: [.10, .08, 0], lft: [.12, .02, .13] };
    J.rel = add(J.rsh, world(L.ua), .29); J.rwr = add(J.rel, world(L.fa), .26);
    J.rkn = add(J.rhip, world(L.th), .40); J.ran = add(J.rkn, world(L.sh), .40);
    J.rft = add(J.ran, world(norm(add(cross(OUT, L.sh), L.sh, .3))), .13);
    const fr = Math.min(...Object.values(J).map(p => p[1]));
    const s = h / 2.1, cx = w / 2, cy = h * .94;
    const pr = p => project([p[0], p[1] - Math.min(0, fr) - .02, p[2]], cx, cy, s, yaw, pitch);
    g.strokeStyle = css('--line'); g.lineWidth = 1;
    for (let k = -3; k <= 3; k++) {
      const a1 = pr([k * .15, 0, -.45]), a2 = pr([k * .15, 0, .45]), b1 = pr([-.45, 0, k * .15]), b2 = pr([.45, 0, k * .15]);
      g.beginPath(); g.moveTo(a1[0], a1[1]); g.lineTo(a2[0], a2[1]); g.moveTo(b1[0], b1[1]); g.lineTo(b2[0], b2[1]); g.stroke();
    }
    g.lineCap = 'round';
    const seg = (a, b, col, wd, dash) => {
      const p = pr(a), q = pr(b); g.strokeStyle = col; g.lineWidth = wd; g.setLineDash(dash || []);
      g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke(); g.setLineDash([]);
    };
    const mute = css('--mute'), acc = css('--acc');
    [['pel', 'lhip'], ['pel', 'rhip'], ['pel', 'neck'], ['neck', 'lsh'], ['neck', 'rsh'], ['lsh', 'lel'], ['lel', 'lwr'],
     ['lhip', 'lkn'], ['lkn', 'lan'], ['lan', 'lft'], ['ran', 'rft']].forEach(([a, b]) => seg(J[a], J[b], mute, 5));
    const hd = pr(J.head); g.strokeStyle = mute; g.lineWidth = 5; g.beginPath(); g.arc(hd[0], hd[1], s * .075, 0, 7); g.stroke();
    const parts = { upper_arm: ['rsh', 'rel'], forearm: ['rel', 'rwr'], thigh: ['rhip', 'rkn'], shank: ['rkn', 'ran'] };
    // the parent bone's line continued: where the bone would point unbent
    const [a, b] = parts[cur];
    if (cur === 'upper_arm' || cur === 'thigh') seg(J[a], add(J[a], [0, -.5, 0]), mute, 2, [5, 6]);
    else { const parent = cur === 'forearm' ? L.ua : L.th; seg(J[a], add(J[a], world(parent), .3), mute, 2, [5, 6]); }
    for (const [k, [p, q]] of Object.entries(parts)) seg(J[p], J[q], k === cur ? acc : css('--sub'), k === cur ? 9 : 5.5);
    g.fillStyle = acc; [J[a], J[b]].forEach(p => { const q = pr(p); g.beginPath(); g.arc(q[0], q[1], 5, 0, 7); g.fill(); });
    g.fillStyle = css('--sub'); g.font = '600 13px Inter, sans-serif'; g.fillText('drag to turn', 12, 22);
  }
  function drawPlot() {
    const [g, w, h] = fit(plotC);
    g.clearRect(0, 0, w, h);
    const Lm = LIMB[cur], [lo, hi] = Lm.range, x0 = pose[Lm.sweep];
    const at = v => limbs({ ...pose, [Lm.sweep]: v })[Lm.vec];
    const L = 44, R = w - 14, top1 = 14, bot1 = h * .40, top2 = h * .52, bot2 = h - 26;
    const X = v => L + (R - L) * (v - lo) / (hi - lo);
    const Ya = v => bot1 - (bot1 - top1) * v / 180, Yd = v => bot2 - (bot2 - top2) * (v + 1) / 2;
    g.font = '12px Inter, sans-serif'; g.fillStyle = css('--sub'); g.strokeStyle = css('--line'); g.lineWidth = 1;
    [[0, 90, 180], [-1, 0, 1]].forEach((ticks, j) => ticks.forEach(v => {
      const y = j ? Yd(v) : Ya(v); g.beginPath(); g.moveTo(L, y); g.lineTo(R, y); g.stroke();
      g.textAlign = 'right'; g.fillText(v + (j ? '' : '°'), L - 6, y + 4);
    }));
    g.textAlign = 'center';
    const stepT = hi - lo === 180 ? 45 : 30;
    for (let v = lo; v <= hi + 1e-6; v += stepT) g.fillText(Math.round(v) + '°', X(v), h - 8);
    const series = [
      [Lm.angle[0], css('--ink'), v => Ya(v), v => v],
      ['forward', '#0b84c6', v => Yd(at(v)[0]), v => at(v)[0]],
      ['up', '#3f8f5b', v => Yd(at(v)[1]), v => at(v)[1]],
      ['outward', '#e07b39', v => Yd(at(v)[2]), v => at(v)[2]],
    ];
    g.lineWidth = 2.4;
    for (const [, col, f] of series) {
      g.strokeStyle = col; g.beginPath();
      for (let k = 0; k <= 90; k++) { const v = lo + (hi - lo) * k / 90, y = f(v); k ? g.lineTo(X(v), y) : g.moveTo(X(v), y); }
      g.stroke();
    }
    g.strokeStyle = css('--pred'); g.lineWidth = 1.5; g.setLineDash([4, 4]);
    g.beginPath(); g.moveTo(X(x0), top1); g.lineTo(X(x0), bot2); g.stroke(); g.setLineDash([]);
    for (const [, col, f] of series) { g.fillStyle = col; g.beginPath(); g.arc(X(x0), f(x0), 4.5, 0, 7); g.fill(); }
    g.textAlign = 'left';
    const tag = (t, y) => {            // a label on a backing, so curves do not run through it
      const tw = g.measureText(t).width;
      g.fillStyle = css('--card'); g.globalAlpha = .85; g.fillRect(L + 3, y - 11, tw + 8, 15); g.globalAlpha = 1;
      g.fillStyle = css('--sub'); g.fillText(t, L + 7, y);
    };
    tag(Lm.angle[0], top1 + 12); tag(`${Lm.label.toLowerCase()} direction (${Lm.frame} frame)`, top2 + 12);
    $('#armRead').innerHTML = series.map(([n, col, , v], i) =>
      `<span><i style="background:${col}"></i>${n} ${i ? v(x0).toFixed(2) : v(x0).toFixed(0) + '°'}</span>`).join('');
  }
  function limbUpdate() { drawFigure(); drawPlot(); }
  function draggable(c, get, set, after) {
    let start = null;
    c.addEventListener('pointerdown', e => { start = [e.clientX, e.clientY, ...get()]; c.setPointerCapture(e.pointerId); });
    c.addEventListener('pointermove', e => {
      if (!start) return;
      set(start[2] + (e.clientX - start[0]) * .01, Math.max(-1.2, Math.min(1.2, start[3] + (e.clientY - start[1]) * .01)));
      after();
    });
    c.addEventListener('pointerup', () => start = null);
    c.addEventListener('pointercancel', () => start = null);
  }
  draggable(armC, () => [yaw, pitch], (y, p) => { yaw = y; pitch = p; }, drawFigure);
  addEventListener('resize', limbUpdate);
  redrawers.push(limbUpdate);
  setLimb('upper_arm');
  // picking an arm or leg bone above shows it here
  showLimbFor = name => { const k = name.replace(/^(left|right)_/, ''); if (LIMB[k]) setLimb(k); };

  /* ----------------------------------------------------------------- demo */
  const dv = $('#demoV'), dc = $('#demoC'), slider = $('#demoSlider'), playB = $('#demoPlay'), ghostT = $('#demoGhost');
  dv.dataset.manual = '1';
  let D = null, dyaw = 0.35, dpitch = 0.12, playing = false;
  fetch('media/demo/pose.json').then(r => r.json()).then(d => {
    D = d; slider.max = d.true.length - 1;
    const first = d.owner.findIndex(o => o >= 0);
    slider.value = Math.max(first, Math.round(d.true.length * .55));
    seek(+slider.value);
  }).catch(() => { $('#demoRead').textContent = 'The demo data could not be loaded.'; });
  function frameNow() { return Math.max(0, Math.min(D.true.length - 1, Math.round(dv.currentTime * D.fps))); }
  function seek(i) { if (!D) return; dv.currentTime = (i + .5) / D.fps; drawDemo(i); }
  function drawDemo(i) {
    if (!D) return;
    const dpr = devicePixelRatio || 1, w = dc.clientWidth, h = dc.clientHeight;
    if (dc.width !== w * dpr || dc.height !== h * dpr) { dc.width = w * dpr; dc.height = h * dpr; }
    const g = dc.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    const tr = D.true[i], s = D.owner[i], rc = s >= 0 ? D.recon[s] : null;
    const c = [tr[0][0] + D.offset[0], tr[0][1] + D.offset[1], tr[0][2] + D.offset[2]];
    const half = w / 2, sc = Math.min(half, h) / (2 * D.extent) * .9;
    const pred = new Set(Object.values(D.predicted).map(p => p.join('-')));
    const acc = css('--acc'), red = css('--pred'), mute = css('--mute');
    const P3 = (p, ox) => project([p[0] - c[0], p[1] - c[1], p[2] - c[2]], ox + half / 2, h / 2 + 10, sc, dyaw, dpitch);
    const seg = (a, b, col, wd, dash) => {
      g.strokeStyle = col; g.lineWidth = wd; g.setLineDash(dash || []);
      g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); g.setLineDash([]);
    };
    g.lineCap = 'round';
    const skel = (J, ox, hot) => {
      for (const [a, b] of D.bones) {
        const isHot = hot && pred.has(a + '-' + b);
        seg(P3(J[a], ox), P3(J[b], ox), isHot ? red : acc, isHot ? 4.2 : 2.6);
      }
      g.fillStyle = acc;
      J.forEach(p => { const q = P3(p, ox); g.beginPath(); g.arc(q[0], q[1], 2.6, 0, 7); g.fill(); });
    };
    skel(tr, 0, false);
    skel(rc || tr, half, !!rc);
    g.strokeStyle = css('--line'); g.lineWidth = 1; g.beginPath(); g.moveTo(half, 16); g.lineTo(half, h - 16); g.stroke();
    if (rc) {
      if (ghostT.checked) for (const [, [a, b]] of Object.entries(D.predicted)) seg(P3(tr[a], half), P3(tr[b], half), mute, 2.4, [5, 5]);
      g.font = '700 12px Inter, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
      for (const [bone, [a, b]] of Object.entries(D.predicted)) {
        const key = bone + '_direction';
        const err = D.readout[key] && D.readout[key].error[s];
        if (err == null) continue;
        const tip = [0, 1, 2].map(k => rc[b][k] + .35 * (rc[b][k] - rc[a][k]));
        const q = P3(tip, half), label = Math.round(err) + '°';
        const tw = g.measureText(label).width + 10;
        g.fillStyle = css('--card'); g.globalAlpha = .9; g.fillRect(q[0] - tw / 2, q[1] - 9, tw, 18); g.globalAlpha = 1;
        g.fillStyle = red; g.fillText(label, q[0], q[1]);
      }
    } else {
      g.fillStyle = css('--sub'); g.font = '13px Inter, sans-serif'; g.textAlign = 'center';
      g.fillText('No prediction at this frame', half * 1.5, h - 18);
    }
    slider.value = i;
    $('#demoRead').innerHTML = s < 0 ? '' : Object.entries(D.readout).map(([n, r]) => r.kind === 'direction'
      ? `<span>${n} <b>off by ${r.error[s].toFixed(1)}°</b></span>`
      : `<span>${n} <b>${r.pred[s].toFixed(0)}°</b> predicted, <em>${r.true[s].toFixed(0)}°</em> true</span>`).join('');
  }
  slider.addEventListener('input', () => { pause(); seek(+slider.value); });
  ghostT.addEventListener('change', () => D && drawDemo(+slider.value));
  function tick() { if (!playing) return; drawDemo(frameNow()); requestAnimationFrame(tick); }
  function play() { if (!D) return; if (+slider.value >= D.true.length - 1) seek(0); playing = true; playB.classList.add('playing'); dv.play().catch(() => {}); tick(); }
  function pause() { playing = false; playB.classList.remove('playing'); dv.pause(); }
  playB.addEventListener('click', () => playing ? pause() : play());
  dv.addEventListener('ended', pause);
  draggable(dc, () => [dyaw, dpitch], (y, p) => { dyaw = y; dpitch = p; }, () => drawDemo(+slider.value));
  addEventListener('resize', () => D && drawDemo(+slider.value));
  redrawers.push(() => D && drawDemo(+slider.value));
  new IntersectionObserver(es => es.forEach(e => { if (!e.isIntersecting) pause(); }), { threshold: 0 }).observe($('#demoBox'));

  /* -------------------------------------------------------------- layouts */
  const LY = {
    A: { H: [['H0', 15.58], ['H1', 6.49], ['H2', 6.33]], LH: [['L0', 10.94], ['LH1', 12.74]] },
    B: { H: [['H0', 14.82], ['H1', 7.80], ['H2', 5.84]], LH: [['L0', 10.94], ['LH1', 7.83], ['LH2', 10.18]] },
    C: { H: [['H0', 19.21], ['H1', 10.08], ['H2', 13.03]], LH: [['L0', 10.94], ['LH1', 8.17], ['LH2', 6.86]] },
  };
  const SERIES = { H: 'Designer alone', LH: 'Designer + LLM' };
  const HINTS = {
    H: 'The designer drew a first layout, read the test report, and revised it twice.',
    LH: 'An LLM proposed the first layout (L0, the same for everyone); the designer then revised it together with the LLM, using the same test reports.',
  };
  let lyD = 'B', lyS = 'H';
  const segBtns = (box, items, cur, set) => {
    box.innerHTML = '';
    items.forEach(([k, label]) => {
      const b = el('button', { type: 'button' }, label);
      b.classList.toggle('on', k === cur());
      b.addEventListener('click', () => { set(k); drawLy(); });
      box.append(b);
    });
  };
  function drawLy() {
    segBtns($('#lyDesigner'), [['A', 'Designer A'], ['B', 'Designer B'], ['C', 'Designer C']], () => lyD, k => lyD = k);
    segBtns($('#lySeries'), Object.entries(SERIES), () => lyS, k => lyS = k);
    const list = LY[lyD][lyS], best = Math.min(...list.map(x => x[1]));
    const row = $('#lyRow'); row.innerHTML = '';
    list.forEach(([name, mp], k) => {
      if (k) row.append(el('div', { class: 'arrow', 'aria-hidden': 'true' }, '→'));
      const file = name === 'L0' ? 'L0' : `${lyD}_${name}`;
      const label = name === 'L0' ? 'LLM layout' : (k === 0 ? 'First layout' : `Revision ${k}`);
      const c = el('figure', { class: 'lcard' + (mp === best ? ' best' : '') },
        `<img src="media/paper/layout_${file}.jpg" alt="${label}" loading="lazy">
         <div class="cap"><b>${label}</b><span class="mp">${mp.toFixed(2)}° mean error</span><div class="err"><i></i></div></div>`);
      row.append(c);
      requestAnimationFrame(() => requestAnimationFrame(() => { $('.err i', c).style.width = Math.min(100, mp / 20 * 100) + '%'; }));
    });
    $('#lyHint').textContent = HINTS[lyS];
  }
  drawLy();

  /* ---------------------------------------------------------------- start */
  const FILES = {
    'sensor.csv': `<span class="c"># one row per sensor sample; any number of channels</span>
EpochTime;S1;S2;S3;S4
1759494276.341718;53769.55;51243.90;36085.11;38830.52
1759494276.455033;53833.33;51542.86;36085.11;38544.36
...`,
    'frames.csv': `<span class="c"># the time of every video frame, on the same clock</span>
FrameIndex;EpochTime
0;1759494279.5043106
1;1759494279.5211549
...`,
    'video.mp4': `<span class="c"># any webcam video of the person; one file per session</span>
<span class="k">session_1/</span>
  video.mp4      <span class="c"># the recording</span>
  sensor.csv     <span class="c"># the garment's sensors</span>
  frames.csv     <span class="c"># when each frame was taken</span>`,
  };
  const ft = $('#fileTabs'), fc = $('#fileCode');
  const showFile = k => { fc.innerHTML = FILES[k]; $$('button', ft).forEach(b => b.classList.toggle('on', b.textContent === k)); };
  Object.keys(FILES).forEach(k => { const b = el('button', { type: 'button' }, k); b.addEventListener('click', () => showFile(k)); ft.append(b); });
  showFile('video.mp4');

  /* ----------------------------------------------------------- video, copy */
  const yt = $('#yt');
  yt.addEventListener('click', () => {
    if ($('iframe', yt)) return;
    yt.append(el('iframe', { src: `https://www.youtube-nocookie.com/embed/${yt.dataset.id}?autoplay=1&rel=0`,
      title: 'Tutorial video', allow: 'autoplay; encrypted-media; picture-in-picture', allowfullscreen: '' }));
    $('.bigplay', yt).remove();
  });
  $$('.copy').forEach(b => b.addEventListener('click', async () => {
    const text = $('#' + b.dataset.copy).textContent;
    try { await navigator.clipboard.writeText(text); } catch (e) {
      const r = document.createRange(); r.selectNodeContents($('#' + b.dataset.copy));
      const s = getSelection(); s.removeAllRanges(); s.addRange(r);
    }
    const t = b.textContent; b.textContent = 'Copied'; b.classList.add('done');
    setTimeout(() => { b.textContent = t; b.classList.remove('done'); }, 1600);
  }));
})();
