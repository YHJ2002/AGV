const ALGORITHM_OPTIONS = {
  scheduler: [
    { value: 'ta', label: 'TA Scheduler' },
    { value: 'random', label: 'Random Scheduler' }
  ],
  planner: [
    { value: 'astar', label: 'A* Planner' },
    { value: 'cbs_fw', label: 'CBS-FW Planner' },
    { value: 'dhc', label: 'DHC Planner' }
  ],
  order_mode: [
    { value: 'oneshot', label: 'One-shot' },
    { value: 'continuous_constant', label: 'Continuous Constant' },
    { value: 'continuous_periodic', label: 'Continuous Periodic' },
    { value: 'continuous_pareto', label: 'Continuous Pareto' },
    { value: 'continuous_burst', label: 'Continuous Burst' }
  ]
};

const algorithmState = {
  current: {
    scheduler: 'ta',
    planner: 'cbs_fw',
    order_mode: 'oneshot'
  },
  draft: {
    scheduler: 'ta',
    planner: 'cbs_fw',
    order_mode: 'oneshot'
  },
  appliesOnReset: true
};

function renderSelectOptions(options, selectedValue) {
  return options
    .map(({ value, label }) =>
      `<option value="${value}"${value === selectedValue ? ' selected' : ''}>${label}</option>`)
    .join('');
}

function initPanel() {
  const panel = document.getElementById('panel');
  panel.innerHTML = `
    <div class="panel-header">
      <p class="panel-eyebrow">Warehouse Operations</p>
      <h2>Control Console</h2>
    </div>

    <div class="panel-section panel-section-primary">
      <div class="section-caption">Simulation controls</div>
      <div class="basic-controls">
        <button id="toggleBtn">Resume</button>
        <button id="stepBtn">Step</button>
      </div>
    </div>

    <div class="panel-section algorithm-section">
      <div class="algorithm-section-header">
        <div>
          <div class="section-caption">Task setup</div>
          <div class="section-title">Algorithm Settings</div>
        </div>
        <span class="algorithm-status-pill" id="algorithmStatusPill">Active</span>
      </div>
      <p class="algorithm-hint" id="algorithmHint">Choose algorithms for the next reset cycle.</p>

      <div class="algorithm-grid">
        <label class="select-field">
          <span>Task Scheduling</span>
          <select id="schedulerSelect">
            ${renderSelectOptions(ALGORITHM_OPTIONS.scheduler, algorithmState.draft.scheduler)}
          </select>
        </label>

        <label class="select-field">
          <span>Path Planning</span>
          <select id="plannerSelect">
            ${renderSelectOptions(ALGORITHM_OPTIONS.planner, algorithmState.draft.planner)}
          </select>
        </label>

        <label class="select-field">
          <span>Order Generation</span>
          <select id="orderModeSelect">
            ${renderSelectOptions(ALGORITHM_OPTIONS.order_mode, algorithmState.draft.order_mode)}
          </select>
        </label>
      </div>

      <div class="algorithm-actions">
        <button id="applyAlgorithmsBtn" class="primary-btn" disabled>Apply</button>
        <span class="algorithm-meta" id="algorithmMeta">Current settings are active.</span>
      </div>
    </div>

    <div class="collapsible" id="displaySettings">
      <div class="collapsible-header" data-icon=">">Display Settings</div>
      <div class="collapsible-content">
        <label><input type="checkbox" id="showAgvId"> Show AGV IDs</label><br>
        <label><input type="checkbox" id="showBoxId"> Show Box IDs</label><br>
        <label><input type="checkbox" id="showRecvId"> Show Receive Zone IDs</label>
      </div>
    </div>

    <div class="collapsible" id="incidentSim">
      <div class="collapsible-header" data-icon=">">Failure Simulation</div>
      <div class="collapsible-content">
        <div class="form-group">
          <input type="number" id="damageAgvId" placeholder="AGV ID to damage">
          <button id="damageBtn">Damage</button>
        </div>
        <div class="form-group">
          <input type="number" id="repairAgvId" placeholder="AGV ID to repair">
          <button id="repairBtn">Repair</button>
        </div>
      </div>
    </div>

    <div class="collapsible" id="metrics">
      <div class="collapsible-header" data-icon=">">Metrics</div>
      <div class="collapsible-content" id="metricsContent">
        <p>AGVs: 0</p>
        <p>Tasks: 0</p>
        <p>FPS: 0</p>
      </div>
    </div>

    <div class="stop-section">
      <button id="stopBtn" class="stop-btn">Stop</button>
      <button id="resetBtn" class="reset-btn">Reset</button>
    </div>
  `;

  setupCollapsibles();
  makePanelDraggable(panel);
  setupAlgorithmControls();

  let isPaused = true;
  const toggleBtn = document.getElementById('toggleBtn');
  const stepBtn = document.getElementById('stepBtn');
  toggleBtn.classList.add('paused');

  toggleBtn.onclick = () => {
    const socket = window.appSocket;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;

    if (isPaused) {
      socket.send(JSON.stringify({ cmd: 'resume' }));
      toggleBtn.textContent = 'Pause';
      toggleBtn.classList.remove('paused');
    } else {
      socket.send(JSON.stringify({ cmd: 'pause' }));
      toggleBtn.textContent = 'Resume';
      toggleBtn.classList.add('paused');
    }
    isPaused = !isPaused;
  };

  stepBtn.onclick = () => {
    const socket = window.appSocket;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    if (!isPaused) {
      socket.send(JSON.stringify({ cmd: 'pause' }));
      toggleBtn.textContent = 'Resume';
      toggleBtn.classList.add('paused');
      isPaused = true;
    }
    socket.send(JSON.stringify({ cmd: 'step' }));
  };

  document.getElementById('damageBtn').onclick = () => {
    const id = parseInt(document.getElementById('damageAgvId').value, 10);
    const socket = window.appSocket;
    if (!Number.isNaN(id) && socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ cmd: 'damage', agv_id: id }));
    }
  };

  document.getElementById('repairBtn').onclick = () => {
    const id = parseInt(document.getElementById('repairAgvId').value, 10);
    const socket = window.appSocket;
    if (!Number.isNaN(id) && socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ cmd: 'repair', agv_id: id }));
    }
  };

  document.getElementById('stopBtn').onclick = () => {
    const socket = window.appSocket;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ cmd: 'stop' }));
    }
  };

  const resetBtn = document.getElementById('resetBtn');
  resetBtn.onclick = () => {
    const socket = window.appSocket;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ cmd: 'reset' }));
      resetBtn.disabled = true;
      resetBtn.textContent = 'Resetting...';
      setTimeout(() => {
        resetBtn.disabled = false;
        resetBtn.textContent = 'Reset';
      }, 1500);
    }
  };

  document.getElementById('showAgvId').addEventListener('change', (e) => {
    window.sceneWorld?.agvs.forEach((agv) => agv.setLabelVisible(e.target.checked));
  });

  document.getElementById('showBoxId').addEventListener('change', (e) => {
    window.sceneWorld?.boxes.forEach((box) => box.setLabelVisible(e.target.checked));
  });

  document.getElementById('showRecvId').addEventListener('change', (e) => {
    window.sceneWorld?.receiveAreas.forEach((area) => area.setLabelVisible(e.target.checked));
  });
}

function setupAlgorithmControls() {
  const schedulerSelect = document.getElementById('schedulerSelect');
  const plannerSelect = document.getElementById('plannerSelect');
  const orderModeSelect = document.getElementById('orderModeSelect');
  const applyBtn = document.getElementById('applyAlgorithmsBtn');

  const syncDraftFromInputs = () => {
    algorithmState.draft.scheduler = schedulerSelect.value;
    algorithmState.draft.planner = plannerSelect.value;
    algorithmState.draft.order_mode = orderModeSelect.value;
    refreshAlgorithmPanel();
  };

  schedulerSelect.addEventListener('change', syncDraftFromInputs);
  plannerSelect.addEventListener('change', syncDraftFromInputs);
  orderModeSelect.addEventListener('change', syncDraftFromInputs);

  applyBtn.addEventListener('click', () => {
    const socket = window.appSocket;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;

    syncDraftFromInputs();
    applyBtn.disabled = true;
    setAlgorithmFeedback({
      status: 'pending',
      message: 'Saving algorithm selection...'
    });
    socket.send(JSON.stringify({
      cmd: 'set_algorithms',
      scheduler: algorithmState.draft.scheduler,
      planner: algorithmState.draft.planner,
      order_mode: algorithmState.draft.order_mode
    }));
  });

  refreshAlgorithmPanel();
}

function hasAlgorithmDraftChanges() {
  return (
    algorithmState.draft.scheduler !== algorithmState.current.scheduler ||
    algorithmState.draft.planner !== algorithmState.current.planner ||
    algorithmState.draft.order_mode !== algorithmState.current.order_mode
  );
}

function setAlgorithmFeedback({ status = 'active', message = '', appliesOnReset = true } = {}) {
  algorithmState.appliesOnReset = appliesOnReset;

  const pill = document.getElementById('algorithmStatusPill');
  const hint = document.getElementById('algorithmHint');
  const meta = document.getElementById('algorithmMeta');
  const applyBtn = document.getElementById('applyAlgorithmsBtn');

  if (pill) {
    pill.dataset.state = status;
    pill.textContent = status === 'dirty'
      ? 'Pending'
      : status === 'pending'
        ? 'Saving'
        : status === 'error'
          ? 'Error'
          : 'Active';
  }

  if (hint) {
    hint.textContent = message || (
      appliesOnReset
        ? 'Choose algorithms for the next reset cycle.'
        : 'Algorithm settings are active now.'
    );
  }

  if (meta) {
    meta.textContent = hasAlgorithmDraftChanges()
      ? 'Selections differ from the active setup.'
      : (appliesOnReset ? 'Current settings are active.' : 'Algorithms are live.');
  }

  if (applyBtn && status !== 'pending') {
    applyBtn.disabled = !hasAlgorithmDraftChanges();
  }
}

function refreshAlgorithmPanel() {
  const schedulerSelect = document.getElementById('schedulerSelect');
  const plannerSelect = document.getElementById('plannerSelect');
  const orderModeSelect = document.getElementById('orderModeSelect');
  const applyBtn = document.getElementById('applyAlgorithmsBtn');

  if (schedulerSelect) schedulerSelect.value = algorithmState.draft.scheduler;
  if (plannerSelect) plannerSelect.value = algorithmState.draft.planner;
  if (orderModeSelect) orderModeSelect.value = algorithmState.draft.order_mode;
  if (applyBtn) applyBtn.disabled = !hasAlgorithmDraftChanges();

  setAlgorithmFeedback({
    status: hasAlgorithmDraftChanges() ? 'dirty' : 'active',
    appliesOnReset: algorithmState.appliesOnReset
  });
}

function updateAlgorithmConfig(config = {}) {
  const {
    scheduler,
    planner,
    order_mode: orderMode,
    status = 'active',
    message = '',
    applies_on_reset: appliesOnReset = true
  } = config;

  if (scheduler) {
    algorithmState.current.scheduler = scheduler;
    algorithmState.draft.scheduler = scheduler;
  }
  if (planner) {
    algorithmState.current.planner = planner;
    algorithmState.draft.planner = planner;
  }
  if (orderMode) {
    algorithmState.current.order_mode = orderMode;
    algorithmState.draft.order_mode = orderMode;
  }

  refreshAlgorithmPanel();
  setAlgorithmFeedback({ status, message, appliesOnReset });
}

function updateMetrics(metrics) {
  const metricsContent = document.getElementById('metricsContent');
  if (!metricsContent) return;

  let html = '';
  for (const [key, value] of Object.entries(metrics)) {
    const formattedKey = key
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (char) => char.toUpperCase());
    html += `<p>${formattedKey}: ${value.toFixed(2)}</p>`;
  }

  metricsContent.innerHTML = html;
}

function setupCollapsibles() {
  const headers = document.querySelectorAll('.collapsible-header');
  headers.forEach((header) => {
    header.addEventListener('click', () => {
      const content = header.nextElementSibling;
      const expanded = header.classList.toggle('expanded');
      header.dataset.icon = expanded ? 'v' : '>';
      content.style.display = expanded ? 'block' : 'none';
    });
  });
}

function makePanelDraggable(panel) {
  let isDragging = false;
  let offsetX = 0;
  let offsetY = 0;

  panel.addEventListener('mousedown', (e) => {
    if (
      e.target.classList.contains('collapsible-header') ||
      e.target.closest('button') ||
      e.target.closest('select') ||
      e.target.closest('input') ||
      e.target.closest('label')
    ) {
      return;
    }

    isDragging = true;
    offsetX = e.clientX - panel.offsetLeft;
    offsetY = e.clientY - panel.offsetTop;
    panel.style.cursor = 'grabbing';
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    panel.style.left = `${e.clientX - offsetX}px`;
    panel.style.top = `${e.clientY - offsetY}px`;
    panel.style.right = 'auto';
  });

  document.addEventListener('mouseup', () => {
    isDragging = false;
    panel.style.cursor = 'grab';
  });
}

export { initPanel, updateAlgorithmConfig, updateMetrics };
