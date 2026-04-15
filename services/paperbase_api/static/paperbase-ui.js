(function () {
  const endpoints = {
    collections: "/api/v1/collections",
    jobs: "/api/v1/jobs",
    search: "/api/v1/search/papers",
    reindex: "/api/v1/search/reindex",
  };

  const state = {
    collections: [],
    selectedCollection: null,
    papers: [],
    searchResults: [],
    selectedPaper: null,
    selectedPaperStructured: null,
    collectionSummary: null,
    jobs: [],
    pollHandle: null,
  };

  const elements = {
    collectionsList: document.getElementById("collections-list"),
    paperList: document.getElementById("paper-list"),
    collectionSummary: document.getElementById("collection-summary"),
    paperDetail: document.getElementById("paper-detail"),
    jobsList: document.getElementById("jobs-list"),
    searchForm: document.getElementById("search-form"),
    searchInput: document.getElementById("paper-search-input"),
    extractButton: document.getElementById("extract-button"),
    reindexButton: document.getElementById("reindex-button"),
    statusBanner: document.getElementById("status-banner"),
  };

  function setStatus(message) {
    elements.statusBanner.textContent = message;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.message || `Request failed: ${response.status}`);
    }
    return payload;
  }

  async function loadCollections() {
    const payload = await fetchJson(endpoints.collections);
    state.collections = payload.data || [];
    if (!state.selectedCollection && state.collections.length > 0) {
      state.selectedCollection = state.collections[0];
      await loadCollectionSurface(state.selectedCollection.id);
    } else if (state.selectedCollection) {
      await loadCollectionSurface(state.selectedCollection.id);
    } else {
      renderCollections();
      renderEmptyCollectionState();
    }
    renderCollections();
  }

  async function loadCollectionSurface(collectionId) {
    const [collectionPayload, papersPayload, summaryPayload] = await Promise.all([
      fetchJson(`/api/v1/collections/${collectionId}`),
      fetchJson(`/api/v1/collections/${collectionId}/papers`),
      fetchJson(`/api/v1/collections/${collectionId}/structured-summary`),
    ]);

    state.selectedCollection = collectionPayload.data;
    state.papers = papersPayload.data || [];
    state.searchResults = [];
    state.collectionSummary = summaryPayload.data;
    state.selectedPaper = null;
    state.selectedPaperStructured = null;

    renderCollections();
    renderPapers();
    renderCollectionSummary();
    renderPaperDetail();
    updateActionButtons();
  }

  async function searchPapers(query) {
    if (!query.trim()) {
      state.searchResults = [];
      renderPapers();
      return;
    }

    const params = new URLSearchParams({ q: query.trim() });
    if (state.selectedCollection) {
      params.set("collection_id", state.selectedCollection.id);
    }
    const payload = await fetchJson(`${endpoints.search}?${params.toString()}`);
    state.searchResults = payload.data || [];
    renderPapers();
  }

  async function loadPaperDetail(paperId) {
    const [paperPayload, structuredPayload] = await Promise.all([
      fetchJson(`/api/v1/papers/${paperId}`),
      fetchJson(`/api/v1/papers/${paperId}/structured-data`),
    ]);

    state.selectedPaper = paperPayload.data;
    state.selectedPaperStructured = structuredPayload.data;
    renderPaperDetail();
  }

  async function loadJobs() {
    const payload = await fetchJson(`${endpoints.jobs}?limit=20`);
    state.jobs = payload.data || [];
    renderJobs();
    updatePolling();
  }

  async function queueReindex() {
    const payload = await fetchJson(endpoints.reindex, { method: "POST" });
    state.jobs.unshift(payload.data);
    renderJobs();
    updatePolling();
    setStatus("Queued a corpus reindex job.");
  }

  async function queueExtraction() {
    if (!state.selectedCollection) {
      return;
    }

    const body = {
      prompt_version: "paperbase-v1",
      schema_version: "paperbase-v1",
    };

    if (state.selectedCollection.extraction_profile_id) {
      body.extraction_profile_id = state.selectedCollection.extraction_profile_id;
    } else {
      body.schema_payload = {
        datasets: true,
        methods: true,
        metrics: true,
        results: true,
        engineering_tricks: true,
      };
    }

    const payload = await fetchJson(
      `/api/v1/collections/${state.selectedCollection.id}/extract`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    );
    state.jobs.unshift(payload.data);
    renderJobs();
    updatePolling();
    setStatus(`Queued extraction for ${state.selectedCollection.title}.`);
  }

  function updateActionButtons() {
    elements.extractButton.disabled = !state.selectedCollection;
  }

  function renderCollections() {
    if (state.collections.length === 0) {
      elements.collectionsList.innerHTML = '<div class="list-card muted">No collections yet.</div>';
      updateActionButtons();
      return;
    }

    elements.collectionsList.innerHTML = state.collections
      .map((collection) => {
        const active = state.selectedCollection && state.selectedCollection.id === collection.id;
        return `
          <div class="list-card" data-active="${active ? "true" : "false"}">
            <button type="button" data-collection-id="${collection.id}">
              <div class="list-title">${escapeHtml(collection.title)}</div>
              <div class="muted">${escapeHtml(collection.description || "Curated field database")}</div>
              <div class="pill-row">
                <span class="pill">${escapeHtml(collection.scope_type)}</span>
                ${(collection.tags || []).slice(0, 3).map((tag) => `<span class="pill">${escapeHtml(tag)}</span>`).join("")}
              </div>
            </button>
          </div>
        `;
      })
      .join("");

    elements.collectionsList.querySelectorAll("[data-collection-id]").forEach((button) => {
      button.addEventListener("click", async () => {
        const collectionId = button.getAttribute("data-collection-id");
        if (!collectionId) {
          return;
        }
        setStatus("Loading collection surface…");
        await loadCollectionSurface(collectionId);
        setStatus(`Loaded ${state.selectedCollection.title}.`);
      });
    });
    updateActionButtons();
  }

  function renderPapers() {
    const items = state.searchResults.length > 0 ? state.searchResults : state.papers.map((membership) => membership.paper);

    if (items.length === 0) {
      elements.paperList.innerHTML = '<div class="list-card muted">No papers match the current view.</div>';
      return;
    }

    elements.paperList.innerHTML = items
      .map((paper) => {
        const active = state.selectedPaper && state.selectedPaper.id === paper.id;
        const meta = [paper.publication_year, paper.venue].filter(Boolean).join(" / ");
        return `
          <div class="list-card" data-active="${active ? "true" : "false"}">
            <button type="button" data-paper-id="${paper.id}">
              <div class="list-title">${escapeHtml(paper.title)}</div>
              <div class="muted">${escapeHtml(meta || paper.provider)}</div>
              <div class="pill-row">
                ${(paper.tags || []).slice(0, 4).map((tag) => `<span class="pill">${escapeHtml(tag)}</span>`).join("")}
              </div>
            </button>
          </div>
        `;
      })
      .join("");

    elements.paperList.querySelectorAll("[data-paper-id]").forEach((button) => {
      button.addEventListener("click", async () => {
        const paperId = button.getAttribute("data-paper-id");
        if (!paperId) {
          return;
        }
        setStatus("Loading structured paper detail…");
        await loadPaperDetail(paperId);
        setStatus(`Opened ${state.selectedPaper.title}.`);
      });
    });
  }

  function renderCollectionSummary() {
    if (!state.selectedCollection || !state.collectionSummary) {
      renderEmptyCollectionState();
      return;
    }

    const summary = state.collectionSummary;
    const metricBoxes = [
      { label: "Papers", value: summary.paper_count },
      { label: "Extracted", value: summary.extracted_paper_count },
      { label: "Methods", value: summary.methods.length },
    ];

    elements.collectionSummary.innerHTML = `
      <div class="metric-grid">
        ${metricBoxes
          .map(
            (item) => `
              <div class="metric-box">
                <span class="metric-value">${escapeHtml(item.value)}</span>
                <div class="muted">${escapeHtml(item.label)}</div>
              </div>
            `,
          )
          .join("")}
      </div>
      <div class="detail-group">
        <div class="list-title">${escapeHtml(state.selectedCollection.title)}</div>
        <p class="summary-copy">${escapeHtml(state.selectedCollection.description || "Curated local collection")}</p>
      </div>
      <div class="detail-group">
        <strong>Datasets</strong>
        <div class="pill-row">${summary.datasets.slice(0, 6).map((item) => `<span class="pill">${escapeHtml(item.display_name)}</span>`).join("") || '<span class="muted">No extracted datasets yet.</span>'}</div>
      </div>
      <div class="detail-group">
        <strong>Metrics</strong>
        <div class="pill-row">${summary.metrics.slice(0, 6).map((item) => `<span class="pill">${escapeHtml(item.display_name)}</span>`).join("") || '<span class="muted">No extracted metrics yet.</span>'}</div>
      </div>
      <div class="detail-group">
        <strong>Top result rows</strong>
        <ul class="detail-list">
          ${summary.top_result_rows
            .slice(0, 5)
            .map(
              (row) => `<li>${escapeHtml(row.paper_title)} — ${escapeHtml(row.method || "Unknown method")} / ${escapeHtml(row.metric || "Unknown metric")} / ${escapeHtml(row.value_numeric ?? row.value_text ?? "n/a")}</li>`,
            )
            .join("") || "<li>No persisted result rows yet.</li>"}
        </ul>
      </div>
    `;
  }

  function renderPaperDetail() {
    if (!state.selectedPaper || !state.selectedPaperStructured) {
      elements.paperDetail.innerHTML = '<div class="muted">Select a paper to inspect its structured surface.</div>';
      return;
    }

    const structured = state.selectedPaperStructured;
    elements.paperDetail.innerHTML = `
      <div class="detail-group">
        <div class="list-title">${escapeHtml(state.selectedPaper.title)}</div>
        <div class="muted">${escapeHtml((state.selectedPaper.authors || []).join(", ") || "Unknown authors")}</div>
      </div>
      <div class="detail-group">
        <strong>Artifacts</strong>
        <div class="pill-row">
          <span class="pill">Datasets ${structured.datasets.length}</span>
          <span class="pill">Methods ${structured.methods.length}</span>
          <span class="pill">Metrics ${structured.metrics.length}</span>
          <span class="pill">Results ${structured.result_rows.length}</span>
          <span class="pill">Tricks ${structured.engineering_tricks.length}</span>
        </div>
      </div>
      <div class="detail-group">
        <strong>Methods</strong>
        <ul class="detail-list">
          ${structured.methods.slice(0, 5).map((item) => `<li>${escapeHtml(item.display_name)}</li>`).join("") || "<li>No extracted methods.</li>"}
        </ul>
      </div>
      <div class="detail-group">
        <strong>Engineering tricks</strong>
        <ul class="detail-list">
          ${structured.engineering_tricks
            .slice(0, 4)
            .map((item) => `<li>${escapeHtml(item.title)} — ${escapeHtml(item.description)}</li>`)
            .join("") || "<li>No extracted engineering tricks.</li>"}
        </ul>
      </div>
    `;
  }

  function renderJobs() {
    if (state.jobs.length === 0) {
      elements.jobsList.innerHTML = '<div class="list-card muted">No jobs queued yet.</div>';
      return;
    }

    elements.jobsList.innerHTML = state.jobs
      .map(
        (job) => `
          <div class="list-card">
            <div class="job-status" data-status="${escapeHtml(job.status)}">${escapeHtml(job.status)}</div>
            <div class="list-title">${escapeHtml(job.job_type)}</div>
            <div class="job-meta">${escapeHtml(job.created_at || "queued")}</div>
            <div class="muted">${escapeHtml(JSON.stringify(job.result || job.payload || {}))}</div>
            ${job.error_message ? `<div class="muted">${escapeHtml(job.error_message)}</div>` : ""}
          </div>
        `,
      )
      .join("");
  }

  function renderEmptyCollectionState() {
    elements.collectionSummary.innerHTML = '<div class="muted">Select a collection to inspect its structured surface.</div>';
    elements.paperList.innerHTML = '<div class="list-card muted">Select a collection to see its papers.</div>';
    elements.paperDetail.innerHTML = '<div class="muted">Select a paper to inspect its structured surface.</div>';
  }

  function updatePolling() {
    const hasActiveJobs = state.jobs.some((job) => job.status === "pending" || job.status === "running");
    if (hasActiveJobs && !state.pollHandle) {
      state.pollHandle = window.setInterval(() => {
        loadJobs().catch((error) => {
          setStatus(error.message);
        });
      }, 3000);
    }
    if (!hasActiveJobs && state.pollHandle) {
      window.clearInterval(state.pollHandle);
      state.pollHandle = null;
    }
  }

  async function initialize() {
    try {
      setStatus("Loading Paperbase Console…");
      await Promise.all([loadCollections(), loadJobs()]);
      renderPaperDetail();
      updateActionButtons();
      setStatus("Paperbase Console ready.");
    } catch (error) {
      setStatus(error.message);
    }
  }

  elements.searchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      setStatus("Searching papers…");
      await searchPapers(elements.searchInput.value);
      setStatus("Search updated.");
    } catch (error) {
      setStatus(error.message);
    }
  });

  elements.reindexButton.addEventListener("click", async () => {
    try {
      await queueReindex();
    } catch (error) {
      setStatus(error.message);
    }
  });

  elements.extractButton.addEventListener("click", async () => {
    try {
      await queueExtraction();
    } catch (error) {
      setStatus(error.message);
    }
  });

  initialize();
})();
