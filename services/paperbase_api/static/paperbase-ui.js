(function () {
  const endpoints = {
    workspaces: "/api/v1/workspaces",
    collections: "/api/v1/collections",
    jobs: "/api/v1/jobs",
    localLibraryUpload: "/api/v1/ingest/local-library-upload",
    localLibraryIngest: "/api/v1/ingest/local-library",
    search: "/api/v1/search/papers",
    searchChunks: "/api/v1/search/chunks",
    searchArtifacts: "/api/v1/search/artifacts",
    reindex: "/api/v1/search/reindex",
    compareFigures: "/api/v1/compare/figures",
    compareTables: "/api/v1/compare/tables",
  };

  const state = {
    workspaces: [],
    selectedWorkspace: null,
    collections: [],
    selectedCollection: null,
    papers: [],
    searchResults: [],
    selectedPaper: null,
    selectedPaperStructured: null,
    collectionSummary: null,
    collectionFigures: [],
    collectionTables: [],
    chunkSearchHits: [],
    artifactSearchHits: [],
    jobs: [],
    pollHandle: null,
  };

  const elements = {
    workspacesList: document.getElementById("workspaces-list"),
    collectionsList: document.getElementById("collections-list"),
    paperList: document.getElementById("paper-list"),
    workspaceTitleInput: document.getElementById("workspace-title-input"),
    workspaceFocusInput: document.getElementById("workspace-focus-input"),
    workspaceMeta: document.getElementById("workspace-meta"),
    collectionSummary: document.getElementById("collection-summary"),
    paperDetail: document.getElementById("paper-detail"),
    artifactSurface: document.getElementById("artifact-surface"),
    chunkSearchSurface: document.getElementById("chunk-search-surface"),
    artifactSearchSurface: document.getElementById("artifact-search-surface"),
    jobsList: document.getElementById("jobs-list"),
    searchForm: document.getElementById("search-form"),
    searchInput: document.getElementById("paper-search-input"),
    localLibraryUploadForm: document.getElementById("local-library-upload-form"),
    localLibraryUploadInput: document.getElementById("local-library-upload-input"),
    localLibraryUploadTitleInput: document.getElementById("local-library-upload-title-input"),
    localLibraryUploadDescriptionInput: document.getElementById("local-library-upload-description-input"),
    localLibraryUploadButton: document.getElementById("local-library-upload-button"),
    localLibraryForm: document.getElementById("local-library-form"),
    localLibrarySourceInput: document.getElementById("local-library-source-input"),
    localLibraryTitleInput: document.getElementById("local-library-title-input"),
    localLibraryDescriptionInput: document.getElementById("local-library-description-input"),
    localLibraryImportButton: document.getElementById("local-library-import-button"),
    saveWorkspaceButton: document.getElementById("save-workspace-button"),
    extractButton: document.getElementById("extract-button"),
    parseButton: document.getElementById("parse-button"),
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

  async function loadWorkspaces(preferredWorkspaceId) {
    const payload = await fetchJson(endpoints.workspaces);
    state.workspaces = payload.data || [];

    const workspaceId = preferredWorkspaceId
      || (state.selectedWorkspace && state.selectedWorkspace.id)
      || (state.workspaces[0] && state.workspaces[0].id);

    renderWorkspaces();
    if (workspaceId) {
      await loadWorkspace(workspaceId);
      return;
    }
    renderWorkspaceDetail();
  }

  async function loadWorkspace(workspaceId) {
    const payload = await fetchJson(`${endpoints.workspaces}/${workspaceId}`);
    state.selectedWorkspace = payload.data;
    renderWorkspaces();
    renderWorkspaceDetail();

    if (state.selectedWorkspace.collection_id) {
      await loadCollectionSurface(state.selectedWorkspace.collection_id);
    }

    elements.searchInput.value = state.selectedWorkspace.saved_query || "";
    if (state.selectedWorkspace.saved_query) {
      await searchPapers(state.selectedWorkspace.saved_query);
    } else {
      state.searchResults = [];
      state.chunkSearchHits = [];
      state.artifactSearchHits = [];
      renderPapers();
      renderSearchSurfaces();
    }

    if (state.selectedWorkspace.pinned_papers && state.selectedWorkspace.pinned_papers.length > 0) {
      await loadPaperDetail(state.selectedWorkspace.pinned_papers[0].id);
    }

    renderWorkspaceDetail();
  }

  async function loadCollectionSurface(collectionId) {
    const [collectionPayload, papersPayload, summaryPayload, figuresPayload, tablesPayload] = await Promise.all([
      fetchJson(`/api/v1/collections/${collectionId}`),
      fetchJson(`/api/v1/collections/${collectionId}/papers`),
      fetchJson(`/api/v1/collections/${collectionId}/structured-summary`),
      fetchJson(endpoints.compareFigures, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ collection_id: collectionId, limit: 8 }),
      }),
      fetchJson(endpoints.compareTables, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ collection_id: collectionId, limit: 8 }),
      }),
    ]);

    state.selectedCollection = collectionPayload.data;
    state.papers = papersPayload.data || [];
    state.searchResults = [];
    state.collectionSummary = summaryPayload.data;
    state.collectionFigures = figuresPayload.data || [];
    state.collectionTables = tablesPayload.data || [];
    state.selectedPaper = null;
    state.selectedPaperStructured = null;

    renderCollections();
    renderPapers();
    renderCollectionSummary();
    renderPaperDetail();
    renderArtifactSurface();
    updateActionButtons();
  }

  async function searchPapers(query) {
    if (!query.trim()) {
      state.searchResults = [];
      state.chunkSearchHits = [];
      state.artifactSearchHits = [];
      renderPapers();
      renderSearchSurfaces();
      return;
    }

    const params = new URLSearchParams({ q: query.trim() });
    if (state.selectedCollection) {
      params.set("collection_id", state.selectedCollection.id);
    }
    const [paperPayload, chunkPayload, artifactPayload] = await Promise.all([
      fetchJson(`${endpoints.search}?${params.toString()}`),
      fetchJson(`${endpoints.searchChunks}?${params.toString()}`),
      fetchJson(`${endpoints.searchArtifacts}?${params.toString()}&kind=all`),
    ]);
    state.searchResults = paperPayload.data || [];
    state.chunkSearchHits = chunkPayload.data || [];
    state.artifactSearchHits = artifactPayload.data || [];
    renderPapers();
    renderSearchSurfaces();
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
    const previouslyHadActiveJobs = state.jobs.some((job) => job.status === "pending" || job.status === "running");
    const payload = await fetchJson(`${endpoints.jobs}?limit=20`);
    state.jobs = payload.data || [];
    const hasActiveJobs = state.jobs.some((job) => job.status === "pending" || job.status === "running");
    renderJobs();
    updatePolling();
    if (previouslyHadActiveJobs && !hasActiveJobs) {
      await refreshAfterJobCompletion();
    }
  }

  async function queueReindex() {
    const payload = await fetchJson(endpoints.reindex, { method: "POST" });
    state.jobs.unshift(payload.data);
    renderJobs();
    updatePolling();
    setStatus("Queued a corpus reindex job.");
  }

  async function queueLocalLibraryIngest() {
    const sourceDir = elements.localLibrarySourceInput.value.trim();
    if (!sourceDir) {
      throw new Error("A local source directory is required.");
    }

    const collectionTitle = elements.localLibraryUploadTitleInput.value.trim();
    const collectionDescription = elements.localLibraryUploadDescriptionInput.value.trim();
    const payload = await fetchJson(endpoints.localLibraryIngest, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_dir: sourceDir,
        collection_title: collectionTitle || null,
        collection_description: collectionDescription || null,
      }),
    });

    state.jobs.unshift(payload.data);
    renderJobs();
    updatePolling();
    setStatus(`Queued local import from ${sourceDir}.`);
  }

  async function queueLocalLibraryUploadIngest() {
    const files = Array.from(elements.localLibraryUploadInput.files || []);
    if (files.length === 0) {
      throw new Error("Select at least one PDF file or folder to upload.");
    }

    const collectionTitle = elements.localLibraryTitleInput.value.trim();
    const collectionDescription = elements.localLibraryDescriptionInput.value.trim();
    const formData = new FormData();
    for (const file of files) {
      const relativePath = file.webkitRelativePath || file.name;
      formData.append("files", file, relativePath);
    }
    if (collectionTitle) {
      formData.append("collection_title", collectionTitle);
    }
    if (collectionDescription) {
      formData.append("collection_description", collectionDescription);
    }

    const response = await fetch(endpoints.localLibraryUpload, {
      method: "POST",
      body: formData,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.message || `Request failed: ${response.status}`);
    }

    state.jobs.unshift(payload.data);
    renderJobs();
    updatePolling();
    setStatus(`Queued upload import for ${files.length} file(s).`);
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

  async function queueParse() {
    if (!state.selectedCollection) {
      return;
    }

    const payload = await fetchJson(
      `/api/v1/collections/${state.selectedCollection.id}/parse`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      },
    );
    state.jobs.unshift(payload.data);
    renderJobs();
    updatePolling();
    setStatus(`Queued parse for ${state.selectedCollection.title}.`);
  }

  async function saveWorkspace() {
    const title = elements.workspaceTitleInput.value.trim()
      || (state.selectedCollection ? `${state.selectedCollection.title} workspace` : "Arxie workspace");
    const body = {
      title,
      description: state.selectedWorkspace ? state.selectedWorkspace.description : null,
      collection_id: state.selectedCollection ? state.selectedCollection.id : null,
      saved_query: elements.searchInput.value.trim() || null,
      focus_note: elements.workspaceFocusInput.value.trim() || null,
      active_filters: state.selectedWorkspace ? (state.selectedWorkspace.active_filters || {}) : {},
      pinned_paper_ids: state.selectedPaper ? [state.selectedPaper.id] : [],
    };

    if (state.selectedWorkspace) {
      await fetchJson(`${endpoints.workspaces}/${state.selectedWorkspace.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      await loadWorkspaces(state.selectedWorkspace.id);
      setStatus(`Updated workspace ${title}.`);
      return;
    }

    const payload = await fetchJson(endpoints.workspaces, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    await loadWorkspaces(payload.data.id);
    setStatus(`Saved workspace ${title}.`);
  }

  function resolvePreferredCollectionIdFromJobs() {
    for (const job of state.jobs) {
      if (job.status !== "completed") {
        continue;
      }
      if (job.job_type === "local_library_ingest") {
        if (job.result && job.result.collection_id) {
          return job.result.collection_id;
        }
        if (job.payload && job.payload.collection_title) {
          const matchingCollection = state.collections.find((collection) => collection.title === job.payload.collection_title);
          if (matchingCollection) {
            return matchingCollection.id;
          }
        }
      }
      if ((job.job_type === "collection_parse" || job.job_type === "collection_extract") && job.result && job.result.collection_id) {
        return job.result.collection_id;
      }
    }
    return state.selectedCollection ? state.selectedCollection.id : null;
  }

  async function refreshAfterJobCompletion() {
    const preferredCollectionId = resolvePreferredCollectionIdFromJobs();
    const preferredWorkspaceId = state.selectedWorkspace ? state.selectedWorkspace.id : null;

    await loadCollections();
    if (
      preferredCollectionId
      && (!state.selectedCollection || state.selectedCollection.id !== preferredCollectionId)
    ) {
      const matchingCollection = state.collections.find((collection) => collection.id === preferredCollectionId);
      if (matchingCollection) {
        await loadCollectionSurface(preferredCollectionId);
      }
    }
    await loadWorkspaces(preferredWorkspaceId);
  }

  function updateActionButtons() {
    elements.extractButton.disabled = !state.selectedCollection;
    elements.parseButton.disabled = !state.selectedCollection;
    elements.localLibraryImportButton.disabled = false;
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

  function renderWorkspaces() {
    if (state.workspaces.length === 0) {
      elements.workspacesList.innerHTML = '<div class="list-card muted">No saved workspaces yet. Search a collection and save the current context.</div>';
      return;
    }

    elements.workspacesList.innerHTML = state.workspaces
      .map((workspace) => {
        const active = state.selectedWorkspace && state.selectedWorkspace.id === workspace.id;
        return `
          <div class="list-card" data-active="${active ? "true" : "false"}">
            <button type="button" data-workspace-id="${workspace.id}">
              <div class="list-title">${escapeHtml(workspace.title)}</div>
              <div class="muted">${escapeHtml(workspace.saved_query || workspace.description || "Saved research context")}</div>
            </button>
          </div>
        `;
      })
      .join("");

    elements.workspacesList.querySelectorAll("[data-workspace-id]").forEach((button) => {
      button.addEventListener("click", async () => {
        const workspaceId = button.getAttribute("data-workspace-id");
        if (!workspaceId) {
          return;
        }
        setStatus("Loading workspace…");
        await loadWorkspace(workspaceId);
        setStatus(`Loaded ${state.selectedWorkspace.title}.`);
      });
    });
  }

  function renderWorkspaceDetail() {
    if (!elements.workspaceTitleInput || !elements.workspaceFocusInput || !elements.workspaceMeta) {
      return;
    }

    if (state.selectedWorkspace) {
      elements.workspaceTitleInput.value = state.selectedWorkspace.title || "";
      elements.workspaceFocusInput.value = state.selectedWorkspace.focus_note || "";
      elements.workspaceMeta.innerHTML = `
        <div class="detail-group">
          <strong>Collection</strong>
          <div class="muted">${escapeHtml((state.selectedWorkspace.collection && state.selectedWorkspace.collection.title) || (state.selectedCollection && state.selectedCollection.title) || "No collection linked")}</div>
        </div>
        <div class="detail-group">
          <strong>Saved query</strong>
          <div class="muted">${escapeHtml(state.selectedWorkspace.saved_query || "No saved query yet")}</div>
        </div>
        <div class="detail-group">
          <strong>Pinned papers</strong>
          <div class="pill-row">
            ${(state.selectedWorkspace.pinned_papers || []).map((paper) => `<span class="pill">${escapeHtml(paper.title)}</span>`).join("") || '<span class="muted">No pinned papers yet.</span>'}
          </div>
        </div>
      `;
      return;
    }

    elements.workspaceTitleInput.value = state.selectedCollection ? `${state.selectedCollection.title} workspace` : "";
    elements.workspaceFocusInput.value = "";
    elements.workspaceMeta.innerHTML = `
      <div class="detail-group">
        <strong>Collection</strong>
        <div class="muted">${escapeHtml((state.selectedCollection && state.selectedCollection.title) || "Choose a collection to start a workspace")}</div>
      </div>
      <div class="detail-group">
        <strong>Saved query</strong>
        <div class="muted">Run a search, open a paper, then save the current research context.</div>
      </div>
    `;
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
      { label: "Limitations", value: summary.limitations.length },
      { label: "Figures", value: summary.figures.length },
      { label: "Tables", value: summary.tables.length },
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
      <div class="detail-group">
        <strong>Limitations</strong>
        <ul class="detail-list">
          ${summary.limitations
            .slice(0, 4)
            .map((item) => `<li>${escapeHtml(item.statement)}</li>`)
            .join("") || "<li>No extracted limitations yet.</li>"}
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
          <span class="pill">Limitations ${structured.limitations.length}</span>
          <span class="pill">Tricks ${structured.engineering_tricks.length}</span>
          <span class="pill">Figures ${structured.figures.length}</span>
          <span class="pill">Tables ${structured.tables.length}</span>
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
      <div class="detail-group">
        <strong>Limitations</strong>
        <ul class="detail-list">
          ${structured.limitations
            .slice(0, 4)
            .map((item) => `<li>${escapeHtml(item.statement)}</li>`)
            .join("") || "<li>No extracted limitations.</li>"}
        </ul>
      </div>
      <div class="detail-group">
        <strong>Figures</strong>
        <ul class="detail-list">
          ${structured.figures
            .slice(0, 4)
            .map((item) => `<li>${escapeHtml(item.figure_label || "Figure")} — ${escapeHtml(item.caption || "No caption")}</li>`)
            .join("") || "<li>No figure artifacts.</li>"}
        </ul>
      </div>
      <div class="detail-group">
        <strong>Tables</strong>
        <ul class="detail-list">
          ${structured.tables
            .slice(0, 4)
            .map((item) => `<li>${escapeHtml(item.table_label || "Table")} — ${escapeHtml(item.caption || "No caption")}</li>`)
            .join("") || "<li>No table artifacts.</li>"}
        </ul>
      </div>
    `;
  }

  function renderArtifactSurface() {
    if (!state.selectedCollection) {
      elements.artifactSurface.innerHTML = '<div class="summary-card muted">Select a collection to inspect figures and tables.</div>';
      return;
    }

    elements.artifactSurface.innerHTML = `
      <div class="summary-card">
        <h3>Figures</h3>
        <ul class="detail-list">
          ${state.collectionFigures
            .slice(0, 6)
            .map((item) => `<li>${escapeHtml(item.paper_title)} — ${escapeHtml(item.figure_label || "Figure")} / ${escapeHtml(item.caption || "No caption")}</li>`)
            .join("") || "<li>No figures in this slice.</li>"}
        </ul>
      </div>
      <div class="summary-card">
        <h3>Tables</h3>
        <ul class="detail-list">
          ${state.collectionTables
            .slice(0, 6)
            .map((item) => `<li>${escapeHtml(item.paper_title)} — ${escapeHtml(item.table_label || "Table")} / ${escapeHtml(item.caption || "No caption")}</li>`)
            .join("") || "<li>No tables in this slice.</li>"}
        </ul>
      </div>
    `;
  }

  function renderSearchSurfaces() {
    elements.chunkSearchSurface.innerHTML = state.chunkSearchHits.length > 0
      ? `
        <ul class="detail-list">
          ${state.chunkSearchHits
            .slice(0, 6)
            .map((item) => `<li>${escapeHtml(item.paper_title)}${item.section_title ? ` / ${escapeHtml(item.section_title)}` : ""} — ${escapeHtml(item.text)}</li>`)
            .join("")}
        </ul>
      `
      : '<div class="muted">Run a paper search to inspect chunk-level hits.</div>';

    elements.artifactSearchSurface.innerHTML = state.artifactSearchHits.length > 0
      ? `
        <ul class="detail-list">
          ${state.artifactSearchHits
            .slice(0, 6)
            .map((item) => `<li>${escapeHtml(item.paper_title)} — ${escapeHtml(item.artifact_type)} / ${escapeHtml(item.label || "Untitled")} / ${escapeHtml(item.caption || "No caption")}</li>`)
            .join("")}
        </ul>
      `
      : '<div class="muted">Run a paper search to inspect figure and table hits.</div>';
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
    elements.artifactSurface.innerHTML = '<div class="summary-card muted">Select a collection to inspect figures and tables.</div>';
    elements.chunkSearchSurface.innerHTML = '<div class="muted">Run a paper search to inspect chunk-level hits.</div>';
    elements.artifactSearchSurface.innerHTML = '<div class="muted">Run a paper search to inspect figure and table hits.</div>';
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
      setStatus("Loading Arxie workspace…");
      await loadCollections();
      await loadWorkspaces();
      await loadJobs();
      renderPaperDetail();
      renderWorkspaceDetail();
      updateActionButtons();
      setStatus("Arxie workspace ready.");
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

  elements.parseButton.addEventListener("click", async () => {
    try {
      await queueParse();
    } catch (error) {
      setStatus(error.message);
    }
  });

  elements.saveWorkspaceButton.addEventListener("click", async () => {
    try {
      await saveWorkspace();
    } catch (error) {
      setStatus(error.message);
    }
  });

  elements.localLibraryForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await queueLocalLibraryIngest();
    } catch (error) {
      setStatus(error.message);
    }
  });

  elements.localLibraryUploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await queueLocalLibraryUploadIngest();
    } catch (error) {
      setStatus(error.message);
    }
  });

  initialize();
})();
