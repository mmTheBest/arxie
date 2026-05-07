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
    searchStatus: "/api/v1/search/status",
    readiness: "/readyz",
    reindex: "/api/v1/search/reindex",
    compareResults: "/api/v1/compare/results",
    compareMethods: "/api/v1/compare/methods",
    compareEngineeringTricks: "/api/v1/compare/engineering-tricks",
    compareFigures: "/api/v1/compare/figures",
    compareTables: "/api/v1/compare/tables",
  };

  const state = {
    activeView: "workspace",
    workspaces: [],
    selectedWorkspace: null,
    collections: [],
    selectedCollection: null,
    papers: [],
    selectedPaperIds: new Set(),
    searchResults: [],
    selectedPaper: null,
    selectedPaperStructured: null,
    collectionSummary: null,
    chunkSearchHits: [],
    artifactSearchHits: [],
    jobs: [],
    system: {
      readiness: null,
      searchStatus: null,
    },
    compare: {
      collectionId: null,
      dataset: "",
      metric: "",
      method: "",
      results: [],
      methods: [],
      engineeringTricks: [],
      figures: [],
      tables: [],
    },
    pollHandle: null,
  };

  const elements = {
    appShell: document.getElementById("app-shell"),
    navTabs: Array.from(document.querySelectorAll("#app-nav [data-view]")),
    sidebarCollectionsList: document.getElementById("sidebar-collections-list"),
    sidebarPapersList: document.getElementById("sidebar-papers-list"),
    sidebarWorkspacesList: document.getElementById("sidebar-workspaces-list"),
    statusBanner: document.getElementById("status-banner"),
    librarySelectedCollectionMeta: document.getElementById("library-selected-collection-meta"),
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
    parseButton: document.getElementById("parse-button"),
    parseSelectedPapersButton: document.getElementById("parse-selected-papers-button"),
    extractButton: document.getElementById("extract-button"),
    libraryJobLog: document.getElementById("library-job-log"),
    workspaceCollectionTitle: document.getElementById("workspace-collection-title"),
    workspaceSearchMeta: document.getElementById("workspace-search-meta"),
    workspaceReadinessBanner: document.getElementById("workspace-readiness-banner"),
    workspaceOpenCompareButton: document.getElementById("workspace-open-compare-button"),
    searchForm: document.getElementById("search-form"),
    searchInput: document.getElementById("paper-search-input"),
    paperList: document.getElementById("paper-list"),
    chunkSearchSurface: document.getElementById("chunk-search-surface"),
    artifactSearchSurface: document.getElementById("artifact-search-surface"),
    workspaceTitleInput: document.getElementById("workspace-title-input"),
    workspaceFocusInput: document.getElementById("workspace-focus-input"),
    workspaceMeta: document.getElementById("workspace-meta"),
    saveWorkspaceButton: document.getElementById("save-workspace-button"),
    collectionSummary: document.getElementById("collection-summary"),
    paperDetail: document.getElementById("paper-detail"),
    compareCollectionContext: document.getElementById("compare-collection-context"),
    compareReadinessBanner: document.getElementById("compare-readiness-banner"),
    compareDatasetInput: document.getElementById("compare-dataset-input"),
    compareMetricInput: document.getElementById("compare-metric-input"),
    compareMethodInput: document.getElementById("compare-method-input"),
    compareRefreshButton: document.getElementById("compare-refresh-button"),
    compareResultsSurface: document.getElementById("compare-results-surface"),
    compareMethodsSurface: document.getElementById("compare-methods-surface"),
    compareTricksSurface: document.getElementById("compare-tricks-surface"),
    compareFiguresSurface: document.getElementById("compare-figures-surface"),
    compareTablesSurface: document.getElementById("compare-tables-surface"),
    refreshJobsButton: document.getElementById("refresh-jobs-button"),
    reindexButton: document.getElementById("reindex-button"),
    jobsList: document.getElementById("jobs-list"),
    settingsSummary: document.getElementById("settings-summary"),
    settingsRuntimeSummary: document.getElementById("settings-runtime-summary"),
  };

  function setStatus(message) {
    elements.statusBanner.textContent = message;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatValue(value) {
    if (value === null || value === undefined || value === "") {
      return "n/a";
    }
    return String(value);
  }

  function uniqueStrings(values) {
    return Array.from(new Set(values.filter(Boolean)));
  }

  async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.message || `Request failed: ${response.status}`);
    }
    return payload;
  }

  function activateView(viewName) {
    state.activeView = viewName;
    if (elements.appShell) {
      elements.appShell.dataset.activeView = viewName;
    }
    document.querySelectorAll(".module-view").forEach((section) => {
      const isActive = section.getAttribute("data-view") === viewName;
      section.hidden = !isActive;
    });
    elements.navTabs.forEach((button) => {
      const isActive = button.getAttribute("data-view") === viewName;
      button.dataset.active = isActive ? "true" : "false";
      button.setAttribute("aria-current", isActive ? "page" : "false");
    });
  }

  function membershipPaperList() {
    return state.papers.map((membership) => membership.paper);
  }

  function visiblePaperList() {
    return state.searchResults.length > 0 ? state.searchResults : membershipPaperList();
  }

  function selectedCollectionPill() {
    if (!state.selectedCollection) {
      return '<span class="pill">No collection selected</span>';
    }
    return `<span class="pill">${escapeHtml(state.selectedCollection.title)}</span>`;
  }

  function pluralize(count, singular, plural) {
    return `${count} ${count === 1 ? singular : plural}`;
  }

  function isActiveJobStatus(status) {
    return status === "pending" || status === "queued" || status === "running";
  }

  function collectionIdFromJob(job) {
    const result = job.result || {};
    const payload = job.payload || {};
    return result.collection_id || payload.collection_id || null;
  }

  function latestCollectionJobStatus(collection, jobs, jobType) {
    if (!collection) {
      return null;
    }
    const match = jobs.find((job) => collectionIdFromJob(job) === collection.id && job.job_type === jobType);
    return match ? match.status : null;
  }

  function collectionFailedJobCount(collection, jobs) {
    if (!collection) {
      return 0;
    }
    return jobs.filter((job) => collectionIdFromJob(job) === collection.id && job.status === "failed").length;
  }

  function getCollectionReadiness(collection, summary, jobs) {
    const paperCount = Number((summary && summary.paper_count) ?? (collection && collection.paper_count) ?? 0);
    const parsedPaperCount = Number(
      (summary && summary.parsed_paper_count) ?? (collection && collection.parsed_paper_count) ?? 0,
    );
    const extractedPaperCount = Number(
      (summary && summary.extracted_paper_count) ?? (collection && collection.extracted_paper_count) ?? 0,
    );
    const latestParseJobStatus = latestCollectionJobStatus(collection, jobs, "collection_parse")
      || (summary && summary.latest_parse_job_status)
      || (collection && collection.latest_parse_job_status);
    const latestExtractionJobStatus = latestCollectionJobStatus(collection, jobs, "collection_extract")
      || (summary && summary.latest_extraction_job_status)
      || (collection && collection.latest_extraction_job_status);
    const latestJobFromJobs = collection ? jobs.find((job) => collectionIdFromJob(job) === collection.id) : null;
    const latestJobStatus = (latestJobFromJobs || {}).status
      || (summary && summary.latest_job_status)
      || (collection && collection.latest_job_status)
      || null;
    const failedJobCount = Math.max(
      Number((summary && summary.failed_job_count) ?? 0),
      Number((collection && collection.failed_job_count) ?? 0),
      collectionFailedJobCount(collection, jobs),
    );

    const readiness = {
      status: "imported",
      label: "Imported",
      detailLabel: "Needs parse",
      nextAction: "parse",
      nextActionLabel: "Run Next Step",
      nextActionDisabled: paperCount === 0,
      isReadyForWorkspace: paperCount > 0 && parsedPaperCount >= paperCount,
      isReadyForCompare: paperCount > 0 && extractedPaperCount >= paperCount,
      paperCount,
      parsedPaperCount,
      extractedPaperCount,
      latestJobStatus,
      latestParseJobStatus,
      latestExtractionJobStatus,
      failedJobCount,
      sidebarLine: `${pluralize(paperCount, "paper", "papers")} · Imported · Needs parse`,
    };

    if (failedJobCount > 0 || latestParseJobStatus === "failed" || latestExtractionJobStatus === "failed") {
      return {
        ...readiness,
        status: "needs_attention",
        label: "Needs attention",
        detailLabel: "Failed",
        nextAction: parsedPaperCount < paperCount ? "parse" : "extract",
        sidebarLine: `${pluralize(paperCount, "paper", "papers")} · Needs attention`,
      };
    }

    if (isActiveJobStatus(latestParseJobStatus) && parsedPaperCount < paperCount) {
      return {
        ...readiness,
        status: "parsing",
        label: "Parsing",
        detailLabel: "Text pending",
        nextAction: "none",
        nextActionDisabled: true,
        sidebarLine: `${pluralize(paperCount, "paper", "papers")} · Parsing · Text pending`,
      };
    }

    if (isActiveJobStatus(latestExtractionJobStatus) && extractedPaperCount < paperCount) {
      return {
        ...readiness,
        status: "extracting",
        label: "Extracting",
        detailLabel: "Evidence pending",
        nextAction: "none",
        nextActionDisabled: true,
        sidebarLine: `${pluralize(paperCount, "paper", "papers")} · Extracting · Evidence pending`,
      };
    }

    if (paperCount === 0) {
      return {
        ...readiness,
        nextAction: "none",
        nextActionDisabled: true,
        detailLabel: "No papers yet",
        sidebarLine: "0 papers · Imported",
      };
    }

    if (parsedPaperCount < paperCount) {
      return readiness;
    }

    if (extractedPaperCount < paperCount) {
      return {
        ...readiness,
        status: "text_ready",
        label: "Text ready",
        detailLabel: "Evidence missing",
        nextAction: "extract",
        sidebarLine: `${pluralize(paperCount, "paper", "papers")} · Text ready · Evidence missing`,
      };
    }

    return {
      ...readiness,
      status: "evidence_ready",
      label: "Evidence ready",
      detailLabel: "Ready",
      nextAction: "none",
      nextActionLabel: "Ready",
      nextActionDisabled: true,
      sidebarLine: `${pluralize(paperCount, "paper", "papers")} · Evidence ready`,
    };
  }

  function selectedCollectionReadiness() {
    return getCollectionReadiness(state.selectedCollection, state.collectionSummary, state.jobs);
  }

  function paperJobApplies(job, paperId) {
    if (!state.selectedCollection || collectionIdFromJob(job) !== state.selectedCollection.id) {
      return false;
    }
    const payload = job.payload || {};
    const paperIds = payload.paper_ids;
    return !Array.isArray(paperIds) || paperIds.length === 0 || paperIds.includes(paperId);
  }

  function latestPaperJob(paperId, jobType) {
    return state.jobs.find((job) => job.job_type === jobType && paperJobApplies(job, paperId)) || null;
  }

  function isStaleJob(job) {
    if (!job || job.status !== "running" || !job.started_at) {
      return false;
    }
    const startedAt = Date.parse(job.started_at);
    if (Number.isNaN(startedAt)) {
      return false;
    }
    return Date.now() - startedAt > 15 * 60 * 1000;
  }

  function paperProcessingState(membership) {
    const paper = membership.paper;
    const latestParseJob = latestPaperJob(paper.id, "collection_parse");
    const latestExtractionJob = latestPaperJob(paper.id, "collection_extract");
    const parseActive = latestParseJob && isActiveJobStatus(latestParseJob.status) && !membership.is_parsed;
    const extractionActive = latestExtractionJob && isActiveJobStatus(latestExtractionJob.status) && membership.is_parsed && !membership.is_extracted;

    if (parseActive) {
      return {
        status: isStaleJob(latestParseJob) ? "needs_attention" : "parsing",
        label: isStaleJob(latestParseJob) ? "Stale parse" : "Parsing",
        detail: isStaleJob(latestParseJob) ? "Started over 15 minutes ago" : "Text extraction running",
        action: "none",
        actionLabel: "Running",
        actionDisabled: true,
      };
    }
    if (extractionActive) {
      return {
        status: isStaleJob(latestExtractionJob) ? "needs_attention" : "extracting",
        label: isStaleJob(latestExtractionJob) ? "Stale extraction" : "Extracting",
        detail: isStaleJob(latestExtractionJob) ? "Started over 15 minutes ago" : "Structured evidence running",
        action: "none",
        actionLabel: "Running",
        actionDisabled: true,
      };
    }
    if (membership.latest_job_error) {
      return {
        status: "needs_attention",
        label: "Needs attention",
        detail: membership.latest_job_error,
        action: membership.is_parsed ? "extract" : "parse",
        actionLabel: membership.is_parsed ? "Extract" : "Parse",
        actionDisabled: false,
      };
    }
    if (!membership.is_parsed) {
      return {
        status: "imported",
        label: "Needs parse",
        detail: "No full text yet",
        action: "parse",
        actionLabel: "Parse",
        actionDisabled: false,
      };
    }
    if (!membership.is_extracted) {
      return {
        status: "text_ready",
        label: "Text ready",
        detail: "Evidence missing",
        action: "extract",
        actionLabel: "Extract",
        actionDisabled: false,
      };
    }
    return {
      status: "evidence_ready",
      label: "Evidence ready",
      detail: "Ready",
      action: "none",
      actionLabel: "Ready",
      actionDisabled: true,
    };
  }

  function unprocessedPaperIds() {
    return state.papers
      .filter((membership) => !membership.is_parsed)
      .map((membership) => membership.paper.id);
  }

  function selectedPaperIdsForAction(action) {
    return state.papers
      .filter((membership) => state.selectedPaperIds.has(membership.paper.id))
      .filter((membership) => {
        if (action === "parse") {
          return !membership.is_parsed;
        }
        if (action === "extract") {
          return membership.is_parsed && !membership.is_extracted;
        }
        return true;
      })
      .map((membership) => membership.paper.id);
  }

  function renderCollectionsSidebar() {
    if (state.collections.length === 0) {
      elements.sidebarCollectionsList.innerHTML = '<div class="list-card muted">No collections yet.</div>';
      return;
    }

    elements.sidebarCollectionsList.innerHTML = state.collections
      .map((collection) => {
        const active = state.selectedCollection && state.selectedCollection.id === collection.id;
        const readiness = getCollectionReadiness(collection, null, state.jobs);
        return `
          <div class="list-card list-card-compact" data-active="${active ? "true" : "false"}">
            <button type="button" data-sidebar-collection-id="${collection.id}">
              <div class="list-card-heading">
                <span class="status-marker" data-status="${escapeHtml(readiness.status)}"></span>
                <span class="list-title">${escapeHtml(collection.title)}</span>
              </div>
              <div class="muted">${escapeHtml(readiness.sidebarLine)}</div>
            </button>
          </div>
        `;
      })
      .join("");

    elements.sidebarCollectionsList.querySelectorAll("[data-sidebar-collection-id]").forEach((button) => {
      button.addEventListener("click", async () => {
        const collectionId = button.getAttribute("data-sidebar-collection-id");
        if (!collectionId) {
          return;
        }
        setStatus("Loading collection…");
        await loadCollectionSurface(collectionId);
        if (state.activeView !== "library") {
          activateView("workspace");
        }
        setStatus(`Loaded ${state.selectedCollection.title}.`);
      });
    });
  }

  function renderSidebarPapers() {
    if (!elements.sidebarPapersList) {
      return;
    }
    if (!state.selectedCollection) {
      elements.sidebarPapersList.innerHTML = '<div class="list-card muted">Select a collection.</div>';
      return;
    }
    if (state.papers.length === 0) {
      elements.sidebarPapersList.innerHTML = '<div class="list-card muted">No papers in this collection.</div>';
      return;
    }

    elements.sidebarPapersList.innerHTML = state.papers
      .map((membership) => {
        const paper = membership.paper;
        const processing = paperProcessingState(membership);
        const checked = state.selectedPaperIds.has(paper.id);
        return `
          <div class="paper-process-row" data-status="${escapeHtml(processing.status)}">
            <label class="paper-select-control">
              <input
                type="checkbox"
                data-sidebar-paper-checkbox="${paper.id}"
                ${checked ? "checked" : ""}
              />
              <span class="status-marker" data-status="${escapeHtml(processing.status)}"></span>
              <span class="paper-row-copy">
                <span class="list-title">${escapeHtml(paper.title)}</span>
                <span class="muted">${escapeHtml(processing.label)} · ${escapeHtml(processing.detail)}</span>
              </span>
            </label>
            <button
              type="button"
              class="action-button paper-row-action"
              data-sidebar-paper-action="${escapeHtml(processing.action)}"
              data-sidebar-paper-id="${paper.id}"
              ${processing.actionDisabled ? "disabled" : ""}
            >
              ${escapeHtml(processing.actionLabel)}
            </button>
          </div>
        `;
      })
      .join("");

    elements.sidebarPapersList.querySelectorAll("[data-sidebar-paper-checkbox]").forEach((checkbox) => {
      checkbox.addEventListener("change", () => {
        const paperId = checkbox.getAttribute("data-sidebar-paper-checkbox");
        if (!paperId) {
          return;
        }
        if (checkbox.checked) {
          state.selectedPaperIds.add(paperId);
        } else {
          state.selectedPaperIds.delete(paperId);
        }
        updateActionButtons();
      });
    });

    elements.sidebarPapersList.querySelectorAll("[data-sidebar-paper-action]").forEach((button) => {
      button.addEventListener("click", async () => {
        try {
          const paperId = button.getAttribute("data-sidebar-paper-id");
          const action = button.getAttribute("data-sidebar-paper-action");
          if (!paperId) {
            return;
          }
          if (action === "parse") {
            await queueParse([paperId]);
          }
          if (action === "extract") {
            await queueExtraction([paperId]);
          }
        } catch (error) {
          setStatus(error.message);
        }
      });
    });
  }

  function renderWorkspacesSidebar() {
    if (state.workspaces.length === 0) {
      elements.sidebarWorkspacesList.innerHTML = '<div class="list-card muted">No saved workspaces yet.</div>';
      return;
    }

    elements.sidebarWorkspacesList.innerHTML = state.workspaces
      .map((workspace) => {
        const active = state.selectedWorkspace && state.selectedWorkspace.id === workspace.id;
        const collection = state.collections.find((item) => item.id === workspace.collection_id);
        const readiness = getCollectionReadiness(collection, null, state.jobs);
        const pinnedCount = (workspace.pinned_paper_ids || []).length;
        const collectionTitle = collection ? collection.title : "No linked collection";
        const queryIndicator = workspace.saved_query ? "saved query" : "no saved query";
        return `
          <div class="list-card list-card-compact" data-active="${active ? "true" : "false"}">
            <button type="button" data-sidebar-workspace-id="${workspace.id}">
              <div class="list-title">${escapeHtml(workspace.title)}</div>
              <div class="muted">${escapeHtml(collectionTitle)} · ${escapeHtml(pluralize(pinnedCount, "pinned", "pinned"))} · ${escapeHtml(queryIndicator)} · ${escapeHtml(readiness.label.toLowerCase())}</div>
            </button>
          </div>
        `;
      })
      .join("");

    elements.sidebarWorkspacesList.querySelectorAll("[data-sidebar-workspace-id]").forEach((button) => {
      button.addEventListener("click", async () => {
        const workspaceId = button.getAttribute("data-sidebar-workspace-id");
        if (!workspaceId) {
          return;
        }
        setStatus("Loading workspace…");
        await openWorkspace(workspaceId);
        activateView("workspace");
        setStatus(`Loaded ${state.selectedWorkspace.title}.`);
      });
    });
  }

  function renderLibraryLogs() {
    if (!elements.libraryJobLog) {
      return;
    }
    elements.librarySelectedCollectionMeta.innerHTML = selectedCollectionPill();
    if (!state.selectedCollection) {
      elements.libraryJobLog.innerHTML = '<div class="list-card muted">Select a collection to inspect Library activity.</div>';
      return;
    }

    const collectionJobs = state.jobs.filter((job) => collectionIdFromJob(job) === state.selectedCollection.id);
    if (collectionJobs.length === 0) {
      elements.libraryJobLog.innerHTML = '<div class="list-card muted">No jobs have run for this collection yet.</div>';
      return;
    }

    elements.libraryJobLog.innerHTML = collectionJobs.slice(0, 6).map((job) => {
      const stale = isStaleJob(job);
      const payload = job.payload || {};
      const selectedCount = Array.isArray(payload.paper_ids) ? payload.paper_ids.length : 0;
      const scope = selectedCount > 0 ? `${selectedCount} selected paper(s)` : "whole collection";
      const result = job.result || {};
      const skippedCount = Array.isArray(result.skipped_paper_ids) ? result.skipped_paper_ids.length : 0;
      return `
        <div class="library-log-entry" data-status="${escapeHtml(stale ? "needs_attention" : job.status)}">
          <div>
            <span class="job-status" data-status="${escapeHtml(stale ? "failed" : job.status)}">${escapeHtml(stale ? "Stale" : job.status)}</span>
            <span class="list-title">${escapeHtml(job.job_type)}</span>
          </div>
          <div class="job-meta">${escapeHtml(scope)} · queued ${escapeHtml(job.created_at || "unknown time")}</div>
          ${job.started_at ? `<div class="muted">Started ${escapeHtml(job.started_at)}</div>` : ""}
          ${job.finished_at ? `<div class="muted">Finished ${escapeHtml(job.finished_at)}</div>` : ""}
          ${stale ? '<div class="job-error">This job has been running for more than 15 minutes. The worker may have stopped or lost the job.</div>' : ""}
          ${job.error_message ? `<div class="job-error">${escapeHtml(job.error_message)}</div>` : ""}
          ${skippedCount > 0 ? `<div class="job-error">${escapeHtml(skippedCount)} paper(s) were skipped. Check failed job details before retrying.</div>` : ""}
        </div>
      `;
    }).join("");
  }

  function renderWorkspaceHeader() {
    if (!state.selectedCollection) {
      elements.workspaceCollectionTitle.textContent = "Search one collection and inspect evidence";
      elements.workspaceSearchMeta.textContent = "Choose a collection in Library to start a focused workspace.";
      return;
    }

    elements.workspaceCollectionTitle.textContent = `Collection: ${state.selectedCollection.title}`;
    const query = elements.searchInput.value.trim();
    elements.workspaceSearchMeta.textContent = query
      ? `Active query: ${query}`
      : "Run a search inside the active collection, then open a paper or save the context.";
  }

  function renderWorkspaceReadiness() {
    if (!elements.workspaceReadinessBanner) {
      return;
    }
    if (!state.selectedCollection) {
      elements.workspaceReadinessBanner.innerHTML = `
        <div class="readiness-banner-content" data-status="imported">
          <span class="status-marker" data-status="imported"></span>
          <span>Select a collection in Library to start a workspace.</span>
        </div>
      `;
      return;
    }

    const readiness = selectedCollectionReadiness();
    let message = `${readiness.label}: ${readiness.sidebarLine}.`;
    if (!readiness.isReadyForWorkspace) {
      message = "This collection needs parsing before full-text search works.";
    } else if (!readiness.isReadyForCompare) {
      message = "This collection needs extraction before structured comparison works.";
    }

    elements.workspaceReadinessBanner.innerHTML = `
      <div class="readiness-banner-content" data-status="${escapeHtml(readiness.status)}">
        <span class="status-marker" data-status="${escapeHtml(readiness.status)}"></span>
        <span>${escapeHtml(message)}</span>
      </div>
    `;

    elements.workspaceOpenCompareButton.dataset.ready = readiness.isReadyForCompare ? "true" : "false";
    elements.workspaceOpenCompareButton.classList.toggle("action-button-primary", readiness.isReadyForCompare);
  }

  function renderPapers() {
    const items = visiblePaperList();
    if (!state.selectedCollection) {
      elements.paperList.innerHTML = '<div class="list-card muted">Select a collection to begin.</div>';
      return;
    }
    if (items.length === 0) {
      elements.paperList.innerHTML = '<div class="list-card muted">No papers match the current slice.</div>';
      return;
    }

    elements.paperList.innerHTML = items
      .map((paper) => {
        const active = state.selectedPaper && state.selectedPaper.id === paper.id;
        const meta = [paper.publication_year, paper.venue].filter(Boolean).join(" / ");
        const authorLine = (paper.authors || []).slice(0, 3).join(", ");
        return `
          <div class="list-card" data-active="${active ? "true" : "false"}">
            <button type="button" data-paper-id="${paper.id}">
              <div class="list-title">${escapeHtml(paper.title)}</div>
              <div class="muted">${escapeHtml(meta || paper.provider || "Paper record")}</div>
              <div class="muted">${escapeHtml(authorLine || "Unknown authors")}</div>
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

  function renderWorkspaceDetail() {
    elements.workspaceTitleInput.value = state.selectedWorkspace
      ? state.selectedWorkspace.title || ""
      : (state.selectedCollection ? `${state.selectedCollection.title} workspace` : "");
    elements.workspaceFocusInput.value = state.selectedWorkspace ? (state.selectedWorkspace.focus_note || "") : "";
    elements.saveWorkspaceButton.textContent = state.selectedWorkspace ? "Update Workspace" : "Save Workspace";

    if (!state.selectedCollection && !state.selectedWorkspace) {
      elements.workspaceMeta.innerHTML = '<div class="muted">Choose a collection, run a search, then save the research context.</div>';
      return;
    }

    const pinnedTitles = state.selectedWorkspace && state.selectedWorkspace.pinned_papers
      ? state.selectedWorkspace.pinned_papers.map((paper) => `<span class="pill">${escapeHtml(paper.title)}</span>`).join("")
      : "";

    elements.workspaceMeta.innerHTML = `
      <div class="detail-group">
        <strong>Linked collection</strong>
        <div class="muted">${escapeHtml((state.selectedWorkspace && state.selectedWorkspace.collection && state.selectedWorkspace.collection.title) || (state.selectedCollection && state.selectedCollection.title) || "None")}</div>
      </div>
      <div class="detail-group">
        <strong>Saved query</strong>
        <div class="muted">${escapeHtml((state.selectedWorkspace && state.selectedWorkspace.saved_query) || elements.searchInput.value.trim() || "No saved query yet")}</div>
      </div>
      <div class="detail-group">
        <strong>Pinned papers</strong>
        <div class="pill-row">
          ${pinnedTitles || '<span class="muted">Open a paper to pin it on save.</span>'}
        </div>
      </div>
    `;
  }

  function renderCollectionSummary() {
    if (!state.selectedCollection || !state.collectionSummary) {
      elements.collectionSummary.innerHTML = '<div class="muted">No collection summary yet.</div>';
      return;
    }

    const summary = state.collectionSummary;
    const readiness = selectedCollectionReadiness();
    const metricBoxes = [
      { label: "Papers", value: summary.paper_count },
      { label: "Parsed", value: summary.parsed_paper_count },
      { label: "Extracted", value: summary.extracted_paper_count },
      { label: "Methods", value: summary.methods.length },
      { label: "Figures", value: summary.figures.length },
      { label: "Tables", value: summary.tables.length },
    ];

    elements.collectionSummary.innerHTML = `
      <div class="metric-grid">
        ${metricBoxes.map((item) => `
          <div class="metric-box">
            <span class="metric-value">${escapeHtml(item.value)}</span>
            <div class="muted">${escapeHtml(item.label)}</div>
          </div>
        `).join("")}
      </div>
      <div class="detail-group">
        <div class="list-title">${escapeHtml(state.selectedCollection.title)}</div>
        <p class="summary-copy">${escapeHtml(state.selectedCollection.description || "Curated local collection")}</p>
        <div class="pill-row">
          <span class="pill">${escapeHtml(readiness.label)}</span>
          ${readiness.latestParseJobStatus ? `<span class="pill">parse ${escapeHtml(readiness.latestParseJobStatus)}</span>` : ""}
          ${readiness.latestExtractionJobStatus ? `<span class="pill">extract ${escapeHtml(readiness.latestExtractionJobStatus)}</span>` : ""}
          ${readiness.failedJobCount > 0 ? `<span class="pill">failed jobs ${escapeHtml(readiness.failedJobCount)}</span>` : ""}
        </div>
      </div>
      <div class="detail-group">
        <strong>Datasets</strong>
        <div class="pill-row">
          ${summary.datasets.slice(0, 6).map((item) => `<span class="pill">${escapeHtml(item.display_name)}</span>`).join("") || '<span class="muted">No datasets yet.</span>'}
        </div>
      </div>
      <div class="detail-group">
        <strong>Metrics</strong>
        <div class="pill-row">
          ${summary.metrics.slice(0, 6).map((item) => `<span class="pill">${escapeHtml(item.display_name)}</span>`).join("") || '<span class="muted">No metrics yet.</span>'}
        </div>
      </div>
      <div class="detail-group">
        <strong>Limitations</strong>
        <ul class="detail-list">
          ${summary.limitations.slice(0, 4).map((item) => `<li>${escapeHtml(item.statement)}</li>`).join("") || "<li>No extracted limitations yet.</li>"}
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
    const evidencePreview = structured.evidence_spans
      .slice(0, 3)
      .map((item) => `<li>${escapeHtml(item.quote_text || "Evidence span")} ${item.page_number ? `(p. ${escapeHtml(item.page_number)})` : ""}</li>`)
      .join("");

    elements.paperDetail.innerHTML = `
      <div class="detail-group">
        <div class="list-title">${escapeHtml(state.selectedPaper.title)}</div>
        <div class="muted">${escapeHtml((state.selectedPaper.authors || []).join(", ") || "Unknown authors")}</div>
      </div>
      <div class="detail-group">
        <strong>Structured surface</strong>
        <div class="pill-row">
          <span class="pill">Datasets ${structured.datasets.length}</span>
          <span class="pill">Methods ${structured.methods.length}</span>
          <span class="pill">Metrics ${structured.metrics.length}</span>
          <span class="pill">Results ${structured.result_rows.length}</span>
          <span class="pill">Limitations ${structured.limitations.length}</span>
        </div>
      </div>
      <div class="detail-group">
        <strong>Methods</strong>
        <ul class="detail-list">
          ${structured.methods.slice(0, 4).map((item) => `<li>${escapeHtml(item.display_name)}</li>`).join("") || "<li>No extracted methods.</li>"}
        </ul>
      </div>
      <div class="detail-group">
        <strong>Result rows</strong>
        <ul class="detail-list">
          ${structured.result_rows.slice(0, 4).map((item) => `<li>${escapeHtml(item.method || "Unknown method")} / ${escapeHtml(item.metric || "Unknown metric")} / ${escapeHtml(formatValue(item.value_numeric ?? item.value_text))}</li>`).join("") || "<li>No result rows yet.</li>"}
        </ul>
      </div>
      <div class="detail-group">
        <strong>Limitations</strong>
        <ul class="detail-list">
          ${structured.limitations.slice(0, 4).map((item) => `<li>${escapeHtml(item.statement)}</li>`).join("") || "<li>No extracted limitations.</li>"}
        </ul>
      </div>
      <div class="detail-group">
        <strong>Figures and tables</strong>
        <ul class="detail-list">
          ${structured.figures.slice(0, 2).map((item) => `<li>${escapeHtml(item.figure_label || "Figure")} — ${escapeHtml(item.caption || "No caption")}</li>`).join("")}
          ${structured.tables.slice(0, 2).map((item) => `<li>${escapeHtml(item.table_label || "Table")} — ${escapeHtml(item.caption || "No caption")}</li>`).join("") || "<li>No extracted artifacts.</li>"}
        </ul>
      </div>
      <div class="detail-group">
        <strong>Evidence preview</strong>
        <ul class="detail-list">
          ${evidencePreview || "<li>No evidence spans persisted yet.</li>"}
        </ul>
      </div>
    `;
  }

  function renderSearchSurfaces() {
    elements.chunkSearchSurface.innerHTML = state.chunkSearchHits.length > 0
      ? `
        <ul class="detail-list">
          ${state.chunkSearchHits.slice(0, 6).map((item) => `
            <li>
              <strong>${escapeHtml(item.paper_title)}</strong>
              ${item.section_title ? `<span class="muted"> / ${escapeHtml(item.section_title)}</span>` : ""}
              <div>${escapeHtml(item.text)}</div>
            </li>
          `).join("")}
        </ul>
      `
      : '<div class="muted">Run a search to inspect chunk-level evidence.</div>';

    elements.artifactSearchSurface.innerHTML = state.artifactSearchHits.length > 0
      ? `
        <ul class="detail-list">
          ${state.artifactSearchHits.slice(0, 6).map((item) => `
            <li>
              <strong>${escapeHtml(item.paper_title)}</strong>
              <span class="muted"> / ${escapeHtml(item.artifact_type)} / ${escapeHtml(item.label || "Untitled")}</span>
              <div>${escapeHtml(item.caption || "No caption")}</div>
            </li>
          `).join("")}
        </ul>
      `
      : '<div class="muted">Run a search to inspect figure and table hits.</div>';
  }

  function renderCompareHeader() {
    if (!state.selectedCollection) {
      elements.compareCollectionContext.innerHTML = '<span class="pill">Select a collection first</span>';
      return;
    }
    const readiness = selectedCollectionReadiness();
    elements.compareCollectionContext.innerHTML = `
      <span class="pill">${escapeHtml(state.selectedCollection.title)}</span>
      <span class="pill">${escapeHtml(readiness.label)}</span>
      ${state.collectionSummary ? `<span class="pill">${escapeHtml(state.collectionSummary.paper_count)} papers</span>` : ""}
    `;
  }

  function renderCompareSurfaces() {
    renderCompareHeader();

    if (!state.selectedCollection) {
      const emptyMessage = '<div class="muted">Select a collection to compare structured evidence.</div>';
      elements.compareReadinessBanner.innerHTML = `
        <div class="readiness-banner-content" data-status="imported">
          <span class="status-marker" data-status="imported"></span>
          <span>Select a collection before comparing structured evidence.</span>
        </div>
      `;
      elements.compareResultsSurface.innerHTML = emptyMessage;
      elements.compareMethodsSurface.innerHTML = emptyMessage;
      elements.compareTricksSurface.innerHTML = emptyMessage;
      elements.compareFiguresSurface.innerHTML = emptyMessage;
      elements.compareTablesSurface.innerHTML = emptyMessage;
      return;
    }

    const readiness = selectedCollectionReadiness();
    if (!readiness.isReadyForCompare) {
      const blockedMessage = `
        <div class="empty-state">
          <strong>Structured evidence is not ready yet. Run extraction in Library.</strong>
          <button type="button" class="action-button action-button-primary" data-open-library-from-compare>Open Library</button>
        </div>
      `;
      elements.compareReadinessBanner.innerHTML = `
        <div class="readiness-banner-content" data-status="${escapeHtml(readiness.status)}">
          <span class="status-marker" data-status="${escapeHtml(readiness.status)}"></span>
          <span>Structured evidence is not ready yet. Run extraction in Library.</span>
        </div>
      `;
      elements.compareResultsSurface.innerHTML = blockedMessage;
      elements.compareMethodsSurface.innerHTML = blockedMessage;
      elements.compareTricksSurface.innerHTML = blockedMessage;
      elements.compareFiguresSurface.innerHTML = blockedMessage;
      elements.compareTablesSurface.innerHTML = blockedMessage;
      document.querySelectorAll("[data-open-library-from-compare]").forEach((button) => {
        button.addEventListener("click", () => {
          activateView("library");
        });
      });
      return;
    }

    elements.compareReadinessBanner.innerHTML = `
      <div class="readiness-banner-content" data-status="${escapeHtml(readiness.status)}">
        <span class="status-marker" data-status="${escapeHtml(readiness.status)}"></span>
        <span>Evidence ready for structured comparison.</span>
      </div>
    `;

    elements.compareDatasetInput.value = state.compare.dataset;
    elements.compareMetricInput.value = state.compare.metric;
    elements.compareMethodInput.value = state.compare.method;

    elements.compareResultsSurface.innerHTML = state.compare.results.length > 0
      ? `
        <ul class="detail-list">
          ${state.compare.results.slice(0, 8).map((item) => `
            <li>
              <strong>${escapeHtml(item.paper_title)}</strong>
              <div>${escapeHtml(item.method || "Unknown method")} / ${escapeHtml(item.metric)} / ${escapeHtml(formatValue(item.value_numeric ?? item.value_text))}</div>
              ${item.evidence_spans && item.evidence_spans.length > 0 ? `<div class="muted">${escapeHtml(item.evidence_spans[0].quote_text || "Evidence available")}</div>` : ""}
            </li>
          `).join("")}
        </ul>
      `
      : '<div class="muted">Provide dataset and metric, then refresh the compare view.</div>';

    elements.compareMethodsSurface.innerHTML = state.compare.methods.length > 0
      ? `
        <ul class="detail-list">
          ${state.compare.methods.slice(0, 8).map((item) => `
            <li>
              <strong>${escapeHtml(item.method)}</strong>
              <div>${escapeHtml(item.paper_count)} papers / ${escapeHtml(item.result_count)} result rows</div>
              ${item.best_result ? `<div class="muted">Best: ${escapeHtml(item.best_result.paper_title)} / ${escapeHtml(item.best_result.metric || "metric")} / ${escapeHtml(formatValue(item.best_result.value_numeric ?? item.best_result.value_text))}</div>` : ""}
            </li>
          `).join("")}
        </ul>
      `
      : '<div class="muted">No method summaries yet for the current collection.</div>';

    elements.compareTricksSurface.innerHTML = state.compare.engineeringTricks.length > 0
      ? `
        <ul class="detail-list">
          ${state.compare.engineeringTricks.slice(0, 8).map((item) => `
            <li>
              <strong>${escapeHtml(item.title)}</strong>
              <div>${escapeHtml(item.description)}</div>
              <div class="muted">${escapeHtml(item.paper_count)} paper(s)</div>
            </li>
          `).join("")}
        </ul>
      `
      : '<div class="muted">No engineering-trick summaries available yet.</div>';

    elements.compareFiguresSurface.innerHTML = state.compare.figures.length > 0
      ? `
        <ul class="detail-list">
          ${state.compare.figures.slice(0, 6).map((item) => `
            <li>
              <strong>${escapeHtml(item.paper_title)}</strong>
              <div>${escapeHtml(item.figure_label || "Figure")} / ${escapeHtml(item.caption || "No caption")}</div>
            </li>
          `).join("")}
        </ul>
      `
      : '<div class="muted">No figure artifacts in the current compare slice.</div>';

    elements.compareTablesSurface.innerHTML = state.compare.tables.length > 0
      ? `
        <ul class="detail-list">
          ${state.compare.tables.slice(0, 6).map((item) => `
            <li>
              <strong>${escapeHtml(item.paper_title)}</strong>
              <div>${escapeHtml(item.table_label || "Table")} / ${escapeHtml(item.caption || "No caption")}</div>
            </li>
          `).join("")}
        </ul>
      `
      : '<div class="muted">No table artifacts in the current compare slice.</div>';
  }

  function renderJobs() {
    if (state.jobs.length === 0) {
      elements.jobsList.innerHTML = '<div class="list-card muted">No jobs queued yet.</div>';
      return;
    }

    const collectionTitle = (collectionId) => {
      const collection = state.collections.find((item) => item.id === collectionId);
      return collection ? collection.title : collectionId;
    };
    const groupDefinitions = [
      { title: "Running", statuses: ["running"] },
      { title: "Queued", statuses: ["pending", "queued"] },
      { title: "Failed", statuses: ["failed"] },
      { title: "Completed", statuses: ["completed"] },
    ];
    const groupedHtml = groupDefinitions
      .map((group) => {
        const jobs = state.jobs.filter((job) => group.statuses.includes(job.status));
        if (jobs.length === 0) {
          return `
            <section class="job-group">
              <h3>${escapeHtml(group.title)}</h3>
              <div class="muted">No ${escapeHtml(group.title.toLowerCase())} jobs.</div>
            </section>
          `;
        }
        return `
          <section class="job-group">
            <h3>${escapeHtml(group.title)}</h3>
            <div class="stacked-list">
              ${jobs.map((job) => {
                const affectedCollectionId = collectionIdFromJob(job);
                return `
                  <div class="list-card">
                    <div class="job-status" data-status="${escapeHtml(job.status)}">${escapeHtml(job.status)}</div>
                    <div class="list-title">${escapeHtml(job.job_type)}</div>
                    <div class="job-meta">${escapeHtml(job.created_at || "queued")}</div>
                    ${affectedCollectionId ? `<div class="muted">Collection: ${escapeHtml(collectionTitle(affectedCollectionId))}</div>` : ""}
                    ${job.error_message ? `<div class="job-error">${escapeHtml(job.error_message)}</div>` : ""}
                    <details class="raw-job-details">
                      <summary>Details</summary>
                      <pre>${escapeHtml(JSON.stringify(job.result || job.payload || {}, null, 2))}</pre>
                    </details>
                  </div>
                `;
              }).join("")}
            </div>
          </section>
        `;
      })
      .join("");
    const uncategorized = state.jobs.filter((job) => !groupDefinitions.some((group) => group.statuses.includes(job.status)));
    elements.jobsList.innerHTML = groupedHtml + (uncategorized.length > 0
      ? `
        <section class="job-group">
          <h3>Other</h3>
          <div class="stacked-list">
            ${uncategorized.map((job) => `
              <div class="list-card">
                <div class="job-status" data-status="${escapeHtml(job.status)}">${escapeHtml(job.status)}</div>
                <div class="list-title">${escapeHtml(job.job_type)}</div>
              </div>
            `).join("")}
          </div>
        </section>
      `
      : "");
  }

  function renderSettings() {
    const selectedPaperTitle = state.selectedPaper ? state.selectedPaper.title : null;
    const readiness = selectedCollectionReadiness();
    const searchStatus = state.system.searchStatus;
    const systemReadiness = state.system.readiness;
    const dependencies = systemReadiness && systemReadiness.dependencies ? systemReadiness.dependencies : [];
    const dependencyStatus = (name) => {
      const item = dependencies.find((dependency) => dependency.name === name);
      if (!item) {
        return "unknown";
      }
      return item.ok ? "ok" : (item.required ? "not ready" : "fallback");
    };
    elements.settingsSummary.innerHTML = `
      <div class="detail-group">
        <strong>Selected collection</strong>
        <div class="muted">${escapeHtml((state.selectedCollection && state.selectedCollection.title) || "None")} ${state.selectedCollection ? `· ${escapeHtml(readiness.label)}` : ""}</div>
      </div>
      <div class="detail-group">
        <strong>Selected workspace</strong>
        <div class="muted">${escapeHtml((state.selectedWorkspace && state.selectedWorkspace.title) || "None")}</div>
      </div>
      <div class="detail-group">
        <strong>Selected paper</strong>
        <div class="muted">${escapeHtml(selectedPaperTitle || "None")}</div>
      </div>
      <div class="detail-group">
        <strong>Current view</strong>
        <div class="muted">${escapeHtml(state.activeView)}</div>
      </div>
      <div class="detail-group">
        <strong>Saved query</strong>
        <div class="muted">${escapeHtml(elements.searchInput.value.trim() || "None")}</div>
      </div>
    `;
    elements.settingsRuntimeSummary.innerHTML = `
      <div class="detail-group">
        <strong>Mode</strong>
        <div class="muted">${escapeHtml(searchStatus && searchStatus.backend_configured ? "Backend-search mode" : "Local fallback mode")}</div>
      </div>
      <div class="detail-group">
        <strong>Search backend</strong>
        <div class="muted">${escapeHtml(searchStatus && searchStatus.backend_type ? searchStatus.backend_type : "Not configured; using database fallback")}</div>
      </div>
      <div class="detail-group">
        <strong>Object store</strong>
        <div class="muted">${escapeHtml(dependencyStatus("object_store"))}</div>
      </div>
      <div class="detail-group">
        <strong>Worker queue</strong>
        <div class="muted">${escapeHtml(state.jobs.some((job) => isActiveJobStatus(job.status)) ? "active jobs present" : "idle or external worker")}</div>
      </div>
      <div class="detail-group">
        <strong>API readiness</strong>
        <div class="muted">${escapeHtml(systemReadiness ? systemReadiness.status : "unknown")}</div>
      </div>
    `;
  }

  function renderAll() {
    renderCollectionsSidebar();
    renderSidebarPapers();
    renderWorkspacesSidebar();
    renderLibraryLogs();
    renderWorkspaceHeader();
    renderWorkspaceReadiness();
    renderPapers();
    renderWorkspaceDetail();
    renderCollectionSummary();
    renderPaperDetail();
    renderSearchSurfaces();
    renderCompareSurfaces();
    renderJobs();
    renderSettings();
    updateActionButtons();
  }

  function updateActionButtons() {
    const collectionSelected = Boolean(state.selectedCollection);
    const readiness = selectedCollectionReadiness();
    elements.parseButton.disabled = !collectionSelected || unprocessedPaperIds().length === 0;
    elements.parseSelectedPapersButton.disabled = !collectionSelected || selectedPaperIdsForAction("parse").length === 0;
    elements.extractButton.disabled = !collectionSelected || selectedPaperIdsForAction("extract").length === 0;
    elements.workspaceOpenCompareButton.disabled = !collectionSelected;
    elements.saveWorkspaceButton.disabled = !collectionSelected;
    elements.compareRefreshButton.disabled = !collectionSelected || !readiness.isReadyForCompare;
  }

  function seedCompareDraft() {
    const currentCollectionId = state.selectedCollection ? state.selectedCollection.id : null;
    const collectionChanged = state.compare.collectionId !== currentCollectionId;
    if (collectionChanged) {
      state.compare.collectionId = currentCollectionId;
      state.compare.results = [];
      state.compare.methods = [];
      state.compare.engineeringTricks = [];
      state.compare.figures = [];
      state.compare.tables = [];
      state.compare.dataset = "";
      state.compare.metric = "";
      state.compare.method = "";
    }
    if (!state.collectionSummary) {
      return;
    }
    if (!state.compare.dataset && state.collectionSummary.datasets.length > 0) {
      state.compare.dataset = state.collectionSummary.datasets[0].display_name;
    }
    if (!state.compare.metric && state.collectionSummary.metrics.length > 0) {
      state.compare.metric = state.collectionSummary.metrics[0].display_name;
    }
    if (!state.compare.method && state.collectionSummary.methods.length > 0) {
      state.compare.method = state.collectionSummary.methods[0].display_name;
    }
  }

  async function loadCollections(preferredCollectionId) {
    const payload = await fetchJson(endpoints.collections);
    state.collections = payload.data || [];

    if (state.collections.length === 0) {
      state.selectedCollection = null;
      state.papers = [];
      state.selectedPaperIds = new Set();
      state.searchResults = [];
      state.selectedPaper = null;
      state.selectedPaperStructured = null;
      state.collectionSummary = null;
      state.chunkSearchHits = [];
      state.artifactSearchHits = [];
      state.compare = {
        collectionId: null,
        dataset: "",
        metric: "",
        method: "",
        results: [],
        methods: [],
        engineeringTricks: [],
        figures: [],
        tables: [],
      };
      renderAll();
      return;
    }

    const collectionIdToOpen = preferredCollectionId
      || (state.selectedCollection && state.collections.some((item) => item.id === state.selectedCollection.id)
        ? state.selectedCollection.id
        : state.collections[0].id);
    await loadCollectionSurface(collectionIdToOpen);
  }

  async function loadCollectionSurface(collectionId, options) {
    const retainPaperSelection = state.selectedCollection && state.selectedCollection.id === collectionId;
    const previousSelectedPaperIds = retainPaperSelection ? new Set(state.selectedPaperIds) : new Set();
    const [collectionPayload, papersPayload, summaryPayload] = await Promise.all([
      fetchJson(`${endpoints.collections}/${collectionId}`),
      fetchJson(`${endpoints.collections}/${collectionId}/papers`),
      fetchJson(`${endpoints.collections}/${collectionId}/structured-summary`),
    ]);

    state.selectedCollection = collectionPayload.data;
    state.papers = papersPayload.data || [];
    const currentPaperIds = new Set(state.papers.map((membership) => membership.paper.id));
    state.selectedPaperIds = new Set(
      Array.from(previousSelectedPaperIds).filter((paperId) => currentPaperIds.has(paperId)),
    );
    state.collectionSummary = summaryPayload.data;
    state.searchResults = [];
    state.chunkSearchHits = [];
    state.artifactSearchHits = [];
    state.selectedPaper = null;
    state.selectedPaperStructured = null;
    const retainWorkspace = Boolean(options && options.retainWorkspace);
    if (
      !retainWorkspace
      && state.selectedWorkspace
      && state.selectedWorkspace.collection_id
      && state.selectedWorkspace.collection_id !== collectionId
    ) {
      state.selectedWorkspace = null;
    }
    seedCompareDraft();
    renderAll();
  }

  async function loadWorkspaces(options) {
    const payload = await fetchJson(endpoints.workspaces);
    state.workspaces = payload.data || [];

    const preferredWorkspaceId = options && options.preferredWorkspaceId ? options.preferredWorkspaceId : null;
    const shouldReloadSelected = !preferredWorkspaceId && state.selectedWorkspace
      && state.workspaces.some((item) => item.id === state.selectedWorkspace.id);

    if (preferredWorkspaceId) {
      await openWorkspace(preferredWorkspaceId, { activate: false });
      return;
    }
    if (shouldReloadSelected) {
      await openWorkspace(state.selectedWorkspace.id, { activate: false });
      return;
    }
    if (state.selectedWorkspace && !state.workspaces.some((item) => item.id === state.selectedWorkspace.id)) {
      state.selectedWorkspace = null;
    }
    renderAll();
  }

  async function openWorkspace(workspaceId, options) {
    const payload = await fetchJson(`${endpoints.workspaces}/${workspaceId}`);
    state.selectedWorkspace = payload.data;

    if (state.selectedWorkspace.collection_id) {
      await loadCollectionSurface(state.selectedWorkspace.collection_id, { retainWorkspace: true });
    }

    elements.searchInput.value = state.selectedWorkspace.saved_query || "";
    if (state.selectedWorkspace.saved_query) {
      await searchPapers(state.selectedWorkspace.saved_query);
    } else {
      state.searchResults = [];
      state.chunkSearchHits = [];
      state.artifactSearchHits = [];
    }

    if (state.selectedWorkspace.pinned_papers && state.selectedWorkspace.pinned_papers.length > 0) {
      await loadPaperDetail(state.selectedWorkspace.pinned_papers[0].id);
    }

    renderAll();
    if (!options || options.activate !== false) {
      activateView("workspace");
    }
  }

  async function searchPapers(query) {
    const trimmed = query.trim();
    renderWorkspaceHeader();
    if (!trimmed) {
      state.searchResults = [];
      state.chunkSearchHits = [];
      state.artifactSearchHits = [];
      renderAll();
      return;
    }

    const params = new URLSearchParams({ q: trimmed });
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
    renderAll();
  }

  async function loadPaperDetail(paperId) {
    const [paperPayload, structuredPayload] = await Promise.all([
      fetchJson(`/api/v1/papers/${paperId}`),
      fetchJson(`/api/v1/papers/${paperId}/structured-data`),
    ]);

    state.selectedPaper = paperPayload.data;
    state.selectedPaperStructured = structuredPayload.data;
    renderAll();
  }

  async function loadJobs() {
    const previouslyHadActiveJobs = state.jobs.some((job) => job.status === "pending" || job.status === "running");
    const payload = await fetchJson(`${endpoints.jobs}?limit=20`);
    state.jobs = payload.data || [];
    renderAll();
    updatePolling();
    const hasActiveJobs = state.jobs.some((job) => job.status === "pending" || job.status === "running");
    if (previouslyHadActiveJobs && !hasActiveJobs) {
      await refreshAfterJobCompletion();
    }
  }

  async function loadSystemStatus() {
    const [searchResult, readinessResult] = await Promise.allSettled([
      fetchJson(endpoints.searchStatus),
      fetch(endpoints.readiness).then(async (response) => response.json().catch(() => ({ status: "unknown" }))),
    ]);
    if (searchResult.status === "fulfilled") {
      state.system.searchStatus = searchResult.value.data;
    }
    if (readinessResult.status === "fulfilled") {
      state.system.readiness = readinessResult.value;
    }
    renderSettings();
  }

  async function queueReindex() {
    const payload = await fetchJson(endpoints.reindex, { method: "POST" });
    state.jobs.unshift(payload.data);
    renderAll();
    updatePolling();
    setStatus("Queued a corpus reindex job.");
  }

  async function queueLocalLibraryIngest() {
    const sourceDir = elements.localLibrarySourceInput.value.trim();
    if (!sourceDir) {
      throw new Error("A local source directory is required.");
    }

    const collectionTitle = elements.localLibraryTitleInput.value.trim();
    const collectionDescription = elements.localLibraryDescriptionInput.value.trim();
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
    renderAll();
    updatePolling();
    setStatus(`Queued local import from ${sourceDir}.`);
  }

  async function queueLocalLibraryUploadIngest() {
    const files = Array.from(elements.localLibraryUploadInput.files || []);
    if (files.length === 0) {
      throw new Error("Select at least one PDF file or folder to upload.");
    }

    const collectionTitle = elements.localLibraryUploadTitleInput.value.trim();
    const collectionDescription = elements.localLibraryUploadDescriptionInput.value.trim();
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
    renderAll();
    updatePolling();
    setStatus(`Queued upload import for ${files.length} file(s).`);
  }

  async function queueExtraction(paperIds) {
    if (!state.selectedCollection) {
      return;
    }
    if (Array.isArray(paperIds) && paperIds.length === 0) {
      setStatus("No selected papers need extraction.");
      return;
    }

    const body = {
      prompt_version: "paperbase-v1",
      schema_version: "paperbase-v1",
    };
    if (Array.isArray(paperIds)) {
      body.paper_ids = paperIds;
    }
    if (state.selectedCollection.extraction_profile_id) {
      body.extraction_profile_id = state.selectedCollection.extraction_profile_id;
    } else {
      body.schema_payload = {
        datasets: true,
        methods: true,
        metrics: true,
        results: true,
        engineering_tricks: true,
        limitations: true,
      };
    }

    const payload = await fetchJson(`/api/v1/collections/${state.selectedCollection.id}/extract`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.jobs.unshift(payload.data);
    renderAll();
    updatePolling();
    const scope = Array.isArray(paperIds) ? `${paperIds.length} paper(s)` : state.selectedCollection.title;
    setStatus(`Queued extraction for ${scope}.`);
  }

  async function queueParse(paperIds) {
    if (!state.selectedCollection) {
      return;
    }
    if (Array.isArray(paperIds) && paperIds.length === 0) {
      setStatus("No selected papers need parsing.");
      return;
    }
    const body = {};
    if (Array.isArray(paperIds)) {
      body.paper_ids = paperIds;
    }
    const payload = await fetchJson(`/api/v1/collections/${state.selectedCollection.id}/parse`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.jobs.unshift(payload.data);
    renderAll();
    updatePolling();
    const scope = Array.isArray(paperIds) ? `${paperIds.length} paper(s)` : state.selectedCollection.title;
    setStatus(`Queued parse for ${scope}.`);
  }

  async function saveWorkspace() {
    const title = elements.workspaceTitleInput.value.trim()
      || (state.selectedCollection ? `${state.selectedCollection.title} workspace` : "Arxie workspace");
    const payload = {
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
        body: JSON.stringify(payload),
      });
      await loadWorkspaces({ preferredWorkspaceId: state.selectedWorkspace.id });
      setStatus(`Updated workspace ${title}.`);
      return;
    }

    const response = await fetchJson(endpoints.workspaces, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await loadWorkspaces({ preferredWorkspaceId: response.data.id });
    setStatus(`Saved workspace ${title}.`);
  }

  async function refreshCompare() {
    if (!state.selectedCollection) {
      renderCompareSurfaces();
      return;
    }
    const readiness = selectedCollectionReadiness();
    if (!readiness.isReadyForCompare) {
      state.compare.results = [];
      state.compare.methods = [];
      state.compare.engineeringTricks = [];
      state.compare.figures = [];
      state.compare.tables = [];
      renderCompareSurfaces();
      return;
    }

    state.compare.dataset = elements.compareDatasetInput.value.trim();
    state.compare.metric = elements.compareMetricInput.value.trim();
    state.compare.method = elements.compareMethodInput.value.trim();

    const collectionId = state.selectedCollection.id;
    const comparePromises = [
      fetchJson(endpoints.compareMethods, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          collection_id: collectionId,
          dataset: state.compare.dataset || null,
          metric: state.compare.metric || null,
          limit: 8,
        }),
      }),
      fetchJson(endpoints.compareEngineeringTricks, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          collection_id: collectionId,
          method: state.compare.method || null,
          limit: 8,
        }),
      }),
      fetchJson(endpoints.compareFigures, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          collection_id: collectionId,
          method: state.compare.method || null,
          limit: 8,
        }),
      }),
      fetchJson(endpoints.compareTables, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          collection_id: collectionId,
          method: state.compare.method || null,
          limit: 8,
        }),
      }),
    ];

    if (state.compare.dataset && state.compare.metric) {
      comparePromises.unshift(
        fetchJson(endpoints.compareResults, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            collection_id: collectionId,
            dataset: state.compare.dataset,
            metric: state.compare.metric,
            include_evidence: true,
          }),
        }),
      );
    } else {
      comparePromises.unshift(Promise.resolve({ data: [] }));
    }

    const [resultsPayload, methodsPayload, tricksPayload, figuresPayload, tablesPayload] = await Promise.all(comparePromises);
    state.compare.results = resultsPayload.data || [];
    state.compare.methods = methodsPayload.data || [];
    state.compare.engineeringTricks = tricksPayload.data || [];
    state.compare.figures = figuresPayload.data || [];
    state.compare.tables = tablesPayload.data || [];
    renderAll();
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
          const match = state.collections.find((collection) => collection.title === job.payload.collection_title);
          if (match) {
            return match.id;
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
    await loadCollections(preferredCollectionId);
    await loadWorkspaces(preferredWorkspaceId ? { preferredWorkspaceId } : undefined);
    if (state.activeView === "compare" && state.selectedCollection) {
      await refreshCompare();
    }
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
      activateView("workspace");
      setStatus("Loading Arxie workspace…");
      await loadCollections();
      await loadWorkspaces();
      await loadJobs();
      await loadSystemStatus();
      renderAll();
      activateView(state.selectedCollection ? "workspace" : "library");
      setStatus("Arxie workspace ready.");
    } catch (error) {
      setStatus(error.message);
    }
  }

  elements.navTabs.forEach((button) => {
    button.addEventListener("click", async () => {
      const nextView = button.getAttribute("data-view");
      if (!nextView) {
        return;
      }
      if (nextView === "compare" && state.selectedCollection) {
        try {
          setStatus("Refreshing compare view…");
          await refreshCompare();
          setStatus("Compare view ready.");
        } catch (error) {
          setStatus(error.message);
        }
      }
      activateView(nextView);
      renderSettings();
    });
  });

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

  elements.workspaceOpenCompareButton.addEventListener("click", async () => {
    try {
      setStatus("Opening compare view…");
      await refreshCompare();
      activateView("compare");
      setStatus("Compare view ready.");
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

  elements.reindexButton.addEventListener("click", async () => {
    try {
      await queueReindex();
    } catch (error) {
      setStatus(error.message);
    }
  });

  elements.refreshJobsButton.addEventListener("click", async () => {
    try {
      setStatus("Refreshing jobs…");
      await loadJobs();
      setStatus("Jobs refreshed.");
    } catch (error) {
      setStatus(error.message);
    }
  });

  elements.parseButton.addEventListener("click", async () => {
    try {
      await queueParse(unprocessedPaperIds());
    } catch (error) {
      setStatus(error.message);
    }
  });

  elements.parseSelectedPapersButton.addEventListener("click", async () => {
    try {
      await queueParse(selectedPaperIdsForAction("parse"));
    } catch (error) {
      setStatus(error.message);
    }
  });

  elements.extractButton.addEventListener("click", async () => {
    try {
      await queueExtraction(selectedPaperIdsForAction("extract"));
    } catch (error) {
      setStatus(error.message);
    }
  });

  elements.localLibraryForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await queueLocalLibraryIngest();
      activateView("jobs");
    } catch (error) {
      setStatus(error.message);
    }
  });

  elements.localLibraryUploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await queueLocalLibraryUploadIngest();
      activateView("jobs");
    } catch (error) {
      setStatus(error.message);
    }
  });

  elements.compareRefreshButton.addEventListener("click", async () => {
    try {
      setStatus("Refreshing compare view…");
      await refreshCompare();
      setStatus("Compare view updated.");
    } catch (error) {
      setStatus(error.message);
    }
  });

  initialize();
})();
