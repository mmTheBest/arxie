import { isActiveJobStatus, pluralize } from "./utils.js";

export function collectionIdFromJob(job) {
  const result = job.result || {};
  const payload = job.payload || {};
  return result.collection_id || payload.collection_id || null;
}

export function preferredJob(jobs) {
  return jobs.find((job) => job.status === "running")
    || jobs.find((job) => job.status === "pending" || job.status === "queued")
    || jobs[0]
    || null;
}

export function latestCollectionJobStatus(collection, jobs, jobType) {
  if (!collection) {
    return null;
  }
  const match = preferredJob(
    jobs.filter((job) => collectionIdFromJob(job) === collection.id && job.job_type === jobType),
  );
  return match ? match.status : null;
}

export function collectionFailedJobCount(collection, jobs) {
  if (!collection) {
    return 0;
  }
  return jobs.filter((job) => collectionIdFromJob(job) === collection.id && job.status === "failed").length;
}

export function getCollectionReadiness(collection, summary, jobs) {
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
