export function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function formatValue(value) {
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  return String(value);
}

export function uniqueStrings(values) {
  return Array.from(new Set(values.filter(Boolean)));
}

export function pluralize(count, singular, plural) {
  return `${count} ${count === 1 ? singular : plural}`;
}

export function urlWithoutQuery(url) {
  return String(url || "").split("?")[0];
}

export function isActiveJobStatus(status) {
  return status === "pending" || status === "queued" || status === "running";
}

export function parseJobTimestamp(value) {
  if (!value) {
    return Number.NaN;
  }
  const hasExplicitTimezone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(value);
  return Date.parse(hasExplicitTimezone ? value : `${value}Z`);
}

export function isStaleJob(job) {
  if (!job || job.status !== "running" || !job.started_at) {
    return false;
  }
  const startedAt = parseJobTimestamp(job.started_at);
  if (Number.isNaN(startedAt)) {
    return false;
  }
  return Date.now() - startedAt > 15 * 60 * 1000;
}
