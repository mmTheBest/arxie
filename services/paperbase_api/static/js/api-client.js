import { state } from "./state.js";
import { urlWithoutQuery } from "./utils.js";

export function currentProjectId() {
  return state.currentProject && state.currentProject.id ? state.currentProject.id : null;
}

export function withProjectHeaders(url, options) {
  const resolvedOptions = options ? { ...options } : {};
  const headers = new Headers(resolvedOptions.headers || {});
  const projectId = currentProjectId();
  if (projectId && urlWithoutQuery(url).indexOf("/api/v1/projects") === -1) {
    headers.set("X-Arxie-Project-Id", projectId);
  }
  resolvedOptions.headers = headers;
  return resolvedOptions;
}

export async function fetchJson(url, options) {
  const resolvedOptions = withProjectHeaders(url, options);
  const response = await fetch(url, resolvedOptions);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.message || `Request failed: ${response.status}`);
  }
  return payload;
}
