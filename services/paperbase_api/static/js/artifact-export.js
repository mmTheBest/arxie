export function csvCell(value) {
  return `"${String(value ?? "").replace(/"/g, '""')}"`;
}

export function artifactToCsv(artifact) {
  const output = artifact.output_payload || {};
  if (Array.isArray(output.comparison_rows) && output.comparison_rows.length > 0) {
    const headers = ["paper_title", "method", "dataset", "metric", "value", "notes"];
    const rows = output.comparison_rows.map((row) => headers.map((header) => csvCell(row[header])).join(","));
    return [headers.join(","), ...rows].join("\n");
  }
  const entries = Object.entries(output)
    .filter(([, value]) => value !== null && value !== undefined)
    .map(([key, value]) => [
      csvCell(key),
      csvCell(Array.isArray(value) || typeof value === "object" ? JSON.stringify(value) : value),
    ].join(","));
  return ["field,value", ...entries].join("\n");
}

export function artifactToMarkdown(artifact, helpers) {
  const output = artifact.output_payload || {};
  const evidence = artifact.evidence_payload || {};
  const lines = [
    `# ${artifact.saved_title || artifact.title}`,
    "",
    `Type: ${helpers.formatArtifactType(artifact.artifact_type)}`,
    "",
  ];
  if (output.summary) {
    lines.push(String(output.summary), "");
  }
  if (Array.isArray(output.comparison_rows) && output.comparison_rows.length > 0) {
    lines.push("| Paper | Method | Dataset | Metric | Value |", "| --- | --- | --- | --- | --- |");
    output.comparison_rows.forEach((row) => {
      lines.push(`| ${row.paper_title || ""} | ${row.method || ""} | ${row.dataset || ""} | ${row.metric || ""} | ${row.value || ""} |`);
    });
    lines.push("");
  }
  const listFields = [
    ["Themes", output.themes],
    ["Hypotheses", output.hypotheses],
    ["Benchmark recommendations", output.benchmark_recommendations],
    ["Revision priorities", output.revision_priorities],
    ["Assumptions to challenge", output.assumptions_to_challenge],
    ["Experiment backlog", output.backlog_items],
    ["Methods", output.method_families],
    ["Datasets", output.datasets],
    ["Metrics", output.metrics],
  ];
  listFields.forEach(([title, value]) => {
    if (!Array.isArray(value) || value.length === 0) {
      return;
    }
    lines.push(`## ${title}`);
    value.slice(0, 12).forEach((item) => {
      if (typeof item === "object") {
        lines.push(`- ${item.title || item.claim || JSON.stringify(item)}`);
      } else {
        lines.push(`- ${item}`);
      }
    });
    lines.push("");
  });
  const papers = evidence.papers || [];
  if (papers.length > 0) {
    lines.push("## Evidence Papers");
    papers.forEach((paper) => {
      lines.push(`- ${paper.title}`);
    });
    lines.push("");
  }
  return lines.join("\n");
}

export function artifactToNote(artifact, helpers) {
  const output = artifact.output_payload || {};
  return [
    artifact.saved_title || artifact.title,
    "",
    output.summary || helpers.previewText(artifact),
    "",
    `Artifact type: ${helpers.formatArtifactType(artifact.artifact_type)}`,
  ].join("\n");
}

export function artifactDownloadText(artifact, format, helpers) {
  if (format === "csv") {
    return artifactToCsv(artifact);
  }
  if (format === "script") {
    return JSON.stringify(artifact.output_payload || {}, null, 2);
  }
  if (format === "note") {
    return artifactToNote(artifact, helpers);
  }
  return artifactToMarkdown(artifact, helpers);
}
