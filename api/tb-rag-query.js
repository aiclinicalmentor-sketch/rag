// api/tb-rag-query.js
// TB Clinical Mentor RAG endpoint with table-aware CSV loading and rendering.

const fs = require("fs");
const path = require("path");
const Papa = require("papaparse");

let RAG_STORE = null;

// ---------- Embedding + RAG store helpers ----------

function normalize(vec) {
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm);
  if (!norm || !Number.isFinite(norm)) return vec.map(() => 0);
  return vec.map((v) => v / norm);
}

function loadRagStore() {
  if (RAG_STORE) return RAG_STORE;

  const ragDir = path.join(process.cwd(), "public", "rag");
  const chunksPath = path.join(ragDir, "chunks.jsonl");
  const embeddingsPath = path.join(ragDir, "embeddings.json");

  const chunksLines = fs
    .readFileSync(chunksPath, "utf-8")
    .split("\n")
    .filter(Boolean);

  const chunks = chunksLines.map((line) => JSON.parse(line));

  const rawEmbeddings = JSON.parse(fs.readFileSync(embeddingsPath, "utf-8"));
  const embeddings = rawEmbeddings.map((vec) => normalize(vec));

  if (embeddings.length !== chunks.length) {
    console.warn(
      "WARNING: embeddings count != chunks count",
      embeddings.length,
      chunks.length
    );
  }

  RAG_STORE = { chunks, embeddings };
  return RAG_STORE;
}

// For normalized vectors, cosine similarity is just their dot product.
function cosineSim(a, b) {
  const len = Math.min(a.length, b.length);
  let dot = 0;
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

async function embedQuestion(question) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is not set");
  }

  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "text-embedding-3-large",
      input: question,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`OpenAI embeddings error: ${response.status} ${text}`);
  }

  const json = await response.json();
  const embedding = json.data[0].embedding;
  return normalize(embedding);
}

// ---------- Scope + doc_hint helpers ----------

function keywordScore(text, keywords) {
  let score = 0;
  for (const kw of keywords) {
    if (text.includes(kw)) score += 1;
  }
  return score;
}

function inferScopeFromQuestion(question) {
  const q = (question || "").toLowerCase();
  if (!q) return null;

  const diagnosisScore = keywordScore(q, [
    "diagnos",
    "algorithm",
    "cxr",
    "x-ray",
    "radiograph",
    "screen",
    "xpert",
    "naat",
    "truenat",
    "lamp",
    "wrd",
    "wrds",
    "lpa",
    "smear",
    "ultra"
  ]);

  const treatmentScore = keywordScore(q, [
    "treatment",
    "regimen",
    "therapy",
    "dosing",
    "dose",
    "4-month",
    "6-month",
    "bpal",
    "bdq",
    "pretomanid",
    "linezolid",
    "dr-tb",
    "drug-resistant"
  ]);

  const safetyScore = keywordScore(q, [
    "adverse",
    "toxicity",
    "monitor",
    "safety",
    "ecg",
    "qt",
    "lft",
    "renal",
    "hepat",
    "side effect",
    "side effects"
  ]);

  const pediScore = keywordScore(q, [
    "child",
    "children",
    "paediatric",
    "pediatric",
    "adolescent",
    "infant",
    "neonate"
  ]);

  const programScore = keywordScore(q, [
    "program",
    "programmatic",
    "supply chain",
    "implementation",
    "health system",
    "adherence support",
    "community"
  ]);

  if (diagnosisScore >= 2 || (diagnosisScore && q.includes("module 3"))) {
    return "diagnosis";
  }
  if (pediScore >= 2) {
    return "pediatrics";
  }
  if (safetyScore >= 2 || (safetyScore && q.includes("monitor"))) {
    return "drug-safety";
  }
  if (programScore >= 2) {
    return "programmatic";
  }
  if (treatmentScore >= 2) {
    return "treatment";
  }

  return null;
}

function filterIndicesByScope(indices, chunks, scope) {
  if (!scope) return indices;

  const s = String(scope).toLowerCase();

  return indices.filter((i) => {
    const c = chunks[i] || {};
    const docId = (c.doc_id || "").toString().toLowerCase();
    const section = (c.section_path || "").toString().toLowerCase();
    const chunkScope = (c.scope || "").toString().toLowerCase();

    if (chunkScope) {
      return chunkScope === s;
    }

    switch (s) {
      case "diagnosis":
        return (
          docId.includes("diag") ||
          section.includes("diagnos") ||
          section.includes("xpert") ||
          section.includes("cxr")
        );
      case "treatment":
        return (
          docId.includes("treat") ||
          section.includes("treatment") ||
          section.includes("regimen")
        );
      case "drug-safety":
        return (
          section.includes("monitor") ||
          section.includes("ecg") ||
          section.includes("adverse") ||
          section.includes("safety")
        );
      case "pediatrics":
        return (
          docId.includes("pediatric") ||
          docId.includes("child") ||
          section.includes("child") ||
          section.includes("adolescent")
        );
      case "programmatic":
        return (
          section.includes("program") ||
          section.includes("implementation") ||
          section.includes("health system")
        );
      default:
        return true;
    }
  });
}

function buildDocHintTokens(hint) {
  if (!hint) return [];
  const raw = String(hint).toLowerCase().trim();
  if (!raw) return [];

  const tokens = new Set();
  tokens.add(raw);
  tokens.add(raw.replace(/\s+/g, ""));
  tokens.add(raw.replace(/\s+/g, "_"));
  tokens.add(raw.replace(/[^a-z0-9]+/g, ""));
  return Array.from(tokens).filter(Boolean);
}

function filterIndicesByDocHint(indices, chunks, hint) {
  if (!hint) return indices;
  const hintTokens = buildDocHintTokens(hint);
  if (!hintTokens.length) return indices;

  const filtered = indices.filter((idx) => {
    const c = chunks[idx] || {};
    const haystacks = [
      (c.doc_id || "").toString().toLowerCase(),
      (c.section_path || "").toString().toLowerCase(),
      (c.guideline_title || "").toString().toLowerCase(),
      (c.scope || "").toString().toLowerCase()
    ];

    return haystacks.some((hay) =>
      hay && hintTokens.some((token) => hay.includes(token))
    );
  });

  return filtered.length ? filtered : indices;
}

const DOC_HINT_ALIASES = [
  {
    target: "module3_diagnosis",
    keywords: ["module 3", "module3", "module iii"],
    requires: ["diagnos", "algorithm", "cxr"]
  },
  {
    target: "module4_treatment",
    keywords: ["module 4", "module4", "module iv"],
    requires: ["treat", "regimen", "therapy"]
  },
  {
    target: "consolidated_module3",
    keywords: ["consolidated", "module 3"],
    requires: ["diagnos"]
  }
];

function inferDocHintFromQuestion(question) {
  const q = (question || "").toLowerCase();
  if (!q) return null;

  for (const alias of DOC_HINT_ALIASES) {
    const hasKeyword = alias.keywords.some((kw) => q.includes(kw));
    if (!hasKeyword) continue;
    const requires = alias.requires || [];
    if (!requires.length || requires.some((kw) => q.includes(kw))) {
      return alias.target;
    }
  }
  return null;
}

function extractSectionKeys(sectionPath) {
  if (!sectionPath) return [];
  const s = String(sectionPath).toLowerCase();

  const keys = new Set();
  keys.add(s.trim());

  const segments = s.split("|").map((seg) => seg.trim()).filter(Boolean);
  for (const seg of segments) keys.add(seg);

  const numMatches = s.match(/\b\d+(?:\.\d+)*\b/g);
  if (numMatches) {
    for (const m of numMatches) keys.add(m);
  }

  return Array.from(keys);
}

// ---------- Table loading + normalization + subtype detection + rendering ----------

// Resolve an attachment_path from chunk metadata to an absolute CSV path
// Deployed CSVs live under: public/rag/tables/<guideline>/Table_X.csv
function resolveTablePath(attachmentPathFromMeta) {
  if (!attachmentPathFromMeta) return null;

  let cleaned = String(attachmentPathFromMeta)
    .replace(/\\\\/g, "/")
    .replace(/\\/g, "/");

  cleaned = cleaned.replace(/^\.\/+/, "");
  cleaned = cleaned.replace(/^\.\.\//, "");

  if (!cleaned.startsWith("public/")) {
    cleaned = path.posix.join("public", cleaned);
  }

  const absolutePath = path.join(process.cwd(), cleaned);
  return absolutePath;
}

// Load raw CSV rows (with generic ColumnA, ColumnB, etc.)
function loadTableRows(attachmentPathFromMeta) {
  const absPath = resolveTablePath(attachmentPathFromMeta);
  if (!absPath) {
    throw new Error("Cannot resolve table path from attachment_path");
  }

  const csv = fs.readFileSync(absPath, "utf8");
  const parsed = Papa.parse(csv, {
    header: true,
    skipEmptyLines: true
  });

  if (parsed.errors && parsed.errors.length) {
    console.warn("CSV parse errors for table:", absPath, parsed.errors);
  }

  return parsed.data || [];
}

// Normalize to "logical" rows using row_index=1 as header row
function normalizeTableRows(rawRows) {
  if (!rawRows || !rawRows.length) {
    return { headerRow: null, logicalRows: [] };
  }

  const headerRow =
    rawRows.find((r) => String(r.row_index) === "1") || rawRows[0];

  const dataRows = rawRows.filter(
    (r) => String(r.row_index) !== String(headerRow.row_index)
  );

  const allKeys = Object.keys(headerRow);
  const contentCols = allKeys.filter((k) => {
    const lower = k.toLowerCase();
    if (!/^columna|columnb|columnc|columnd|columne|columnf|columng|columnh|columni|columnj|columnk|columnl|columnm|columnn|columno|columnp|columnq|columnr|columns|columnt|columnu|columnv|columnw|columnx|columny|columnz$/.test(
      lower
    )) {
      return false;
    }
    const headerVal = headerRow[k];
    return headerVal && String(headerVal).trim() !== "";
  });

  const logicalRows = dataRows.map((row) => {
    const logical = {};
    for (const col of contentCols) {
      const headerLabel = String(headerRow[col] || "").trim();
      if (!headerLabel) continue;
      logical[headerLabel] = row[col];
    }
    logical._row_index = row.row_index;
    return logical;
  });

  return { headerRow, logicalRows };
}

function detectTableSubtype(chunk, logicalRows, headerRow) {
  const caption = (chunk.caption || "").toLowerCase();
  const section = (chunk.section_path || "").toLowerCase();
  const headers = logicalRows.length
    ? Object.keys(logicalRows[0]).map((h) => h.toLowerCase())
    : [];

  const hasHeader = (pattern) => headers.some((h) => pattern.test(h));

  // 1) Dosing tables (adult + pediatric)
  if (
    /dose|dosage|mg\/kg|mg\/ day|mg per kg|weight band|weight-band/.test(
      caption
    ) ||
    headers.some((h) => /kg/.test(h))
  ) {
    if (
      /child|paediatric|pediatric|infant|neonate/.test(caption) ||
      section.includes("child")
    ) {
      return "peds_dosing";
    }
    return "dosing";
  }

  // 2) Eligibility / decision / IF-THEN tables
  if (
    /eligib|criteria|if.*then|decision|indication|when to/.test(caption) ||
    hasHeader(/criteria|recommendation|option|action/)
  ) {
    return "decision";
  }

  // 3) Regimen composition tables (need regimen + drug-ish header)
  if (
    hasHeader(/regimen/) &&
    hasHeader(/drug|component|medicine|combination/)
  ) {
    return "regimen";
  }

  // 4) Timeline / monitoring tables
  if (
    /monitoring|follow-up|follow up|timeline|schedule/.test(caption) ||
    headers.some((h) => /baseline|month|week|visit|timepoint/.test(h))
  ) {
    return "timeline";
  }

  // 5) Drug-drug interaction tables
  if (
    /interaction|drug-drug|drug drug|qt|contraindication/.test(caption) ||
    headers.some((h) => /interaction|effect|recommendation/.test(h))
  ) {
    return "interaction";
  }

  // 6) Toxicity / adverse event management
  if (
    /toxicity|adverse event|side effect|hepatotox|neuropathy|ae management/.test(
      caption
    ) ||
    headers.some((h) => /grade|toxicity|ctcae/.test(h))
  ) {
    return "toxicity";
  }

  return "generic";
}


// ---- Helpers + renderers for high-value table types ----

function nonEmpty(val) {
  return val !== undefined && val !== null && String(val).trim() !== "";
}

function getTableHeaders(logicalRows) {
  if (!logicalRows || !logicalRows.length) return [];
  return Object.keys(logicalRows[0] || {}).filter((h) => h !== "_row_index");
}

// ---- Timeline helpers ----
function looksTimeLike(value) {
  if (!value) return false;
  const s = String(value).toLowerCase();

  if (
    /baseline|month|week|day|visit|timepoint|end of treatment|posttreatment|follow[- ]?up/.test(
      s
    )
  ) {
    return true;
  }

  if (/(month|week|day)\s*\d+/.test(s)) return true;

  return false;
}

function guessTimelineOrientation(logicalRows) {
  const headers = getTableHeaders(logicalRows);
  if (!headers.length) return { orientation: "unknown" };

  // A) timepoints in COLUMNS → several time-like headers
  const timeHeaders = headers.filter((h) => looksTimeLike(h));
  if (timeHeaders.length >= 2) {
    const entityHeader =
      headers.find((h) => !timeHeaders.includes(h)) || headers[0];
    return { orientation: "cols", entityHeader, timeHeaders };
  }

  // B) timepoints in ROWS → first column values look time-like
  const firstCol = headers[0];
  const sampleRows = logicalRows.slice(0, 6);
  const timeLikeCount = sampleRows.filter((r) => looksTimeLike(r[firstCol]))
    .length;

  if (timeLikeCount >= 2) {
    const entityHeaders = headers.slice(1);
    return { orientation: "rows", timeKey: firstCol, entityHeaders };
  }

  // C) unknown / weird table
  return { orientation: "unknown" };
}

// ---- Renderers for high-value table types ----

function renderDosingTable(chunk, logicalRows, headerRow) {
  const caption = chunk.caption || "Dosing table";
  const headers = getTableHeaders(logicalRows);
  if (!logicalRows.length || !headers.length) {
    return {
      text: `${caption}. (No rows found.)`,
      debug: { renderer: "dosing", header_keys: headers, row_count: logicalRows.length }
    };
  }

  const medKey =
    headers.find((h) =>
      /medicine|drug|product|regimen|group|tb drug/i.test(h.toLowerCase())
    ) || headers[0];

  const weightBandKeys = headers.filter((h) =>
    /(kg|weight band|weight-band|weight range| to <| to ≤|≥|<=|>=)/i.test(h)
  );

  // If we couldn't confidently find weight-band columns, just fall back to generic.
  if (!weightBandKeys.length) {
    return {
      text: renderGenericTable(chunk, logicalRows, headerRow),
      debug: {
        renderer: "dosing",
        header_keys: headers,
        row_count: logicalRows.length,
        note: "fallback_generic_no_weight_bands"
      }
    };
  }

  const lines = [];
  lines.push(`${caption}. Weight-band dosing summary:`);

  logicalRows.forEach((row) => {
    const med = row[medKey];
    if (!nonEmpty(med)) return;

    const bandParts = weightBandKeys
      .map((k) => {
        const val = row[k];
        if (!nonEmpty(val)) return null;
        return `${k}: ${String(val).trim()}`;
      })
      .filter(Boolean);

    if (!bandParts.length) {
      const cells = headers
        .map((h) => {
          const v = row[h];
          if (!nonEmpty(v)) return null;
          return `${h}: ${String(v).trim()}`;
        })
        .filter(Boolean);
      if (!cells.length) return;
      lines.push(`${med}: ${cells.join("; ")}`);
      return;
    }

    lines.push(`${med}: ${bandParts.join("; ")}`);
  });

  return {
    text: lines.join("\n"),
    debug: {
      renderer: "dosing",
      header_keys: headers,
      row_count: logicalRows.length,
      med_key: medKey,
      weight_band_keys: weightBandKeys
    }
  };
}

function renderPedsDosingTable(chunk, logicalRows, headerRow) {
  const caption = chunk.caption || "Paediatric dosing table";
  const headers = getTableHeaders(logicalRows);
  if (!logicalRows.length || !headers.length) {
    return {
      text: `${caption}. (No rows found.)`,
      debug: { renderer: "peds_dosing", header_keys: headers, row_count: logicalRows.length }
    };
  }

  const medKey =
    headers.find((h) =>
      /medicine|drug|product|regimen|group|tb drug/i.test(h.toLowerCase())
    ) || headers[0];

  const formKey =
    headers.find((h) =>
      /formulation|form|dispersible|fdc/i.test(h.toLowerCase())
    ) || null;

  const weightBandKeys = headers.filter((h) =>
    /(kg|weight band|weight-band|weight range| to <| to ≤|≥|<=|>=)/i.test(h)
  );

  if (!weightBandKeys.length) {
    return {
      text: renderGenericTable(chunk, logicalRows, headerRow),
      debug: {
        renderer: "peds_dosing",
        header_keys: headers,
        row_count: logicalRows.length,
        note: "fallback_generic_no_weight_bands"
      }
    };
  }

  const lines = [];
  lines.push(`${caption}. Paediatric weight-band dosing summary:`);

  logicalRows.forEach((row) => {
    const med = row[medKey];
    if (!nonEmpty(med)) return;

    const form = formKey ? row[formKey] : null;

    const bandParts = weightBandKeys
      .map((k) => {
        const v = row[k];
        if (!nonEmpty(v)) return null;
        return `${k}: ${String(v).trim()}`;
      })
      .filter(Boolean);

    if (!bandParts.length) {
      const cells = headers
        .map((h) => {
          const v = row[h];
          if (!nonEmpty(v)) return null;
          return `${h}: ${String(v).trim()}`;
        })
        .filter(Boolean);
      if (!cells.length) return;
      let line = `${med}`;
      if (form) line += ` (${form})`;
      line += ` — ${cells.join("; ")}`;
      lines.push(line);
      return;
    }

    let line = `${med}`;
    if (form) line += ` (${form})`;
    line += ` — ${bandParts.join("; ")}`;
    lines.push(line);
  });

  return {
    text: lines.join("\n"),
    debug: {
      renderer: "peds_dosing",
      header_keys: headers,
      row_count: logicalRows.length,
      med_key: medKey,
      form_key: formKey,
      weight_band_keys: weightBandKeys
    }
  };
}

function renderRegimenTable(chunk, logicalRows, headerRow) {
  const caption = chunk.caption || "Regimen composition table";
  const headers = getTableHeaders(logicalRows);
  if (!logicalRows.length || !headers.length) {
    return {
      text: `${caption}. (No rows found.)`,
      debug: { renderer: "regimen", header_keys: headers, row_count: logicalRows.length }
    };
  }

  const regimenKey =
    headers.find((h) => /regimen|name|strategy|option/i.test(h.toLowerCase())) ||
    headers[0];

  const drugsKey =
    headers.find((h) => /drug|component|medicine|composition/i.test(h.toLowerCase())) ||
    null;

  const durationKey =
    headers.find((h) =>
      /duration|months|weeks|days|length of treatment/i.test(h.toLowerCase())
    ) || null;

  const lines = [];
  lines.push(`${caption}. Regimen components and options:`);

  logicalRows.forEach((row, idx) => {
    const regimen = row[regimenKey];
    if (!nonEmpty(regimen)) {
      const cells = headers
        .map((h) => {
          const v = row[h];
          if (!nonEmpty(v)) return null;
          return `${h}: ${String(v).trim()}`;
        })
        .filter(Boolean);
      if (!cells.length) return;
      lines.push(`Row ${idx + 1}: ${cells.join("; ")}`);
      return;
    }

    const parts = [];
    if (drugsKey && nonEmpty(row[drugsKey])) {
      parts.push(String(row[drugsKey]).trim());
    }
    if (durationKey && nonEmpty(row[durationKey])) {
      parts.push(`duration: ${String(row[durationKey]).trim()}`);
    }

    if (!parts.length) {
      const cells = headers
        .map((h) => {
          const v = row[h];
          if (!nonEmpty(v)) return null;
          return `${h}: ${String(v).trim()}`;
        })
        .filter(Boolean);
      lines.push(`${regimen}: ${cells.join("; ")}`);
    } else {
      lines.push(`${regimen} — ${parts.join("; ")}`);
    }
  });

  return {
    text: lines.join("\n"),
    debug: {
      renderer: "regimen",
      header_keys: headers,
      row_count: logicalRows.length,
      regimen_key: regimenKey,
      drugs_key: drugsKey,
      duration_key: durationKey
    }
  };
}

function renderDecisionTable(chunk, logicalRows, headerRow) {
  const caption = chunk.caption || "Decision table";
  const headers = getTableHeaders(logicalRows);
  if (!logicalRows.length || !headers.length) {
    return {
      text: `${caption}. (No rows found.)`,
      debug: { renderer: "decision", header_keys: headers, row_count: logicalRows.length }
    };
  }

  const isConditionHeader = (h) =>
    /if|criteria|condition|situation|scenario|finding|result|status|baseline|risk/i.test(
      h.toLowerCase()
    );
  const isActionHeader = (h) =>
    /then|recommendation|action|management|treatment|decision|next step|regimen/i.test(
      h.toLowerCase()
    );

  const lines = [];
  lines.push(`${caption}. IF–THEN decision rules:`);

  let rowsWithIfThen = 0;

  logicalRows.forEach((row, idx) => {
    const conditionParts = [];
    const actionParts = [];
    const otherParts = [];

    headers.forEach((h) => {
      const v = row[h];
      if (!nonEmpty(v)) return;
      const label = `${h}: ${String(v).trim()}`;

      if (isActionHeader(h)) actionParts.push(label);
      else if (isConditionHeader(h)) conditionParts.push(label);
      else otherParts.push(label);
    });

    if (!conditionParts.length && !actionParts.length) {
      if (!otherParts.length) return;
      lines.push(`Row ${idx + 1}: ${otherParts.join("; ")}`);
      return;
    }

    if (!conditionParts.length && otherParts.length) {
      conditionParts.push(...otherParts);
    } else if (!actionParts.length && otherParts.length) {
      actionParts.push(...otherParts);
    }

    const condText = conditionParts.length
      ? conditionParts.join("; ")
      : "the criteria in this row are met";
    const actionText = actionParts.length
      ? actionParts.join("; ")
      : "see other columns in this row for recommended action";

    lines.push(`IF ${condText} THEN ${actionText}`);
    rowsWithIfThen += 1;
  });

  const debug = {
    renderer: "decision",
    header_keys: headers,
    row_count: logicalRows.length,
    rows_with_if_then: rowsWithIfThen
  };

  if (rowsWithIfThen === 0) {
    return {
      text: renderGenericTable(chunk, logicalRows, headerRow),
      debug: { ...debug, note: "fallback_generic_no_if_then_rows" }
    };
  }

  return {
    text: lines.join("\n"),
    debug
  };
}

function renderTimelineTable(chunk, logicalRows, headerRow) {
  const caption = chunk.caption || "Monitoring schedule";
  const headers = getTableHeaders(logicalRows);
  if (!logicalRows.length || !headers.length) {
    return {
      text: `${caption}. (No rows found.)`,
      debug: { renderer: "timeline", header_keys: headers, row_count: logicalRows.length }
    };
  }

  const orientationInfo = guessTimelineOrientation(logicalRows);
  const { orientation, entityHeader, timeHeaders, timeKey, entityHeaders } =
    orientationInfo;

  const lines = [];
  lines.push(`${caption}. Follow-up schedule over time:`);

  // A) Timepoints in COLUMNS (WHO Table 2.9.2 style)
  if (orientation === "cols") {
    logicalRows.forEach((row) => {
      const entity = row[entityHeader];
      if (!nonEmpty(entity)) return;

      const timeParts = timeHeaders
        .map((h) => {
          const v = row[h];
          if (!nonEmpty(v)) return null;
          return `${h}: ${String(v).trim()}`;
        })
        .filter(Boolean);
      if (!timeParts.length) return;

      lines.push(`For ${entity}: ${timeParts.join("; ")}`);
    });

    if (lines.length > 1) {
      return {
        text: lines.join("\n"),
        debug: {
          renderer: "timeline",
          header_keys: headers,
          row_count: logicalRows.length,
          orientation,
          entity_header: entityHeader,
          time_headers: timeHeaders
        }
      };
    }
  }

  // B) Timepoints in ROWS (time down first column)
  if (orientation === "rows") {
    logicalRows.forEach((row) => {
      const when = row[timeKey];
      if (!nonEmpty(when)) return;

      const parts = entityHeaders
        .map((h) => {
          const v = row[h];
          if (!nonEmpty(v)) return null;
          return `${h}: ${String(v).trim()}`;
        })
        .filter(Boolean);
      if (!parts.length) return;

      lines.push(`At ${when}: ${parts.join("; ")}`);
    });

    if (lines.length > 1) {
      return {
        text: lines.join("\n"),
        debug: {
          renderer: "timeline",
          header_keys: headers,
          row_count: logicalRows.length,
          orientation,
          time_key: timeKey,
          entity_headers: entityHeaders
        }
      };
    }
  }

  // C) Unknown pattern → generic safe summary
  return {
    text: renderGenericTable(chunk, logicalRows, headerRow),
    debug: {
      renderer: "timeline",
      header_keys: headers,
      row_count: logicalRows.length,
      orientation: orientationInfo.orientation,
      note: "fallback_generic_unknown_orientation"
    }
  };
}

function renderInteractionTable(chunk, logicalRows, headerRow) {
  const caption = chunk.caption || "Drug–drug interaction table";
  const headers = getTableHeaders(logicalRows);
  if (!logicalRows.length || !headers.length) {
    return {
      text: `${caption}. (No rows found.)`,
      debug: { renderer: "interaction", header_keys: headers, row_count: logicalRows.length }
    };
  }

  const isDrugHeader = (h) =>
    /drug ?1|drug ?2|drug a|drug b|medicine 1|medicine 2|comedication|arv|antiretroviral|tb drug|rifampin|rifampicin|rifapentine/i.test(
      h.toLowerCase()
    ) || /drug|medicine|regimen/i.test(h.toLowerCase());

  const effectKey =
    headers.find((h) =>
      /interaction|effect|impact|change in level/i.test(h.toLowerCase())
    ) || null;

  const recKey =
    headers.find((h) =>
      /recommendation|management|action|dose adjustment|avoid/i.test(
        h.toLowerCase()
      )
    ) || null;

  const drugHeaders = headers.filter(isDrugHeader);

  if (!drugHeaders.length) {
    return {
      text: renderGenericTable(chunk, logicalRows, headerRow),
      debug: {
        renderer: "interaction",
        header_keys: headers,
        row_count: logicalRows.length,
        note: "fallback_generic_no_drug_headers"
      }
    };
  }

  const lines = [];
  lines.push(`${caption}. Drug combinations and recommendations:`);

  let rowsWithCombos = 0;

  logicalRows.forEach((row, idx) => {
    const drugs = drugHeaders
      .map((h) => row[h])
      .filter(nonEmpty)
      .map((v) => String(v).trim());

    const effect = effectKey && nonEmpty(row[effectKey])
      ? String(row[effectKey]).trim()
      : null;
    const rec = recKey && nonEmpty(row[recKey])
      ? String(row[recKey]).trim()
      : null;

    if (!drugs.length && !effect && !rec) {
      const cells = headers
        .map((h) => {
          const v = row[h];
          if (!nonEmpty(v)) return null;
          return `${h}: ${String(v).trim()}`;
        })
        .filter(Boolean);
      if (!cells.length) return;
      lines.push(`Row ${idx + 1}: ${cells.join("; ")}`);
      return;
    }

    let line = "";
    if (drugs.length) line += `Combination ${drugs.join(" + ")}`;
    if (effect) line += (line ? " — " : "") + `effect: ${effect}`;
    if (rec) line += (line ? "; " : "") + `recommendation: ${rec}`;

    lines.push(line || `Row ${idx + 1}: (see table)`);
    if (drugs.length) rowsWithCombos += 1;
  });

  const debug = {
    renderer: "interaction",
    header_keys: headers,
    row_count: logicalRows.length,
    drug_headers: drugHeaders,
    effect_key: effectKey,
    recommendation_key: recKey,
    rows_with_combos: rowsWithCombos
  };

  if (rowsWithCombos === 0) {
    return {
      text: renderGenericTable(chunk, logicalRows, headerRow),
      debug: { ...debug, note: "fallback_generic_no_combos" }
    };
  }

  return {
    text: lines.join("\n"),
    debug
  };
}

function renderToxicityTable(chunk, logicalRows, headerRow) {
  const caption = chunk.caption || "Toxicity / adverse event table";
  const headers = getTableHeaders(logicalRows);
  if (!logicalRows.length || !headers.length) {
    return {
      text: `${caption}. (No rows found.)`,
      debug: { renderer: "toxicity", header_keys: headers, row_count: logicalRows.length }
    };
  }

  const gradeKey =
    headers.find((h) => /grade|severity|ctcae/i.test(h.toLowerCase())) ||
    headers[0];
  const descKey =
    headers.find((h) =>
      /description|finding|toxicity|event|symptom/i.test(h.toLowerCase())
    ) || null;
  const mgmtKey =
    headers.find((h) =>
      /management|action|recommendation|dose adjustment|stop/i.test(
        h.toLowerCase()
      )
    ) || null;

  const lines = [];
  lines.push(`${caption}. Toxicity grades and management:`);

  let rowsWithGrades = 0;

  logicalRows.forEach((row, idx) => {
    const grade = row[gradeKey];
    const desc = descKey ? row[descKey] : null;
    const mgmt = mgmtKey ? row[mgmtKey] : null;

    if (!nonEmpty(grade) && !nonEmpty(desc) && !nonEmpty(mgmt)) {
      const cells = headers
        .map((h) => {
          const v = row[h];
          if (!nonEmpty(v)) return null;
          return `${h}: ${String(v).trim()}`;
        })
        .filter(Boolean);
      if (!cells.length) return;
      lines.push(`Row ${idx + 1}: ${cells.join("; ")}`);
      return;
    }

    const parts = [];
    if (nonEmpty(desc)) parts.push(String(desc).trim());
    if (nonEmpty(mgmt)) parts.push(`management: ${String(mgmt).trim()}`);

    if (!parts.length) {
      lines.push(`Grade ${grade}: see table row ${idx + 1}`);
    } else {
      lines.push(`Grade ${grade}: ${parts.join(" — ")}`);
    }

    if (nonEmpty(grade)) rowsWithGrades += 1;
  });

  const debug = {
    renderer: "toxicity",
    header_keys: headers,
    row_count: logicalRows.length,
    grade_key: gradeKey,
    description_key: descKey,
    management_key: mgmtKey,
    rows_with_grades: rowsWithGrades
  };

  if (rowsWithGrades === 0) {
    return {
      text: renderGenericTable(chunk, logicalRows, headerRow),
      debug: { ...debug, note: "fallback_generic_no_grades" }
    };
  }

  return {
    text: lines.join("\n"),
    debug
  };
}

function renderGenericTable(chunk, logicalRows, headerRow) {
  const caption = chunk.caption || "Table";
  const headers = getTableHeaders(logicalRows);
  if (!logicalRows.length || !headers.length) {
    return {
      text: `${caption}. (No rows found.)`,
      debug: { renderer: "generic", header_keys: headers, row_count: logicalRows.length }
    };
  }

  const lines = [];
  lines.push(`${caption}. Columns: ${headers.join(", ")}`);

  logicalRows.forEach((row, idx) => {
    const cells = headers
      .map((h) => `${h}: ${row[h] ?? ""}`)
      .join("; ");
    lines.push(`Row ${idx + 1}: ${cells}`);
  });

  return {
    text: lines.join("\n"),
    debug: { renderer: "generic", header_keys: headers, row_count: logicalRows.length }
  };
}

function renderTable(chunk, rawRows) {
  const { headerRow, logicalRows } = normalizeTableRows(rawRows);
  const subtype = detectTableSubtype(chunk, logicalRows, headerRow) || "generic";

  // Base debug info that is always present
  const baseDebug = {
    subtype_guess: subtype,
    row_count: logicalRows.length,
    has_header_row: !!headerRow
  };

  switch (subtype) {
    case "dosing": {
      const { text, debug } = renderDosingTable(chunk, logicalRows, headerRow);
      return { subtype, tableText: text, debug: { ...baseDebug, ...debug } };
    }
    case "peds_dosing": {
      const { text, debug } = renderPedsDosingTable(chunk, logicalRows, headerRow);
      return { subtype, tableText: text, debug: { ...baseDebug, ...debug } };
    }
    case "regimen": {
      const { text, debug } = renderRegimenTable(chunk, logicalRows, headerRow);
      return { subtype, tableText: text, debug: { ...baseDebug, ...debug } };
    }
    case "decision": {
      const { text, debug } = renderDecisionTable(chunk, logicalRows, headerRow);
      return { subtype, tableText: text, debug: { ...baseDebug, ...debug } };
    }
    case "timeline": {
      const { text, debug } = renderTimelineTable(chunk, logicalRows, headerRow);
      return { subtype, tableText: text, debug: { ...baseDebug, ...debug } };
    }
    case "interaction": {
      const { text, debug } = renderInteractionTable(chunk, logicalRows, headerRow);
      return { subtype, tableText: text, debug: { ...baseDebug, ...debug } };
    }
    case "toxicity": {
      const { text, debug } = renderToxicityTable(chunk, logicalRows, headerRow);
      return { subtype, tableText: text, debug: { ...baseDebug, ...debug } };
    }
    default: {
      const { text, debug } = renderGenericTable(chunk, logicalRows, headerRow);
      return {
        subtype: "generic",
        tableText: text,
        debug: { ...baseDebug, ...debug }
      };
    }
  }
}


function enrichChunkWithTable(chunk) {
  const ct = (chunk.content_type || "").toLowerCase();
  const attachmentPath = chunk.attachment_path;

  if (ct !== "table" || !attachmentPath) {
    return chunk;
  }

  try {
    const rawRows = loadTableRows(attachmentPath);
    const { subtype, tableText, debug } = renderTable(chunk, rawRows);

    return {
      ...chunk,
      table_subtype: subtype,
      table_text: tableText,
      table_rows: rawRows,
      table_row_count: Array.isArray(rawRows) ? rawRows.length : 0,
      table_debug: debug || null
    };
  } catch (err) {
    console.error(
      "Failed to load or render table for chunk",
      chunk.chunk_id,
      "path:",
      attachmentPath,
      err
    );
    return chunk;
  }
}

// ---------- Main handler ----------

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    res.statusCode = 405;
    res.setHeader("Content-Type", "application/json");
    res.setHeader("Allow", "POST");
    res.end(JSON.stringify({ error: "Use POST to query the TB RAG store." }));
    return;
  }

  try {
    const body =
      typeof req.body === "string" ? JSON.parse(req.body) : req.body || {};

    const question = body.question;
    let finalTopK = typeof body.top_k === "number" ? body.top_k : 8;
    let scope = body.scope || null;
    let documentHint =
      typeof body.document_hint === "string" ? body.document_hint : null;

    if (!scope) {
      scope = inferScopeFromQuestion(question);
    }
    if (!documentHint) {
      documentHint = inferDocHintFromQuestion(question);
    }

    if (!question || typeof question !== "string" || !question.trim()) {
      res.statusCode = 400;
      res.setHeader("Content-Type", "application/json");
      res.end(
        JSON.stringify({
          error: "Missing or empty 'question' string in request body."
        })
      );
      return;
    }

    finalTopK = Math.max(1, Math.min(finalTopK, 8));

    const { chunks, embeddings } = loadRagStore();

    if (!embeddings.length || !chunks.length) {
      throw new Error("RAG store is empty or failed to load.");
    }

    finalTopK = Math.min(finalTopK, embeddings.length);

    const qEmbedding = await embedQuestion(question);

    console.log("Query embedding length:", qEmbedding.length);
    console.log("First chunk embedding length:", embeddings[0].length);
    console.log("Total chunks:", embeddings.length);

    const fullIndices = embeddings.map((_, idx) => idx);
    let scopedIndices = filterIndicesByScope(fullIndices, chunks, scope);
    if (!scopedIndices.length) {
      scopedIndices = fullIndices;
    }

    let indices = filterIndicesByDocHint(scopedIndices, chunks, documentHint);
    if (!indices.length) {
      indices = scopedIndices;
    }

    // Dual-channel retrieval: text/prose vs tables
    const textIndices = indices.filter((i) => {
      const ct = (chunks[i].content_type || "").toLowerCase();
      return ct !== "table";
    });

    const tableIndices = indices.filter((i) => {
      const ct = (chunks[i].content_type || "").toLowerCase();
      return ct === "table";
    });

    let scoredText = textIndices.map((idx) => ({
      index: idx,
      score: cosineSim(qEmbedding, embeddings[idx])
    }));
    scoredText.sort((a, b) => b.score - a.score);

    let scoredTables = tableIndices.map((idx) => ({
      index: idx,
      score: cosineSim(qEmbedding, embeddings[idx])
    }));
    scoredTables.sort((a, b) => b.score - a.score);

    // --- Optional table-aware boosting based on section proximity ---
    const anchorCount = Math.min(10, scoredText.length);
    const anchors = scoredText.slice(0, anchorCount).map(({ index, score }) => {
      const c = chunks[index] || {};
      return {
        index,
        score,
        doc_id: c.doc_id || null,
        section_path: c.section_path || "",
        content_type: (c.content_type || "").toLowerCase(),
        sectionKeys: extractSectionKeys(c.section_path || "")
      };
    });

    const anchorDocSectionMap = [];
    for (const a of anchors) {
      if (!a.doc_id) continue;
      anchorDocSectionMap.push(a);
    }

    const maxTextScore = scoredText.length ? scoredText[0].score : 1.0;

    // Boost table chunks that share a doc + section key with top text anchors
    for (const entry of scoredTables) {
      const c = chunks[entry.index] || {};
      const tableDoc = c.doc_id || null;
      if (!tableDoc) continue;

      const tableKeys = extractSectionKeys(c.section_path || "");
      let isNeighbor = false;
      for (const a of anchorDocSectionMap) {
        if (a.doc_id !== tableDoc) continue;
        if (a.sectionKeys.some((key) => tableKeys.includes(key))) {
          isNeighbor = true;
          break;
        }
      }

      if (!isNeighbor) continue;
      entry.score = Math.max(entry.score, maxTextScore * 0.98);
    }

    scoredTables.sort((a, b) => b.score - a.score);
    // --- End table-aware boosting ---

    // Take top-N from each channel before merging
    const TEXT_LIMIT = 10;
    const TABLE_LIMIT = 5;

    const topText = scoredText.slice(
      0,
      Math.min(TEXT_LIMIT, scoredText.length)
    );
    const topTables = scoredTables.slice(
      0,
      Math.min(TABLE_LIMIT, scoredTables.length)
    );

    let combined = topText.concat(topTables);
    combined.sort((a, b) => b.score - a.score);

    const seen = new Set();
    const deduped = [];
    for (const entry of combined) {
      const c = chunks[entry.index] || {};
      const id = c.chunk_id;
      if (!id || seen.has(id)) continue;
      seen.add(id);
      deduped.push(entry);
    }
    }

    const top = deduped.slice(0, finalTopK);

    const results = top.map(({ index, score }) => {
      const baseChunk = chunks[index] || {};
      const c = enrichChunkWithTable(baseChunk);

      return {
        doc_id: c.doc_id,
        guideline_title: c.guideline_title ?? null,
        year: c.year ?? null,
        chunk_id: c.chunk_id,
        section_path: c.section_path,
        text: c.text,
        content_type: c.content_type ?? null,
        attachment_id: c.attachment_id ?? null,
        attachment_path: c.attachment_path ?? null,
        table_subtype: c.table_subtype ?? null,
        table_text: c.table_text ?? null,
        table_rows: c.table_rows ?? null,
        score
      };
    });

    res.statusCode = 200;
    res.setHeader("Content-Type", "application/json");
    res.end(
      JSON.stringify({
        question,
        top_k: finalTopK,
        scope: scope || null,
        document_hint: documentHint || null,
        results
      })
    );
  } catch (err) {
    console.error(err);
    res.statusCode = 500;
    res.setHeader("Content-Type", "application/json");
    res.end(
      JSON.stringify({
        error: String(err.message || err)
      })
    );
  }
};