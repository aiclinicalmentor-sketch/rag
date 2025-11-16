// api/tb-rag-query.js
//
// Vercel Node function that powers the TB Clinical Mentor RAG Action.
// Loads precomputed embeddings + chunks from public/rag and returns the
// top-K passages for a clinician question. Now table-aware: it can load
// CSV tables from public/rag/tables and enrich table chunks with
// GPT-ready summaries and raw rows.

const fs = require("fs");
const path = require("path");
const Papa = require("papaparse");

let RAG_STORE = null;

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
  const embeddingsPath = path.join(ragDir, "embeddings.json"); // from your Python conversion

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

function filterIndicesByScope(indices, chunks, scope) {
  if (!scope) return indices;

  const s = String(scope).toLowerCase();

  return indices.filter((i) => {
    const c = chunks[i] || {};
    const docId = (c.doc_id || "").toString().toLowerCase();
    const section = (c.section_path || "").toString().toLowerCase();
    const chunkScope = (c.scope || "").toString().toLowerCase();

    // If chunks have an explicit scope field, prefer that.
    if (chunkScope) {
      return chunkScope === s;
    }

    // Otherwise, use simple heuristics based on doc_id / section_path.
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
      (c.scope || "").toString().toLowerCase(),
    ];

    return haystacks.some((hay) =>
      hay && hintTokens.some((token) => hay.includes(token))
    );
  });

  return filtered.length ? filtered : indices;
}

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
    "ultra",
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
    "drug-resistant",
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
    "side effects",
  ]);

  const pediScore = keywordScore(q, [
    "child",
    "children",
    "paediatric",
    "pediatric",
    "adolescent",
    "infant",
    "neonate",
  ]);

  const programScore = keywordScore(q, [
    "program",
    "programmatic",
    "supply chain",
    "implementation",
    "health system",
    "adherence support",
    "community",
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

const DOC_HINT_ALIASES = [
  {
    target: "module3_diagnosis",
    keywords: ["module 3", "module3", "module iii"],
    requires: ["diagnos", "algorithm", "cxr"],
  },
  {
    target: "module4_treatment",
    keywords: ["module 4", "module4", "module iv"],
    requires: ["treat", "regimen", "therapy"],
  },
  {
    target: "consolidated_module3",
    keywords: ["consolidated", "module 3"],
    requires: ["diagnos"],
  },
];

function extractSectionKeys(sectionPath) {
  if (!sectionPath) return [];
  const s = String(sectionPath).toLowerCase();

  const keys = new Set();

  // Whole string as a coarse key
  keys.add(s.trim());

  // Individual segments split by '|'
  const segments = s.split("|").map((seg) => seg.trim()).filter(Boolean);
  for (const seg of segments) {
    keys.add(seg);
  }

  // Extract numeric section patterns like "5.2", "2.5.2", etc.
  const numMatches = s.match(/\b\d+(?:\.\d+)*\b/g);
  if (numMatches) {
    for (const m of numMatches) {
      keys.add(m);
    }
  }

  return Array.from(keys);
}

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

// ---------- Table loading + subtype detection + rendering ----------

// Resolve an attachment_path from chunk metadata to an absolute CSV path
// Deployed CSVs live under: public/rag/tables/<guideline>/Table_X.csv
function resolveTablePath(attachmentPathFromMeta) {
  if (!attachmentPathFromMeta) return null;

  // Normalize slashes
  let cleaned = String(attachmentPathFromMeta).replace(/\\\\/g, "/").replace(/\\/g, "/");

  // Remove leading ./ or ../ if present
  cleaned = cleaned.replace(/^\.\/+/, "");
  cleaned = cleaned.replace(/^\.\.\//, "");

  // If it doesn't already start with "public/", prepend it
  if (!cleaned.startsWith("public/")) {
    cleaned = path.posix.join("public", cleaned);
  }

  const absolutePath = path.join(process.cwd(), cleaned);
  return absolutePath;
}

function loadTableRows(attachmentPathFromMeta) {
  const absPath = resolveTablePath(attachmentPathFromMeta);
  if (!absPath) {
    throw new Error("Cannot resolve table path from attachment_path");
  }

  const csv = fs.readFileSync(absPath, "utf8");
  const parsed = Papa.parse(csv, {
    header: true,
    skipEmptyLines: true,
  });

  if (parsed.errors && parsed.errors.length) {
    console.warn("CSV parse errors for table:", absPath, parsed.errors);
  }

  return parsed.data || [];
}

function detectTableSubtype(chunk, rows) {
  const caption = (chunk.caption || "").toLowerCase();
  const section = (chunk.section_path || "").toLowerCase();
  const headers = rows.length ? Object.keys(rows[0]).map((h) => h.toLowerCase()) : [];

  const hasHeader = (pattern) => headers.some((h) => pattern.test(h));

  // 1) Dosing tables (adult + pediatric)
  if (
    /dose|dosing|mg\/kg|weight-based|weight band|weight-band/.test(caption) ||
    hasHeader(/dose|mg\/kg|weight|kg|tablet/)
  ) {
    // Distinguish peds formulations if obviously pediatric
    if (
      /child|paediatric|pediatric|infant|neonate/.test(caption) ||
      section.includes("child") ||
      hasHeader(/dispersible|fdc|rh?z/)
    ) {
      return "peds_dosing";
    }
    return "dosing";
  }

  // 2) Regimen composition tables
  if (
    /regimen|composition|components|backbone/.test(caption) ||
    hasHeader(/regimen|drugs?|medicine/)
  ) {
    return "regimen";
  }

  // 3) Eligibility / decision / IF-THEN tables
  if (
    /eligib|criteria|if.*then|decision|indication|when to/.test(caption) ||
    hasHeader(/criteria|if|then|recommendation|option/)
  ) {
    return "decision";
  }

  // 4) Timeline / monitoring tables
  if (
    /monitoring|follow-up|follow up|timeline|schedule/.test(caption) ||
    hasHeader(/month|week|visit|timepoint|baseline/)
  ) {
    return "timeline";
  }

  // 5) Drug-drug interaction tables
  if (
    /interaction|drug-drug|drug drug|qt|contraindication/.test(caption) ||
    hasHeader(/drug_a|drug_b|drug1|drug2|interaction|effect/)
  ) {
    return "interaction";
  }

  // 6) Toxicity / adverse event management
  if (
    /toxicity|adverse event|side effect|hepatotox|neuropathy|ae management/.test(caption) ||
    hasHeader(/grade|toxicity|ctcae/)
  ) {
    return "toxicity";
  }

  return "generic";
}

// ---- Renderers for high-value table types ----

function renderDosingTable(chunk, rows) {
  const caption = chunk.caption || "Dosing table";
  const lines = [];
  lines.push(`${caption}. Weight- or age-based dosing:`);

  for (const row of rows) {
    const weightBand =
      row["weight_band"] ||
      row["Weight_band"] ||
      row["Weight band"] ||
      row["Weight (kg)"] ||
      row["weight"] ||
      "";
    const ageBand = row["Age_band"] || row["Age band"] || row["Age"] || "";
    const drug = row["drug"] || row["Drug"] || row["medicine"] || row["Medicine"] || "";
    const dose = row["dose"] || row["Dose"] || row["dose_mg"] || row["Dose (mg)"] || "";
    const freq =
      row["frequency"] ||
      row["Frequency"] ||
      row["freq"] ||
      row["Freq"] ||
      "";
    const notes = row["notes"] || row["Notes"] || row["comment"] || row["Comment"] || "";

    if (!drug && !dose) continue;

    let subject = weightBand
      ? `For weight band ${weightBand}`
      : ageBand
      ? `For age band ${ageBand}`
      : "Recommended dose";

    let line = `${subject}: ${drug ? drug + " " : ""}${dose || ""}`;
    if (freq) line += `, ${freq}`;
    if (notes) line += ` — ${notes}`;

    lines.push(line.trim());
  }

  return lines.join("\n");
}

function renderPedsDosingTable(chunk, rows) {
  const caption = chunk.caption || "Paediatric dosing table";
  const lines = [];
  lines.push(`${caption}. Paediatric weight-band dosing with formulations:`);

  for (const row of rows) {
    const weightBand =
      row["weight_band"] ||
      row["Weight_band"] ||
      row["Weight band"] ||
      row["Weight (kg)"] ||
      row["weight"] ||
      "";
    const formulation =
      row["formulation"] ||
      row["Formulation"] ||
      row["product"] ||
      row["Product"] ||
      "";
    const tablets =
      row["tablets"] ||
      row["Tablets"] ||
      row["tabs"] ||
      row["Tabs"] ||
      row["No. of tablets"] ||
      "";
    const totalDose =
      row["total_dose"] ||
      row["Total dose"] ||
      row["Total (mg)"] ||
      row["total_mg"] ||
      "";
    const notes = row["notes"] || row["Notes"] || "";

    if (!weightBand && !formulation && !totalDose && !tablets) continue;

    let line = `For weight band ${weightBand || "—"}:`;
    if (formulation) line += ` ${formulation}`;
    if (tablets) line += ` — ${tablets} tablet(s)`;
    if (totalDose) line += ` (total ${totalDose})`;
    if (notes) line += ` — ${notes}`;

    lines.push(line.trim());
  }

  return lines.join("\n");
}

function renderRegimenTable(chunk, rows) {
  const caption = chunk.caption || "Regimen composition table";
  const lines = [];
  lines.push(`${caption}. Regimen components and options:`);

  for (const row of rows) {
    const regimen =
      row["regimen"] ||
      row["Regimen"] ||
      row["Regimen name"] ||
      row["Name"] ||
      "";
    const components =
      row["components"] ||
      row["Components"] ||
      row["drugs"] ||
      row["Drugs"] ||
      row["backbone"] ||
      row["Backbone"] ||
      "";
    const duration =
      row["duration"] ||
      row["Duration"] ||
      row["phase_duration"] ||
      row["Phase duration"] ||
      "";
    const notes = row["notes"] || row["Notes"] || "";

    if (!regimen && !components) continue;

    let line = regimen ? `${regimen}: ` : "";
    if (components) line += `${components}`;
    if (duration) line += ` (duration: ${duration})`;
    if (notes) line += ` — ${notes}`;

    lines.push(line.trim());
  }

  return lines.join("\n");
}

function renderDecisionTable(chunk, rows) {
  const caption = chunk.caption || "Decision table";
  const lines = [];
  lines.push(`${caption}. IF–THEN decision rules:`);

  for (const row of rows) {
    const condition =
      row["if"] ||
      row["If"] ||
      row["criteria"] ||
      row["Criteria"] ||
      row["Condition"] ||
      "";
    const thenVal =
      row["then"] ||
      row["Then"] ||
      row["recommendation"] ||
      row["Recommendation"] ||
      row["Action"] ||
      "";
    const notes = row["notes"] || row["Notes"] || "";

    if (!condition && !thenVal) continue;

    let line = "IF ";
    line += condition || "the criteria in this row are met";
    line += " THEN ";
    line += thenVal || "see notes";
    if (notes) line += ` — Notes: ${notes}`;

    lines.push(line.trim());
  }

  return lines.join("\n");
}

function renderTimelineTable(chunk, rows) {
  const caption = chunk.caption || "Monitoring schedule";
  const lines = [];
  lines.push(`${caption}. Follow-up schedule over time:`);

  for (const row of rows) {
    const when =
      row["time_point"] ||
      row["Timepoint"] ||
      row["Time point"] ||
      row["Visit"] ||
      row["Month"] ||
      row["Week"] ||
      "";
    const tests =
      row["tests"] ||
      row["Tests"] ||
      row["investigations"] ||
      row["Investigations"] ||
      row["Monitoring"] ||
      "";
    const notes = row["notes"] || row["Notes"] || "";

    if (!when && !tests) continue;

    let line = `${when || "Time"}: ${tests || "—"}`;
    if (notes) line += ` — Notes: ${notes}`;

    lines.push(line.trim());
  }

  return lines.join("\n");
}

function renderInteractionTable(chunk, rows) {
  const caption = chunk.caption || "Drug–drug interaction table";
  const lines = [];
  lines.push(`${caption}. Drug combinations and recommendations:`);

  for (const row of rows) {
    const drugA =
      row["drug_a"] ||
      row["Drug_A"] ||
      row["Drug A"] ||
      row["Drug1"] ||
      row["drug1"] ||
      "";
    const drugB =
      row["drug_b"] ||
      row["Drug_B"] ||
      row["Drug B"] ||
      row["Drug2"] ||
      row["drug2"] ||
      "";
    const effect = row["effect"] || row["Effect"] || "";
    const recommendation =
      row["recommendation"] || row["Recommendation"] || row["Action"] || "";
    const notes = row["notes"] || row["Notes"] || "";

    if (!drugA && !drugB) continue;

    let line = `${drugA || "Drug A"} + ${drugB || "Drug B"}: `;
    if (effect) line += `${effect}. `;
    if (recommendation) line += `Recommendation: ${recommendation}. `;
    if (notes) line += `Notes: ${notes}.`;

    lines.push(line.trim());
  }

  return lines.join("\n");
}

function renderToxicityTable(chunk, rows) {
  const caption = chunk.caption || "Toxicity / adverse event table";
  const lines = [];
  lines.push(`${caption}. Toxicity grades and management:`);

  for (const row of rows) {
    const grade =
      row["grade"] ||
      row["Grade"] ||
      row["severity"] ||
      row["Severity"] ||
      "";
    const finding =
      row["finding"] ||
      row["Finding"] ||
      row["toxicity"] ||
      row["Toxicity"] ||
      row["Description"] ||
      "";
    const management =
      row["management"] ||
      row["Management"] ||
      row["action"] ||
      row["Action"] ||
      "";
    const notes = row["notes"] || row["Notes"] || "";

    if (!grade && !finding && !management) continue;

    let line = "";
    if (grade) line += `Grade ${grade}: `;
    if (finding) line += `${finding}. `;
    if (management) line += `Management: ${management}. `;
    if (notes) line += `Notes: ${notes}.`;

    lines.push(line.trim());
  }

  return lines.join("\n");
}

function renderGenericTable(chunk, rows) {
  const caption = chunk.caption || "Table";
  if (!rows || rows.length === 0) {
    return `${caption}. (No rows found in CSV.)`;
  }

  const headers = Object.keys(rows[0] || {});
  const lines = [];
  lines.push(`${caption}. Columns: ${headers.join(", ")}`);

  rows.forEach((row, idx) => {
    const cells = headers
      .map((h) => `${h}: ${row[h] ?? ""}`)
      .join("; ");
    lines.push(`Row ${idx + 1}: ${cells}`);
  });

  return lines.join("\n");
}

function renderTable(chunk, rows) {
  const subtype = detectTableSubtype(chunk, rows) || "generic";

  switch (subtype) {
    case "dosing":
      return { subtype, tableText: renderDosingTable(chunk, rows) };
    case "peds_dosing":
      return { subtype, tableText: renderPedsDosingTable(chunk, rows) };
    case "regimen":
      return { subtype, tableText: renderRegimenTable(chunk, rows) };
    case "decision":
      return { subtype, tableText: renderDecisionTable(chunk, rows) };
    case "timeline":
      return { subtype, tableText: renderTimelineTable(chunk, rows) };
    case "interaction":
      return { subtype, tableText: renderInteractionTable(chunk, rows) };
    case "toxicity":
      return { subtype, tableText: renderToxicityTable(chunk, rows) };
    default:
      return { subtype: "generic", tableText: renderGenericTable(chunk, rows) };
  }
}

function enrichChunkWithTable(chunk) {
  const ct = (chunk.content_type || "").toLowerCase();
  const attachmentPath = chunk.attachment_path;

  if (ct !== "table" || !attachmentPath) {
    return chunk;
  }

  try {
    const rows = loadTableRows(attachmentPath);
    const { subtype, tableText } = renderTable(chunk, rows);

    return {
      ...chunk,
      table_subtype: subtype,
      table_text: tableText,
      table_rows: rows,
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

// ---------- End of table helpers ----------

// Vercel Node function entry point
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
    let topK = typeof body.top_k === "number" ? body.top_k : 5;
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
          error: "Missing or empty 'question' string in request body.",
        })
      );
      return;
    }

    // Clamp topK between 1 and 10 and not more than corpus size
    topK = Math.max(1, Math.min(topK, 10));

    const { chunks, embeddings } = loadRagStore();

    if (!embeddings.length || !chunks.length) {
      throw new Error("RAG store is empty or failed to load.");
    }

    topK = Math.min(topK, embeddings.length);

    const qEmbedding = await embedQuestion(question);

    console.log("Query embedding length:", qEmbedding.length);
    console.log("First chunk embedding length:", embeddings[0].length);
    console.log("Total chunks:", embeddings.length);

    // Start from all indices, then filter by scope if requested.
    const fullIndices = embeddings.map((_, idx) => idx);
    let scopedIndices = filterIndicesByScope(fullIndices, chunks, scope);
    if (!scopedIndices.length) {
      scopedIndices = fullIndices;
    }

    let indices = filterIndicesByDocHint(scopedIndices, chunks, documentHint);
    if (!indices.length) {
      indices = scopedIndices;
    }

    let scored = indices.map((idx) => ({
      index: idx,
      score: cosineSim(qEmbedding, embeddings[idx]),
    }));

    // --- Table-aware boosting based on section proximity ---
    // 1. Take the top text-like chunks as "anchors"
    const anchorCount = Math.min(10, scored.length);
    const anchors = scored.slice(0, anchorCount).map(({ index, score }) => {
      const c = chunks[index] || {};
      return {
        index,
        score,
        doc_id: c.doc_id || null,
        section_path: c.section_path || "",
        content_type: (c.content_type || "").toLowerCase(),
        sectionKeys: extractSectionKeys(c.section_path || ""),
      };
    });

    const anchorDocSectionMap = [];
    for (const a of anchors) {
      if (!a.doc_id) continue;
      anchorDocSectionMap.push(a);
    }

    // 2. For each table in the same doc + overlapping section keys, add/boost it
    const maxScore = scored.length ? scored[0].score : 1.0;

    const additionalTableEntries = [];
    for (let i = 0; i < chunks.length; i++) {
      const c = chunks[i] || {};
      const ct = (c.content_type || "").toLowerCase();
      if (ct !== "table") continue;

      const tableDoc = c.doc_id || null;
      if (!tableDoc) continue;

      const tableKeys = extractSectionKeys(c.section_path || "");

      let isNeighbor = false;
      for (const a of anchorDocSectionMap) {
        if (a.doc_id !== tableDoc) continue;
        // Check overlap in section keys
        if (a.sectionKeys.some((key) => tableKeys.includes(key))) {
          isNeighbor = true;
          break;
        }
      }

      if (!isNeighbor) continue;

      // See if this table is already in scored
      const existing = scored.find((entry) => entry.index === i);
      if (existing) {
        // Boost its score but keep relative ordering reasonable
        existing.score = Math.max(existing.score, maxScore * 0.98);
      } else {
        additionalTableEntries.push({
          index: i,
          score: maxScore * 0.97,
        });
      }
    }

    if (additionalTableEntries.length) {
      scored = scored.concat(additionalTableEntries);
    }
    // --- End table-aware boosting ---

    // Sort and dedupe by chunk_id
    scored.sort((a, b) => b.score - a.score);
    const seen = new Set();
    const deduped = [];
    for (const entry of scored) {
      const c = chunks[entry.index] || {};
      const id = c.chunk_id;
      if (!id || seen.has(id)) continue;
      seen.add(id);
      deduped.push(entry);
    }

    const top = deduped.slice(0, topK);

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
        score,
      };
    });

    res.statusCode = 200;
    res.setHeader("Content-Type", "application/json");
    res.end(
      JSON.stringify({
        question,
        top_k: topK,
        scope: scope || null,
        document_hint: documentHint || null,
        results,
      })
    );
  } catch (err) {
    console.error(err);
    res.statusCode = 500;
    res.setHeader("Content-Type", "application/json");
    res.end(
      JSON.stringify({
        error: String(err.message || err),
      })
    );
  }
};
