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

async function loadEmbeddings(ragDir) {
  const embeddingsJsonPath = path.join(ragDir, "embeddings.json");
  const embeddingsNpyPath = path.join(ragDir, "embeddings.npy");

  if (fs.existsSync(embeddingsJsonPath)) {
    const raw = fs.readFileSync(embeddingsJsonPath, "utf-8");
    const trimmed = raw.trimStart();
    const isLfsPointer = trimmed.startsWith(
      "version https://git-lfs.github.com/spec/v1"
    );

    if (!isLfsPointer) {
      try {
        return JSON.parse(raw);
      } catch (err) {
        console.warn(
          "Failed to parse embeddings.json; will try .npy fallback",
          err
        );
      }
    } else {
      console.warn(
        "embeddings.json appears to be a Git LFS pointer; falling back to embeddings.npy"
      );
    }
  }

  if (fs.existsSync(embeddingsNpyPath)) {
    const { default: Npyjs } = require("npyjs");
    const npy = new Npyjs();
    const buf = fs.readFileSync(embeddingsNpyPath);
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    const arr = await npy.load(ab);
    const [rows, cols] = arr.shape || [];
    if (!rows || !cols) {
      throw new Error("Invalid shape from embeddings.npy");
    }

    const embeddings = [];
    for (let i = 0; i < rows; i++) {
      const start = i * cols;
      const end = start + cols;
      embeddings.push(Array.from(arr.data.slice(start, end)));
    }
    return embeddings;
  }

  throw new Error(
    `No embeddings found. Expected at ${embeddingsJsonPath} (JSON) or ${embeddingsNpyPath} (.npy).`
  );
}

async function loadRagStore() {
  if (RAG_STORE) return RAG_STORE;

  const ragDir = path.join(process.cwd(), "public", "rag");
  const chunksPath = path.join(ragDir, "chunks.jsonl");

  const chunksLines = fs
    .readFileSync(chunksPath, "utf-8")
    .split("\n")
    .filter(Boolean);

  const chunks = chunksLines.map((line) => JSON.parse(line));

  const rawEmbeddings = await loadEmbeddings(ragDir);
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

function inferScopeFromQuestion(question, intentFlags) {
  const q = (question || "").toLowerCase();
  const flags = Array.isArray(intentFlags) ? intentFlags : [];

  // 1) Prefer topic-level intent flags to choose a high-level scope.
  //    We intentionally do NOT allow population/context flags
  //    (special_populations, comorbidities) to become scope.
  const hasFlag = (name) => flags.includes(name);

  const topicCandidates = new Set();

  // Module 1: TB infection, preventive treatment, IPC
  if (hasFlag("prevention_and_ltbi") || hasFlag("infection_prevention_and_control")) {
    topicCandidates.add("prevention");
  }

  // Module 2: systematic screening / triage
  if (hasFlag("screening")) {
    topicCandidates.add("screening");
  }

  // Module 3: diagnosis / diagnostic workup
  if (hasFlag("diagnostic_workup")) {
    topicCandidates.add("diagnosis");
  }

  // Module 4: treatment-related questions (any treatment subtask)
  if (
    hasFlag("regimen_selection") ||
    hasFlag("regimen_modification") ||
    hasFlag("toxicity_management") ||
    hasFlag("monitoring_schedule") ||
    hasFlag("treatment_failure_or_reversion")
  ) {
    topicCandidates.add("treatment");
  }

  // Priority order if multiple topic candidates are present.
  const topicPriority = ["treatment", "diagnosis", "prevention", "screening"];
  for (const t of topicPriority) {
    if (topicCandidates.has(t)) {
      return t;
    }
  }

  // 2) Fallback: keyword-based heuristic (no population/comorbidity scopes).
  if (!q) return null;

  const preventionScore = keywordScore(q, [
    "prevention",
    "tb infection",
    "latent tb",
    "tpt",
    "preventive treatment",
    "contact management",
    "household contacts"
  ]);

  const screeningScore = keywordScore(q, [
    "screening",
    "systematic screening",
    "triage",
    "ai-assisted",
    "cxr screening",
    "screen for tb"
  ]);

  const diagnosisScore = keywordScore(q, [
    "diagnos",
    "algorithm",
    "cxr",
    "x-ray",
    "radiograph",
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

  if (preventionScore >= 2 || q.includes("module 1")) {
    return "prevention";
  }
  if (screeningScore >= 2 || (screeningScore && q.includes("module 2"))) {
    return "screening";
  }
  if (diagnosisScore >= 2 || (diagnosisScore && q.includes("module 3"))) {
    return "diagnosis";
  }
  if (treatmentScore >= 2 || q.includes("module 4")) {
    return "treatment";
  }

  return null;
}

function inferIntentFlags(question) {
  const q = (question || "").toLowerCase();
  if (!q) return [];

  const flags = [];

  const has = (words) =>
    words.some((w) => q.includes(w));

  // Module 2: Screening
  if (
    has([
      "screening",
      "systematic screening",
      "triage",
      "ai-assisted",
      "ai assisted",
      "cxr screening",
      "screen for tb"
    ])
  ) {
    flags.push("screening");
  }

  // Module 3: Diagnosis / diagnostic workup
  if (
    has([
      "diagnos",
      "algorithm",
      "cxr",
      "x-ray",
      "xray",
      "radiograph",
      "xpert",
      "ultra",
      "naat",
      "truenat",
      "lamp",
      "wrd",
      "wrds",
      "lpa",
      "smear",
      "culture",
      "dst",
      "drug susceptibility"
    ])
  ) {
    flags.push("diagnostic_workup");
  }

  // Module 1: TB infection & preventive treatment
  if (
    has([
      "tpt",
      "preventive treatment",
      "tb preventive treatment",
      "tb infection",
      "latent tb",
      "ltbi",
      "isoniazid preventive",
      "inh preventive"
    ])
  ) {
    flags.push("prevention_and_ltbi");
  }

  // Infection prevention and control
  if (
    has([
      "infection prevention",
      "ipc",
      "airborne infection",
      "ventilation",
      "uvgi",
      "n95",
      "respirator",
      "masking",
      "administrative controls",
      "environmental controls"
    ])
  ) {
    flags.push("infection_prevention_and_control");
  }

  // Treatment: regimen selection vs modification
  if (
    has([
      "start treatment",
      "starting treatment",
      "initial regimen",
      "which regimen",
      "what regimen",
      "choose regimen",
      "choice of regimen",
      "regimen choice"
    ])
  ) {
    flags.push("regimen_selection");
  }

  if (
    has([
      "change regimen",
      "switch regimen",
      "modify regimen",
      "regimen modification",
      "adjust regimen",
      "add drug",
      "add a drug",
      "remove drug",
      "stop drug",
      "substitute",
      "substitution"
    ])
  ) {
    flags.push("regimen_modification");
  }

  // Toxicity and adverse events
  if (
    has([
      "toxicity",
      "adverse event",
      "side effect",
      "peripheral neuropathy",
      "neuropathy",
      "optic neuritis",
      "myelosuppression",
      "anaemia",
      "anemia",
      "thrombocytopenia",
      "hepatotoxicity",
      "liver toxicity",
      "vomiting",
      "nausea",
      "abdominal pain",
      "hyperpigmentation",
      "qt prolongation",
      "qt-prolonging",
      "cardiac toxicity"
    ])
  ) {
    flags.push("toxicity_management");
  }

  // Monitoring and follow-up
  if (
    has([
      "monitoring",
      "follow-up",
      "follow up",
      "schedule",
      "how often",
      "how frequently",
      "baseline tests",
      "baseline investigations",
      "routine monitoring",
      "post-treatment monitoring"
    ])
  ) {
    flags.push("monitoring_schedule");
  }

  // Treatment failure, culture reversion, relapse
  if (
    has([
      "treatment failure",
      "failed treatment",
      "failure of treatment",
      "culture reversion",
      "remains culture-positive",
      "remains culture positive",
      "persistent positive",
      "recurrent tb",
      "relapse"
    ])
  ) {
    flags.push("treatment_failure_or_reversion");
  }

  // Special populations (pregnancy, children, adolescents)
  if (
    has([
      "pregnant",
      "pregnancy",
      "breastfeeding",
      "lactating",
      "postpartum",
      "child",
      "children",
      "paediatric",
      "pediatric",
      "adolescent",
      "infant",
      "neonate"
    ])
  ) {
    flags.push("special_populations");
  }

  
  // Drug-resistant TB / MDR/RR/XDR intent
  if (
    has([
      "mdr-tb",
      "mdr tb",
      "xdr-tb",
      "xdr tb",
      "pre-xdr",
      "pre xdr",
      "rr-tb",
      "rr tb",
      "rifampicin-resistant",
      "rifampin-resistant",
      "drug-resistant tb",
      "drug resistant tb",
      "fluoroquinolone-resistant",
      "fluoroquinolone resistance"
    ])
  ) {
    flags.push("drug_resistance");
  }

  // TB preventive treatment / TPT intent
  if (
    has([
      "tpt",
      "tb preventive treatment",
      "tb preventive therapy",
      "preventive treatment",
      "preventive therapy",
      "ipt",
      "isoniazid preventive therapy",
      "3hp",
      "1hp",
      "3hr",
      "4r",
      "latent tb infection treatment",
      "ltbi treatment",
      "tb infection treatment",
      "tb infection therapy"
    ])
  ) {
    flags.push("tpt");
  }

  // DR-TPT (preventive treatment for contacts of MDR/RR-TB, levofloxacin, etc.)
  if (
    has([
      "levofloxacin preventive",
      "levofloxacin prophylaxis",
      "6lfx",
      "6 months of levofloxacin",
      "mdr-tb contact",
      "mdr tb contact",
      "rr-tb contact",
      "rr tb contact",
      "contact of mdr-tb",
      "household contact of mdr-tb"
    ])
  ) {
    flags.push("dr_tpt");
    flags.push("tpt");
  }

// Comorbidities (HIV, diabetes, renal, liver, mental health, substance use)
  if (
    has([
      "hiv",
      "plhiv",
      "cd4",
      "viral load",
      "diabetes",
      "diabetic",
      "renal failure",
      "kidney disease",
      "ckd",
      "cirrhosis",
      "liver disease",
      "alcohol use",
      "harmful use",
      "substance use",
      "drug use",
      "depression",
      "anxiety",
      "mental health",
      "hepatitis",
      "hbv",
      "hcv"
    ])
  ) {
    flags.push("comorbidities");
  }

  return Array.from(new Set(flags));
}

function inferPopulationContext(question, intentFlags) {
  const q = (question || "").toLowerCase();
  const flags = Array.isArray(intentFlags) ? intentFlags : [];

  // Population/context-level signals. These are NOT used to choose scope,
  // but can be logged and later used for boosting.
  const context = {
    special_populations: flags.includes("special_populations") || false,
    comorbidities: flags.includes("comorbidities") || false
  };

  return context;
}

function isDrugSusceptibleChunk(chunk) {
  if (!chunk) return false;

  const doc = (chunk.doc_id || "").toString().toLowerCase();
  const section = (chunk.section_path || "").toString().toLowerCase();
  const scope = (chunk.scope || "").toString().toLowerCase();
  const text = (chunk.text || "").toString().toLowerCase();

  const dsSignals = [
    "drug - susceptible",
    "drug-susceptible",
    "drug susceptible",
    "ds-tb",
    "ds tb"
  ];

  const hasSignal = (s) => dsSignals.some((sig) => s.includes(sig));

  if (hasSignal(scope)) return true;
  if (hasSignal(section)) return true;
  if (hasSignal(text)) return true;

  // Module 4 chapter 1 is DS-TB; down-weight it for DR intent.
  if (
    doc.includes("module4") &&
    doc.includes("treat") &&
    section.includes("chapter 1")
  ) {
    return true;
  }

  return false;
}


function filterIndicesByScope(indices, chunks, scope) {
  if (!scope) return indices;

  const s = String(scope).toLowerCase();

  return indices.filter((i) => {
    const c = chunks[i] || {};
    const docId = (c.doc_id || "").toString().toLowerCase();
    const section = (c.section_path || "").toString().toLowerCase();
    const chunkScope = (c.scope || "").toString().toLowerCase();

    // If a chunk has an explicit scope tag, trust it.
    if (chunkScope) {
      return chunkScope === s;
    }

    switch (s) {
      case "prevention":
        // Module 1: TB infection & TPT, contact management, infection prevention
        return (
          section.includes("prevention") ||
          section.includes("tb infection") ||
          section.includes("tpt") ||
          section.includes("preventive treatment") ||
          section.includes("contact management") ||
          section.includes("latent tb") ||
          docId.includes("module1")
        );
      case "screening":
        // Module 2: systematic screening / triage
        return (
          section.includes("screening") ||
          section.includes("systematic screening") ||
          section.includes("triage") ||
          section.includes("ai-assisted") ||
          section.includes("cxr screening") ||
          docId.includes("module2")
        );
      case "diagnosis":
        // Module 3: diagnostic algorithms, Xpert, smear, radiography
        return (
          docId.includes("diag") ||
          docId.includes("module3") ||
          section.includes("diagnos") ||
          section.includes("xpert") ||
          section.includes("cxr") ||
          section.includes("smear") ||
          section.includes("naat")
        );
      case "treatment":
        // Module 4: treatment regimens for DS- and DR-TB
        return (
          docId.includes("treat") ||
          docId.includes("module4") ||
          section.includes("treatment") ||
          section.includes("regimen") ||
          section.includes("drug-resistant tb") ||
          section.includes("dr-tb") ||
          section.includes("mdr-tb") ||
          section.includes("rifampicin-resistant")
        );
      case "pediatrics":
        // Module 5: child & adolescent TB
        return (
          docId.includes("pediatric") ||
          docId.includes("child") ||
          docId.includes("module5") ||
          section.includes("child") ||
          section.includes("children") ||
          section.includes("adolescent") ||
          section.includes("paediatric") ||
          section.includes("pediatric")
        );
      case "comorbidities":
        // Module 6: TB with HIV, diabetes, other comorbidities
        return (
          docId.includes("module6") ||
          section.includes("hiv") ||
          section.includes("diabetes") ||
          section.includes("comorbid") ||
          section.includes("substance use") ||
          section.includes("alcohol use") ||
          section.includes("mental health") ||
          section.includes("hepatitis") ||
          section.includes("hcv")
        );
      default:
        return true;
    }
  });
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

  if (cleaned.startsWith("public/rag/")) {
    // already rooted correctly
  } else if (cleaned.startsWith("public/")) {
    // has public prefix but missing rag
    cleaned = cleaned.replace(/^public\//, "public/rag/");
  } else if (cleaned.startsWith("rag/")) {
    cleaned = path.posix.join("public", cleaned);
  } else {
    // default: assume chunker paths are relative to public/rag
    cleaned = path.posix.join("public", "rag", cleaned);
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
    /dose|dosage|dosing|mg\/kg|mg\/ day|mg per kg|weight band|weight-band/.test(
      caption
    ) ||
    headers.some((h) => /kg|dose|weight band|weight-band/.test(h))
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
    /eligib|criteria|if.*then|decision|indication|when to|option(s)?/.test(
      caption
    ) ||
    hasHeader(/criteria|recommendation|option|action/)
  ) {
    return "decision";
  }

  // 3) Regimen composition tables (need regimen + drug-ish header)
  if (
    /regimen/.test(caption) ||
    (hasHeader(/regimen/) && hasHeader(/drug|component|medicine|combination/))
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
    /interaction|drug-drug|drug drug|ddi|qt|contraindication/.test(caption) ||
    headers.some((h) => /interaction|effect|recommendation|ddi/.test(h))
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

  // A)   // A) timepoints in COLUMNS -> several time-like headers
  const timeHeaders = headers.filter((h) => looksTimeLike(h));
  if (timeHeaders.length >= 2) {
    const entityHeader =
      headers.find((h) => !timeHeaders.includes(h)) || headers[0];
    return { orientation: "cols", entityHeader, timeHeaders };
  }

  // B)   // B) timepoints in ROWS -> first column values look time-like
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
    /(kg|weight band|weight-band|weight range| to <|<=|>=)/i.test(h)
  );

  // If we couldn't confidently find weight-band columns, just fall back to generic.
  if (!weightBandKeys.length) {
    const generic = renderGenericTable(chunk, logicalRows, headerRow);
    return {
      text: generic.text,
      debug: { ...generic.debug, renderer: "dosing", note: "fallback_generic_no_weight_bands" }
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
    /(kg|weight band|weight-band|weight range| to <|<=|>=)/i.test(h)
  );
  const otherKeys = headers.filter(
    (h) => h !== medKey && h !== formKey && !weightBandKeys.includes(h)
  );
  const medIsWeight = weightBandKeys.includes(medKey);
  const extraWeightKeys = medIsWeight ? weightBandKeys.filter((k) => k !== medKey) : [];

  if (!weightBandKeys.length) {
    const generic = renderGenericTable(chunk, logicalRows, headerRow);
    return {
      text: generic.text,
      debug: { ...generic.debug, renderer: "peds_dosing", note: "fallback_generic_no_weight_bands" }
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
    const doseParts = otherKeys
      .concat(extraWeightKeys)
      .map((k) => {
        const v = row[k];
        if (!nonEmpty(v)) return null;
        return `${k}: ${String(v).trim()}`;
      })
      .filter(Boolean);

    if (!bandParts.length && !doseParts.length) {
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
      line += ` -> ${cells.join("; ")}`;
      lines.push(line);
      return;
    }

    let line = `${med}`;
    if (form) line += ` (${form})`;
    const parts = [];
    if (medIsWeight) {
      parts.push(`${medKey}: ${String(med).trim()}`);
    } else {
      parts.push(...bandParts);
    }
    parts.push(...doseParts);
    line += ` -> ${parts.join("; ")}`;
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
      lines.push(`${regimen} - ${parts.join("; ")}`);
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
  lines.push(`${caption}. IF-THEN decision rules:`);

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
    const generic = renderGenericTable(chunk, logicalRows, headerRow);
    return {
      text: generic.text,
      debug: { ...generic.debug, renderer: "decision", note: "fallback_generic_no_if_then_rows" }
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

  // C) Unknown pattern -> generic safe summary
  const generic = renderGenericTable(chunk, logicalRows, headerRow);
  return {
    text: generic.text,
    debug: {
      ...generic.debug,
      renderer: "timeline",
      note: "fallback_generic_unknown_orientation"
    }
  };
}

function renderInteractionTable(chunk, logicalRows, headerRow) {
  const caption = chunk.caption || "Drug-drug interaction table";
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
    const generic = renderGenericTable(chunk, logicalRows, headerRow);
    return {
      text: generic.text,
      debug: { ...generic.debug, renderer: "interaction", note: "fallback_generic_no_drug_headers" }
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
    if (effect) line += (line ? " - " : "") + `effect: ${effect}`;
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
    const generic = renderGenericTable(chunk, logicalRows, headerRow);
    return {
      text: generic.text,
      debug: { ...generic.debug, renderer: "interaction", note: "fallback_generic_no_combos" }
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
      lines.push(`Grade ${grade}: ${parts.join(" - ")}`);
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
    const generic = renderGenericTable(chunk, logicalRows, headerRow);
    return {
      text: generic.text,
      debug: { ...generic.debug, renderer: "toxicity", note: "fallback_generic_no_grades" }
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


function enrichChunkWithTable(chunk, options = {}) {
  const includeTableRows = options.includeTableRows === true;
  const tableRowLimit =
    typeof options.tableRowLimit === "number"
      ? Math.max(1, Math.floor(options.tableRowLimit))
      : 0;
  const ct = (chunk.content_type || "").toLowerCase();
  const attachmentPath = chunk.attachment_path;

  if (ct !== "table" || !attachmentPath) {
    return chunk;
  }

  try {
    const rawRows = loadTableRows(attachmentPath);
    const limitedRows =
      includeTableRows && Array.isArray(rawRows)
        ? rawRows.slice(0, tableRowLimit || rawRows.length)
        : null;
    const { subtype, tableText, debug } = renderTable(chunk, rawRows);

    return {
      ...chunk,
      table_subtype: subtype,
      table_text: tableText,
      ...(includeTableRows ? { table_rows: limitedRows } : {}),
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

function formatRetrievalEntry(chunk, score) {
  if (!chunk) return null;

  const previewSource =
    typeof chunk.text === "string" && chunk.text.trim()
      ? chunk.text
      : typeof chunk.table_text === "string" && chunk.table_text.trim()
        ? chunk.table_text
        : "";
  const preview = previewSource
    ? previewSource.replace(/\s+/g, " ").slice(0, 240)
    : null;

  return {
    chunk_id: chunk.chunk_id ?? null,
    doc_id: chunk.doc_id ?? null,
    guideline_title: chunk.guideline_title ?? null,
    section_path: chunk.section_path ?? null,
    content_type: chunk.content_type ?? null,
    table_subtype: chunk.table_subtype ?? null,
    score: typeof score === "number" ? Number(score.toFixed(4)) : null,
    preview
  };
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
    const includeTableRows = body.include_table_rows === true;
    const tableRowLimitRaw = Number(body.table_row_limit);
    const tableRowLimit = Number.isFinite(tableRowLimitRaw)
      ? Math.max(1, Math.min(500, Math.floor(tableRowLimitRaw)))
      : 150;
    const retrievalLog = [];
    const addLog = (stage, payload) => {
      retrievalLog.push({ stage, ...(payload || {}) });
    };

    const intentFlags = inferIntentFlags(question || "");
    const populationContext = inferPopulationContext(question || "", intentFlags);
    const hasFlag = (f) => Array.isArray(intentFlags) && intentFlags.includes(f);
    const isPediatric = hasFlag("special_populations");
    const hasDrugResistanceIntent = hasFlag("drug_resistance");
    const hasTptIntent = hasFlag("tpt");
    const hasDrTptIntent = hasFlag("dr_tpt");

    if (!scope) {
      scope = inferScopeFromQuestion(question, intentFlags);
    }

    const isDiagnosisScope = scope === "diagnosis";
    const isTreatmentScope = scope === "treatment";
    const isPreventionScope = scope === "prevention";

    addLog("request", {
      question_preview:
        typeof question === "string" ? question.slice(0, 200) : null,
      requested_top_k: body.top_k ?? null,
      scope_used: scope || null,
      include_table_rows: includeTableRows,
      table_row_limit: includeTableRows ? tableRowLimit : null
    });

    addLog("intent", {
      intent_flags: intentFlags,
      topic_scope: scope || null,
      population_context: populationContext
    });


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

    const { chunks, embeddings } = await loadRagStore();

    if (!embeddings.length || !chunks.length) {
      throw new Error("RAG store is empty or failed to load.");
    }

    addLog("store_loaded", {
      chunk_count: chunks.length,
      embedding_count: embeddings.length,
      embedding_dimensions: embeddings[0]?.length || null
    });

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

    const indices = scopedIndices;

    addLog("filtering", {
      scope_used: scope || null,
      scope_match_count: scopedIndices.length
    });



    // Dual-channel retrieval: text/prose vs tables
    const textIndices = indices.filter((i) => {
      const ct = (chunks[i].content_type || "").toLowerCase();
      return ct !== "table";
    });

    const tableIndices = indices.filter((i) => {
      const ct = (chunks[i].content_type || "").toLowerCase();
      return ct === "table";
    });

    addLog("channels", {
      text_candidates: textIndices.length,
      table_candidates: tableIndices.length
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

    // --- Module-aware boosts: recency/authority & pediatric/TPT routing ---
    const dsPenaltyFactor = 0.6;
    let dsPenaltyText = 0;
    for (const entry of scoredText) {
      const c = chunks[entry.index] || {};
      const doc = (c.doc_id || "").toLowerCase();
      const section = (c.section_path || "").toLowerCase();

      if (isTreatmentScope) {
        // 2025 treatment module (Module 4)
        if (doc.includes("module4") && doc.includes("treatment") && doc.includes("2025")) {
          entry.score *= 1.03;
        }

        // Pediatric DS-TB treatment: Module 5, section 5.2
        if (
          isPediatric &&
          !hasDrugResistanceIntent &&
          doc.includes("module5") &&
          doc.includes("pediatr") &&
          (
            section.includes("5.2.") ||
            section.includes("treatment of drug-susceptible tb in children")
          )
        ) {
          entry.score *= 1.03;
        }
      }

      // Pediatric diagnosis: favor Module 3 (diagnosis)
      if (isDiagnosisScope && isPediatric) {
        if (
          doc.includes("module3") &&
          (doc.includes("diag") || section.includes("diagnosis"))
        ) {
          entry.score *= 1.03;
        }
      }

      // TB preventive treatment: favor Module 1 TPT 2024
      if (isPreventionScope) {
        if (doc.includes("module1") && doc.includes("tpt") && doc.includes("2024")) {
          entry.score *= 1.03;
        }
      }

      if (hasDrugResistanceIntent && isDrugSusceptibleChunk(c)) {
        entry.score *= dsPenaltyFactor;
        dsPenaltyText += 1;
      }
    }

    let dsPenaltyTables = 0;
    for (const entry of scoredTables) {
      const c = chunks[entry.index] || {};
      const doc = (c.doc_id || "").toLowerCase();
      const section = (c.section_path || "").toLowerCase();

      if (isTreatmentScope) {
        if (doc.includes("module4") && doc.includes("treatment") && doc.includes("2025")) {
          entry.score *= 1.03;
        }

        if (
          isPediatric &&
          !hasDrugResistanceIntent &&
          doc.includes("module5") &&
          doc.includes("pediatr") &&
          section.includes("5.2.")
        ) {
          entry.score *= 1.03;
        }
      }

      if (isDiagnosisScope && isPediatric) {
        if (
          doc.includes("module3") &&
          (doc.includes("diag") || section.includes("diagnosis"))
        ) {
          entry.score *= 1.03;
        }
      }

      if (isPreventionScope) {
        if (doc.includes("module1") && doc.includes("tpt") && doc.includes("2024")) {
          entry.score *= 1.03;
        }
      }

      if (hasDrugResistanceIntent && isDrugSusceptibleChunk(c)) {
        entry.score *= dsPenaltyFactor;
        dsPenaltyTables += 1;
      }
    }
    // --- end module-aware boosts ---

    if (hasDrugResistanceIntent) {
      addLog("ds_downweight", {
        intent_flag_present: hasDrugResistanceIntent,
        penalized_text: dsPenaltyText,
        penalized_tables: dsPenaltyTables,
        factor: dsPenaltyFactor
      });
    }

    // Resort after boosting to respect adjusted scores
    scoredText.sort((a, b) => b.score - a.score);
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
    let hasModule4DrAnchor = false;
    let hasModule1TptAnchor = false;

    for (const a of anchors) {
      if (!a.doc_id) continue;
      anchorDocSectionMap.push(a);
      const docId = (a.doc_id || "").toLowerCase();
      const sectionPath = (a.section_path || "").toLowerCase();
      if (
        docId.includes("module4") &&
        docId.includes("treatment") &&
        sectionPath.includes("chapter 2 : drug-resistant tb treatment")
      ) {
        hasModule4DrAnchor = true;
      }
      if (
        docId.includes("module1") &&
        docId.includes("tpt") &&
        sectionPath.includes("tb preventive treatment")
      ) {
        hasModule1TptAnchor = true;
      }
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


    // Penalize outdated pediatric MDR regimen tables from Module 5 when Module 4 DR content is present
    if (isTreatmentScope && isPediatric && hasDrugResistanceIntent && hasModule4DrAnchor) {
      for (const entry of scoredTables) {
        const c = chunks[entry.index] || {};
        const doc = (c.doc_id || "").toLowerCase();
        const ct = (c.content_type || "").toLowerCase();
        const subtype = (c.table_subtype || "").toLowerCase();
        if (!doc.includes("module5") || !doc.includes("pediatr")) continue;
        if (ct !== "table") continue;
        if (!(subtype === "regimen" || subtype === "decision")) continue;
        entry.score *= 0.97;
      }
      scoredTables.sort((a, b) => b.score - a.score);
    }

    // Penalize outdated TPT regimen tables from Module 5 when Module 1 TPT content is present
    if (isPreventionScope && hasTptIntent && hasModule1TptAnchor) {
      for (const entry of scoredTables) {
        const c = chunks[entry.index] || {};
        const doc = (c.doc_id || "").toLowerCase();
        const ct = (c.content_type || "").toLowerCase();
        const subtype = (c.table_subtype || "").toLowerCase();
        const section = (c.section_path || "").toLowerCase();
        if (!doc.includes("module5") || !doc.includes("pediatr")) continue;
        if (ct !== "table") continue;
        if (!(subtype === "regimen" || subtype === "decision")) continue;
        if (
          !section.includes("3.3.5") &&
          !section.includes("3.3.6") &&
          !section.includes("preventive treatment")
        ) {
          continue;
        }
        entry.score *= 0.97;
      }
      scoredTables.sort((a, b) => b.score - a.score);
    }


    // Take top-N from each channel before merging
    const TEXT_LIMIT = 20;
    const TABLE_LIMIT = 8;

    const topText = scoredText.slice(
      0,
      Math.min(TEXT_LIMIT, scoredText.length)
    );
    const topTables = scoredTables.slice(
      0,
      Math.min(TABLE_LIMIT, scoredTables.length)
    );

    const textLog = topText
      .map(({ index, score }) => formatRetrievalEntry(chunks[index], score))
      .filter(Boolean);
    const tableLog = topTables
      .map(({ index, score }) => formatRetrievalEntry(chunks[index], score))
      .filter(Boolean);

    addLog("top_candidates", {
      text: textLog,
      tables: tableLog
    });

    let combined = topText.concat(topTables);
    combined.sort((a, b) => b.score - a.score);

    // Ensure at least a couple of Module 4 DR-TB text chunks for pediatric MDR/RR-TB queries
    if (isTreatmentScope && isPediatric && hasDrugResistanceIntent) {
      const module4DrCandidates = scoredText.filter(({ index }) => {
        const c = chunks[index] || {};
        const doc = (c.doc_id || "").toLowerCase();
        const section = (c.section_path || "").toLowerCase();
        return (
          doc.includes("module4") &&
          doc.includes("treatment") &&
          section.includes("chapter 2 : drug-resistant tb treatment")
        );
      });

      let needed = 2;
      for (const cand of module4DrCandidates) {
        if (needed <= 0) break;
        const alreadyIn = combined.some((e) => e.index === cand.index);
        if (!alreadyIn) {
          combined.push(cand);
          needed -= 1;
        }
      }
      combined.sort((a, b) => b.score - a.score);
    }

    // Ensure at least a couple of Module 1 TPT text chunks for TPT queries
    if (isPreventionScope && hasTptIntent) {
      const module1TptCandidates = scoredText.filter(({ index }) => {
        const c = chunks[index] || {};
        const doc = (c.doc_id || "").toLowerCase();
        const section = (c.section_path || "").toLowerCase();
        return (
          doc.includes("module1") &&
          doc.includes("tpt") &&
          doc.includes("2024") &&
          (
            section.includes("tb preventive treatment") ||
            section.includes("tb infection")
          )
        );
      });

      let needed = 2;
      for (const cand of module1TptCandidates) {
        if (needed <= 0) break;
        const alreadyIn = combined.some((e) => e.index === cand.index);
        if (!alreadyIn) {
          combined.push(cand);
          needed -= 1;
        }
      }
      combined.sort((a, b) => b.score - a.score);
    }


    const seen = new Set();
    const deduped = [];
    for (const entry of combined) {
      const c = chunks[entry.index] || {};
      const id = c.chunk_id;
      if (!id || seen.has(id)) continue;
      seen.add(id);
      deduped.push(entry);
    }

    const top = deduped.slice(0, finalTopK);

    const enrichmentOptions = { includeTableRows, tableRowLimit };
    const results = top.map(({ index, score }) => {
      const baseChunk = chunks[index] || {};
      const c = enrichChunkWithTable(baseChunk, enrichmentOptions);

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
        table_rows: includeTableRows ? c.table_rows ?? null : null,
        table_row_count: c.table_row_count ?? null,
        score
      };
    });

    addLog("final_results", {
      top_k: finalTopK,
      results: results
        .map((r) => formatRetrievalEntry(r, r.score))
        .filter(Boolean)
    });

    // Emit retrieval log to server logs for observability.
    console.log(
      "retrieval_log",
      JSON.stringify(
        {
          question_preview:
            typeof question === "string" ? question.slice(0, 120) : null,
          entries: retrievalLog
        },
        null,
        2
      )
    );

    res.statusCode = 200;
    res.setHeader("Content-Type", "application/json");
    res.end(
      JSON.stringify({
        question,
        top_k: finalTopK,
        scope: scope || null,
        results,
        retrieval_log: retrievalLog
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
