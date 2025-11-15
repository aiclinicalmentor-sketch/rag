// api/tb-rag-query.js
//
// Vercel Node function that powers the TB Clinical Mentor RAG Action.
// Loads precomputed embeddings + chunks from public/rag and returns the
// top-K passages for a clinician question.

const fs = require("fs");
const path = require("path");

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
      model: "text-embedding-3-small",
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
    const scope = body.scope || null;

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

    // Start from all indices, then filter by scope if requested.
    let indices = embeddings.map((_, idx) => idx);
    indices = filterIndicesByScope(indices, chunks, scope);

    if (!indices.length) {
      // Fallback: if scope filtering removed everything, use full corpus.
      indices = embeddings.map((_, idx) => idx);
    }

    const scored = indices.map((idx) => ({
      index: idx,
      score: cosineSim(qEmbedding, embeddings[idx]),
    }));

    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, topK);

    const results = top.map(({ index, score }) => {
      const c = chunks[index] || {};
      return {
        doc_id: c.doc_id,
        guideline_title: c.guideline_title ?? null,
        year: c.year ?? null,
        chunk_id: c.chunk_id,
        section_path: c.section_path,
        text: c.text,
        attachment_id: c.attachment_id ?? null,
        attachment_path: c.attachment_path ?? null,
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
