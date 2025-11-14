// api/tb-rag-query.js

const fs = require("fs");
const path = require("path");

let RAG_STORE = null;

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
  const embeddings = JSON.parse(fs.readFileSync(embeddingsPath, "utf-8"));

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

function cosineSim(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const va = a[i];
    const vb = b[i];
    dot += va * vb;
    na += va * va;
    nb += vb * vb;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
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
  return json.data[0].embedding;
}

// Vercel Node function entry point
module.exports = async (req, res) => {
  if (req.method !== "POST") {
    res.statusCode = 405;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ error: "Use POST" }));
    return;
  }

  try {
    const body =
      typeof req.body === "string" ? JSON.parse(req.body) : req.body || {};
    const question = body.question;
    const topK = body.top_k || 5;

    if (!question || typeof question !== "string") {
      res.statusCode = 400;
      res.end(
        JSON.stringify({ error: "Missing 'question' string in request body" })
      );
      return;
    }

    const { chunks, embeddings } = loadRagStore();
    const qEmbedding = await embedQuestion(question);

    const scored = embeddings.map((vec, idx) => ({
      index: idx,
      score: cosineSim(qEmbedding, vec),
    }));

    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, topK);

    const results = top.map(({ index, score }) => {
      const c = chunks[index];
      return {
        doc_id: c.doc_id,
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
        results,
      })
    );
  } catch (err) {
    console.error(err);
    res.statusCode = 500;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ error: String(err.message || err) }));
  }
};
