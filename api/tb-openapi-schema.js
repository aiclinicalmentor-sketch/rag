// api/tb-openapi-schema.js
//
// Serves the OpenAPI schema used by the TB Clinical Mentor GPT Action so the
// Actions UI and runtime can discover how to call the retrieval endpoint.

const openApiDocument = {
  openapi: "3.1.0",
  info: {
    title: "TB Clinical Mentor Actions API",
    version: "1.0.0",
    description:
      "Action surface for the TB Clinical Mentor GPT. Clinicians supply a free-form question about a tuberculosis case, and the action returns the most relevant sections of the TB guidelines/handbooks stored in the Retrieval-Augmented Generation (RAG) store.",
    contact: {
      name: "TB Clinical Mentor",
      url: "https://chatgpt.com/g/g-68e126775224819193328c1977555fd0-tb-clinical-mentor-v2",
    },
  },
  servers: [
    {
      url: "https://rag-two-rouge.vercel.app/",
      description:
        "This is the production domain that hosts this API.",
    },
  ],
  paths: {
    "/api/tb-rag-query": {
      post: {
        operationId: "fetchRelevantTbGuidance",
        summary: "Retrieve relevant TB guidance passages",
        description:
          "Calculates an embedding for the clinician question and returns the top matching guideline or handbook passages from the on-device RAG store. Use this action whenever additional TB-specific context is needed before giving clinical advice.",
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: {
                $ref: "#/components/schemas/QueryRequest",
              },
            },
          },
        },
        responses: {
          "200": {
            description:
              "Top passages ordered by cosine similarity to the clinician question.",
            content: {
              "application/json": {
                schema: {
                  $ref: "#/components/schemas/QueryResponse",
                },
                examples: {
                  default: {
                    summary: "Example response",
                    value: {
                      question:
                        "What are the recommended monitoring steps for a patient on bedaquiline?",
                      top_k: 3,
                      results: [
                        {
                          doc_id: "tb_handbook_v3",
                          chunk_id: "chunk-01752",
                          section_path: "Chapter 4 > Drug-Resistant TB > Monitoring",
                          text:
                            "For patients receiving bedaquiline, monitor liver function tests weekly for the first month, then monthly thereafter. Electrocardiograms should be obtained ....",
                          attachment_id: "bedaquiline-monitoring.pdf",
                          attachment_path: "attachments/bedaquiline-monitoring.pdf",
                          score: 0.8123,
                        },
                      ],
                    },
                  },
                },
              },
            },
          },
          "400": {
            description:
              "Missing or malformed question. The response includes an `error` string.",
            content: {
              "application/json": {
                schema: {
                  $ref: "#/components/schemas/ErrorResponse",
                },
              },
            },
          },
          "405": {
            description: "Only POST is supported on this endpoint.",
            content: {
              "application/json": {
                schema: {
                  $ref: "#/components/schemas/ErrorResponse",
                },
              },
            },
          },
          "500": {
            description:
              "Unexpected retrieval or embedding failure. The `error` string will include details suitable for logs.",
            content: {
              "application/json": {
                schema: {
                  $ref: "#/components/schemas/ErrorResponse",
                },
              },
            },
          },
        },
      },
    },
  },
  components: {
    schemas: {
      QueryRequest: {
        type: "object",
        additionalProperties: false,
        required: ["question"],
        properties: {
          question: {
            type: "string",
            description:
              "Free-form clinician question about a TB patient. This text is embedded and compared against the stored TB guideline passages.",
            minLength: 1,
            examples: [
              "How should I adjust the regimen for a TB/HIV co-infected patient with renal impairment?",
            ],
          },
          top_k: {
            type: "integer",
            description:
              "Maximum number of passages to return. Defaults to 5 and supports up to 10 results for GPT Actions.",
            minimum: 1,
            maximum: 10,
            default: 5,
            examples: [5],
          },
        },
      },
      QueryResponse: {
        type: "object",
        required: ["question", "top_k", "results"],
        properties: {
          question: {
            type: "string",
            description: "Echo of the received clinician question.",
          },
          top_k: {
            type: "integer",
            description: "Number of passages returned in `results`.",
          },
          results: {
            type: "array",
            description: "Relevant TB guidance passages ordered by similarity.",
            items: {
              $ref: "#/components/schemas/QueryResult",
            },
          },
        },
      },
      QueryResult: {
        type: "object",
        required: ["doc_id", "chunk_id", "section_path", "text", "score"],
        properties: {
          doc_id: {
            type: "string",
            description:
              "Identifier of the source document within the RAG corpus.",
          },
          chunk_id: {
            type: "string",
            description: "Unique identifier of the chunk within the document.",
          },
          section_path: {
            type: "string",
            description:
              "Breadcrumb-style path pointing to the relevant guideline section.",
          },
          text: {
            type: "string",
            description: "Excerpt of the guideline passage to show to clinicians.",
          },
          attachment_id: {
            type: "string",
            nullable: true,
            description:
              "Optional attachment identifier if additional files are associated with the passage.",
          },
          attachment_path: {
            type: "string",
            nullable: true,
            description:
              "Optional relative path to download the associated attachment.",
          },
          score: {
            type: "number",
            format: "float",
            description:
              "Cosine similarity score where higher values mean a closer match.",
          },
        },
      },
      ErrorResponse: {
        type: "object",
        required: ["error"],
        properties: {
          error: {
            type: "string",
            description: "Human-readable error message.",
          },
        },
      },
    },
  },
};

module.exports = (req, res) => {
  if (req.method !== "GET") {
    res.statusCode = 405;
    res.setHeader("Allow", "GET");
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ error: "Use GET to retrieve the OpenAPI schema" }));
    return;
  }

  res.statusCode = 200;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(openApiDocument));
};
