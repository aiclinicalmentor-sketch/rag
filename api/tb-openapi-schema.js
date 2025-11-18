// api/tb-openapi-schema.js
//
// Serves the OpenAPI schema (in YAML) for the TB Clinical Mentor GPT Action.

const openApiYaml = `openapi: 3.1.0
info:
  title: TB Clinical Mentor Actions API
  version: 1.2.1
  description: >
    Action surface for the TB Clinical Mentor GPT. Clinicians supply a free-form
    question about a tuberculosis case, and the action returns the most relevant
    sections of the TB guidelines/handbooks stored in the Retrieval-Augmented
    Generation (RAG) store.
  contact:
    name: TB Clinical Mentor
    url: https://chatgpt.com/g/g-68e126775224819193328c1977555fd0-tb-clinical-mentor-v2
  license:
    name: Creative Commons Attribution 4.0 International
    url: https://creativecommons.org/licenses/by/4.0/
security: []
servers:
  - url: "https://rag-two-rouge.vercel.app"
    description: This is the production domain that hosts this API.
paths:
  /api/tb-rag-query:
    post:
      operationId: fetchRelevantTbGuidance
      summary: Retrieve relevant TB guidance passages
      description: >
        Calculates an embedding for the clinician question and returns the top
        matching guideline or handbook passages from the on-device RAG store.

        WHEN TO USE:
        - Use this action whenever the user is asking about tuberculosis
          diagnosis, treatment, monitoring, adverse effects, or programmatic
          management of a patient.
        - Call this action BEFORE giving specific clinical advice so you can
          base your reasoning on up-to-date WHO TB guidance.

        HOW TO CALL:
        - Put the full clinician question (including patient details and
          comorbidities) in "question".
        - Use "top_k" between 3 and 8 to control how many final passages are
          returned. If omitted, it defaults to 8. Internally, the action may
          retrieve a larger candidate set (for example, ~15 chunks) and then
          return the best-scoring passages.
        - Optionally set "scope" to focus on a subset of the corpus:
          "prevention", "screening", "diagnosis", "treatment", "pediatrics", or "comorbidities".

        HOW TO USE THE RESPONSE:
        - Review the "results" array in order.
        - Cite doc_id, guideline_title (if present), year (if present), and
          section_path when explaining your answer.
        - Quote or paraphrase key lines from "text" before providing your
          own clinical reasoning.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/QueryRequest'
      responses:
        "200":
          description: >
            Top passages ordered by cosine similarity to the clinician question.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/QueryResponse'
              examples:
                default:
                  summary: Example response
                  value:
                    question: What are the recommended monitoring steps for a patient on bedaquiline?
                    top_k: 8
                    scope: "treatment"
                    results:
                      - doc_id: tb_handbook_v3
                        guideline_title: WHO operational handbook on tuberculosis, Module 4: treatment
                        year: 2022
                        chunk_id: chunk-01752
                        section_path: Chapter 4 > Drug-Resistant TB > Monitoring
                        text: >
                          For patients receiving bedaquiline, monitor liver function tests weekly for
                          the first month, then monthly thereafter. Electrocardiograms should be
                          obtained ...
                        attachment_id: bedaquiline-monitoring.pdf
                        attachment_path: attachments/bedaquiline-monitoring.pdf
                        score: 0.8123
        "400":
          description: >
            Missing or malformed question (for example, empty or non-string
            "question" field). The response includes an "error" string. The
            assistant should apologize and attempt to answer based on its
            internal TB knowledge without calling the action again.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        "405":
          description: Only POST is supported on this endpoint.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        "500":
          description: >
            Unexpected retrieval or embedding failure. The "error" string will
            include details suitable for logs. The assistant should apologize
            and either try again later or answer based on its internal TB
            knowledge without calling the action again.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
components:
  schemas:
    QueryRequest:
      type: object
      additionalProperties: false
      required:
        - question
      properties:
        question:
          type: string
          description: >
            Free-form clinician question about a TB patient. This text is embedded
            and compared against the stored TB guideline passages. Include enough
            clinical context (age, HIV status, comorbidities, prior treatment) to
            retrieve relevant guidance.
          minLength: 1
          examples:
            - How should I adjust the regimen for a TB/HIV co-infected patient with renal impairment?
        top_k:
          type: integer
          description: >
            Desired number of passages to return (final top-K). Defaults to 8 and
            supports between 1 and 8 results. Internally, the service may
            retrieve more candidates (for example, ~15 text/table chunks) and
            then return the best ones.
          minimum: 1
          maximum: 8
          default: 8
          examples:
            - 8
        scope:
          type: string
          description: >
            Optional hint to focus retrieval on a subset of the TB corpus.
            Suggested values:
            - "prevention" for TB infection, TPT, and contact management.
            - "screening" for systematic screening and triage.
            - "diagnosis" for smear-negative/Xpert/CXR algorithms and diagnostic
              pathways.
            - "treatment" for regimen selection, dosing, and duration.
            - "pediatrics" for child and adolescent TB guidance.
            - "comorbidities" for TB/HIV, diabetes, and other comorbid conditions.
          enum: [prevention, screening, diagnosis, treatment, pediatrics, comorbidities]
          examples:
            - diagnosis

    QueryResponse:
      type: object
      additionalProperties: false
      required:
        - question
        - top_k
        - results
      properties:
        question:
          type: string
          description: Echo of the received clinician question.
        top_k:
          type: integer
          description: >
            Number of passages requested (after capping to the supported range)
            and returned in "results".
        results:
          type: array
          description: Relevant TB guidance passages ordered by similarity.
          items:
            $ref: '#/components/schemas/QueryResult'
    QueryResult:
      type: object
      additionalProperties: false
      required:
        - doc_id
        - chunk_id
        - section_path
        - text
        - score
      properties:
        doc_id:
          type: string
          description: Identifier of the source document within the RAG corpus.
        guideline_title:
          type:
            - string
            - "null"
          description: >
            Human-readable title of the guideline or handbook, if available.
        year:
          type:
            - integer
            - "null"
          description: Publication year of the guideline module, if available.
        chunk_id:
          type: string
          description: Unique identifier of the chunk within the document.
        section_path:
          type: string
          description: Breadcrumb-style path pointing to the relevant guideline section.
        text:
          type: string
          description: Excerpt of the guideline passage to show to clinicians.
        attachment_id:
          type:
            - string
            - "null"
          description: Optional attachment identifier if files are associated with the passage.
        attachment_path:
          type:
            - string
            - "null"
          description: Optional relative path to download the associated attachment.
        score:
          type: number
          format: float
          description: Cosine similarity score where higher values mean a closer match.
    ErrorResponse:
      type: object
      additionalProperties: false
      required:
        - error
      properties:
        error:
          type: string
          description: >
            Human-readable error message. The assistant should apologize and,
            when appropriate, answer based on its internal TB knowledge instead
            of repeatedly calling the action.
`;

module.exports = (req, res) => {
  if (req.method !== "GET") {
    res.statusCode = 405;
    res.setHeader("Allow", "GET");
    res.setHeader("Content-Type", "application/json");
    res.end(
      JSON.stringify({ error: "Use GET to retrieve the OpenAPI schema for this API." })
    );
    return;
  }

  res.statusCode = 200;
  res.setHeader("Content-Type", "application/yaml");
  res.end(openApiYaml);
};
