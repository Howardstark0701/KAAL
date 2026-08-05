# Requirements Document

## Introduction

Phase 11 delivers the KAAL web frontend — a Next.js 14 single-page application that provides a browser-based interface to the Phase 10 FastAPI backend. The frontend allows ML practitioners to upload models and datasets, run full adversarial audits, monitor live progress, view vulnerability results with interactive charts, generate adversarial patches, and compare two audit reports side by side. All pages consume the REST and WebSocket APIs exposed at `http://localhost:8080`.

## Glossary

- **Frontend**: The Next.js 14 / React 18 application running at `http://localhost:3000`.
- **Backend**: The KAAL FastAPI server running at `http://localhost:8080`.
- **API_Client**: The `lib/api.ts` module responsible for all HTTP and WebSocket calls to the Backend.
- **Job**: A long-running server-side task identified by a `job_id` UUID string. Status values: `pending`, `running`, `complete`, `failed`.
- **KVS_Score**: The KAAL Vulnerability Score, a float from 0.0 to 10.0 representing overall model risk across five dimensions.
- **Dimension_Score**: One of five sub-scores that compose the KVS_Score (e.g., FGSM Vulnerability, PGD Vulnerability, Patch Vulnerability, Physical Robustness, Confidence Stability).
- **Radar_Chart**: A Recharts RadarChart visualizing the five Dimension_Scores as a pentagon.
- **GradCAM_Image**: A heatmap PNG returned inside the audit result JSON, encoded as a base64 data URL or served from a backend endpoint.
- **Audit_Result**: The full JSON object returned by `GET /api/audit/result/{job_id}`.
- **Progress_Bar**: A UI component showing integer percentage 0–100 and the current step label.
- **Upload_Zone**: A drag-and-drop file input component that also accepts click-to-browse.
- **Risk_Badge**: A coloured label derived from the KVS_Score range (Robust / Low Risk / Medium Risk / High Risk / Critical / Catastrophic).
- **WebSocket_Client**: The browser WebSocket connection to `ws://localhost:8080/ws/{job_id}`.
- **Compare_View**: The `/compare` page that displays two Audit_Results side by side.
- **Patch_Job**: A Job of type `patch` started by `POST /api/patch/generate`.

---

## Requirements

### Requirement 1: Project Bootstrap and Shared Infrastructure

**User Story:** As a developer, I want a properly configured Next.js 14 project with shared layout, API client, and type definitions, so that all pages have a consistent foundation.

#### Acceptance Criteria

1. THE Frontend SHALL serve pages from the `web/frontend/` directory with Next.js 14 using the Pages Router (`pages/` directory).
2. THE Frontend SHALL apply Tailwind CSS 3.4 utility classes for all styling with no additional CSS framework.
3. THE API_Client SHALL send all HTTP requests to `http://localhost:8080` using the native `fetch` API.
4. THE API_Client SHALL open WebSocket connections to `ws://localhost:8080/ws/{job_id}` using the native browser `WebSocket` API.
5. THE Frontend SHALL export TypeScript interfaces for all Backend request and response shapes used by the application.
6. THE Frontend SHALL render a persistent top navigation bar on every page with links to `/`, `/audit`, `/results`, `/patch`, and `/compare`.
7. IF a Backend request returns a non-2xx HTTP status, THEN THE API_Client SHALL throw a typed `ApiError` containing the status code and the `detail` field from the JSON response body.
8. THE Frontend SHALL display a global error toast notification WHEN THE API_Client throws an `ApiError`, so that users receive feedback on request failures.

---

### Requirement 2: Landing Page (`/`)

**User Story:** As a new user, I want a landing page that explains KAAL and links to the key workflows, so that I can quickly understand the tool and start an audit.

#### Acceptance Criteria

1. WHEN a user visits `/`, THE Frontend SHALL render a hero section containing the KAAL name, tagline ("What cannot be seen, cannot be defended."), and a call-to-action button linking to `/audit`.
2. THE Frontend SHALL display a feature summary section listing the four attack modules (FGSM, PGD, Patch, Physical) with a brief description of each.
3. THE Frontend SHALL display the KVS score range table showing all six risk bands (Robust, Low Risk, Medium Risk, High Risk, Critical, Catastrophic) with their numeric ranges.
4. THE Frontend SHALL display navigation links to `/audit`, `/patch`, and `/compare` workflows from the landing page.

---

### Requirement 3: Audit Page (`/audit`) — Upload and Configuration

**User Story:** As an ML engineer, I want to upload a model file and image dataset, configure attack parameters, and start an audit, so that I can assess my model's adversarial robustness.

#### Acceptance Criteria

1. THE Frontend SHALL render an Upload_Zone for the model file that accepts `.pt`, `.pth`, `.h5`, `.keras`, `.onnx`, and `.tflite` file extensions.
2. WHEN a user drops or selects a model file, THE Frontend SHALL call `POST /api/upload/model` and display the returned `framework`, `input_shape`, and `num_classes` as confirmation metadata.
3. THE Frontend SHALL render an Upload_Zone for dataset images that accepts multiple files with extensions `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.
4. WHEN a user drops or selects dataset images, THE Frontend SHALL call `POST /api/upload/dataset` and display the returned image count and format breakdown.
5. THE Frontend SHALL render attack configuration controls including: a multi-select for attacks (`fgsm`, `pgd`, `patch`, `physical`) with initial state of all four selected, an epsilon slider with range 0.001–1.0 and initial value 0.03, a PGD steps input with range 1–200 and initial value 40, and a no-GradCAM toggle with initial state off; users may change any of these values before starting an audit.
6. WHEN a user clicks "Start Audit", THE Frontend SHALL call `POST /api/audit/start` with the stored `model_id`, `dataset_id`, and configuration values, then navigate to `/results?job_id={job_id}`.
7. IF the model Upload_Zone or dataset Upload_Zone has not received a successful upload, THEN THE Frontend SHALL disable the "Start Audit" button.
8. IF `POST /api/upload/model` returns an error, THEN THE Frontend SHALL display the error message inside the model Upload_Zone and clear any previously stored `model_id`.
9. IF `POST /api/upload/dataset` returns an error, THEN THE Frontend SHALL display the error message inside the dataset Upload_Zone and clear any previously stored `dataset_id`.

---

### Requirement 4: Results Page (`/results`) — Live Progress and Audit Results

**User Story:** As an ML engineer, I want to monitor live audit progress and view the full results once complete, so that I can understand my model's vulnerability profile.

#### Acceptance Criteria

1. WHEN the Results page loads with a `job_id` query parameter, THE Frontend SHALL open a WebSocket_Client connection to `ws://localhost:8080/ws/{job_id}`.
2. WHILE the WebSocket_Client is connected and the Job status is `running` or `pending`, THE Frontend SHALL display a Progress_Bar updated by each incoming WebSocket message's `progress_pct` and `step_name` fields.
3. WHEN a WebSocket message with `type` equal to `done` is received, THE Frontend SHALL call `GET /api/audit/result/{job_id}` and render the Audit_Result.
4. WHEN a WebSocket message with `type` equal to `error` is received, THE Frontend SHALL display the `message` field as an error state and stop polling.
5. THE Frontend SHALL display the KVS_Score as a large numeric value alongside its Risk_Badge derived from the score range.
6. THE Frontend SHALL render a Radar_Chart using Recharts with the five Dimension_Scores from the Audit_Result.
7. IF the Audit_Result contains a GradCAM_Image, THEN THE Frontend SHALL display the GradCAM heatmap image.
8. THE Frontend SHALL provide a "Download PDF Report" button that triggers a browser download of `GET /api/report/{job_id}/pdf`.
9. THE Frontend SHALL provide a "Download Patch PNG" button that triggers a browser download of `GET /api/patch/{job_id}/png` WHEN the audit included the patch attack.
10. IF no `job_id` query parameter is present, THEN THE Frontend SHALL display guidance directing the user to start an audit from `/audit`.
11. IF the WebSocket connection fails to establish or is dropped before completion, THEN THE Frontend SHALL fall back to polling `GET /api/audit/status/{job_id}` every 3 seconds.

---

### Requirement 5: Patch Page (`/patch`) — Adversarial Patch Generator

**User Story:** As a security researcher, I want to generate an adversarial patch for a specific target class without running a full audit, so that I can produce a printable patch for physical-world testing.

#### Acceptance Criteria

1. THE Frontend SHALL render an Upload_Zone for the model file accepting the same extensions as Requirement 3.1.
2. THE Frontend SHALL render an Upload_Zone for dataset images accepting the same extensions as Requirement 3.3.
3. THE Frontend SHALL render patch configuration controls including: a target class integer input with minimum 0, a patch fraction slider with range 0.001–0.5 and default 0.05, an iterations input with minimum 1 and default 500, and a print size (cm) input with minimum 0.1 and default 15.0.
4. WHEN a user clicks "Generate Patch", THE Frontend SHALL call `POST /api/patch/generate` and display a Progress_Bar connected to the resulting job via WebSocket_Client.
5. WHEN the patch Job completes, THE Frontend SHALL display the patch result metrics: `attack_success_rate`, `avg_confidence_on_target`, `patch_fraction_used`, `iterations_used`, and `plain_english` summary.
6. THE Frontend SHALL provide a "Download Patch PNG" button linking to `GET /api/patch/{job_id}/png` WHEN the job is complete.
7. THE Frontend SHALL provide a "Download Printable PDF" button linking to `GET /api/patch/{job_id}/printable` WHEN the job is complete.
8. IF both the model Upload_Zone and the dataset Upload_Zone have not received successful uploads, THEN THE Frontend SHALL disable the "Generate Patch" button.

---

### Requirement 6: Compare Page (`/compare`) — Side-by-Side Audit Comparison

**User Story:** As an ML engineer, I want to compare two audit results side by side, so that I can measure whether model changes improved or worsened adversarial robustness.

#### Acceptance Criteria

1. THE Frontend SHALL render two job ID input fields labelled "Before Audit" and "After Audit".
2. WHEN both job ID fields contain non-empty values and a user clicks "Compare", THE Frontend SHALL call `GET /api/compare?before_id={before_id}&after_id={after_id}`.
3. THE Frontend SHALL display the `before` and `after` KVS_Score values side by side with their respective Risk_Badges.
4. THE Frontend SHALL render an overlaid Radar_Chart on the Compare_View showing both audit's Dimension_Scores using two differently coloured series.
5. THE Frontend SHALL display the `delta.overall` KVS score change with a colour-coded indicator: green for negative delta (improvement) and red for positive delta (regression).
6. THE Frontend SHALL display per-dimension delta values for each of the five Dimension_Scores.
7. IF `GET /api/compare` returns a 404 error, THEN THE Frontend SHALL display a message stating that one or both job IDs were not found or not yet complete.

---

### Requirement 7: KVS Score Display and Risk Badge

**User Story:** As an ML engineer, I want consistent KVS score presentation across all pages, so that I can immediately understand risk level without consulting the legend.

#### Acceptance Criteria

1. THE Frontend SHALL implement colour coding logic that assigns Risk_Badge colours based on KVS_Score ranges: green for Robust (0.0–2.0), yellow for Low Risk (2.1–4.0), orange for Medium Risk (4.1–6.0), red for High Risk (6.1–8.0), dark red for Critical (8.1–9.5), and black for Catastrophic (9.6–10.0).
2. THE Frontend SHALL derive the Risk_Badge label deterministically from the KVS_Score value using the six defined ranges.
3. THE Frontend SHALL display the KVS_Score rounded to two decimal places wherever it appears.

---

### Requirement 8: Responsive Layout and Accessibility

**User Story:** As a user on various devices, I want the frontend to be usable on both desktop and tablet screen widths, so that I can access audit results from different contexts.

#### Acceptance Criteria

1. THE Frontend SHALL apply responsive Tailwind CSS breakpoints such that all pages remain usable at viewport widths from 768px to 1920px.
2. THE Frontend SHALL ensure all interactive elements (buttons, inputs, file zones) have visible focus indicators meeting WCAG 2.1 AA contrast requirements.
3. THE Frontend SHALL associate every form input with a `<label>` element using `htmlFor` / `id` linkage.
4. THE Frontend SHALL provide `aria-label` or `aria-labelledby` attributes on all icon-only buttons and Upload_Zone components.
