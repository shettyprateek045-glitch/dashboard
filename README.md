# TALENTRANK AI — Recruitment Intelligence Platform

TalentRank AI is a recruitment intelligence workspace. It opens directly to a login screen, then presents screening, candidates, analytics, AI insights, and fairness audit tools. The sign-in screen currently uses demo authentication: the API validates email syntax and a minimum password length but does not verify an account. Configure a real identity provider before using it to protect data. Password recovery is not configured.

A human-designed, enterprise SaaS recruitment intelligence platform built with **TypeScript**, **React**, **Tailwind CSS**, and **FastAPI**.

Inspired by the design quality and craftsmanship of **Linear**, **Stripe**, **Notion**, **Vercel**, and **Raycast**, TalentRank AI delivers precision semantic screening, algorithmic explainability, and responsible AI auditing (EEOC & NYC Local Law 144 compliance).

---

## Architecture Overview

```
hack/
├── backend/                  # Python FastAPI Backend
│   ├── main.py               # REST API endpoints (Screening, Candidates, Analytics, Fairness)
│   ├── nlp_engine.py         # Pure Python TF-IDF, semantic cosine alignment & explainability
│   ├── fairness_engine.py    # Demographic Parity, 4/5ths Rule & JD Linguistic Bias Scanner
│   ├── synthetic_data.py     # Authentic candidate profiles, job requisitions & audit cohorts
│   └── requirements.txt      # Python dependencies
│
└── frontend/                 # TypeScript + React + Vite Frontend
    ├── src/
    │   ├── components/
    │   │   ├── Navbar.tsx             # Top navigation, active job picker, blind screening toggle
    │   │   ├── Sidebar.tsx            # Minimal Linear-style navigation sidebar
    │   │   ├── Overview.tsx           # Hiring overview, 4 KPIs, screening chart, donut distribution
    │   │   ├── Screening.tsx          # Two-column screening workspace with multi-stage progress
    │   │   ├── Candidates.tsx         # Sophisticated candidate table, search, filters, CSV export
    │   │   ├── CandidateDrawer.tsx    # Slide-over candidate profile, score breakdown, resume viewer
    │   │   ├── CandidateCompareModal.tsx # Side-by-side head-to-head candidate comparison matrix
    │   │   ├── AiAnalysis.tsx         # Model weight simulator, 2D semantic scatter plot
    │   │   ├── FairnessAudit.tsx      # Responsible AI center, group selection rates, NYC LL144 cert
    │   │   ├── Analytics.tsx          # Hiring funnel, score histogram, talent market gap
    │   │   ├── Reports.tsx            # Executive screening dossier generator with print styling
    │   │   └── Settings.tsx           # Algorithmic thresholds & blind mode controls
    │   ├── api.ts                     # Typed API client with seamless offline/mock fallback
    │   ├── types.ts                   # Comprehensive TypeScript type definitions
    │   ├── mockData.ts                # High-fidelity typed synthetic candidate & audit database
    │   ├── App.tsx                    # Main application coordinator
    │   ├── main.tsx                   # React root entry point
    │   ├── index.css                  # Tailored design system styling & custom scrollbars
    │   └── vite-env.d.ts              # Vite TypeScript declarations
    ├── tailwind.config.js             # Refined neutral color palette & typography tokens
    ├── tsconfig.json                  # TypeScript compiler configuration
    └── package.json
```

---

## Visual Design System & Palette

- **Background:** `#F8FAFC`
- **Sidebar & Cards:** `#FFFFFF`
- **Primary Text:** `#0F172A`
- **Secondary Text:** `#475569`
- **Borders:** `#E2E8F0`
- **Accent:** `#2563EB`
- **Success:** `#16845B`
- **Warning:** `#C58A29`
- **Danger:** `#C94B4B`
- **Corner Radii:** 6–10px
- **Typography:** Inter, Geist, and JetBrains Mono

---

## Key Features

1. **Overview Dashboard:**
   - 4 KPIs with trend indicators: Resumes Analyzed, Shortlisted Candidates, Average Relevance Score, Active Job Descriptions.
   - Interactive Candidate Screening Activity line chart with 7D/30D/90D ranges.
   - Restrained candidate match distribution donut chart.
   - Recent screening activity log and top-ranked candidates list.

2. **Screening Workspace:**
   - Two-column layout: Job description editor on left, resume upload panel on right.
   - Supports Drag & Drop for PDF, DOCX, and TXT resumes.
   - Real-time linguistic bias checker for job descriptions.
   - Multi-stage screening progress: Ingestion → Parsing → Semantic Embedding → Calibration → Fairness Pre-check.

3. **Candidate Management:**
   - Sortable, searchable, and multi-filtered table (by Score, Status, Experience, Skills).
   - Multi-select checkbox with **Head-to-Head Candidate Comparison**.
   - One-click **Export to CSV**.
   - Detailed candidate profile slide-over with verified skills, gap analysis, and original resume viewer.

4. **AI Analysis & Transparency:**
   - **Interactive Model Weight Simulator:** Recalculates candidate rankings dynamically by adjusting Semantic, Keyword, and Seniority weights.
   - **2D Semantic vs. Lexical Alignment Scatter Plot:** Identifies non-traditional talent with high semantic fit.
   - Plain-English explainability for every score and recommendation.

5. **Responsible AI & Fairness Audit:**
   - Decoupled from candidate selection: Zero protected attributes in scoring algorithms.
   - Demographic Parity Difference (DPD) and Disparate Impact Ratio (4/5ths Rule) evaluation.
   - Group selection rates bar chart across Gender, Education, and Experience cohorts.
   - Annual NYC Local Law 144 Compliance Certificate preview and verification.
   - Global **Blind Screening** toggle to mask names, schools, and graduation years.

---

## How to Run

Run commands from the repository root.

### 1. Frontend (TypeScript + React + Vite)
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### 2. Backend (FastAPI)
```bash
python -m uvicorn backend.main:app --port 8000 --reload
```
API documentation available at [http://localhost:8000/docs](http://localhost:8000/docs).
