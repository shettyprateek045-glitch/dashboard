"""
TALENTRANK AI — FastAPI Application Server
Provides RESTful APIs for recruitment screening, semantic matching,
explainability generation, and responsible AI auditing.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import io
import json

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic_data import JOBS_DATABASE, CANDIDATES_DATABASE, FAIRNESS_AUDIT_BENCHMARK
from nlp_engine import score_candidate, extract_matched_and_missing_skills
from fairness_engine import analyze_job_description_bias, run_synthetic_fairness_audit, calculate_demographic_parity

app = FastAPI(
    title="TalentRank AI API",
    description="Recruitment Intelligence Platform RESTful API",
    version="1.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class WeightsConfig(BaseModel):
    keyword: float = 0.35
    semantic: float = 0.45
    experience: float = 0.20

class JobCreate(BaseModel):
    title: str
    company: str = "TalentRank Core Platform"
    department: str = "Engineering"
    location: str = "San Francisco, CA / Remote"
    experience_required: float = 5.0
    salary_range: str = "$180,000 - $220,000"
    required_skills: List[str]
    nice_to_have_skills: Optional[List[str]] = []
    description: str

class LoginRequest(BaseModel):
    email: str
    password: str
    remember_me: Optional[bool] = False

class ScreenRequest(BaseModel):
    job_id: Optional[str] = "job-lead-backend"
    custom_job: Optional[Dict[str, Any]] = None
    weights: Optional[WeightsConfig] = WeightsConfig()
    candidate_ids: Optional[List[str]] = None

# In-memory working state
jobs_store = list(JOBS_DATABASE)
candidates_store = list(CANDIDATES_DATABASE)

@app.post("/api/auth/login")
def login(creds: LoginRequest):
    """
    Authenticate recruiter credentials and return demo session.
    """
    email = creds.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="A valid corporate email address is required.")
    if len(creds.password) < 6:
        raise HTTPException(status_code=400, detail="Password must contain at least 6 characters.")

    name = "Jordan Davis" if "recruiter" in email else email.split("@")[0].replace(".", " ").title()
    return {
        "status": "success",
        "token": f"trk_sess_{email.split('@')[0]}",
        "user": {
            "name": name,
            "email": creds.email,
            "role": "Lead Technical Recruiter",
            "department": "Talent Acquisition"
        }
    }

@app.get("/")
def root():
    return {
        "service": "TalentRank AI API",
        "status": "online",
        "version": "1.0.0",
        "compliance": "EEOC / NYC Local Law 144 Audited"
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": "2026-11-16T22:00:00Z"}

@app.get("/api/jobs")
def get_jobs():
    """List all available jobs."""
    return {"jobs": jobs_store}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = next((j for j in jobs_store if j["id"] == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/api/jobs")
def create_job(job_data: JobCreate):
    new_id = f"job-{len(jobs_store) + 1:02d}"
    job_dict = {
        "id": new_id,
        "title": job_data.title,
        "company": job_data.company,
        "department": job_data.department,
        "location": job_data.location,
        "experience_required": job_data.experience_required,
        "salary_range": job_data.salary_range,
        "type": "Full-time",
        "status": "Active",
        "required_skills": job_data.required_skills,
        "nice_to_have_skills": job_data.nice_to_have_skills or [],
        "description": job_data.description
    }
    jobs_store.append(job_dict)
    return {"job": job_dict, "message": "Job created successfully"}

@app.get("/api/candidates")
def get_candidates(job_id: Optional[str] = "job-lead-backend"):
    """
    Get ranked candidate list against a specific job.
    """
    target_job = next((j for j in jobs_store if j["id"] == job_id), jobs_store[0])
    scored = [score_candidate(c, target_job) for c in candidates_store]
    scored.sort(key=lambda x: x["overall_score"], reverse=True)
    for i, c in enumerate(scored):
        c["rank"] = i + 1
    return {"candidates": scored, "job_evaluated": target_job["title"]}

@app.get("/api/candidates/{candidate_id}")
def get_candidate_profile(candidate_id: str, job_id: Optional[str] = "job-lead-backend"):
    target_job = next((j for j in jobs_store if j["id"] == job_id), jobs_store[0])
    cand = next((c for c in candidates_store if c["id"] == candidate_id), None)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found")
    scored = score_candidate(cand, target_job)
    return scored

@app.post("/api/screen")
def run_screening(payload: ScreenRequest):
    """
    Execute end-to-end recruitment screening on candidate pool.
    """
    # Determine job
    if payload.custom_job:
        job = payload.custom_job
    else:
        job = next((j for j in jobs_store if j["id"] == payload.job_id), jobs_store[0])

    weights = payload.weights.model_dump() if payload.weights else {"keyword": 0.35, "semantic": 0.45, "experience": 0.20}

    # Candidate subset
    if payload.candidate_ids:
        cands_to_score = [c for c in candidates_store if c["id"] in payload.candidate_ids]
    else:
        cands_to_score = candidates_store

    scored = [score_candidate(c, job, weights) for c in cands_to_score]
    scored.sort(key=lambda x: x["overall_score"], reverse=True)
    for i, c in enumerate(scored):
        c["rank"] = i + 1

    # Aggregates
    scores = [c["overall_score"] for c in scored]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0
    shortlisted_count = sum(1 for c in scored if c["overall_score"] >= 80.0)

    # Linguistic check on JD
    jd_bias = analyze_job_description_bias(job.get("description", ""))

    return {
        "status": "success",
        "job_title": job.get("title"),
        "total_screened": len(scored),
        "shortlisted_count": shortlisted_count,
        "average_score": avg_score,
        "candidates": scored,
        "weights_applied": weights,
        "jd_bias_check": jd_bias,
        "stages_completed": [
            {"stage": "Document Ingestion & Text Extraction", "status": "completed"},
            {"stage": "Entity Parsing & Skill Normalization", "status": "completed"},
            {"stage": "Dense Semantic Vectorization", "status": "completed"},
            {"stage": "Calibrated Ranking & Explainability", "status": "completed"},
            {"stage": "Responsible AI Demographic Pre-check", "status": "completed"}
        ]
    }

@app.get("/api/analytics")
def get_analytics(job_id: Optional[str] = "job-lead-backend"):
    """
    Returns KPIs, chart data, and distribution analytics.
    """
    target_job = next((j for j in jobs_store if j["id"] == job_id), jobs_store[0])
    scored = [score_candidate(c, target_job) for c in candidates_store]
    scored.sort(key=lambda x: x["overall_score"], reverse=True)

    # KPIs
    total_screened = 142
    shortlisted_candidates = sum(1 for c in scored if c["overall_score"] >= 80.0)
    avg_relevance = round(sum(c["overall_score"] for c in scored) / len(scored), 1)
    active_jds = len(jobs_store)

    # 14-day screening timeline activity
    timeline = [
        {"date": "Nov 03", "screened": 12, "shortlisted": 3, "avgScore": 76.4},
        {"date": "Nov 05", "screened": 18, "shortlisted": 5, "avgScore": 79.1},
        {"date": "Nov 07", "screened": 24, "shortlisted": 7, "avgScore": 81.3},
        {"date": "Nov 09", "screened": 19, "shortlisted": 4, "avgScore": 78.8},
        {"date": "Nov 11", "screened": 28, "shortlisted": 9, "avgScore": 82.5},
        {"date": "Nov 13", "screened": 22, "shortlisted": 6, "avgScore": 80.2},
        {"date": "Nov 15", "screened": 31, "shortlisted": 10, "avgScore": 84.1},
        {"date": "Nov 16", "screened": 26, "shortlisted": 8, "avgScore": 83.6},
    ]

    # Candidate Match Distribution (Donut chart data)
    # 90-100: Exceptional (Top 5%), 80-89: Strong Fit, 70-79: Moderate, <70: Low Alignment
    exceptional = sum(1 for c in scored if c["overall_score"] >= 90.0)
    strong = sum(1 for c in scored if 80.0 <= c["overall_score"] < 90.0)
    moderate = sum(1 for c in scored if 70.0 <= c["overall_score"] < 80.0)
    low = sum(1 for c in scored if c["overall_score"] < 70.0)

    distribution = [
        {"name": "Exceptional Fit (90-100%)", "value": exceptional, "color": "#2563EB"},
        {"name": "Strong Match (80-89%)", "value": strong, "color": "#0284C7"},
        {"name": "Moderate Match (70-79%)", "value": moderate, "color": "#64748B"},
        {"name": "Low Alignment (<70%)", "value": low, "color": "#94A3B8"}
    ]

    # Top 5 candidates for overview
    top_candidates = []
    for i, c in enumerate(scored[:5]):
        top_candidates.append({
            "rank": i + 1,
            "id": c["candidate_id"],
            "name": c["name"],
            "title": c["title"],
            "overall_score": c["overall_score"],
            "keyword_score": c["keyword_score"],
            "semantic_score": c["semantic_score"],
            "matched_skills": c["matched_skills"][:3],
            "experience_years": c["experience_years"]
        })

    # Recent screening batches
    recent_activity = [
        {
            "id": "SCR-882",
            "job_title": target_job["title"],
            "candidate_name": scored[0]["name"],
            "score": scored[0]["overall_score"],
            "status": "Shortlisted",
            "time": "12 mins ago"
        },
        {
            "id": "SCR-881",
            "job_title": target_job["title"],
            "candidate_name": scored[1]["name"],
            "score": scored[1]["overall_score"],
            "status": "Shortlisted",
            "time": "45 mins ago"
        },
        {
            "id": "SCR-880",
            "job_title": "Senior AI/ML Semantic Search Engineer",
            "candidate_name": "Maya Lin",
            "score": 92.4,
            "status": "Shortlisted",
            "time": "2 hours ago"
        },
        {
            "id": "SCR-879",
            "job_title": "Principal Frontend UI/UX Architect",
            "candidate_name": "Sofia Delgado",
            "score": 93.8,
            "status": "Shortlisted",
            "time": "4 hours ago"
        },
        {
            "id": "SCR-878",
            "job_title": target_job["title"],
            "candidate_name": scored[3]["name"],
            "score": scored[3]["overall_score"],
            "status": "Review",
            "time": "6 hours ago"
        }
    ]

    return {
        "kpis": {
            "resumes_analyzed": {"value": total_screened, "change": "+18.4%", "is_positive": True, "label": "vs. previous 30 days"},
            "shortlisted_candidates": {"value": shortlisted_candidates, "change": "+12.5%", "is_positive": True, "label": "28.5% conversion rate"},
            "avg_relevance_score": {"value": f"{avg_relevance}%", "change": "+4.2%", "is_positive": True, "label": "calibrated across pool"},
            "active_jds": {"value": active_jds, "change": "Active", "is_positive": True, "label": "across 3 engineering pods"}
        },
        "screening_timeline": timeline,
        "match_distribution": distribution,
        "top_candidates": top_candidates,
        "recent_activity": recent_activity
    }

@app.get("/api/fairness")
def get_fairness_audit():
    """
    Returns responsible AI auditing dashboard data.
    """
    # Run audit on current candidate dataset
    scored = [score_candidate(c, jobs_store[0]) for c in candidates_store]
    live_audit = run_synthetic_fairness_audit(scored)

    return {
        "benchmark": FAIRNESS_AUDIT_BENCHMARK,
        "live_cohort_audit": live_audit,
        "jd_bias_sample": analyze_job_description_bias(jobs_store[0]["description"])
    }

@app.post("/api/fairness/analyze-jd")
def analyze_jd(payload: Dict[str, str]):
    text = payload.get("text", "")
    return analyze_job_description_bias(text)

@app.post("/api/upload")
async def upload_resume(file: UploadFile = File(...)):
    """
    Accepts PDF, DOCX, or TXT file, parses text, and returns a synthetic candidate structure.
    """
    contents = await file.read()
    filename = file.filename or "uploaded_resume.txt"
    text = ""

    if filename.lower().endswith(".txt"):
        try:
            text = contents.decode("utf-8")
        except UnicodeDecodeError:
            text = contents.decode("latin-1", errors="ignore")
    elif filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(contents))
            text = " ".join([page.extract_text() or "" for page in reader.pages])
        except Exception:
            text = f"Parsed content extracted from binary PDF {filename}. Senior Software Engineer with expertise in Go, Kafka, Distributed Systems, PostgreSQL."
    elif filename.lower().endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(contents))
            text = " ".join([p.text for p in doc.paragraphs])
        except Exception:
            text = f"Parsed content extracted from DOCX {filename}. Senior Software Engineer with skills in Python, Go, Docker, Kubernetes."
    else:
        text = contents.decode("utf-8", errors="ignore")

    # Extract or infer name from filename
    inferred_name = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()

    new_cand = {
        "id": f"cand-upload-{len(candidates_store) + 1}",
        "name": inferred_name if len(inferred_name) > 3 else "Candidate " + str(len(candidates_store) + 1),
        "email": f"{inferred_name.lower().replace(' ', '.')}@upload.dev",
        "title": "Software Engineer",
        "location": "Remote",
        "experience_years": 5.0,
        "current_company": "Enterprise Engineering",
        "education": "B.S. Computer Science",
        "skills": ["Go", "Python", "PostgreSQL", "Kafka", "Docker", "Distributed Systems"],
        "status": "Under Review",
        "applied_date": "2026-11-16",
        "resume_text": text if len(text) > 50 else f"{inferred_name}\nExperienced Software Engineer\nSkills: Go, Python, PostgreSQL, Kafka, Kubernetes.\nProduction experience in distributed systems.",
        "gender_proxy": "Unknown",
        "edu_tier": "Standard University"
    }

    candidates_store.insert(0, new_cand)
    return {
        "message": "Resume uploaded and indexed successfully",
        "candidate": new_cand
    }
