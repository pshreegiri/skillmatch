from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import re

app = Flask(__name__)

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------
# PDF TEXT EXTRACTION
# ---------------------------
def extract_text_pdf(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text.lower()


# ---------------------------
# EXTRACT JD REQUIREMENTS
# ---------------------------
def extract_requirements(text):
    sentences = re.split(r"[.\n]", text)

    trigger_words = [
        "experience", "knowledge", "understanding", "familiarity",
        "ability", "skills", "using", "develop", "design",
        "implement", "integrate", "build", "deploy",
        "debug", "maintain", "optimize",
        "analyze", "train", "evaluate", "create", "manage"
    ]

    strong, weak = [], []

    for s in sentences:
        s = s.strip()
        if len(s.split()) < 5:
            continue

        clean = re.sub(r"[^a-zA-Z0-9\s]", "", s)

        if any(t in s for t in trigger_words):
            strong.append(clean)
        else:
            weak.append(clean)

    return list(set(strong)) if strong else list(set(weak[:12]))


# ---------------------------
# EXTRACT MISSING KEYWORDS
# ---------------------------
def extract_keywords(phrases, resume_text):
    resume_words = set(resume_text.split())

    stopwords = {
        "using", "with", "such", "and", "or", "to", "of", "for", "from",
        "ability", "skills", "experience", "knowledge", "understanding",
        "develop", "design", "implement", "maintain", "validate",
        "think", "logically", "problem", "learn",
        "guidance", "mentorship", "candidate",
        "should", "have", "has", "will", "can", "you", "your",
        "work", "working", "team", "environment", "opportunity",
        "role", "responsibility", "plus", "strong", "ideal"
    }

    important_skill_map = {
        "debug": "Frontend Testing & Debugging",
        "test": "Frontend Testing & Debugging",
        "testing": "Frontend Testing & Debugging",
        "accessibility": "Accessibility",
        "performance": "Performance Optimization",
        "designers": "UI/UX Collaboration",
        "designer": "UI/UX Collaboration",
        "product": "Product Team Collaboration",
        "components": "Reusable Components",
        "component": "Reusable Components",
        "frontend": "Frontend Development",
        "react": "React.js",
        "state": "State Management",
        "redux": "Redux",
        "git": "Git/GitHub",
        "github": "Git/GitHub",
        "api": "REST APIs",
        "backend": "Backend Collaboration",
        "developers": "Backend Collaboration",
        "modern": "Modern Web Practices",
        "creative": "Creative Thinking",
        "creatively": "Creative Thinking",
        "fresh": "Creative Thinking"
    }

    keywords = set()

    for phrase in phrases:
        words = [
            w for w in phrase.split()
            if w.isalpha()
            and len(w) > 3
            and w not in stopwords
            and w not in resume_words
        ]

        found_skill = None

        for word in words:
            if word in important_skill_map:
                found_skill = important_skill_map[word]
                break

        if found_skill:
            keywords.add(found_skill)
        elif len(words) >= 2:
            keywords.add(" ".join(words[:2]).title())
        elif len(words) == 1:
            keywords.add(words[0].title())

    return sorted(keywords)


# ---------------------------
# SECTION-WISE ATS ANALYSIS
# ---------------------------
def section_wise_analysis(resume_text, jd_text):

    section_keywords = {
        "Skills Match": [
            "python", "java", "sql", "react", "node", "mongodb",
            "aws", "html", "css", "javascript", "machine learning"
        ],
        "Projects Match": [
            "project", "developed", "built", "application",
            "system", "implementation", "api", "website"
        ],
        "Experience Match": [
            "experience", "internship", "worked", "company",
            "developer", "engineer", "team", "client"
        ],
        "Education Match": [
            "bachelor", "master", "degree", "cgpa",
            "college", "university", "education"
        ],
        "Soft Skills Match": [
            "communication", "leadership", "teamwork",
            "collaboration", "problem solving", "creativity"
        ]
    }

    scores = {}

    for section, words in section_keywords.items():
        matched = 0

        for word in words:
            if word in jd_text:
                if word in resume_text:
                    matched += 1

        total = sum(1 for word in words if word in jd_text)

        if total == 0:
            scores[section] = "N/A"
        else:
            scores[section] = round((matched / total) * 100, 2)

    return scores


# ---------------------------
# RESUME IMPROVEMENT TIPS
# ---------------------------
def generate_resume_tips(section_scores, missing_keywords):
    tips = []

    skills_score = section_scores.get("Skills Match")
    projects_score = section_scores.get("Projects Match")
    experience_score = section_scores.get("Experience Match")
    soft_skills_score = section_scores.get("Soft Skills Match")

    if isinstance(skills_score, (int, float)) and skills_score < 60:
        tips.append({
            "priority": "High",
            "message": "Add more technical skills that directly match the job description."
        })

    if isinstance(projects_score, (int, float)) and projects_score < 60:
        tips.append({
            "priority": "High",
            "message": "Include stronger project descriptions with technologies and measurable outcomes."
        })

    if isinstance(experience_score, (int, float)) and experience_score < 50:
        tips.append({
            "priority": "Medium",
            "message": "Mention internships, freelance work, team collaboration, or practical experience."
        })

    if isinstance(soft_skills_score, (int, float)) and soft_skills_score < 50:
        tips.append({
            "priority": "Low",
            "message": "Add communication, leadership, teamwork, and collaboration points."
        })

    if missing_keywords:
        tips.append({
            "priority": "High",
            "message": f"Try adding keywords like: {', '.join(missing_keywords[:5])}"
        })

    return tips[:5]


# ---------------------------
# CORE ATS SCORING FUNCTION
# ---------------------------
def calculate_ats_score(resume_text, jd_text):

    jd_reqs = extract_requirements(jd_text)
    if not jd_reqs:
        return 0, [], []

    jd_emb = model.encode(jd_reqs, convert_to_tensor=True)

    resume_sentences = [
        s.strip() for s in re.split(r"[.\n]", resume_text)
        if len(s.split()) > 5
    ] or [resume_text]

    resume_emb = model.encode(resume_sentences, convert_to_tensor=True)

    matched = 0
    missing = []

    for i, req in enumerate(jd_reqs):
        sim = util.cos_sim(jd_emb[i], resume_emb)
        if sim.max() >= 0.5:
            matched += 1
        else:
            missing.append(req)

    ats_score = round((matched / len(jd_reqs)) * 100, 2)
    keywords = extract_keywords(missing, resume_text)

    return ats_score, missing, keywords


# ---------------------------
# DASHBOARD ROUTE
# ---------------------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# ---------------------------
# SINGLE RESUME MODE
# ---------------------------
@app.route("/personal", methods=["GET", "POST"])
def personal():

    if request.method == "POST":
        resume_pdf = request.files["resume_pdf"]
        jd_pdf = request.files["jd_pdf"]

        resume_text = extract_text_pdf(resume_pdf)
        jd_text = extract_text_pdf(jd_pdf)

        ats_score, missing, keywords = calculate_ats_score(resume_text, jd_text)

        section_scores = section_wise_analysis(resume_text, jd_text)
        tips = generate_resume_tips(section_scores, keywords)

        if ats_score >= 70:
            recommendation = "Great match! Your resume is well aligned."
        elif ats_score >= 40:
            recommendation = "It's good but there is still room for improvement."
        else:
            recommendation = "Low match. Improve job-specific skills."

        return render_template(
            "home.html",
            msg=f"ATS Match Score: {ats_score}%",
            recommendation=recommendation,
            missing=keywords[:8],
            section_scores=section_scores,
            tips=tips
        )

    return render_template("home.html")


# ---------------------------
# BULK UPLOAD PAGE
# ---------------------------
@app.route("/bulk")
def bulk():
    return render_template("bulk_ranking.html")


# ---------------------------
# BULK RANKING LOGIC
# ---------------------------
@app.route("/rank_resumes", methods=["POST"])
def rank_resumes():

    jd_pdf = request.files["job_description"]
    resume_files = request.files.getlist("resumes")

    if len(resume_files) > 15:
        return render_template("bulk_ranking.html", error="Maximum 15 resumes allowed.")

    jd_text = extract_text_pdf(jd_pdf)
    jd_requirements = extract_requirements(jd_text)

    results = []
    all_missing_keywords = []
    all_present_keywords = []

    for resume in resume_files:
        resume_text = extract_text_pdf(resume)

        score, missing, keywords = calculate_ats_score(resume_text, jd_text)

        results.append({
            "resume_name": resume.filename,
            "score": score
        })

        all_missing_keywords.extend(keywords)

        for req in jd_requirements:
            clean_req = req.strip().lower()
            if clean_req and clean_req in resume_text:
                all_present_keywords.append(clean_req)

    ranked_results = sorted(results, key=lambda x: x["score"], reverse=True)

    missing_count = {}
    for keyword in all_missing_keywords:
        missing_count[keyword] = missing_count.get(keyword, 0) + 1

    top_missing_skills = sorted(
        missing_count.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    present_count = {}
    for keyword in all_present_keywords:
        present_count[keyword] = present_count.get(keyword, 0) + 1

    top_present_skills = sorted(
        present_count.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    high_ready = len([r for r in ranked_results if r["score"] >= 70])
    moderate_ready = len([r for r in ranked_results if 40 <= r["score"] < 70])
    low_ready = len([r for r in ranked_results if r["score"] < 40])

    training_topics = [skill[0] for skill in top_missing_skills[:4]]

    return render_template(
        "ranking_result.html",
        results=ranked_results,
        top_missing_skills=top_missing_skills,
        top_present_skills=top_present_skills,
        high_ready=high_ready,
        moderate_ready=moderate_ready,
        low_ready=low_ready,
        training_topics=training_topics
    )


if __name__ == "__main__":
    app.run(debug=True)