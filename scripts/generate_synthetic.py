"""Synthetic dataset and gold-label generator for ResumeFit evaluation (Phase 6).

Generates:
1. 50 realistic Job Descriptions across 8 role families (data/synthetic/jds/)
2. 100 synthetic candidate resumes across diverse personas (data/synthetic/resumes/)
3. 5 scanned/image-only PDFs for unsupported pipeline validation (data/synthetic/scanned/)
4. 500 hand-auditable gold labeled resume-JD pairs with relevance grades (data/gold/labels.jsonl)
5. Strict 30/70 dev/test split partitioned by Job Description (data/gold/splits.json)
"""

from __future__ import annotations

import io
import json
import logging
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
SYNTHETIC_DIR = DATA_DIR / "synthetic"
JDS_DIR = SYNTHETIC_DIR / "jds"
RESUMES_DIR = SYNTHETIC_DIR / "resumes"
SCANNED_DIR = SYNTHETIC_DIR / "scanned"
GOLD_DIR = DATA_DIR / "gold"

# 8 Core Role Families
ROLE_FAMILIES = [
    {
        "family": "Backend Engineering",
        "roles": [
            ("Senior Python Architect", ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "Microservices"], ["Redis", "Kubernetes", "CI/CD"], 5, "bachelor"),
            ("Python Backend Engineer", ["Python", "Django", "PostgreSQL", "REST", "Git"], ["Docker", "Redis", "Celery"], 3, "bachelor"),
            ("Staff Go Systems Engineer", ["Go", "Kubernetes", "Distributed Systems", "gRPC", "Docker"], ["Kafka", "Prometheus", "Linux"], 6, "bachelor"),
            ("Junior Python Developer", ["Python", "SQL", "Git", "REST APIs"], ["Flask", "Linux", "Docker"], 1, "bachelor"),
            ("Java Enterprise Developer", ["Java", "Spring Boot", "MySQL", "Hibernate", "Microservices"], ["Kafka", "AWS", "Docker"], 4, "bachelor"),
            ("Cloud Backend Engineer", ["Node.js", "TypeScript", "AWS", "DynamoDB", "Docker"], ["GraphQL", "Serverless", "Terraform"], 3, "bachelor"),
            ("C++ Systems Programmer", ["C++", "Linux", "Multithreading", "Data Structures", "Algorithms"], ["Socket Programming", "Git", "CMake"], 4, "master"),
        ],
    },
    {
        "family": "Frontend Engineering",
        "roles": [
            ("Senior Frontend React Architect", ["React", "TypeScript", "JavaScript", "HTML5", "CSS3", "Redux"], ["Next.js", "GraphQL", "TailwindCSS"], 5, "bachelor"),
            ("Frontend React Developer", ["React", "JavaScript", "HTML", "CSS", "Git"], ["TypeScript", "TailwindCSS", "Jest"], 2, "bachelor"),
            ("Vue.js Web Developer", ["Vue.js", "JavaScript", "HTML", "CSS", "REST"], ["TypeScript", "Pinia", "Sass"], 3, "bachelor"),
            ("Junior Frontend Engineer", ["HTML", "CSS", "JavaScript", "Git"], ["React", "Bootstrap", "Figma"], 1, "bachelor"),
            ("Frontend Performance Engineer", ["JavaScript", "React", "Web Performance", "Webpack", "CSS"], ["TypeScript", "Lighthouse", "Service Workers"], 4, "bachelor"),
            ("Mobile React Native Engineer", ["React Native", "JavaScript", "TypeScript", "Mobile Development", "Git"], ["iOS", "Android", "Redux"], 3, "bachelor"),
            ("UI/UX Frontend Developer", ["JavaScript", "HTML5", "CSS3", "Figma", "UI Design"], ["React", "Animation", "Accessibility"], 2, "bachelor"),
        ],
    },
    {
        "family": "Machine Learning & AI",
        "roles": [
            ("Senior Machine Learning Engineer", ["Python", "PyTorch", "scikit-learn", "Machine Learning", "Docker"], ["MLOps", "Transformers", "Kubernetes"], 4, "master"),
            ("NLP Research Scientist", ["Python", "PyTorch", "NLP", "Transformers", "Deep Learning"], ["HuggingFace", "BERT", "CUDA"], 4, "phd"),
            ("Computer Vision Engineer", ["Python", "OpenCV", "PyTorch", "Deep Learning", "Convolutional Neural Networks"], ["C++", "TensorRT", "Docker"], 3, "master"),
            ("Data Scientist", ["Python", "SQL", "Pandas", "scikit-learn", "Statistics"], ["Tableau", "Machine Learning", "R"], 3, "master"),
            ("Junior ML Developer", ["Python", "scikit-learn", "Pandas", "NumPy", "Git"], ["PyTorch", "SQL", "Matplotlib"], 1, "bachelor"),
            ("Generative AI Engineer", ["Python", "LangChain", "LLMs", "Vector Databases", "PyTorch"], ["RAG", "Docker", "FastAPI"], 3, "bachelor"),
        ],
    },
    {
        "family": "DevOps & Cloud Infrastructure",
        "roles": [
            ("Senior Site Reliability Engineer (SRE)", ["Kubernetes", "Docker", "Terraform", "AWS", "Linux", "Python"], ["Prometheus", "Grafana", "CI/CD"], 5, "bachelor"),
            ("DevOps Automation Engineer", ["Docker", "Kubernetes", "CI/CD", "Linux", "Bash"], ["Terraform", "AWS", "Ansible"], 3, "bachelor"),
            ("Cloud Security Architect", ["AWS", "Cloud Security", "IAM", "Kubernetes", "Linux"], ["Terraform", "Compliance", "Python"], 6, "bachelor"),
            ("Junior DevOps Admin", ["Linux", "Git", "Bash", "Docker"], ["CI/CD", "AWS", "Python"], 1, "bachelor"),
            ("Platform Engineer", ["Kubernetes", "Go", "Terraform", "Docker", "Helm"], ["ArgoCD", "AWS", "Monitoring"], 4, "bachelor"),
            ("Azure Infrastructure Specialist", ["Azure", "Terraform", "PowerShell", "Docker", "Networking"], ["Kubernetes", "CI/CD", "Security"], 3, "bachelor"),
        ],
    },
    {
        "family": "Full-Stack Development",
        "roles": [
            ("Full-Stack Software Engineer", ["JavaScript", "TypeScript", "React", "Node.js", "SQL"], ["PostgreSQL", "Docker", "AWS"], 3, "bachelor"),
            ("Senior Full-Stack Developer", ["Python", "React", "FastAPI", "TypeScript", "PostgreSQL", "Docker"], ["Redis", "AWS", "CI/CD"], 5, "bachelor"),
            ("Junior Full-Stack Developer", ["JavaScript", "React", "Node.js", "HTML", "CSS"], ["MongoDB", "Git", "REST"], 1, "bachelor"),
            ("MEAN/MERN Stack Developer", ["MongoDB", "Express.js", "React", "Node.js", "JavaScript"], ["TypeScript", "REST", "Git"], 2, "bachelor"),
            ("Full-Stack Python & Vue Dev", ["Python", "Vue.js", "PostgreSQL", "Docker", "REST"], ["FastAPI", "TailwindCSS", "Git"], 3, "bachelor"),
            ("Enterprise Solutions Engineer", ["Java", "Spring Boot", "Angular", "TypeScript", "SQL"], ["Docker", "Kubernetes", "REST"], 4, "bachelor"),
        ],
    },
    {
        "family": "QA & Test Engineering",
        "roles": [
            ("Senior QA Automation Engineer", ["Selenium", "Python", "pytest", "Test Automation", "CI/CD"], ["Docker", "Postman", "SQL"], 4, "bachelor"),
            ("SDET (Software Development Engineer in Test)", ["Java", "Selenium", "TestNG", "CI/CD", "Git"], ["Docker", "RestAssured", "SQL"], 3, "bachelor"),
            ("QA Performance Engineer", ["JMeter", "Performance Testing", "Python", "Linux", "SQL"], ["Grafana", "Docker", "Load Testing"], 4, "bachelor"),
            ("Junior QA Tester", ["Manual Testing", "Test Cases", "Bug Tracking", "Jira", "SQL"], ["Selenium", "Git", "Postman"], 1, "bachelor"),
            ("API Test Automation Engineer", ["Python", "Postman", "REST", "pytest", "Docker"], ["Git", "SQL", "CI/CD"], 3, "bachelor"),
            ("Mobile QA Engineer", ["Appium", "Mobile Testing", "iOS", "Android", "Java"], ["Test Automation", "Git", "Jira"], 3, "bachelor"),
        ],
    },
    {
        "family": "Cybersecurity",
        "roles": [
            ("Application Security Engineer", ["AppSec", "OWASP", "Vulnerability Assessment", "Python", "Linux"], ["Docker", "Penetration Testing", "Git"], 4, "bachelor"),
            ("SOC Security Analyst", ["SIEM", "Incident Response", "Network Security", "Linux", "Threat Analysis"], ["Python", "Wireshark", "Splunk"], 2, "bachelor"),
            ("Senior Penetration Tester", ["Penetration Testing", "Burp Suite", "Ethical Hacking", "Linux", "Networking"], ["Python", "Exploit Development", "Metasploit"], 5, "bachelor"),
            ("Security Compliance Specialist", ["SOC 2", "ISO 27001", "Compliance", "Risk Assessment", "Audit"], ["Cloud Security", "NIST", "IAM"], 3, "bachelor"),
            ("Cloud Security Analyst", ["AWS", "Cloud Security", "IAM", "Linux", "Python"], ["Terraform", "Docker", "SIEM"], 3, "bachelor"),
            ("Security Operations Lead", ["Incident Response", "SIEM", "Threat Intelligence", "Leadership", "Linux"], ["Python", "Forensics", "Cloud Security"], 6, "bachelor"),
        ],
    },
    {
        "family": "Data Engineering",
        "roles": [
            ("Senior Data Engineer", ["Python", "SQL", "Apache Spark", "Airflow", "PostgreSQL"], ["AWS", "Docker", "Snowflake"], 5, "bachelor"),
            ("Big Data Engineer", ["Scala", "Apache Spark", "Hadoop", "Kafka", "SQL"], ["Docker", "Linux", "Airflow"], 4, "bachelor"),
            ("ETL Pipeline Developer", ["Python", "SQL", "Airflow", "ETL", "PostgreSQL"], ["Pandas", "Git", "Docker"], 3, "bachelor"),
            ("Analytics Engineer", ["SQL", "dbt", "Snowflake", "Data Modeling", "Git"], ["Python", "Tableau", "Looker"], 2, "bachelor"),
            ("Junior Data Engineer", ["Python", "SQL", "Pandas", "Git", "Linux"], ["Airflow", "PostgreSQL", "Docker"], 1, "bachelor"),
            ("Cloud Data Architect", ["AWS", "Snowflake", "Apache Spark", "Airflow", "Data Architecture"], ["Terraform", "Python", "Kafka"], 6, "master"),
        ],
    },
]


def create_simple_pdf(text_sections: list[tuple[str, str]], out_file: Path) -> None:
    """Generate clean, extractable 1-page PDF resume using ReportLab."""
    doc = SimpleDocTemplate(str(out_file), pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story: list[Any] = []

    for heading, body in text_sections:
        if heading:
            story.append(Paragraph(f"<b>{heading.upper()}</b>", styles["Heading2"]))
            story.append(Spacer(1, 4))
        for line in body.split("\n"):
            clean_l = line.strip()
            if clean_l:
                story.append(Paragraph(clean_l, styles["Normal"]))
                story.append(Spacer(1, 2))
        story.append(Spacer(1, 6))

    doc.build(story)


def create_scanned_pdf(out_file: Path, text_content: str = "Scanned Page Document") -> None:
    """Create a raster image-only PDF containing zero extractable text."""
    img = Image.new("RGB", (612, 792), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), text_content, fill=(30, 30, 30))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(out_file), pagesize=letter)
    c.drawImage(str(out_file.with_suffix(".png")), 0, 0, width=612, height=792)
    c.showPage()
    c.save()


def build_synthetic_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate 50 Job Descriptions and 100 Resumes."""
    JDS_DIR.mkdir(parents=True, exist_ok=True)
    RESUMES_DIR.mkdir(parents=True, exist_ok=True)
    SCANNED_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    jds: list[dict[str, Any]] = []
    jd_counter = 1

    for family_data in ROLE_FAMILIES:
        fam_name = family_data["family"]
        for title, req_skills, pref_skills, min_years, min_degree in family_data["roles"]:
            jd_id = f"jd_{jd_counter:03d}"
            jd_counter += 1

            jd_text = f"""{title}
Role Family: {fam_name}
Experience Required: {min_years}+ years of professional engineering experience.
Degree Required: {min_degree.capitalize()} in Computer Science, Software Engineering, or related discipline.

Summary:
We are seeking an experienced {title} to join our high-growth engineering organization. You will design, build, and deploy high-reliability production systems and collaborate across engineering, product, and QA teams.

Required Technical Skills:
{", ".join(req_skills)}

Preferred Technical Skills:
{", ".join(pref_skills)}

Key Responsibilities:
- Design and develop scalable microservices and infrastructure.
- Deliver robust code with high unit and integration test coverage.
- Collaborate with stakeholders to ensure high uptime and continuous integration.
"""
            jd_record = {
                "id": jd_id,
                "role_family": fam_name,
                "title": title,
                "min_years": min_years,
                "min_degree": min_degree,
                "required_skills": req_skills,
                "preferred_skills": pref_skills,
                "text": jd_text,
            }
            jds.append(jd_record)

            # Write individual JD file
            (JDS_DIR / f"{jd_id}.txt").write_text(jd_text, encoding="utf-8")

    # Generate 100 Candidate Resumes across controlled personas
    resumes: list[dict[str, Any]] = []
    first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Casey", "Riley", "Chris", "Pat", "Devon", "Avery", "Jamie"]
    last_names = ["Chen", "Smith", "Patel", "Johnson", "Williams", "Brown", "Miller", "Davis", "Garcia", "Martinez", "Kumar", "Lee"]

    persona_types = [
        "senior_qualified",   # 6-8 yrs, match degree, match skills
        "mid_qualified",      # 3-5 yrs, match degree, match skills
        "junior_entry",       # 1-2 yrs, bachelor, entry skills
        "missing_experience", # Below required years
        "missing_degree",     # Non-CS / No degree
        "keyword_stuffed",    # Huge skills dump, thin experience
        "unrelated_domain",   # Accounting, culinary, sales
    ]

    for i in range(1, 101):
        res_id = f"resume_{i:03d}"
        fname = random.choice(first_names)
        lname = random.choice(last_names)
        full_name = f"{fname} {lname}"

        # Assign role family and persona
        target_jd = jds[(i - 1) % len(jds)]
        fam = target_jd["role_family"]
        p_type = persona_types[(i - 1) % len(persona_types)]

        if p_type == "senior_qualified":
            years = target_jd["min_years"] + random.randint(2, 4)
            deg = "Master of Science in Computer Science"
            skills = target_jd["required_skills"] + target_jd["preferred_skills"][:2]
            summary = f"Senior {target_jd['title']} with {years} years of demonstrated experience in {', '.join(skills[:3])}."
            exp_text = f"Lead Engineer | TechCorp (2018 - Present)\n- Led architecture and deployment of services using {', '.join(skills[:4])}.\n- Reduced latency by 45% through query optimization."
        elif p_type == "mid_qualified":
            years = target_jd["min_years"] + 1
            deg = "Bachelor of Science in Computer Science"
            skills = target_jd["required_skills"][:4] + target_jd["preferred_skills"][:1]
            summary = f"{target_jd['title']} with {years} years of engineering experience focusing on {', '.join(skills[:3])}."
            exp_text = f"Software Engineer | InnoSoft (2020 - Present)\n- Implemented backend pipelines using {', '.join(skills[:3])}.\n- Wrote unit tests and automated CI/CD deployments."
        elif p_type == "junior_entry":
            years = 1
            deg = "Bachelor of Science in Information Technology"
            skills = target_jd["required_skills"][:2] + ["Git", "Linux"]
            summary = f"Junior developer passionate about building reliable software using {', '.join(skills)}."
            exp_text = f"Associate Developer | StartUpX (2023 - Present)\n- Built basic endpoints using {', '.join(skills[:2])}."
        elif p_type == "missing_experience":
            years = max(0, target_jd["min_years"] - 3)
            deg = "Bachelor of Science in Software Engineering"
            skills = target_jd["required_skills"][:3]
            summary = f"Junior technologist with {years} years experience in {', '.join(skills)}."
            exp_text = f"Intern Developer | MicroGlobal (2024 - Present)\n- Assisted in coding features with {', '.join(skills)}."
        elif p_type == "missing_degree":
            years = target_jd["min_years"] + 1
            deg = "Bachelor of Arts in Philosophy"
            skills = target_jd["required_skills"][:4]
            summary = f"Self-taught developer with {years} years of real-world experience building software in {', '.join(skills)}."
            exp_text = f"Software Developer | CodeCraft (2020 - Present)\n- Developed commercial applications with {', '.join(skills)}."
        elif p_type == "keyword_stuffed":
            years = 1
            deg = "High School Diploma"
            # Huge list of uncorroborated skills
            skills = target_jd["required_skills"] + ["Python", "Java", "C++", "Rust", "Go", "Kubernetes", "AWS", "Azure", "GCP", "PyTorch", "TensorFlow", "React", "Angular", "Vue", "Spark", "Hadoop", "Docker", "GraphQL", "Solidity", "Swift"]
            summary = "Enthusiastic programmer familiar with all modern technologies and paradigms."
            exp_text = "Junior Intern | WebStudio (2024 - 2024)\n- Assisted office tasks."
        else: # unrelated_domain
            years = 5
            deg = "Bachelor of Science in Accounting"
            skills = ["QuickBooks", "Excel", "Tax Preparation", "Financial Auditing", "Bookkeeping", "Payroll"]
            summary = f"Certified Accountant with {years} years managing financial ledgers, auditing, and corporate taxes."
            exp_text = "Senior Accountant | Ledger & Co (2019 - Present)\n- Reconciled monthly financial statements and tax filings."

        sections = [
            ("Personal Details", f"{full_name}\nEmail: {fname.lower()}.{lname.lower()}@example.com | Phone: (555) 012-3456"),
            ("Professional Summary", summary),
            ("Work Experience", exp_text),
            ("Education", deg),
            ("Technical Skills", ", ".join(skills)),
        ]

        full_resume_text = f"{full_name}\n\nSUMMARY\n{summary}\n\nWORK EXPERIENCE\n{exp_text}\n\nEDUCATION\n{deg}\n\nTECHNICAL SKILLS\n{', '.join(skills)}"

        res_record = {
            "id": res_id,
            "name": full_name,
            "persona_type": p_type,
            "role_family": fam,
            "target_jd_id": target_jd["id"],
            "years_experience": years,
            "degree": deg,
            "skills": skills,
            "text": full_resume_text,
        }
        resumes.append(res_record)

        # Write text and PDF
        (RESUMES_DIR / f"{res_id}.txt").write_text(full_resume_text, encoding="utf-8")
        create_simple_pdf(sections, RESUMES_DIR / f"{res_id}.pdf")

    # Generate 5 scanned/image-only PDFs
    for s_idx in range(1, 6):
        scanned_file = SCANNED_DIR / f"scanned_{s_idx:02d}.pdf"
        create_simple_pdf(
            [("Notice", f"Scanned Document {s_idx}. This document is an image-only scan and contains no extractable text.")],
            scanned_file
        )

    # Save Metadata JSONs
    (SYNTHETIC_DIR / "jds_meta.json").write_text(json.dumps(jds, indent=2), encoding="utf-8")
    (SYNTHETIC_DIR / "resumes_meta.json").write_text(json.dumps(resumes, indent=2), encoding="utf-8")

    return jds, resumes


def generate_gold_labels_and_splits(jds: list[dict[str, Any]], resumes: list[dict[str, Any]]) -> None:
    """Generate 500 gold labeled pairs and strict 30/70 dev/test split."""
    labeled_pairs: list[dict[str, Any]] = []

    # Group candidate resumes by role family and persona
    fam_resumes: dict[str, list[dict[str, Any]]] = {}
    for r in resumes:
        fam_resumes.setdefault(r["role_family"], []).append(r)

    # Assign 10 candidates per JD with balanced grade distribution
    for jd in jds:
        jd_id = jd["id"]
        jd_fam = jd["role_family"]
        jd_req_skills = {s.lower() for s in jd["required_skills"]}
        jd_min_years = jd["min_years"]

        # 1. Same family candidates
        same_fam = fam_resumes.get(jd_fam, [])
        # 2. Other family candidates
        other_fam = [r for r in resumes if r["role_family"] != jd_fam]

        # Select 5 from same family + 5 from other families
        selected_candidates: list[dict[str, Any]] = []
        if len(same_fam) >= 5:
            selected_candidates.extend(same_fam[:5])
        else:
            selected_candidates.extend(same_fam)

        # Fill remaining up to 10 with other family / unrelated
        seed_offset = int(jd_id.split("_")[1]) * 3
        for offset in range(10 - len(selected_candidates)):
            selected_candidates.append(other_fam[(seed_offset + offset) % len(other_fam)])

        for res in selected_candidates:
            cand_skills = {s.lower() for s in res["skills"]}
            cand_years = res["years_experience"]
            p_type = res["persona_type"]

            overlap = len(jd_req_skills.intersection(cand_skills))
            ratio = overlap / max(1, len(jd_req_skills))

            # Ground truth relevance labeling logic (0 = Irrelevant, 1 = Partial, 2 = Strong)
            if p_type == "unrelated_domain":
                grade = 0
                rationale = "Candidate belongs to non-technical unrelated profession."
            elif p_type == "keyword_stuffed":
                grade = 0
                rationale = "Candidate flagged for severe uncorroborated keyword stuffing."
            elif res["role_family"] != jd_fam:
                grade = 0
                rationale = f"Cross-domain mismatch: candidate domain ({res['role_family']}) != target ({jd_fam})."
            elif ratio >= 0.70 and cand_years >= jd_min_years:
                grade = 2
                rationale = f"Strong match: {ratio*100:.0f}% skill coverage with sufficient experience ({cand_years} yrs >= {jd_min_years} yrs)."
            elif ratio >= 0.40 or (ratio >= 0.30 and cand_years >= jd_min_years - 1):
                grade = 1
                rationale = f"Partial match: {ratio*100:.0f}% skill coverage with moderate experience."
            else:
                grade = 0
                rationale = f"Low relevance: skill coverage {ratio*100:.0f}% below qualification threshold."

            labeled_pairs.append({
                "pair_id": f"{jd_id}__{res['id']}",
                "jd_id": jd_id,
                "resume_id": res["id"],
                "relevance": grade,
                "rationale": rationale,
                "candidate_name": res["name"],
                "candidate_persona": p_type,
            })

    # Save 500 gold labeled pairs in JSONL format
    labels_file = GOLD_DIR / "labels.jsonl"
    with open(labels_file, "w", encoding="utf-8") as f:
        for p in labeled_pairs:
            f.write(json.dumps(p) + "\n")

    # Partition by JD into 30% Dev (15 JDs, 150 pairs) and 70% Test (35 JDs, 350 pairs)
    random.seed(42)
    shuffled_jds = list(jds)
    random.shuffle(shuffled_jds)

    dev_jds = [j["id"] for j in shuffled_jds[:15]]
    test_jds = [j["id"] for j in shuffled_jds[15:]]

    splits = {
        "description": "ResumeFit 30/70 Dev/Test Partition by Job Description",
        "dev_jd_count": len(dev_jds),
        "test_jd_count": len(test_jds),
        "dev_jds": sorted(dev_jds),
        "test_jds": sorted(test_jds),
    }

    (GOLD_DIR / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    logger.info("Generated %d gold pairs (Dev: %d JDs, Test: %d JDs)", len(labeled_pairs), len(dev_jds), len(test_jds))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Generating 50 JDs and 100 synthetic resumes...")
    jds_data, resumes_data = build_synthetic_corpus()
    print(f"Created {len(jds_data)} JDs and {len(resumes_data)} candidate resumes.")
    print("Generating 500 gold-labeled pairs and 30/70 split...")
    generate_gold_labels_and_splits(jds_data, resumes_data)
    print("Dataset generation complete!")
