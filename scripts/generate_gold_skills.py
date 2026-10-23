"""Script to generate 20 hand-labeled gold-standard resumes for skill extraction evaluation (P2.10).

Covers diverse real-world profiles:
- Backend Engineers (Python, Go, Java, .NET)
- Data Scientists & ML Engineers (PyTorch, scikit-learn, NLP, LLM, Vector Search)
- DevOps & Cloud Infrastructure Engineers (Docker, Kubernetes, AWS, Terraform, CI/CD)
- Frontend & Full-Stack Engineers (React, TypeScript, Next.js, Node.js)
"""

from __future__ import annotations

import json
from pathlib import Path

TARGET_DIR = Path("data/gold/skill_labels")

BENCHMARK_PROFILES = [
    {
        "id": "resume_001",
        "domain": "backend_python",
        "text": """
Aarav Sharma | Senior Backend Developer
SUMMARY
Experienced backend engineer specializing in distributed Python microservices, PostgreSQL databases, and high-throughput REST APIs.

WORK EXPERIENCE
Senior Backend Engineer | CloudScale Tech (2022 - Present)
• Architected and deployed microservices handling 20,000 requests/sec using Flask, PostgreSQL, and Redis.
• Implemented automated CI/CD pipelines with Docker and GitHub Actions, cutting release cycles by 40%.
• Optimized SQL queries and added pgvector indexing for semantic document similarity search.

Junior Developer | InfoSystems (2020 - 2022)
• Developed web APIs using Python, Flask, and SQLAlchemy for transaction ledger services.
• Wrote automated unit test suites with pytest.

TECHNICAL SKILLS
Languages: Python, SQL, Bash
Frameworks: Flask, SQLAlchemy, FastAPI
Databases: PostgreSQL, Redis, pgvector
DevOps & Tools: Docker, Git, GitHub Actions, Linux
""",
        "expected_skills": [
            "Python",
            "SQL",
            "Bash Scripting",
            "Flask",
            "SQLAlchemy",
            "FastAPI",
            "PostgreSQL",
            "Redis",
            "pgvector",
            "Docker",
            "Git",
            "GitHub Actions",
            "Linux",
            "Microservices Architecture",
            "CI/CD",
            "RESTful API",
            "Pytest",
        ],
    },
    {
        "id": "resume_002",
        "domain": "ml_engineer",
        "text": """
Priya Patel | Machine Learning Engineer
SUMMARY
Machine Learning Engineer with 4 years of experience building NLP systems, predictive models, and vector search pipelines.

WORK EXPERIENCE
ML Engineer | DataCore Systems (2022 - Present)
• Trained deep learning models using PyTorch and Hugging Face Transformers for sentiment analysis.
• Built vector search retrieval pipelines with Sentence-Transformers and FAISS for 10M documents.
• Conducted model evaluation and hyperparameter tuning using scikit-learn and MLflow.

Data Scientist | Alpha Analytics (2020 - 2022)
• Developed tabular predictive models with XGBoost and Pandas.
• Performed exploratory data analysis with NumPy, Matplotlib, and Seaborn.

TECHNICAL SKILLS
Languages: Python, SQL
ML & AI: PyTorch, scikit-learn, Hugging Face, Sentence-Transformers, XGBoost, FAISS, MLflow
Data Analysis: Pandas, NumPy, Matplotlib, Seaborn
Tools: Git, Docker, Linux
""",
        "expected_skills": [
            "Python",
            "SQL",
            "PyTorch",
            "scikit-learn",
            "Hugging Face",
            "Sentence-Transformers",
            "XGBoost",
            "FAISS",
            "MLflow",
            "Pandas",
            "NumPy",
            "Matplotlib",
            "Seaborn",
            "Git",
            "Docker",
            "Linux",
            "Natural Language Processing",
            "Deep Learning",
            "Machine Learning",
            "Vector Similarity Search",
            "Model Evaluation",
            "Hyperparameter Tuning",
        ],
    },
    {
        "id": "resume_003",
        "domain": "devops_cloud",
        "text": """
Rohan Verma | DevOps & Cloud Platform Engineer
SUMMARY
Cloud Infrastructure Engineer with deep expertise in Kubernetes orchestration, AWS cloud provisioning, and CI/CD pipelines.

WORK EXPERIENCE
DevOps Engineer | Scalenet Labs (2021 - Present)
• Architected and provisioned multi-region AWS cloud infrastructure using Terraform and Ansible.
• Deployed and managed production Kubernetes clusters across Amazon EKS.
• Automated end-to-end continuous integration and deployment with GitLab CI and Helm.
• Implemented cluster observability and alerting using Prometheus and Grafana.

SysAdmin | InfraSolutions (2019 - 2021)
• Administered Linux Ubuntu servers and automated maintenance tasks with Bash scripting.
• Managed container deployments using Docker and Docker Compose.

TECHNICAL SKILLS
Cloud: Amazon Web Services, Amazon EKS, Amazon S3, Amazon EC2
Orchestration: Kubernetes, Docker, Helm, Docker Compose
IaC & Automation: Terraform, Ansible, GitLab CI
Monitoring: Prometheus, Grafana
OS & Scripting: Linux, Ubuntu Linux, Bash Scripting, Git
""",
        "expected_skills": [
            "Amazon Web Services",
            "Amazon EKS",
            "Amazon S3",
            "Amazon EC2",
            "Kubernetes",
            "Docker",
            "Helm",
            "Docker Compose",
            "Terraform",
            "Ansible",
            "GitLab CI",
            "Prometheus",
            "Grafana",
            "Linux",
            "Ubuntu Linux",
            "Bash Scripting",
            "Git",
            "CI/CD",
            "Infrastructure as Code",
        ],
    },
    {
        "id": "resume_004",
        "domain": "frontend_web",
        "text": """
Neha Gupta | Senior Frontend Engineer
SUMMARY
Frontend specialist with 5 years experience creating responsive, accessible web applications in React and TypeScript.

WORK EXPERIENCE
Senior Frontend Developer | WebCraft Digital (2022 - Present)
• Built server-side rendered web applications using Next.js, React.js, and TypeScript.
• Architected client state management with Zustand and TanStack Query.
• Styled responsive UI design systems using Tailwind CSS and modern CSS3.
• Wrote unit and component integration tests using Jest and Playwright.

Frontend Developer | Pixel Studio (2019 - 2022)
• Developed interactive single page applications using React, HTML5, and JavaScript.
• Integrated RESTful API and WebSocket endpoints for live real-time chats.

TECHNICAL SKILLS
Languages: TypeScript, JavaScript, HTML5, CSS3
Frameworks: React.js, Next.js, Tailwind CSS
State & Data: Zustand, TanStack Query, Redux
Testing & Tooling: Jest, Playwright, Vite, Webpack, Git
Protocols: RESTful API, WebSockets
""",
        "expected_skills": [
            "TypeScript",
            "JavaScript",
            "HTML5",
            "CSS3",
            "React.js",
            "Next.js",
            "Tailwind CSS",
            "Zustand",
            "TanStack Query",
            "Redux",
            "Jest",
            "Playwright",
            "Vite",
            "Webpack",
            "Git",
            "RESTful API",
            "WebSockets",
        ],
    },
    {
        "id": "resume_005",
        "domain": "fullstack_node",
        "text": """
Ananya Sen | Full Stack Developer
SUMMARY
Full Stack Engineer skilled in Node.js, Express.js, React, and MongoDB database modeling.

WORK EXPERIENCE
Full Stack Engineer | FinScale (2021 - Present)
• Developed modular microservices using Node.js, Express.js, and TypeScript.
• Built customer-facing responsive dashboards with React and Redux.
• Designed NoSQL database schemas and aggregation queries in MongoDB.
• Implemented authentication using JSON Web Tokens (JWT) and OAuth 2.0.

TECHNICAL SKILLS
Languages: JavaScript, TypeScript, Python
Backend: Node.js, Express.js, RESTful API
Frontend: React.js, HTML5, CSS3, Bootstrap (CSS)
Databases: MongoDB, Redis
DevOps: Docker, Git, GitHub
Security: JSON Web Tokens (JWT), OAuth 2.0
""",
        "expected_skills": [
            "JavaScript",
            "TypeScript",
            "Python",
            "Node.js",
            "Express.js",
            "RESTful API",
            "React.js",
            "HTML5",
            "CSS3",
            "Bootstrap (CSS)",
            "MongoDB",
            "Redis",
            "Docker",
            "Git",
            "GitHub",
            "JSON Web Tokens (JWT)",
            "OAuth 2.0",
            "Microservices Architecture",
        ],
    },
    {
        "id": "resume_006",
        "domain": "backend_go",
        "text": """
Vikram Nair | Cloud Backend Engineer
SUMMARY
Systems engineer with 4 years building high-concurrency microservices in Go and Rust.

WORK EXPERIENCE
Backend Engineer | CloudGrid (2022 - Present)
• Developed distributed streaming services in Go and gRPC handling 50k events/sec.
• Integrated Apache Kafka message queues for asynchronous order processing.
• Maintained containerized workloads on Kubernetes clusters in Google Cloud Platform.

Systems Programmer | NetworkEdge (2020 - 2022)
• Built low-latency networking utilities in C++ and Go.
• Optimized PostgreSQL database queries and connection pools.

TECHNICAL SKILLS
Languages: Go, C++, SQL, Bash Scripting
Frameworks & Protocols: gRPC, Protocol Buffers, RESTful API
Messaging & Storage: Apache Kafka, PostgreSQL, Redis
Cloud & Ops: Google Cloud Platform, Kubernetes, Docker, Git
""",
        "expected_skills": [
            "Go",
            "C++",
            "SQL",
            "Bash Scripting",
            "gRPC",
            "Protocol Buffers",
            "RESTful API",
            "Apache Kafka",
            "PostgreSQL",
            "Redis",
            "Google Cloud Platform",
            "Kubernetes",
            "Docker",
            "Git",
            "Microservices Architecture",
        ],
    },
    {
        "id": "resume_007",
        "domain": "data_engineer",
        "text": """
Siddharth Rao | Big Data Engineer
SUMMARY
Data Engineer with 5 years experience designing scalable data lakehouses and ETL pipelines using Apache Spark and Snowflake.

WORK EXPERIENCE
Senior Data Engineer | DataWave (2022 - Present)
• Built batch and real-time ETL pipelines with PySpark and Apache Spark.
• Orchestrated automated DAG workflows with Apache Airflow on Amazon Web Services.
• Architected analytical data warehouse models in Snowflake and Google BigQuery.

Data Engineer | MetricStream (2019 - 2022)
• Ingested streaming telemetry using Apache Kafka and Python.
• Maintained relational databases in PostgreSQL and MySQL.

TECHNICAL SKILLS
Languages: Python, SQL, Scala
Data Processing: Apache Spark, PySpark, Apache Airflow
Storage & Warehousing: Snowflake, Google BigQuery, PostgreSQL, MySQL
Cloud & Messaging: Amazon Web Services, Apache Kafka, Docker, Git
""",
        "expected_skills": [
            "Python",
            "SQL",
            "Scala",
            "Apache Spark",
            "PySpark",
            "Apache Airflow",
            "Snowflake",
            "Google BigQuery",
            "PostgreSQL",
            "MySQL",
            "Amazon Web Services",
            "Apache Kafka",
            "Docker",
            "Git",
            "Data Analysis",
        ],
    },
    {
        "id": "resume_008",
        "domain": "backend_java",
        "text": """
Aditi Deshmukh | Java Backend Developer
SUMMARY
Enterprise Java developer with 6 years experience in Spring Boot microservices, Hibernate ORM, and Oracle Database.

WORK EXPERIENCE
Senior Java Developer | GlobalBank Tech (2021 - Present)
• Architected enterprise financial services using Java, Spring Boot, and Spring Framework.
• Implemented ORM persistence layers with Hibernate ORM and PostgreSQL.
• Orchestrated messaging and decoupling using RabbitMQ.
• Practiced test-driven development using JUnit and Mockito.

Software Engineer | TechCorp (2018 - 2021)
• Built RESTful API microservices in Java and MySQL.
• Configured build automation with Apache Maven and Jenkins.

TECHNICAL SKILLS
Languages: Java (programming language), SQL
Frameworks: Spring Boot, Spring Framework, Hibernate ORM
Databases: PostgreSQL, MySQL, Oracle Database
Messaging: RabbitMQ, Apache Kafka
DevOps: Docker, Jenkins, Apache Maven, Git
Methodologies: Test-Driven Development, Agile, Scrum
""",
        "expected_skills": [
            "Java (programming language)",
            "SQL",
            "Spring Boot",
            "Spring Framework",
            "Hibernate ORM",
            "PostgreSQL",
            "MySQL",
            "Oracle Database",
            "RabbitMQ",
            "Apache Kafka",
            "Docker",
            "Jenkins",
            "Apache Maven",
            "Git",
            "Test-Driven Development",
            "Agile",
            "Scrum",
            "Microservices Architecture",
            "RESTful API",
        ],
    },
    {
        "id": "resume_009",
        "domain": "ai_rag_specialist",
        "text": """
Karan Mehta | Generative AI & RAG Engineer
SUMMARY
AI Engineer specializing in Large Language Models (LLM), Retrieval-Augmented Generation (RAG), and vector database infrastructure.

WORK EXPERIENCE
AI Research Engineer | SynthAI (2023 - Present)
• Architected production Retrieval-Augmented Generation pipelines using LangChain, LlamaIndex, and pgvector.
• Fine-tuned open-source Large Language Models using PyTorch and Hugging Face Transformers.
• Deployed low-latency inference services with vLLM and FastAPI.
• Optimized vector similarity search across 5M document embeddings in ChromaDB and Qdrant.

TECHNICAL SKILLS
Languages: Python, SQL
AI & LLM: Large Language Models, Retrieval-Augmented Generation, LangChain, LlamaIndex, PyTorch, Hugging Face
Vector DBs: pgvector, ChromaDB, Qdrant
APIs & Deployment: FastAPI, Docker, Git, Linux
""",
        "expected_skills": [
            "Python",
            "SQL",
            "Large Language Models",
            "Retrieval-Augmented Generation",
            "LangChain",
            "LlamaIndex",
            "PyTorch",
            "Hugging Face",
            "pgvector",
            "ChromaDB",
            "Qdrant",
            "FastAPI",
            "Docker",
            "Git",
            "Linux",
            "Deep Learning",
            "Natural Language Processing",
            "Vector Similarity Search",
        ],
    },
    {
        "id": "resume_010",
        "domain": "systems_cpp",
        "text": """
Manish Kulkarni | C++ Systems Engineer
SUMMARY
Low-latency C++ developer specializing in high-performance computing, memory optimization, and Linux systems programming.

WORK EXPERIENCE
Systems Engineer | HighFreq Trading (2021 - Present)
• Engineered low-latency trading execution gateways in C++ on Linux.
• Implemented concurrent multithreaded order books with lock-free data structures and algorithms.
• Automated performance profiling and unit tests using Google Test and CMake.

Firmware Engineer | EmbeddedTech (2019 - 2021)
• Programmed microcontrollers in C (programming language) and modern C++.
• Developed hardware communication drivers over TCP/IP and CAN bus.

TECHNICAL SKILLS
Languages: C++, C (programming language), Python, Bash Scripting
Systems: Linux, algorithms and data structures, Multi-threading
Tools: Git, Docker, GDB
""",
        "expected_skills": [
            "C++",
            "C (programming language)",
            "Python",
            "Bash Scripting",
            "Linux",
            "algorithms and data structures",
            "Git",
            "Docker",
        ],
    },
    {
        "id": "resume_011",
        "domain": "dotnet_engineer",
        "text": """
Deepak Joshi | Senior .NET Backend Developer
SUMMARY
Full lifecycle software engineer building cloud services on Microsoft C#, ASP.NET Core, and Microsoft Azure.

WORK EXPERIENCE
Senior .NET Engineer | EnterpriseCloud (2021 - Present)
• Built distributed microservices in C# and ASP.NET Core.
• Implemented relational persistence using Entity Framework and Microsoft SQL Server.
• Provisioned cloud infrastructure across Microsoft Azure using Azure Kubernetes Service.
• Configured CI/CD release pipelines with Azure DevOps and Docker.

TECHNICAL SKILLS
Languages: C#, SQL, TypeScript
Frameworks: .NET, ASP.NET Core, Entity Framework
Cloud & DB: Microsoft Azure, Microsoft SQL Server, Redis
DevOps: Docker, Kubernetes, Git
Methodologies: Agile, Scrum, Domain-Driven Design
""",
        "expected_skills": [
            "C#",
            "SQL",
            "TypeScript",
            ".NET",
            "ASP.NET Core",
            "Entity Framework",
            "Microsoft Azure",
            "Microsoft SQL Server",
            "Redis",
            "Docker",
            "Kubernetes",
            "Git",
            "Agile",
            "Scrum",
            "Domain-Driven Design",
            "Microservices Architecture",
        ],
    },
    {
        "id": "resume_012",
        "domain": "qa_automation",
        "text": """
Divya Nambiar | QA Automation Engineer
SUMMARY
Quality Assurance Engineer with 4 years experience creating automated testing frameworks and CI integration tests.

WORK EXPERIENCE
QA Automation Lead | TestMatrix (2021 - Present)
• Developed automated end-to-end testing suites using Playwright, Cypress, and TypeScript.
• Built API test frameworks in Python using Pytest and Requests.
• Integrated automated test execution in GitHub Actions continuous integration pipelines.
• Executed cross-browser functional testing with Selenium.

TECHNICAL SKILLS
Languages: Python, JavaScript, TypeScript
Test Frameworks: Playwright, Cypress, Selenium, Pytest
CI/CD: GitHub Actions, Jenkins, Git
Methodologies: Test-Driven Development, Behavior-Driven Development, Agile
""",
        "expected_skills": [
            "Python",
            "JavaScript",
            "TypeScript",
            "Playwright",
            "Cypress",
            "Selenium",
            "Pytest",
            "GitHub Actions",
            "Jenkins",
            "Git",
            "Test-Driven Development",
            "Behavior-Driven Development",
            "Agile",
            "CI/CD",
        ],
    },
    {
        "id": "resume_013",
        "domain": "mobile_developer",
        "text": """
Arjun Kapoor | Mobile Application Developer
SUMMARY
Mobile app engineer with 4 years developing cross-platform applications in Flutter, React Native, and native iOS.

WORK EXPERIENCE
Mobile Engineer | AppVenture (2022 - Present)
• Developed cross-platform mobile apps using Flutter and Dart (programming language).
• Built responsive React Native mobile clients with Redux state management.
• Integrated RESTful API backend microservices and Firebase cloud notifications.

TECHNICAL SKILLS
Languages: Dart (programming language), JavaScript, TypeScript, Swift (programming language)
Mobile Frameworks: Flutter, React Native
State & Backend: Redux, Google Firebase, RESTful API
Tools: Git, Docker, Figma
""",
        "expected_skills": [
            "Dart (programming language)",
            "JavaScript",
            "TypeScript",
            "Swift (programming language)",
            "Flutter",
            "React Native",
            "Redux",
            "Google Firebase",
            "RESTful API",
            "Git",
            "Docker",
            "Figma",
        ],
    },
    {
        "id": "resume_014",
        "domain": "cybersecurity_devsecops",
        "text": """
Sunita Rao | DevSecOps & Security Engineer
SUMMARY
Security specialist focusing on cloud security posture, secrets management, and automated vulnerability scanning.

WORK EXPERIENCE
DevSecOps Engineer | SecureCloud (2021 - Present)
• Configured automated secrets management with HashiCorp Vault across Kubernetes.
• Implemented vulnerability scanning in GitLab CI pipelines with Docker container scanning.
• Provisioned identity and access management policies across Amazon Web Services.
• Maintained security incident monitoring using Splunk and Datadog.

TECHNICAL SKILLS
Cloud & Security: Amazon Web Services, HashiCorp Vault, Kubernetes, Docker
CI/CD & Monitoring: GitLab CI, Splunk, Datadog, Git
OS: Linux, Ubuntu Linux, Bash Scripting
""",
        "expected_skills": [
            "Amazon Web Services",
            "HashiCorp Vault",
            "Kubernetes",
            "Docker",
            "GitLab CI",
            "Splunk",
            "Datadog",
            "Git",
            "Linux",
            "Ubuntu Linux",
            "Bash Scripting",
            "CI/CD",
        ],
    },
    {
        "id": "resume_015",
        "domain": "mlops_engineer",
        "text": """
Tarun Nair | MLOps & Platform Engineer
SUMMARY
MLOps practitioner experienced in machine learning deployment, model registry pipelines, and distributed GPU training.

WORK EXPERIENCE
MLOps Engineer | DeepScale (2022 - Present)
• Built automated machine learning training and evaluation pipelines using MLflow, DVC, and Ray.
• Packaged and served deep learning models using Docker, Triton Inference Server, and FastAPI.
• Orchestrated model tracking and experiment logging with Weights & Biases.
• Provisioned cloud GPU clusters on Amazon Web Services using Kubernetes.

TECHNICAL SKILLS
MLOps: MLflow, Data Version Control (DVC), Weights & Biases, Triton Inference Server, Ray
ML Frameworks: PyTorch, scikit-learn, Hugging Face
Cloud & Infrastructure: Amazon Web Services, Kubernetes, Docker, FastAPI, Git, Linux
""",
        "expected_skills": [
            "MLflow",
            "Data Version Control (DVC)",
            "Weights & Biases",
            "Triton Inference Server",
            "PyTorch",
            "scikit-learn",
            "Hugging Face",
            "Amazon Web Services",
            "Kubernetes",
            "Docker",
            "FastAPI",
            "Git",
            "Linux",
            "MLOps",
            "Machine Learning",
            "Deep Learning",
        ],
    },
    {
        "id": "resume_016",
        "domain": "data_analyst",
        "text": """
Pooja Bhatt | Senior Data Analyst
SUMMARY
Data Analyst with 4 years experience translating business requirements into dashboards, SQL pipelines, and predictive insights.

WORK EXPERIENCE
Senior Analyst | RetailInsights (2021 - Present)
• Built automated reporting pipelines in Python with Pandas, NumPy, and Statsmodels.
• Optimized complex analytical queries and database aggregation in PostgreSQL and Snowflake.
• Developed interactive KPI visual dashboards using Plotly and Tableau.
• Conducted exploratory data analysis and statistical modeling for customer retention.

TECHNICAL SKILLS
Languages: Python, SQL, R (programming language)
Data & Stats: Pandas, NumPy, Statsmodels, exploratory data analysis
Visualization: Plotly, Matplotlib, Seaborn
Databases: PostgreSQL, Snowflake
Tools: Git, Excel
""",
        "expected_skills": [
            "Python",
            "SQL",
            "R (programming language)",
            "Pandas",
            "NumPy",
            "Statsmodels",
            "Plotly",
            "Matplotlib",
            "Seaborn",
            "PostgreSQL",
            "Snowflake",
            "Git",
            "data analysis",
        ],
    },
    {
        "id": "resume_017",
        "domain": "sre_engineer",
        "text": """
Varun Menon | Site Reliability Engineer
SUMMARY
SRE with 5 years experience ensuring 99.99% availability of microservices on multi-cloud environments.

WORK EXPERIENCE
Site Reliability Engineer | CloudStream (2021 - Present)
• Architected telemetry and distributed tracing systems using OpenTelemetry, Prometheus, and Grafana.
• Automated incident reporting and alerting pipelines with Datadog and Sentry.
• Managed containerized Kubernetes clusters in Google Cloud Platform and Amazon Web Services.
• Scripted operational automation using Python and Bash Scripting.

TECHNICAL SKILLS
Observability: OpenTelemetry, Prometheus, Grafana, Datadog, Sentry
Cloud & Containers: Kubernetes, Docker, Google Cloud Platform, Amazon Web Services
Languages: Python, Go, Bash Scripting
Methodologies: Site Reliability Engineering, High Availability, CI/CD, Git
""",
        "expected_skills": [
            "OpenTelemetry",
            "Prometheus",
            "Grafana",
            "Datadog",
            "Sentry (monitoring)",
            "Kubernetes",
            "Docker",
            "Google Cloud Platform",
            "Amazon Web Services",
            "Python",
            "Go",
            "Bash Scripting",
            "Site Reliability Engineering",
            "High Availability",
            "CI/CD",
            "Git",
        ],
    },
    {
        "id": "resume_018",
        "domain": "blockchain_web3",
        "text": """
Nikhil Jain | Web3 & Distributed Systems Developer
SUMMARY
Software developer with 3 years building smart contract integrations and decentralized web applications.

WORK EXPERIENCE
Blockchain Developer | ChainCraft (2022 - Present)
• Developed client integration libraries using TypeScript, Node.js, and WebSockets.
• Built decentralized frontend interfaces with React.js, Next.js, and Tailwind CSS.
• Integrated relational caching and document storage with PostgreSQL and Redis.

TECHNICAL SKILLS
Languages: TypeScript, JavaScript, Rust (programming language), Python
Frontend & Fullstack: React.js, Next.js, Tailwind CSS, Node.js
Databases: PostgreSQL, Redis
Protocols & Tools: WebSockets, RESTful API, Docker, Git
""",
        "expected_skills": [
            "TypeScript",
            "JavaScript",
            "Rust (programming language)",
            "Python",
            "React.js",
            "Next.js",
            "Tailwind CSS",
            "Node.js",
            "PostgreSQL",
            "Redis",
            "WebSockets",
            "RESTful API",
            "Docker",
            "Git",
        ],
    },
    {
        "id": "resume_019",
        "domain": "nlp_engineer",
        "text": """
Shreya Iyer | NLP & Computational Linguistics Specialist
SUMMARY
Computational linguist and NLP Engineer with 4 years experience creating multilingual classification and search engines.

WORK EXPERIENCE
NLP Engineer | LinguaTech (2022 - Present)
• Developed entity extraction and topic modeling pipelines with spaCy, NLTK, and scikit-learn.
• Trained neural cross-encoders with PyTorch, Sentence-Transformers, and Hugging Face.
• Built lexical search and semantic ranking pipelines with Elasticsearch and pgvector.

TECHNICAL SKILLS
NLP & AI: Natural Language Processing, spaCy, NLTK, Hugging Face, Sentence-Transformers, PyTorch, scikit-learn
Search & Databases: Elasticsearch, pgvector, PostgreSQL
Languages & Tools: Python, SQL, Docker, Git, Linux
""",
        "expected_skills": [
            "Natural Language Processing",
            "spaCy",
            "NLTK",
            "Hugging Face",
            "Sentence-Transformers",
            "PyTorch",
            "scikit-learn",
            "Elasticsearch",
            "pgvector",
            "PostgreSQL",
            "Python",
            "SQL",
            "Docker",
            "Git",
            "Linux",
            "Deep Learning",
            "Machine Learning",
        ],
    },
    {
        "id": "resume_020",
        "domain": "backend_architect",
        "text": """
Rajesh Sundaram | Principal Backend Architect
SUMMARY
Principal Software Architect with 8 years experience leading enterprise digital transformations, event-driven architectures, and high-throughput databases.

WORK EXPERIENCE
Principal Architect | Enterprise Core (2020 - Present)
• Architected enterprise event-driven architectures processing 100M events/day using Apache Kafka, RabbitMQ, and microservices.
• Modeled distributed persistence across PostgreSQL, Apache Cassandra, and Redis.
• Enforced Clean Architecture, Domain-Driven Design, and SOLID Principles across 6 agile engineering squads.
• Deployed microservices on Kubernetes across Amazon Web Services and Google Cloud Platform.

TECHNICAL SKILLS
Architecture & Design: Microservices Architecture, Event-Driven Architecture, Domain-Driven Design, Clean Architecture, SOLID Principles, System Design
Messaging: Apache Kafka, RabbitMQ
Databases: PostgreSQL, Apache Cassandra, Redis
Cloud: Amazon Web Services, Google Cloud Platform, Kubernetes, Docker, Git
Methodologies: Agile, Scrum, Code Review
""",
        "expected_skills": [
            "Microservices Architecture",
            "Event-Driven Architecture",
            "Domain-Driven Design",
            "Clean Architecture",
            "SOLID Principles",
            "System Design",
            "Apache Kafka",
            "RabbitMQ",
            "PostgreSQL",
            "Apache Cassandra",
            "Redis",
            "Amazon Web Services",
            "Google Cloud Platform",
            "Kubernetes",
            "Docker",
            "Git",
            "Agile",
            "Scrum",
            "Code Review",
        ],
    },
]


def generate_gold_resumes() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for profile in BENCHMARK_PROFILES:
        filepath = TARGET_DIR / f"{profile['id']}.json"
        with open(filepath, mode="w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
    print(f"Generated {len(BENCHMARK_PROFILES)} gold-standard resumes in {TARGET_DIR.resolve()}")


if __name__ == "__main__":
    generate_gold_resumes()
