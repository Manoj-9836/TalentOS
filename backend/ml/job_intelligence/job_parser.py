from __future__ import annotations

import re
from typing import Any


COMMON_SKILLS = [
    "python", "java", "javascript", "typescript", "sql", "nosql", "postgresql", "mysql",
    "mongodb", "redis", "elasticsearch", "aws", "azure", "gcp", "docker", "kubernetes",
    "terraform", "ansible", "ci/cd", "jenkins", "gitlab", "github actions", "spark",
    "hadoop", "kafka", "airflow", "dbt", "snowflake", "bigquery", "databricks",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow",
    "pytorch", "scikit-learn", "pandas", "numpy", "langchain", "rag", "llm", "llms",
    "react", "vue", "angular", "next.js", "node.js", "django", "flask", "fastapi",
    "spring boot", "go", "rust", "c++", "c#", ".net", "graphql", "rest api",
    "microservices", "event-driven", "message queues", "rabbitmq", "kafka",
    "agile", "scrum", "kanban", "jira", "confluence", "tableau", "power bi",
    "looker", "etl", "data pipelines", "data warehouse", "data lake", "dbt",
    "salesforce", "hubspot", "marketo", "pandas", "excel", "powerpoint",
    "project management", "stakeholder management", "product management",
    "business analysis", "requirements gathering", "user stories", "api design",
    "system design", "architecture", "scalability", "performance optimization",
    "security", "authentication", "authorization", "oauth", "jwt", "saml",
    "testing", "unit testing", "integration testing", "e2e testing", "pytest",
    "jest", "cypress", "playwright", "selenium", "devops", "sre", "observability",
    "prometheus", "grafana", "datadog", "logging", "monitoring", "alerting",
    "git", "github", "gitlab", "bitbucket", "code review", "design patterns",
    "clean code", "refactoring", "legacy modernization", "cloud migration",
    "serverless", "lambda", "cloud functions", "api gateway", "load balancing",
    "caching", "cdn", "web sockets", "grpc", "protobuf", "avro", "parquet",
    "delta lake", "iceberg", "hudi", "mlops", "kubeflow", "mlflow", "weights & biases",
    "feature store", "model registry", "experiment tracking", "hyperparameter tuning",
    "auto ml", "model deployment", "model monitoring", "drift detection",
    "data quality", "data governance", "data lineage", "data catalog",
    "privacy", "gdpr", "ccpa", "hipaa", "pci dss", "soc2", "iso 27001",
    "linux", "bash", "shell scripting", "vim", "tmux", "ssh", "networking",
    "dns", "http", "https", "tcp/ip", "ssl", "tls", "vpn", "firewall",
    "load testing", "stress testing", "chaos engineering", "disaster recovery",
    "backup", "high availability", "multi-region", "active-active", "active-passive"
]

EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)",
    r"(\d+)\+?\s*yrs?\s*(?:of\s*)?(?:experience|exp)",
    r"minimum\s+(\d+)\s*years?",
    r"at least\s+(\d+)\s*years?",
    r"(\d+)\s*-\s*(\d+)\s*years?",
    r"senior|lead|principal|staff|architect|manager|director|vp|head of",
]

INDUSTRY_KEYWORDS = {
    "fintech": ["fintech", "financial technology", "banking", "payments", "trading", "wealth management", "lending", "insurance", "insurtech"],
    "healthcare": ["healthcare", "health tech", "medtech", "biotech", "pharma", "clinical", "hospital", "medical", "digital health"],
    "ecommerce": ["e-commerce", "ecommerce", "retail", "marketplace", "shopify", "amazon", "shopping", "merchant"],
    "saas": ["saas", "software as a service", "b2b saas", "enterprise software", "platform"],
    "ai/ml": ["ai", "artificial intelligence", "machine learning", "ml", "deep learning", "generative ai", "genai", "llm", "nlp", "computer vision"],
    "transportation": ["transportation", "logistics", "ride sharing", "delivery", "fleet", "supply chain", "autonomous vehicles"],
    "food delivery": ["food delivery", "restaurant", "meal delivery", "uber eats", "doordash", "grubhub"],
    "edtech": ["edtech", "education technology", "online learning", "e-learning", "course", "training"],
    "gaming": ["gaming", "video games", "game development", "unity", "unreal engine", "esports"],
    "crypto": ["crypto", "cryptocurrency", "blockchain", "web3", "defi", "nft", "smart contracts"],
    "cybersecurity": ["cybersecurity", "security", "infosec", "penetration testing", "vulnerability", "soc", "siem"],
    "media": ["media", "streaming", "content", "entertainment", "video", "music", "publishing"],
    "real estate": ["real estate", "proptech", "property", "rental", "mortgage"],
    "travel": ["travel", "hospitality", "booking", "airline", "hotel", "tourism"],
    "manufacturing": ["manufacturing", "industrial", "iot", "industry 4.0", "factory"],
    "energy": ["energy", "renewable", "solar", "wind", "battery", "grid", "utilities"],
    "telecom": ["telecom", "telecommunications", "5g", "network", "wireless"],
    "automotive": ["automotive", "auto", "vehicle", "car", "electric vehicle", "ev"],
    "government": ["government", "public sector", "federal", "state", "municipal", "defense"],
    "non-profit": ["non-profit", "nonprofit", "ngo", "charity", "social impact"],
}

ROLE_TYPE_KEYWORDS = {
    "engineering": ["engineer", "developer", "software", "backend", "frontend", "fullstack", "full-stack", "devops", "sre", "platform", "infrastructure", "data engineer", "ml engineer", "machine learning engineer", "ai engineer", "architect", "tech lead", "engineering manager", "cto", "vp engineering"],
    "data": ["data scientist", "data analyst", "business analyst", "analytics engineer", "data engineer", "ml engineer", "machine learning engineer", "ai researcher", "research scientist", "quantitative analyst", "bi analyst", "data architect"],
    "product": ["product manager", "product owner", "product lead", "group product manager", "senior product manager", "principal product manager", "head of product", "vp product", "cpo"],
    "design": ["designer", "ux", "ui", "product designer", "ux researcher", "design lead", "design manager", "creative director"],
    "management": ["manager", "director", "vp", "head of", "chief", "lead", "supervisor", "team lead", "engineering manager", "product manager", "project manager", "program manager", "delivery manager"],
    "sales": ["sales", "account executive", "account manager", "business development", "sales engineer", "solutions engineer", "pre-sales", "customer success", "customer support"],
    "marketing": ["marketing", "growth", "demand generation", "content", "seo", "sem", "brand", "communications", "public relations", "social media"],
    "operations": ["operations", "ops", "operational", "supply chain", "logistics", "procurement", "vendor management"],
    "finance": ["finance", "accounting", "fp&a", "treasury", "tax", "audit", "controller", "cfo", "financial analyst"],
    "hr": ["human resources", "hr", "recruiting", "talent acquisition", "people operations", "learning and development", "compensation", "benefits"],
    "legal": ["legal", "counsel", "attorney", "lawyer", "compliance", "regulatory", "privacy", "contracts"],
}


def extract_skills(text: str) -> list[str]:
    text_lower = text.lower()
    found_skills = []
    for skill in COMMON_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    return found_skills


def extract_experience_level(text: str) -> int:
    text_lower = text.lower()
    years = []
    
    for pattern in EXPERIENCE_PATTERNS[:4]:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                years.extend([int(m) for m in match if m.isdigit()])
            elif match.isdigit():
                years.append(int(match))
    
    range_match = re.search(r"(\d+)\s*-\s*(\d+)\s*years?", text_lower)
    if range_match:
        years.append(int(range_match.group(1)))
        years.append(int(range_match.group(2)))
    
    senior_keywords = ["principal", "staff", "architect", "lead", "senior"]
    for keyword in senior_keywords:
        if keyword in text_lower:
            years.append(7)
            break
    
    manager_keywords = ["manager", "director", "vp", "head of", "chief"]
    for keyword in manager_keywords:
        if keyword in text_lower:
            years.append(8)
            break
    
    if years:
        return max(min(max(years), 20), 0)
    
    return 3


def extract_industry(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in text_lower:
                score += 1
        if score > 0:
            scores[industry] = score
    
    if scores:
        return max(scores, key=scores.get)
    
    return "technology"


def extract_role_type(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    
    for role_type, keywords in ROLE_TYPE_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in text_lower:
                score += 1
        if score > 0:
            scores[role_type] = score
    
    if scores:
        return max(scores, key=scores.get)
    
    return "engineering"


def parse_job_description(job_data: dict[str, Any]) -> dict[str, Any]:
    description = job_data.get("description", "")
    title = job_data.get("title", "")
    company = job_data.get("company", "")
    industry = job_data.get("industry", "")
    location = job_data.get("location", "")
    company_size = job_data.get("company_size", "")
    
    full_text = f"{title} {company} {industry} {location} {company_size} {description}"
    
    skills = extract_skills(full_text)
    experience = extract_experience_level(full_text)
    detected_industry = extract_industry(full_text) if not industry else industry
    role_type = extract_role_type(full_text)
    
    return {
        "skills": skills,
        "experience": experience,
        "industry": detected_industry,
        "role_type": role_type,
        "title": title,
        "company": company,
        "location": location,
        "company_size": company_size,
        "description": description,
    }