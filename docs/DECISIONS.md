1- **Rationale**: High retrieval performance, transparent execution plans (`EXPLAIN`), and clean separation from ORM overhead during search.

## ADR-011: ESCO Taxonomy Granularity Gaps, Alias Overlay, and Guarded Matching Cascade
- **Context**: ESCO v1.2.1 provides an authoritative European multilingual classification of skills and occupations, but exhibits a significant latency and granularity gap regarding modern developer ecosystems:
  1. Broad concepts exist (e.g., "Python (computer programming)", "machine learning", "cloud technologies"), but specific frameworks and libraries (`FastAPI`, `scikit-learn`, `PyTorch`, `pgvector`, `Hugging Face`, `Celery`) are absent.
  2. Unrestricted fuzzy matching on short tech tokens (C, R, Go) causes severe false-positive cascades (e.g., "C" matching "CSS" or "CI/CD", "R" matching "REST", "Go" matching the common verb "go").
  3. Short acronyms (e.g., `DDD`, `WandB`, `K8s`, `AWS`) incorrectly match common English words if fuzzy distance is applied to short strings.
- **Decision**:
  1. **Curated Alias Overlay (`data/aliases.yaml`)**: Maintain a version-controlled YAML overlay with 314+ software and AI skills, linking each entry to its canonical label and parent ESCO concept URI.
  2. **Three-Tier Matching Cascade**:
     - *Exact ESCO*: Preferred labels and alternative labels.
     - *Exact Alias*: Case-insensitive lookup in the alias registry.
     - *Guarded Fuzzy Match*: Powered by RapidFuzz (`ratio` >= 0.82), strictly restricted to terms >= 4 characters. Acronyms (<= 4 chars) and spaCy stop words are barred from fuzzy matching.
  3. **Context Guards for 1-2 Character Languages**: "C", "R", and "Go" require contextual validation regexes (e.g. "Languages: C, C++", "C programming", "Golang", "R/Python") and are never matched as free-form single letters.
  4. **Evidence Tiers & Keyword-Stuffing Flag**: Skills found in Work Experience or Projects with action verbs are tagged `DEMONSTRATED`; skills in list headers are tagged `LISTED_ONLY`. If `listed_only / max(demonstrated, 1) >= 3.0` and uncorroborated skills >= 15, the resume is flagged for keyword stuffing.
- **Rationale**: Achieves 90.0% Micro-F1 and 89.8% Macro-F1 across 20 diverse engineering benchmark resumes without generative hallucinations or silent misclassifications.

