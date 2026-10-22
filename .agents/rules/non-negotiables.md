# ResumeFit — Non-Negotiable Principles

These rules apply to ALL code, documentation, and UI produced for the ResumeFit project.

1. **Explainability first**: every number in the UI must be traceable to a component or an evidence span in the resume text (store character offsets).
2. **Fit analysis, not hiring authority**: a persistent disclaimer in the UI and API responses. The word "probability" is never used for the fit index.
3. **Honest engineering**: unsupported inputs are flagged loudly, never silently degraded.
4. **Reproducibility**: fixed seeds, pinned dependencies, and every report and evaluation run records a manifest (model name+version, ESCO version, config hash, α/β, git commit).
5. **Privacy by default**: resumes are processed in memory; persistence is opt-in; PII (names, emails, phones, addresses) is redacted before embedding and never written to logs.
6. **Config over constants**: α, β, thresholds, section weights, penalties live in `config/default.yaml`, validated at startup. Nothing magic in code.
7. **No dead code, no TODO-comment stubs, no placeholder lorem text, no fake data in shipped paths.**
8. **Every module is independently testable with a clear typed interface.**

## Additional Engineering Rules

- Type hints on all public functions; docstrings explain WHY, not what.
- No commented-out code. No generic names like `utils.py` or `helpers.py` dumping grounds.
- Errors are typed and handled at boundaries; no bare except; no swallowed exceptions.
- Tests assert behaviour, not implementation; include negative and edge cases.
- No fabricated sample outputs in docs or UI: anything shown comes from a real run.
- UI copy, README, and docs must read like a careful human wrote them: specific, concise, no hype adjectives, no emoji-decorated headings.
- Prefer boring, well-understood solutions over clever ones; every dependency must earn its place.
- The word "probability" is NEVER used for the fit index. Use "Fit Index" or "fit score".
- Never claim something works unless it was actually run and verified.
- Never hardcode, invent, or "illustrate" evaluation numbers. Every metric must come from a real evaluation run.
