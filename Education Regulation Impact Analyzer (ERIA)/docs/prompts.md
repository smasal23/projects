# Prompt Design (ERIA)

## Core Principle

Use a single structured prompt to extract all insights.

---

## Master Prompt

You are an education policy expert.

Analyze the following regulation and return STRICT JSON output with:

- category
- summary
- stakeholders (students, faculty, institutions)
- benefits
- risks
- impact (short, medium, long term)

Rules:
- Use simple language
- Be precise
- Do not hallucinate
- Output only valid JSON

---

## Best Practices

- Keep prompts structured
- Avoid multiple LLM calls
- Use clear instructions
- Control output format

---

## Future Improvements

- Few-shot prompting
- Dynamic prompt injection
- Context-aware prompting
