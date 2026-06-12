# M-SHIELD

**A Risk-Aware Multimodal Defense Framework Against Indirect Prompt Injection in Agentic AI Systems**

---

## What is M-SHIELD?

M-SHIELD is an external, model-agnostic defense framework that protects AI agents from indirect prompt injection attacks across image, audio, and text modalities — without modifying the underlying model.

### Three Core Innovations

| Innovation | What it does |
|---|---|
| Visibility Trust Scoring | Detects text hidden from human perception in images |
| Cross-Modal Agreement Checking | Finds injections by comparing vision model output vs OCR |
| Risk-Aware Tool Restriction | Limits agent tool access proportional to detected risk |

---

## Live Demo

[Deploy on Streamlit](https://streamlit.io)

---

## How to Run Locally

```bash
git clone https://github.com/yourusername/mshield
cd mshield
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
mshield/
├── app.py           ← Streamlit frontend
├── mshield.py       ← Core M-SHIELD pipeline
├── requirements.txt ← Dependencies
└── README.md
```

---

## Key Results

| Attack Type | Detection Rate | False Positive |
|---|---|---|
| Low opacity injection | 100% | 0% |
| Tiny font injection | 80% | 0% |
| White-on-white injection | 40% | 0% |
| Clean documents | N/A | 0% |

---

## Research Paper

*M-SHIELD: A Risk-Aware Multimodal Defense Framework Against Indirect Prompt Injection in Agentic AI Systems*

---

## Benchmark — MIPI-Bench

MIPI-Bench is the first unified benchmark for indirect prompt injection evaluation across image, audio, and text modalities, comprising 418 labelled samples.

| Modality | Source | Samples |
|---|---|---|
| Image | MM-SafetyBench + Synthetic | 276 |
| Audio | SACRED-Bench | 80 |
| Text | InjecAgent | 62 |
