# 🩺 RheumaRepo

**Digital and AI education for rheumatologists: tutorials, tools, tweaks and scripts.**

[![Focus](https://img.shields.io/badge/focus-rheumatology-8b5cf6)](#)
[![AI](https://img.shields.io/badge/topic-applied_AI-blue)](#)
[![Hands on](https://img.shields.io/badge/style-hands--on-green)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Most AI education for clinicians stops at slideware. This repo goes the other way:
working code, reproducible recipes and honest benchmarks, written by a rheumatologist
for rheumatologists who want to actually build with this technology, not just read about it.

## 📦 What's inside

| Project | What it is | Why a clinician should care |
|---|---|---|
| [`ornith-blackwell-vllm/`](ornith-blackwell-vllm/) | Serve a 35B open model on two consumer GPUs: 838 tok/s across 8 parallel agents, 262k context each | Run capable AI **privately**, on hardware you control, with no patient data leaving the room. Includes an 8-agent demo grounded on the full GRAPPA 2021 PsA recommendations |

More coming: prompt patterns for clinical literature work, retrieval setups for guideline
corpora, agent workflows for systematic reviews, and small scripts that remove friction
from academic life.

## 🧭 Principles

1. **Grounding over recall.** A language model asked to recite guidelines will invent
   plausible nonsense (we watched one invent an enthesitis index that does not exist).
   Every clinical demo here feeds the model the actual source document and instructs it
   to answer only from it. The difference is the whole game.
2. **Verify like a reviewer.** Outputs are checked against the source tables before
   anything is published, the same way you would check a trainee's data extraction.
3. **Private by default.** Local and self-hosted setups are first-class citizens:
   clinical curiosity should not require sending text to a third party.
4. **Show the failures.** The recipes document what broke and why, because the errors
   are where the education lives.

## 👤 Who I am

**Vincenzo Venerito, MD**, rheumatologist and assistant professor at the University of Bari
Aldo Moro, working on applied AI in rheumatology.

[![Google Scholar](https://img.shields.io/badge/Google_Scholar-profile-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=UitwlOAAAAAJ)
[![PubMed](https://img.shields.io/badge/PubMed-publications-326599?logo=pubmed&logoColor=white)](https://pubmed.ncbi.nlm.nih.gov/?term=Venerito+Vincenzo%5BAuthor%5D)
[![X](https://img.shields.io/badge/X-@drvincentvenus-000000?logo=x&logoColor=white)](https://x.com/drvincentvenus)

## ⚠️ Disclaimer

Everything here is educational material about technology. Nothing is medical advice,
and no AI output should inform patient care without verification against primary
sources and clinical judgment.

## 🤝 Contributing

Questions, ideas and pull requests from the rheumatology community are welcome.
Open an issue or reach out on X.

## 📄 License

MIT
