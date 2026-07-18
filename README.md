# СУДРЫН ХЭЛ — Ganjuur Digital Humanities Platform

> 108 volumes of the Mongolian woodblock *Ganjuur*, carved 1717–1720 under
> the Manchu Emperor Enkh-Amgalan. Nearly every page is now embedded,
> searchable, and permanently attributed to its BDRC source.

---

## What the crisis looks like

- Mongolia's 7 daily newspapers together cycle through only **~2,944 unique
  words per issue** (after deduplication and removal of proper nouns).
- Government documents operate within a working vocabulary of roughly **300 words**.
- The *Ganjuur* alone contains **41,588 pages** of the deepest, most logically
  structured Mongolian ever carved — philosophical, grammatical, ritual,
  poetic — most of it **unreadable to a modern audience**.

The widest point of this gap is where the *СУДРЫН ХЭЛ* project works.

## What the platform does

Ganjuur is a visual-similarity search engine over the woodblock manuscript.
A researcher uploads a cropped glyph or page and the system returns
visually and structurally similar forms across all 108 volumes, using
**DINOv2-base 768-dimensional cosine vectors** computed on an NVIDIA RTX 3060.

Alongside search the app records human-entered **transliterations
byte-for-byte** — no normalization, no tokenization, no silent correction —
so the reading of each form is preserved exactly as entered.

### Current production state (live at `192.168.0.55`)

- All **108 volumes** ingested from BDRC (83,466 content pages).
- Qdrant running in Docker (`qdrant_ganjuur`), HTTP on `127.0.0.1:6333`.
- Gradio app served at `:7860`, managed by `systemd ganjuur.service`
  (runs `longcat.py` as the production entrypoint).
- Three collections: `ganjuur_frames`, `ganjuur_genealogy`, `ganjuur_words`.
- Every vector point carries BDRC attribution in its payload
  (`bdrc_url`, `bdrc_resource_id`, `bdrc_volume`, `bdrc_access`).

---

## Repository layout

```
.
├── README.md                  # this file — start here
├── .gitignore                 # excludes debug artifacts, .env, scan dumps
│
├── src/                       # production Python source
│   ├── longcat.py             # ★ PRODUCTION — the running entrypoint (systemd)
│   ├── gpt.py                 # alternative app build (research/experimental)
│   ├── app.py                 # alternative app build (research/experimental)
│   ├── ingest_from_bdrc.py    # BDRC IIIF batch ingester (108 volumes → Qdrant)
│   ├── appvmeta.py            # app variant / metadata tooling
│   ├── snapshot_v5.py         # production snapshot builder (COW hardlink clones)
│   ├── grad_patch.py          # shared Gradio compatibility patches
│   ├── config.py              # centralized paths/settings (env-var overridable)
│   └── archive/               # older single-file builds (kept for reference)
│       ├── app.py.orig           # original copy before refactor
│       ├── gpt.py.orig           # original copy before refactor
│       ├── longcat.py.orig       # original copy before refactor
│       ├── app06_24.py
│       ├── app06_24_3.py
│       └── ganjuur_v2.py
│
├── scripts/                   # deployment and operations
│   ├── snapshot.sh            # launch snapshot_v5.py on the server
│   ├── deploy_remote.sh       # rsync + restart the live app
│   ├── redeploy.sh            # full redeploy from local
│   ├── update_service.sh      # patch systemd unit and reload
│   ├── upload_and_restart.sh  # upload code, restart service
│   ├── migrate.sh             # data-migration helper
│   ├── preflight_check.ps1    # pre-deploy validation (Windows side)
│   ├── deploy_snap.ps1        # deploy snapshot script to server
│   ├── do_deploy.ps1          # orchestrate a full deploy
│   ├── enc_*.ps1              # base64-pipe helpers (plink/pscp staging)
│   ├── run_last.ps1           # re-run the last deploy step
│   ├── mon.py                 # snapshot-process health monitor
│   ├── verify_final.py        # post-snapshot integrity check
│   ├── Code-improvements.sh   # server-side code-improvement runner
│   └── Code-improvements.html # code-improvement report viewer
│
├── docs/                      # documentation and fundraising
│   ├── GANJUUR_PROJECT_HANDOFF.md   # production handoff — read before any change
│   ├── SudrynKhel_Pitch_Deck.md     # investor pitch deck (narrative, no tech)
│   ├── SudrynKhel_Presentation_Script.md  # 11-slide speaker script
│   ├── Ganjuur_Pitch_Deck_MN.md     # Mongolian-Cyrillic pitch deck
│   ├── GEMINI_MASTER_PROMPT.md      # full-context prompt to regenerate all materials
│   ├── ganjuur.docx                 # founding concept document (5 products, 5 activities)
│   └── ganjuur_gariin_avlaga.html   # in-app Mongolian user manual
│
└── assets/                    # static assets
    ├── favicon.png
    └── EMBEDDING_MAP.png      # UMAP visualization of the frame collection
```

---

## Founding concept

The original vision — **СУДРЫН ХЭЛ** ("Language of the Scriptures") — is
documented in `docs/ganjuur.docx`. It defines five products and five
activities:

**Products**
1. High-resolution digital edition of all 108 volumes.
2. AI-assisted search and automatic Cyrillic transliteration platform.
3. Print and digital editions in both traditional Mongolian and Cyrillic.
4. End-of-volume thesaurus linking each term to modern usage.
5. Open online dictionary — "Судрын хэлний үгийн сан".

**Activities**
1. Thesaurus construction (Sanskrit / Tibetan / Manchu / Chinese / Old Uyghur
   etymologies).
2. Per-volume introductions explaining content and linguistic features.
3. International ISBN/ISSN registration and national-archive deposit.
4. Cross-lingual reading pairs (6 languages).
5. Curriculum integration (Р3 program).

---

## Critical protection rules

Read `docs/GANJUUR_PROJECT_HANDOFF.md` in full before changing anything.
The short version:

1. **Never** delete, overwrite, mount over, or reset `vectordb/` or
   `vectordb_pre_docker_backup/`.
2. **Never** drop or recreate `ganjuur_frames` without verifying counts first.
3. **Never** remove the compatibility patches at the top of `app.py` / `gpt.py`.
4. **Never** normalize, trim, tokenize, or alter transcription text before
   storage or export.
5. **Never** replace the `systemd` service with a screen-only deployment.
6. **Never** commit `.env`, secrets, API keys, or passwords.

---

## Data attribution

All manuscript images are sourced from the **Buddhist Digital Resource Center**
(BDRC), Lokesh Chandra Collection (`bdr:MW4CZ5370`), used under the BDRC
Open-Access policy. Every vector point in Qdrant carries the source URL,
resource ID, volume, and access level in its payload. When surfacing results
to users, link back to the BDRC original.

---

## License

Code in this repository is the intellectual property of the СУДРЫН ХЭЛ project.
Manuscript images © BDRC and respective rights holders. Used with permission
for non-commercial research and cultural-preservation purposes.
