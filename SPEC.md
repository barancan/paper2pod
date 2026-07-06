# Paper2Pod — Technical Specification

Version 0.1 · 2026-07-02

## 1. Overview

Paper2Pod is a Python CLI service. Input: a research paper or research summary as a `.md` file. Output: a Two Minute Papers style narration, rendered to a realistic AI voice recording, uploaded to Supabase Storage.

Pipeline: `parse → generate transcript → synthesize audio → upload → done`.

Target runtime per paper: under 90 seconds on a standard laptop, network permitting.

## 2. Goals

1. One command converts a `.md` or `.pdf` file into an uploaded audio file. PDFs
   are sent natively to Claude (Anthropic provider only).
2. Transcript model is configurable (provider + model string) via config file.
3. TTS is free by default, with a pluggable provider interface for keyed services.
4. Live progress in the terminal for each pipeline stage.
5. Recording named `[TITLE-OF-THE-RESEARCH] - [AUTHORS]`.
6. All errors logged to file with context.

Non-goals (v0.1): batch processing, video generation, background music, a web UI.

## 3. Architecture

```
paper2pod/
├── config.yaml              # model, voice, bucket, style settings
├── .env                     # secrets (API keys, Supabase creds)
├── paper2pod/
│   ├── cli.py               # Typer entrypoint, orchestration, progress UI
│   ├── config.py            # pydantic-settings config loader (yaml + env)
│   ├── parser.py            # md loading, frontmatter + LLM metadata extraction
│   ├── pdf.py               # pdf loading, native Anthropic document block + metadata
│   ├── transcript.py        # LLM transcript generation (provider-agnostic)
│   ├── tts/
│   │   ├── base.py          # TTSProvider protocol
│   │   ├── edge.py          # default: edge-tts (free, no key)
│   │   └── elevenlabs.py    # optional keyed provider
│   ├── storage.py           # Supabase upload
│   └── logging_setup.py     # console + rotating file logger
├── tests/
├── pyproject.toml
└── README.md
```

Stack: Python 3.11+, Typer (CLI), Rich (progress), pydantic-settings (config), `anthropic` / `openai` SDKs (transcript), `edge-tts` (audio), `supabase` (storage).

## 4. Pipeline stages

### 4.1 Parse (`parser.py`)

- Read the `.md` file (UTF-8, fail with clear error on missing/binary file).
- Extract metadata in two passes:
  1. YAML frontmatter if present (`title`, `authors`).
  2. Fallback: a cheap LLM call that returns `{title, authors[]}` as JSON from the first ~2,000 tokens of the document.
- If neither yields a title, abort with an actionable error ("add a `# Title` heading or frontmatter").
- Truncate document body to a configurable max token budget (default 12,000 tokens) before transcript generation, keeping abstract/intro/results sections preferentially if headings are detectable.

### 4.2 Transcript generation (`transcript.py`)

- Provider-agnostic interface: `generate(paper_text, metadata, style_config) -> Transcript`.
- Supported providers v0.1: `anthropic`, `openai`. Provider and model string come from `config.yaml`; keys from `.env`.
- Style: Two Minute Papers narration. The system prompt encodes:
  - Enthusiastic, accessible tone; second-person address ("Now, hold on to your papers…").
  - Structure: hook → what the researchers did → why it's hard → key result with a concrete number → limitation → closing ("What a time to be alive!" style sign-off, but varied, not verbatim every time).
  - Length target: 320–420 words (~2–3 minutes at ~150 wpm). Hard cap enforced; if the model overruns, one revision pass asks it to compress.
  - Plain speech only: no markdown, no citations, no URLs, numbers written for the ear ("thirty-five thousand" not "35,000").
- Output object: `{text, title, authors, word_count, estimated_duration_s}`.

### 4.3 TTS (`tts/`)

- `TTSProvider` protocol: `synthesize(text: str, out_path: Path) -> Path`.
- **Default provider: `edge-tts`** (Microsoft Edge neural voices). Rationale: genuinely free, no API key, no rate-limit friction at this scale, and voice quality is realistic neural TTS — the proven free option. Default voice `en-US-GuyNeural` (energetic male, closest fit to the reference style); configurable.
  - Known risk: edge-tts is an unofficial client and Microsoft can break it. Mitigated by the provider interface — swapping providers is a config change.
- Optional keyed provider: ElevenLabs (`tts.provider: elevenlabs`, key in `.env`). Local alternative worth noting for full offline use: Kokoro-82M.
- Output format: MP3, 48 kbps mono minimum (edge-tts default is fine). Written to a temp dir before upload.

### 4.4 Storage (`storage.py`)

- Upload to Supabase Storage bucket (default `recordings`, configurable).
- Object name: `{sanitized_title} - {sanitized_authors}.mp3`.
  - Sanitization: strip characters illegal in object keys (`/ \ : * ? " < > |`), collapse whitespace, trim to 180 chars total, join multiple authors with `, ` and truncate to first 3 + `et al.` if longer.
  - Example: `Diffusion Models Beat GANs - Dhariwal, Nichol.mp3`.
- Upsert behavior configurable (`storage.upsert: true|false`, default false → append ` (2)` on collision).
- On success, print the public or signed URL depending on bucket visibility.

## 5. Configuration

`config.yaml` (non-secret):

```yaml
transcript:
  provider: anthropic # anthropic | openai
  model: claude-sonnet-4-6
  max_input_tokens: 12000
  target_words: [320, 420]

tts:
  provider: edge # edge | elevenlabs
  voice: en-US-GuyNeural
  rate: "+8%" # slight pace-up to match the reference style

storage:
  bucket: recordings
  upsert: false

logging:
  level: INFO
  file: logs/paper2pod.log
```

`.env` (secrets, never committed):

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
ELEVENLABS_API_KEY=          # only if tts.provider=elevenlabs
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
```

Config precedence: CLI flags > env vars > `config.yaml` > defaults. Validation at startup via pydantic; missing required keys fail fast with the exact variable name.

## 6. CLI and progress

```
paper2pod run paper.md [--config config.yaml] [--voice ...] [--model ...] [--dry-run]
```

Progress via Rich: one line per stage with spinner → checkmark and timing:

```
✔ Parsed paper.md (title: "Diffusion Models Beat GANs", 2 authors)   0.8s
✔ Transcript generated (387 words, ~2m 35s)                          6.2s
⠸ Synthesizing audio (edge-tts, en-US-GuyNeural)…
```

`--dry-run` stops after transcript generation and prints it to stdout (useful for style iteration without burning TTS/upload time).

## 7. Error handling and logging

- Rotating file logger (`logs/paper2pod.log`, 5 MB × 3 files). Console shows WARNING+ only; file gets DEBUG+.
- Every stage wraps errors into typed exceptions (`ParseError`, `TranscriptError`, `TTSError`, `StorageError`) with the input filename and stage context.
- LLM and TTS calls: 3 retries with exponential backoff on transient errors (429/5xx/network); non-retryable errors (401, invalid config) fail immediately with a remediation hint.
- Exit codes: 0 success, 2 config error, 3 parse error, 4 transcript error, 5 TTS error, 6 storage error.

## 8. Testing (v0.1 minimum)

- Unit: filename sanitization, config precedence, frontmatter extraction, word-count enforcement.
- Integration (mocked network): full pipeline with stubbed LLM + TTS + Supabase.
- One live smoke test behind an env flag.

## 9. Open questions / risks

1. **edge-tts fragility.** Unofficial API; pin the package version and keep the ElevenLabs adapter as the documented fallback.
2. **Long papers.** v0.1 truncates. A map-reduce summarization pass is the v0.2 answer if quality suffers.
3. **Author extraction accuracy.** LLM extraction from arbitrary markdown is best-effort; frontmatter is the reliable path and should be documented as the recommended input format.
4. **Voice licensing.** Edge voices are fine for personal/prototype use; check Microsoft terms before commercial distribution of the recordings.
