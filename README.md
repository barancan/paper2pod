# paper2pod

paper2pod is a Python CLI that turns a research paper (as a `.md` file) into
an energetic, ear-friendly narration, renders it to a realistic AI voice
recording, and uploads the result to Supabase Storage.

Pipeline: parse the paper, generate a narration transcript with an LLM,
synthesize audio, upload it, done. One command, under 90 seconds on a
typical laptop, network permitting.

## Prerequisites

- Python 3.11 or newer.
- A Supabase project with a Storage bucket (see Quickstart below).
- An API key for at least one transcript provider: Anthropic or OpenAI.
- Optional: an ElevenLabs API key if you want keyed TTS instead of the free
  default (edge-tts, no key required).

## Install

```bash
git clone <this-repo-url>
cd paper2pod
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Then open `.env` and fill in the keys you plan to use (see Configuration
reference below). Never commit `.env`, it's already covered by
`.gitignore`.

## Quickstart

1. **Create the Supabase bucket.** In the Supabase dashboard, go to
   Storage, click "New bucket", and create one named `recordings` (or
   whatever you set `storage.bucket` to in `config.yaml`). Mark it public if
   you want paper2pod to print a plain public URL; leave it private if you'd
   rather it print a signed URL that expires after an hour.
2. **Fill in `.env`** with `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` if you
   set `transcript.provider: openai`), `SUPABASE_URL`, and
   `SUPABASE_SERVICE_ROLE_KEY`.
3. **Preview a transcript without spending on TTS or storage:**
   ```bash
   paper2pod run tests/fixtures/sample_paper.md --dry-run
   ```
4. **Run the full pipeline:**
   ```bash
   paper2pod run path/to/your_paper.md
   ```
   On success this prints the public or signed URL of the uploaded MP3.

## Configuration reference

Precedence, highest to lowest: CLI flags, then environment variables, then
`config.yaml`, then built-in defaults. Missing required secrets fail fast
at startup with the exact variable name.

`config.yaml`:

| Key | Default | Notes |
|---|---|---|
| `transcript.provider` | `anthropic` | `anthropic` or `openai` |
| `transcript.model` | `claude-sonnet-4-6` | model string passed to the provider |
| `transcript.max_input_tokens` | `12000` | paper body is truncated to roughly this many tokens |
| `transcript.target_words` | `[320, 420]` | narration length target |
| `tts.provider` | `edge` | `edge` (free) or `elevenlabs` (keyed) |
| `tts.voice` | `en-US-GuyNeural` | edge-tts voice name, or an ElevenLabs voice ID when `tts.provider=elevenlabs` |
| `tts.rate` | `+8%` | edge-tts speaking rate adjustment |
| `storage.bucket` | `recordings` | Supabase Storage bucket name |
| `storage.upsert` | `false` | overwrite on name collision instead of appending " (2)" |
| `logging.level` | `INFO` | reserved, see Decisions |
| `logging.file` | `logs/paper2pod.log` | rotating log file path |

Any of these can also be set as an environment variable using the pattern
`PAPER2POD_<SECTION>__<KEY>`, for example `PAPER2POD_TTS__VOICE`. The CLI
flags `--config`, `--voice`, and `--model` override everything else.

`.env` (secrets, never committed):

| Variable | Required when |
|---|---|
| `ANTHROPIC_API_KEY` | `transcript.provider=anthropic` |
| `OPENAI_API_KEY` | `transcript.provider=openai` |
| `ELEVENLABS_API_KEY` | `tts.provider=elevenlabs` (skipped entirely on `--dry-run`) |
| `SUPABASE_URL` | always, unless `--dry-run` |
| `SUPABASE_SERVICE_ROLE_KEY` | always, unless `--dry-run` |

## Switching providers

**Transcript:** set `transcript.provider: openai` in `config.yaml` (or pass
`--model gpt-4o` etc.) and put `OPENAI_API_KEY` in `.env`. Both providers
implement the same `generate()` interface, so no code changes are needed.

**TTS:** set `tts.provider: elevenlabs` in `config.yaml`, put
`ELEVENLABS_API_KEY` in `.env`, and set `tts.voice` to an ElevenLabs voice
ID. This is the documented fallback if edge-tts ever breaks (see
Troubleshooting).

## Troubleshooting

- **401 / authentication errors:** the CLI fails immediately (no retries)
  and names which provider rejected the key. Double-check the corresponding
  variable in `.env` is present, current, and has no leading/trailing
  whitespace.
- **edge-tts stops working:** edge-tts is an unofficial client for
  Microsoft's service and can break without notice. Switch to
  `tts.provider: elevenlabs` in `config.yaml` (see Switching providers
  above) as an immediate workaround.
- **Missing bucket / storage errors:** confirm the bucket named in
  `storage.bucket` exists in your Supabase project and that
  `SUPABASE_SERVICE_ROLE_KEY` (not the anon key) is set.
- **"Could not determine a title" parse error:** add a `title:` field to
  YAML frontmatter at the top of the file, or a `# Title` heading. See
  `tests/fixtures/` for examples.
- All errors are also written with full context to `logs/paper2pod.log`
  (rotated at 5 MB, 3 files kept); the console only shows warnings and
  above.

## External API reference

`postman_collection.json` in the repo root documents the raw HTTP requests
paper2pod makes to Anthropic, OpenAI, ElevenLabs, and the Supabase Storage
REST API, useful for debugging those integrations directly. The default
edge-tts provider uses a websocket protocol rather than plain REST, so it
isn't included.

## Development

```bash
ruff check .
pytest
```

The integration suite mocks the LLM, TTS, and Supabase clients. One live
smoke test hits real providers and is skipped unless `PAPER2POD_LIVE=1` is
set alongside a valid `.env`.

## Decisions

Choices made where the spec was silent:

- **Token budgeting** uses a fixed ~4-characters-per-token heuristic rather
  than a real tokenizer, since no tokenizer package was in the dependency
  list. This governs both the LLM metadata-extraction excerpt size (~2000
  tokens) and the `max_input_tokens` truncation.
- **Truncation** splits the paper on markdown headings and keeps
  abstract/introduction/results/conclusion sections first when the body
  exceeds the token budget, preserving original order among kept sections.
- **`logging.level`** in `config.yaml` is accepted but currently unused:
  the file handler is fixed at DEBUG and the console at WARNING, matching
  the literal levels spec §7 specifies. The key is reserved for future use.
- **Config environment overrides** use the convention
  `PAPER2POD_<SECTION>__<KEY>` (e.g. `PAPER2POD_TTS__VOICE`), separate from
  the plain-named secrets in `.env`.
- **`--dry-run` secret validation** only requires the transcript provider's
  API key; TTS and Supabase secrets are not required, since those stages
  never run in dry-run mode.
- **ElevenLabs** is implemented as a direct REST client (via `urllib`)
  rather than an SDK dependency, since no ElevenLabs package was in the
  spec's dependency list.
- **Filename sanitization** replaces illegal characters with a space
  (rather than deleting them) so words don't get accidentally joined, then
  collapses whitespace.
- **Collision handling** increments beyond " (2)" to " (3)", " (4)", and so
  on if further collisions occur, rather than giving up after one retry.
- **Bucket visibility** (public vs. signed URL) is determined by querying
  the bucket's `public` flag via the Supabase client at upload time.
- **The local temp MP3** is deleted after a successful upload, and left on
  disk if the upload step fails, so a failed run can be retried without
  resynthesizing audio.
- **The Postman collection** covers only the REST-based external APIs
  (Anthropic, OpenAI, ElevenLabs, Supabase Storage); edge-tts, the default
  TTS provider, has no REST equivalent to document.
