# paper2pod

paper2pod is a Python CLI that turns a research paper (as a `.md` or `.pdf`
file) or an OpenLabs project into an energetic narration, renders it to
a realistic AI voice recording, and uploads the result to Supabase Storage.

Pipeline: get the source content (parse a paper, or fetch an OpenLabs
project), generate a narration transcript with an LLM, append a spoken call
to action, synthesize audio, upload it, done. One command, under 90 seconds
on a typical laptop, network permitting. PDFs are sent natively to Claude, so
that path requires the `anthropic` transcript provider.

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
2. **Create the `episodes` table.** One-time manual step: open the
   Supabase dashboard's SQL Editor (New query) and paste in the contents of
   `supabase/schema.sql`, then run it. This is what every successful run
   gets recorded into (see "Episode records" below); paper2pod does not
   create the table for you.
3. **Fill in `.env`** with `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` if you
   set `transcript.provider: openai`), `SUPABASE_URL`, and
   `SUPABASE_SERVICE_ROLE_KEY`.
4. **Preview a transcript without spending on TTS or storage:**
   ```bash
   paper2pod run tests/fixtures/sample_paper.md --dry-run
   ```
5. **Run the full pipeline** (accepts a `.md` or `.pdf`; the extension picks
   the ingestion path):
   ```bash
   paper2pod run path/to/your_paper.md
   paper2pod run path/to/your_paper.pdf
   ```
   On success this prints the public or signed URL of the uploaded MP3, and
   records the episode so it's browsable with `paper2pod list`/`paper2pod show`.

### OpenLabs project briefs

`paper2pod openlabs <project-url>` takes the same flags as `run` (plus
`--no-cache`) and produces a spoken brief for an OpenLabs project instead of
a paper. Grab any project URL from https://openlabs.bio.xyz/projects, for
example:

```bash
paper2pod openlabs https://openlabs.bio.xyz/projects/aab54f59-abc7-44b9-8736-4d6d56111f37 --dry-run
paper2pod openlabs https://openlabs.bio.xyz/projects/aab54f59-abc7-44b9-8736-4d6d56111f37
```

Add `--no-cache` to force a fresh fetch instead of reading the previous
result from `cache/openlabs/`:

```bash
paper2pod openlabs https://openlabs.bio.xyz/projects/aab54f59-abc7-44b9-8736-4d6d56111f37 --dry-run --no-cache
```

Both commands share the same transcript/CTA/TTS/upload pipeline and produce
files named the same way; only how the source content is obtained differs.

## Configuration reference

Precedence, highest to lowest: CLI flags, then environment variables, then
`config.yaml`, then built-in defaults. Missing required secrets fail fast
at startup with the exact variable name.

`config.yaml`:

| Key                           | Default                    | Notes                                                                                                                                                   |
| ----------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `transcript.provider`         | `anthropic`                | `anthropic` or `openai`                                                                                                                                 |
| `transcript.model`            | `claude-sonnet-4-6`        | fallback model string, used only if `transcript.models` has no entry for `provider`                                                                     |
| `transcript.models`           | see `config.yaml`          | optional per-provider model map, e.g. `{anthropic: ..., openai: ...}`; the entry for the active `provider` wins over `transcript.model`                 |
| `transcript.max_input_tokens` | `12000`                    | paper body is truncated to roughly this many tokens                                                                                                     |
| `transcript.target_words`     | `[320, 420]`               | narration length target                                                                                                                                 |
| `tts.provider`                | `edge`                     | `edge` (free) or `elevenlabs` (keyed)                                                                                                                   |
| `tts.voice`                   | `en-US-GuyNeural`          | edge-tts voice name, or an ElevenLabs voice ID when `tts.provider=elevenlabs`                                                                           |
| `tts.rate`                    | `+8%`                      | edge-tts speaking rate adjustment                                                                                                                       |
| `storage.bucket`              | `recordings`               | Supabase Storage bucket name                                                                                                                            |
| `storage.upsert`              | `false`                    | overwrite on name collision instead of appending " (2)"                                                                                                 |
| `logging.level`               | `INFO`                     | reserved, see Decisions                                                                                                                                 |
| `logging.file`                | `logs/paper2pod.log`       | rotating log file path                                                                                                                                  |
| `cta.enabled`                 | `true`                     | append a spoken call to action after every transcript                                                                                                   |
| `cta.text`                    | see `config.yaml`          | the CTA text, spoken verbatim, never sent to the LLM. Non-empty and 80 words or fewer are enforced at startup; a console warning appears above 60 words |
| `openlabs.base_url`           | `https://openlabs.bio.xyz` | the OpenLabs app URL; the API host is derived from it                                                                                                   |
| `openlabs.cache_ttl_hours`    | `24`                       | how long a fetched project is cached before refetching                                                                                                  |
| `openlabs.min_content_words`  | `200`                      | minimum usable content required, or the fetch fails                                                                                                     |

Any of these can also be set as an environment variable using the pattern
`PAPER2POD_<SECTION>__<KEY>`, for example `PAPER2POD_TTS__VOICE`. The CLI
flags `--config`, `--voice`, and `--model` override everything else.

`.env` (secrets, never committed):

| Variable                    | Required when                                               |
| --------------------------- | ----------------------------------------------------------- |
| `ANTHROPIC_API_KEY`         | `transcript.provider=anthropic`                             |
| `OPENAI_API_KEY`            | `transcript.provider=openai`                                |
| `ELEVENLABS_API_KEY`        | `tts.provider=elevenlabs` (skipped entirely on `--dry-run`) |
| `SUPABASE_URL`              | always, unless `--dry-run`                                  |
| `SUPABASE_SERVICE_ROLE_KEY` | always, unless `--dry-run`                                  |

## Switching providers

**Transcript:** set `transcript.provider: openai` in `config.yaml` and put
`OPENAI_API_KEY` in `.env`. Both providers implement the same `generate()`
interface, so no code changes are needed. If you keep a `transcript.models`
map (the shipped `config.yaml` does, with entries for both providers),
switching `provider` automatically switches to the right model too, with no
need to also update `transcript.model` by hand. `--model` on the command
line always overrides both for that one run.

**TTS:** set `tts.provider: elevenlabs` in `config.yaml`, put
`ELEVENLABS_API_KEY` in `.env`, and set `tts.voice` to an ElevenLabs voice
ID. This is the documented fallback if edge-tts ever breaks (see
Troubleshooting).

## Fetching OpenLabs projects

The OpenLabs app (`openlabs.bio.xyz`) is a client-rendered Next.js app: a
plain GET on a project page returns no project data in the HTML. Rather than
driving a headless browser, paper2pod calls the same public JSON API the
app's own frontend calls, at `api.openlabs.bio.xyz` (no API key required).
This was found by downloading the app's JS bundles and grepping them for API
routes and the API host string. `sources/openlabs.py` fetches the project
detail, its collaborators (for the byline), and its update posts (combined
with the summary to reliably clear `openlabs.min_content_words`, since the
summary alone is usually short), then caches the result at
`cache/openlabs/{project_id}.json` for `openlabs.cache_ttl_hours`.

This is unofficial and undocumented, like the OpenLabs frontend's use of it.
**It depends entirely on OpenLabs' current API and page structure and may
need maintenance if they change either.** If the API ever disappears, the
next fallback (not currently implemented) would be a headless-browser
render behind an optional `paper2pod[browser]` extra, exactly as sketched in
the original request, kept as an optional dependency so the `run` command
for local `.md` files keeps working without it installed.

## Episode records

Every successful `run` or `openlabs` episode is recorded as a row in a
Supabase Postgres `episodes` table (schema in `supabase/schema.sql`, see the
Quickstart step above): the full transcript text (CTA included), the
title/authors/team, the transcript and TTS config actually used for that
run, and where the audio landed in Storage. This makes past episodes
browsable straight from the Supabase dashboard, and from the CLI:

```bash
paper2pod list
paper2pod list --limit 5
paper2pod show "Diffusion Models Beat GANs"
paper2pod show 7ed1e5eb-20d6-4943-9fd7-5548f45e8bf4
```

`list` prints a table of recent episodes (created time, episode name,
source type, duration), most recent first. `show` is the hand-testing entry
point: it prints the full transcript text under `--- TRANSCRIPT ---`
(copy-pasteable elsewhere, e.g. into a video tool) and the audio's URL under
`--- AUDIO ---`. It accepts an episode id, an exact episode name, or a
partial, case-insensitive name match (`"diffusion models"` finds the
closest episode by that substring).

Recording an episode is best-effort and never blocks a successful pipeline
run: if the Supabase write fails right after the audio uploads (network
blip, table missing, etc.), the run still exits 0, prints the audio URL
it already has, and logs the failure -- nothing is lost, just not indexed
for browsing.

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
- **`openlabs` fetch fails (exit code 7):** either the project URL is wrong
  (404, "Project not found") or the project genuinely has too little
  content to narrate ("insufficient content"). Try `--no-cache` if you
  suspect a stale cached fetch is the problem.
- **"Episode record failed" after a successful upload:** the audio is safe
  (the URL printed right above is real); only the browsable record in the
  `episodes` table failed to write. Check that `supabase/schema.sql` has
  actually been run in your project's SQL Editor -- a missing table is the
  most common cause.
- **`list`/`show` exit 2:** these two commands only need
  `SUPABASE_URL`/`SUPABASE_SERVICE_ROLE_KEY`, not a transcript provider key;
  the error names whichever of the two is missing.
- **`show` exits 1 with "No episode found matching":** the id/exact
  name/partial name didn't match anything recorded. Run `paper2pod list`
  to see what's actually there.
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
- **The CTA's system-prompt instruction is static**, not conditional on
  `cta.enabled`: the narrator is always asked to end on a natural closing
  sentence rather than a catchphrase, whether or not a CTA actually follows.
  A single prompt is simpler than branching prompt text per config, and a
  natural closing sentence reads fine either way.
- **`cta.text` is stripped once**, in the config model, so every downstream
  comparison (including "byte-equal to config" in tests) is exact without
  repeated `.strip()` calls at each use site.
- **`ProjectContent.team_or_authors` is `list[str]`**, not a bare string,
  specifically so it's a drop-in replacement for `PaperMetadata.authors`.
  That's what lets the `openlabs` command reuse `build_object_name()` and
  the shared publish pipeline completely unchanged.
- **The OpenLabs API host is derived, not configured separately**: swapping
  `openlabs.bio.xyz` for `api.openlabs.bio.xyz` in `_derive_api_base()`
  keeps `openlabs.base_url` as the one user-facing value to change (e.g. for
  a staging environment) rather than needing two URLs kept in sync.
- **Collaborators and update posts are best-effort**, not fatal, on fetch
  failure: only the primary project-detail call blocks the pipeline.
  Missing collaborators fall back to the project creator, then to
  `"OpenLabs"`; missing updates just leave `body_text` as the summary alone
  (which may then legitimately trip the `min_content_words` check).
- **The OpenLabs cache key is the project's UUID**, not a human-readable
  slug, since that's what OpenLabs project URLs actually contain.
- **`Paper2PodError`'s `__str__` suffix changed from `(file: ...)` to
  `(input: ...)`**, since `input_file` now also carries a URL for
  `SourceError`; no code or test depended on the old literal wording.
- **Per-provider models (`transcript.models`) default to an empty map in
  code**, not to a pre-filled `{anthropic: ..., openai: ...}` dict: the
  shipped `config.yaml` is what actually populates it, same as every other
  section. Defaulting it non-empty in code would make the per-provider
  entry win unconditionally, silently overriding a plain `--model` flag,
  `PAPER2POD_TRANSCRIPT__MODEL`, or a hand-edited flat `transcript.model` any
  time its provider happened to already have a code-level default.
- **Episode records use Supabase Postgres, not a local SQLite file**:
  Supabase already holds the audio and `supabase-py` is already a
  dependency, so this makes records browsable straight from the Supabase
  dashboard without any extra tooling -- useful for exactly the hand-testing
  workflow `show` exists for.
- **No new `storage.public` config flag** for `audio_public_url`:
  `storage.py` already detects bucket visibility dynamically per-upload
  (`client.storage.get_bucket(bucket).public`); `upload()` now returns that
  bit directly via `UploadResult` instead of duplicating the check behind a
  second, config-level source of truth that could drift from the bucket's
  actual setting.
- **`episode_name` is derived from the actual uploaded object path**
  (`Path(upload_result.object_path).stem`), not the pre-upload requested
  name, so it's always the real filename stem even when a " (2)"-style
  collision suffix was appended during upload.
- **`authors_or_team` reuses `storage.format_authors()`**, the same
  first-3-plus-"et al." truncation already used in the audio filename,
  rather than a separate full join -- keeps the record's byline consistent
  with what's actually in the filename rather than a second, longer value.
- **`list`/`show` failures exit 1**, not a new dedicated exit code: they're
  read-only commands outside the stage-based pipeline (0/2/3/4/5/6/7), and
  `RecordError` deliberately has no pipeline exit code of its own (a record
  failure must never fail an otherwise-successful `run`/`openlabs`).
- **This is unrelated to a separate, not-yet-built HTTP API feature** (with
  its own ephemeral `jobs.db` for request/job status). `db.py`'s functions
  take an explicit `client` + typed values with no CLI-specific coupling,
  specifically so that future feature can call `record_episode()` directly
  once it exists -- but nothing here depends on or references it.
