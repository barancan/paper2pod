-- paper2pod: episodes table
--
-- One-time manual step: paste this into the Supabase SQL Editor for your
-- project (Dashboard -> SQL Editor -> New query) and run it. paper2pod does
-- not create this table for you.

create table if not exists episodes (
  id uuid primary key default gen_random_uuid(),
  episode_name text not null,              -- "[TITLE] - [AUTHORS]", matches audio filename stem
  source_type text not null check (source_type in ('markdown', 'openlabs')),
  source_reference text not null,          -- original file path or OpenLabs project URL
  title text not null,
  authors_or_team text not null,
  transcript_text text not null,           -- full generated script, CTA included
  cta_text text,                           -- snapshot of CTA used, null if cta.enabled was false
  word_count integer not null,
  estimated_duration_seconds integer not null,
  transcript_provider text not null,       -- e.g. "anthropic"
  transcript_model text not null,          -- e.g. "claude-sonnet-4-6"
  tts_voice text not null,                 -- e.g. "en-US-GuyNeural"
  audio_bucket text not null,
  audio_object_path text not null,         -- key within the bucket
  audio_public_url text,                   -- populated only if bucket is public
  created_at timestamptz not null default now()
);

create index if not exists episodes_created_at_idx on episodes (created_at desc);
create index if not exists episodes_episode_name_idx on episodes (episode_name);
