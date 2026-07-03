from types import SimpleNamespace

import pytest

from paper2pod.logging_setup import StorageError
from paper2pod.storage import (
    build_object_name,
    format_authors,
    resolve_url,
    sanitize_component,
    upload,
)


def test_sanitize_component_strips_illegal_chars_and_collapses_whitespace():
    assert sanitize_component('A/B: Test?  Name<>|"*') == "A B Test Name"


def test_sanitize_component_preserves_unicode_title():
    result = sanitize_component("Attention Is All You Need: 注意力评估 \U0001f680")
    assert ":" not in result
    assert "注意力评估" in result
    assert "\U0001f680" in result


def test_format_authors_two_authors():
    assert format_authors(["Prafulla Dhariwal", "Alex Nichol"]) == "Prafulla Dhariwal, Alex Nichol"


def test_format_authors_truncates_ten_authors_to_first_three_et_al():
    authors = [f"Author {i}" for i in range(10)]
    result = format_authors(authors)
    assert result == "Author 0, Author 1, Author 2 et al."


def test_format_authors_empty_list():
    assert format_authors([]) == ""


def test_build_object_name_matches_spec_example():
    name = build_object_name("Diffusion Models Beat GANs", ["Dhariwal", "Nichol"])
    assert name == "Diffusion Models Beat GANs - Dhariwal, Nichol.mp3"


def test_build_object_name_no_authors_omits_separator():
    name = build_object_name("Solo Title", [])
    assert name == "Solo Title.mp3"


def test_build_object_name_truncates_to_180_chars_total():
    long_title = "A" * 300
    name = build_object_name(long_title, ["Author One"])
    stem = name.removesuffix(".mp3")
    assert len(stem) <= 180
    assert name.endswith(".mp3")


def test_build_object_name_strips_illegal_chars_from_title_and_authors():
    name = build_object_name('Weird: Title / With * Illegal? Chars', ['Auth/or "One"'])
    for bad_char in '/\\:*?"<>|':
        assert bad_char not in name


class FakeStorageApiError(Exception):
    pass


class FakeBucketFileApi:
    def __init__(self, existing=None, fail_message=None):
        self.existing = set(existing or [])
        self.uploaded = []
        self.fail_message = fail_message

    def upload(self, path, file, file_options=None):
        if self.fail_message:
            raise FakeStorageApiError(self.fail_message)
        upsert = (file_options or {}).get("upsert") == "true"
        if path in self.existing and not upsert:
            raise FakeStorageApiError("The resource already exists")
        self.existing.add(path)
        self.uploaded.append(path)
        return SimpleNamespace(path=path)

    def get_public_url(self, path):
        return f"https://fake.supabase.co/public/{path}"

    def create_signed_url(self, path, expires_in):
        url = f"https://fake.supabase.co/signed/{path}"
        return {"signedURL": url, "signedUrl": url}


class FakeStorageNamespace:
    def __init__(self, bucket_api, public):
        self._bucket_api = bucket_api
        self._public = public

    def from_(self, bucket):
        return self._bucket_api

    def get_bucket(self, bucket):
        return SimpleNamespace(public=self._public)


class FakeSupabaseClient:
    def __init__(self, bucket_api, public=False):
        self.storage = FakeStorageNamespace(bucket_api, public)


def test_upload_returns_public_url_when_bucket_public(tmp_path):
    local = tmp_path / "audio.mp3"
    local.write_bytes(b"fake mp3")
    bucket_api = FakeBucketFileApi()
    client = FakeSupabaseClient(bucket_api, public=True)

    result = upload(client, "recordings", "Title - Author.mp3", local)

    assert result.url == "https://fake.supabase.co/public/Title - Author.mp3"
    assert result.object_path == "Title - Author.mp3"
    assert result.is_public is True
    assert bucket_api.uploaded == ["Title - Author.mp3"]


def test_upload_returns_signed_url_when_bucket_private(tmp_path):
    local = tmp_path / "audio.mp3"
    local.write_bytes(b"fake mp3")
    bucket_api = FakeBucketFileApi()
    client = FakeSupabaseClient(bucket_api, public=False)

    result = upload(client, "recordings", "Title - Author.mp3", local)

    assert result.url == "https://fake.supabase.co/signed/Title - Author.mp3"
    assert result.is_public is False


def test_upload_appends_2_suffix_on_collision_when_upsert_false(tmp_path):
    local = tmp_path / "audio.mp3"
    local.write_bytes(b"fake mp3")
    bucket_api = FakeBucketFileApi(existing={"Title - Author.mp3"})
    client = FakeSupabaseClient(bucket_api, public=True)

    result = upload(client, "recordings", "Title - Author.mp3", local, upsert=False)

    assert bucket_api.uploaded == ["Title - Author (2).mp3"]
    assert result.object_path == "Title - Author (2).mp3"
    assert result.url.endswith("Title - Author (2).mp3")


def test_upload_overwrites_on_collision_when_upsert_true(tmp_path):
    local = tmp_path / "audio.mp3"
    local.write_bytes(b"fake mp3")
    bucket_api = FakeBucketFileApi(existing={"Title - Author.mp3"})
    client = FakeSupabaseClient(bucket_api, public=True)

    result = upload(client, "recordings", "Title - Author.mp3", local, upsert=True)

    assert bucket_api.uploaded == ["Title - Author.mp3"]
    assert result.object_path == "Title - Author.mp3"
    assert result.url.endswith("Title - Author.mp3")


def test_upload_raises_storage_error_on_persistent_failure(tmp_path):
    local = tmp_path / "audio.mp3"
    local.write_bytes(b"fake mp3")
    bucket_api = FakeBucketFileApi(fail_message="network unreachable")
    client = FakeSupabaseClient(bucket_api, public=True)

    with pytest.raises(StorageError, match="Supabase upload failed"):
        upload(client, "recordings", "Title - Author.mp3", local)


def test_resolve_url_standalone_public_bucket():
    bucket_api = FakeBucketFileApi()
    client = FakeSupabaseClient(bucket_api, public=True)

    url, is_public = resolve_url(client, "recordings", "Some Episode.mp3")

    assert url == "https://fake.supabase.co/public/Some Episode.mp3"
    assert is_public is True


def test_resolve_url_standalone_private_bucket():
    bucket_api = FakeBucketFileApi()
    client = FakeSupabaseClient(bucket_api, public=False)

    url, is_public = resolve_url(client, "recordings", "Some Episode.mp3")

    assert url == "https://fake.supabase.co/signed/Some Episode.mp3"
    assert is_public is False
