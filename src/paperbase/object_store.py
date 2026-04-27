"""Object-store adapters for canonical PDF and artifact persistence."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from shutil import copy2
from typing import Protocol
from urllib.parse import urlparse

from paperbase.config import PaperbaseConfig


class ObjectStore(Protocol):
    bucket: str

    def ensure_bucket(self) -> None: ...

    def put_file(self, *, key: str, source_path: Path, content_type: str | None = None) -> str: ...

    def put_bytes(self, *, key: str, content: bytes, content_type: str | None = None) -> str: ...

    def download_to_path(self, *, storage_uri: str, destination_path: Path) -> Path: ...


def build_storage_uri(bucket: str, key: str) -> str:
    normalized_key = key.lstrip("/")
    return f"s3://{bucket}/{normalized_key}"


def parse_storage_uri(storage_uri: str) -> tuple[str, str]:
    parsed = urlparse(storage_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Unsupported storage uri: {storage_uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid storage uri: {storage_uri}")
    return bucket, key


class FilesystemObjectStore:
    """Local filesystem-backed object store used for single-user dev/test flows."""

    def __init__(self, *, root_dir: Path, bucket: str) -> None:
        self.root_dir = root_dir
        self.bucket = bucket

    def ensure_bucket(self) -> None:
        (self.root_dir / self.bucket).mkdir(parents=True, exist_ok=True)

    def put_file(self, *, key: str, source_path: Path, content_type: str | None = None) -> str:
        del content_type
        self.ensure_bucket()
        destination = self.root_dir / self.bucket / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(source_path, destination)
        return build_storage_uri(self.bucket, key)

    def put_bytes(self, *, key: str, content: bytes, content_type: str | None = None) -> str:
        del content_type
        self.ensure_bucket()
        destination = self.root_dir / self.bucket / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return build_storage_uri(self.bucket, key)

    def download_to_path(self, *, storage_uri: str, destination_path: Path) -> Path:
        bucket, key = parse_storage_uri(storage_uri)
        source = self.root_dir / bucket / key
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, destination_path)
        return destination_path


class S3ObjectStore:
    """S3-compatible object store using the MinIO Python client."""

    def __init__(
        self,
        *,
        endpoint: str,
        bucket: str,
        access_key: str,
        secret_key: str,
    ) -> None:
        from minio import Minio  # type: ignore[import-untyped]

        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported object store endpoint: {endpoint}")
        netloc = parsed.netloc or parsed.path
        if not netloc:
            raise ValueError(f"Unable to resolve object store endpoint: {endpoint}")

        self.bucket = bucket
        self.client = Minio(
            netloc,
            access_key=access_key,
            secret_key=secret_key,
            secure=parsed.scheme == "https",
        )

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def put_file(self, *, key: str, source_path: Path, content_type: str | None = None) -> str:
        self.ensure_bucket()
        self.client.fput_object(
            self.bucket,
            key,
            str(source_path),
            content_type=content_type,
        )
        return build_storage_uri(self.bucket, key)

    def put_bytes(self, *, key: str, content: bytes, content_type: str | None = None) -> str:
        self.ensure_bucket()
        self.client.put_object(
            self.bucket,
            key,
            data=BytesIO(content),
            length=len(content),
            content_type=content_type,
        )
        return build_storage_uri(self.bucket, key)

    def download_to_path(self, *, storage_uri: str, destination_path: Path) -> Path:
        bucket, key = parse_storage_uri(storage_uri)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.fget_object(bucket, key, str(destination_path))
        return destination_path


def build_object_store(config: PaperbaseConfig) -> ObjectStore:
    backend = config.object_store_backend.strip().lower()
    if backend == "s3":
        if not config.object_store_access_key or not config.object_store_secret_key:
            raise ValueError("S3 object store requires access and secret keys.")
        return S3ObjectStore(
            endpoint=config.object_store_endpoint,
            bucket=config.object_store_bucket,
            access_key=config.object_store_access_key,
            secret_key=config.object_store_secret_key,
        )
    return FilesystemObjectStore(
        root_dir=Path(config.object_store_local_root),
        bucket=config.object_store_bucket,
    )
