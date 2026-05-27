"""Streaming multipart upload helpers for Paperbase ingest routes."""

from __future__ import annotations

import codecs
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from uuid import uuid4

from fastapi import Request
import python_multipart
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header

from services.paperbase_api.errors import PaperbaseAPIError

_FORM_FIELD_MAX_BYTES = 64 * 1024
_FORM_FIELD_MAX_COUNT = 32
_MULTIPART_HEADER_COUNT_MAX = 32
_MULTIPART_HEADER_NAME_MAX_BYTES = 128
_MULTIPART_HEADER_VALUE_MAX_BYTES = 8 * 1024
_MULTIPART_HEADER_TOTAL_MAX_BYTES = 16 * 1024


@dataclass(frozen=True, slots=True)
class StreamedLocalLibraryUpload:
    staged_dir: Path
    owner_id: str
    collection_title: str | None
    collection_description: str | None


@dataclass(slots=True)
class _MultipartPart:
    field_name: str | None = None
    filename: str | None = None
    headers: list[tuple[bytes, bytes]] = field(default_factory=list)
    partial_header_name: bytes = b""
    partial_header_value: bytes = b""
    header_bytes: int = 0
    field_data: bytearray = field(default_factory=bytearray)
    file_bytes: int = 0
    should_store: bool = False
    handle: BinaryIO | None = None

    @property
    def is_file(self) -> bool:
        return self.filename is not None


def _safe_uploaded_relative_path(filename: str | None) -> Path:
    raw_name = (filename or "").replace("\\", "/").strip()
    candidate = PurePosixPath(raw_name or "upload.pdf")
    safe_parts = [part for part in candidate.parts if part not in {"", ".", "..", "/"}]
    if not safe_parts:
        safe_parts = ["upload.pdf"]
    return Path(*safe_parts)


def _decode_multipart_value(value: bytes, charset: str) -> str:
    try:
        return value.decode(charset)
    except LookupError as exc:
        raise PaperbaseAPIError(
            status_code=400,
            error="invalid_input",
            message="Multipart upload specifies an unsupported charset.",
        ) from exc
    except UnicodeDecodeError:
        return value.decode("latin-1")


def _validated_charset(value: bytes) -> str:
    charset = _decode_multipart_value(value, "latin-1")
    try:
        codecs.lookup(charset)
    except LookupError as exc:
        raise PaperbaseAPIError(
            status_code=400,
            error="invalid_input",
            message="Multipart upload specifies an unsupported charset.",
        ) from exc
    return charset


def _upload_too_large(message: str) -> PaperbaseAPIError:
    return PaperbaseAPIError(status_code=400, error="upload_too_large", message=message)


class _StreamingLocalLibraryUploadParser:
    def __init__(
        self,
        *,
        request: Request,
        staging_root: Path,
        max_file_count: int,
        max_single_file_bytes: int,
        max_total_bytes: int,
    ) -> None:
        self.request = request
        self.staging_root = staging_root
        self.staged_dir = staging_root / uuid4().hex
        self.max_file_count = max_file_count
        self.max_single_file_bytes = max_single_file_bytes
        self.max_total_bytes = max_total_bytes
        self.charset = "utf-8"
        self.fields: dict[str, str] = {}
        self.file_count = 0
        self.field_count = 0
        self.total_request_bytes = 0
        self.reached_final_boundary = False
        self.stored_any = False
        self.part = _MultipartPart()

    async def parse(self) -> StreamedLocalLibraryUpload:
        content_type_header = self.request.headers.get("content-type")
        content_type, params = parse_options_header(content_type_header)
        if content_type.lower() != b"multipart/form-data":
            raise PaperbaseAPIError(
                status_code=400,
                error="invalid_input",
                message="local-library-upload requires multipart/form-data.",
            )
        boundary = params.get(b"boundary")
        if not boundary:
            raise PaperbaseAPIError(
                status_code=400,
                error="invalid_input",
                message="Multipart upload is missing a boundary.",
            )
        raw_charset = params.get(b"charset")
        if raw_charset:
            self.charset = _validated_charset(raw_charset)

        self.staged_dir.mkdir(parents=True, exist_ok=True)
        parser = python_multipart.MultipartParser(boundary, self._callbacks())
        try:
            async for chunk in self.request.stream():
                self._record_request_bytes(chunk)
                parser.write(chunk)
            parser.finalize()
            if not self.reached_final_boundary:
                raise PaperbaseAPIError(
                    status_code=400,
                    error="invalid_input",
                    message="Multipart upload is missing the final boundary.",
                )
            if not self.stored_any:
                raise PaperbaseAPIError(
                    status_code=400,
                    error="invalid_input",
                    message="At least one PDF file is required for local library upload.",
                )
        except PaperbaseAPIError:
            self._close_current_handle()
            self._cleanup()
            raise
        except MultipartParseError as exc:
            self._close_current_handle()
            self._cleanup()
            raise PaperbaseAPIError(
                status_code=400,
                error="invalid_input",
                message="Invalid multipart upload.",
            ) from exc
        except Exception:
            self._close_current_handle()
            self._cleanup()
            raise

        return StreamedLocalLibraryUpload(
            staged_dir=self.staged_dir,
            owner_id=self.fields.get("owner_id") or "local-user",
            collection_title=self.fields.get("collection_title"),
            collection_description=self.fields.get("collection_description"),
        )

    def _callbacks(self) -> dict[str, object]:
        return {
            "on_part_begin": self._on_part_begin,
            "on_part_data": self._on_part_data,
            "on_part_end": self._on_part_end,
            "on_header_field": self._on_header_field,
            "on_header_value": self._on_header_value,
            "on_header_end": self._on_header_end,
            "on_headers_finished": self._on_headers_finished,
            "on_end": self._on_end,
        }

    def _on_part_begin(self) -> None:
        self.part = _MultipartPart()

    def _on_header_field(self, data: bytes, start: int, end: int) -> None:
        fragment = data[start:end]
        self._record_header_bytes(fragment)
        if len(self.part.partial_header_name) + len(fragment) > _MULTIPART_HEADER_NAME_MAX_BYTES:
            raise PaperbaseAPIError(
                status_code=400,
                error="invalid_input",
                message="Multipart header name exceeds the allowed size.",
            )
        self.part.partial_header_name += fragment

    def _on_header_value(self, data: bytes, start: int, end: int) -> None:
        fragment = data[start:end]
        self._record_header_bytes(fragment)
        if len(self.part.partial_header_value) + len(fragment) > _MULTIPART_HEADER_VALUE_MAX_BYTES:
            raise PaperbaseAPIError(
                status_code=400,
                error="invalid_input",
                message="Multipart header value exceeds the allowed size.",
            )
        self.part.partial_header_value += fragment

    def _on_header_end(self) -> None:
        if len(self.part.headers) >= _MULTIPART_HEADER_COUNT_MAX:
            raise PaperbaseAPIError(
                status_code=400,
                error="invalid_input",
                message="Multipart part includes too many headers.",
            )
        self.part.headers.append(
            (
                self.part.partial_header_name.lower(),
                self.part.partial_header_value,
            )
        )
        self.part.partial_header_name = b""
        self.part.partial_header_value = b""

    def _on_headers_finished(self) -> None:
        headers = dict(self.part.headers)
        disposition = headers.get(b"content-disposition")
        if not disposition:
            raise PaperbaseAPIError(
                status_code=400,
                error="invalid_input",
                message="Multipart part is missing content disposition.",
            )
        _, options = parse_options_header(disposition)
        raw_name = options.get(b"name")
        if raw_name is None:
            raise PaperbaseAPIError(
                status_code=400,
                error="invalid_input",
                message="Multipart part is missing a field name.",
            )
        self.part.field_name = _decode_multipart_value(raw_name, self.charset)

        raw_filename = options.get(b"filename")
        if raw_filename is None:
            self.field_count += 1
            if self.field_count > _FORM_FIELD_MAX_COUNT:
                raise PaperbaseAPIError(
                    status_code=400,
                    error="invalid_input",
                    message="Multipart upload includes too many form fields.",
                )
            return

        self.part.filename = _decode_multipart_value(raw_filename, self.charset)
        if self.part.field_name != "files":
            raise PaperbaseAPIError(
                status_code=400,
                error="invalid_input",
                message="Unexpected file field in local library upload.",
            )
        self.file_count += 1
        if self.file_count > self.max_file_count:
            raise _upload_too_large(
                f"Upload includes too many files; maximum is {self.max_file_count}."
            )

        relative_path = _safe_uploaded_relative_path(self.part.filename)
        self.part.should_store = relative_path.suffix.lower() == ".pdf"
        if self.part.should_store:
            destination = self.staged_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.part.handle = destination.open("wb")

    def _on_part_data(self, data: bytes, start: int, end: int) -> None:
        message_bytes = data[start:end]
        if self.part.is_file:
            self._record_file_bytes(message_bytes)
            if self.part.should_store:
                if self.part.handle is None:
                    raise RuntimeError("PDF upload part is missing a destination handle.")
                self.part.handle.write(message_bytes)
            return

        if len(self.part.field_data) + len(message_bytes) > _FORM_FIELD_MAX_BYTES:
            raise PaperbaseAPIError(
                status_code=400,
                error="invalid_input",
                message="Multipart form field exceeds the allowed size.",
            )
        self.part.field_data.extend(message_bytes)

    def _record_request_bytes(self, message_bytes: bytes) -> None:
        self.total_request_bytes += len(message_bytes)
        if self.total_request_bytes > self.max_total_bytes:
            raise _upload_too_large(
                "Multipart request exceeds the configured total "
                f"limit of {self.max_total_bytes} bytes."
            )

    def _record_file_bytes(self, message_bytes: bytes) -> None:
        self.part.file_bytes += len(message_bytes)
        if self.part.file_bytes > self.max_single_file_bytes:
            raise _upload_too_large(
                "Uploaded file exceeds the configured per-file "
                f"limit of {self.max_single_file_bytes} bytes."
            )

    def _on_part_end(self) -> None:
        if self.part.is_file:
            if self.part.should_store:
                self._close_current_handle()
                self.stored_any = True
            return

        if self.part.field_name in {
            "owner_id",
            "collection_title",
            "collection_description",
        }:
            self.fields[self.part.field_name] = _decode_multipart_value(
                bytes(self.part.field_data),
                self.charset,
            )

    def _record_header_bytes(self, message_bytes: bytes) -> None:
        self.part.header_bytes += len(message_bytes)
        if self.part.header_bytes > _MULTIPART_HEADER_TOTAL_MAX_BYTES:
            raise PaperbaseAPIError(
                status_code=400,
                error="invalid_input",
                message="Multipart part headers exceed the allowed size.",
            )

    def _on_end(self) -> None:
        self.reached_final_boundary = True

    def _close_current_handle(self) -> None:
        if self.part.handle is not None:
            self.part.handle.close()
            self.part.handle = None

    def _cleanup(self) -> None:
        shutil.rmtree(self.staged_dir, ignore_errors=True)


async def stage_streamed_local_library_upload(
    *,
    request: Request,
    staging_root: Path,
    max_file_count: int,
    max_single_file_bytes: int,
    max_total_bytes: int,
) -> StreamedLocalLibraryUpload:
    parser = _StreamingLocalLibraryUploadParser(
        request=request,
        staging_root=staging_root,
        max_file_count=max_file_count,
        max_single_file_bytes=max_single_file_bytes,
        max_total_bytes=max_total_bytes,
    )
    return await parser.parse()
