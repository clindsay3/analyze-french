#!/usr/bin/env python3
"""Audit JMdict French glosses against Japanese forms and English glosses.

The workflow supports direct Responses API calls and optional Batch API runs:

1. download-jmdict: fetch the current official JMdict.gz and decompress it.
2. prepare: parse entries with French glosses and create request JSONL.
3. run: audit prepared entries directly, with saved progress and --resume.
4. submit/status/collect: alternatively run and collect an OpenAI Batch job.
5. summarize: regenerate CSV and Markdown summaries from collected JSONL.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable


XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
DEFAULT_JMDICT_URL = "http://ftp.edrdg.org/pub/Nihongo/JMdict.gz"
AUDIT_PROMPT_VERSION = "2"

SYSTEM_PROMPT = """You are a bilingual lexicography QA reviewer for Japanese-to-French dictionary entries.

You receive Japanese forms, English reference glosses, and current French glosses from JMdict.
Judge the French against the Japanese meaning. English is supporting evidence, not an exhaustive checklist
or an infallible authority. If the Japanese meaning or English reference is uncertain, do not invent a correction.

Evaluate semantic coverage across ALL French glosses in the entry together. Languages may group senses
differently: one French gloss can cover several English senses or synonyms. Sense indices are source
locations, not a required one-to-one alignment between languages. Before claiming an omission, consider
the ordinary literal and figurative meanings of every existing French gloss in context.
Ordinary polysemy counts without requiring parenthetical explanations. Do not report a concept as absent
after acknowledging that an existing gloss naturally expresses it in the relevant context. A desire
for a clearer explanatory label is not a semantic gap. If coverage is genuinely uncertain, use
needs_review rather than likely_error and identify the unresolved substantive question.

Respect concise dictionary conventions. A French noun can adequately translate a Japanese suru noun
or na-adjective. Do not require separate noun, verb, adjective, adverb, or productive-prefix renderings
of the same concept. Missing French POS annotations are not a defect. Flag part_of_speech_mismatch
only when the rendering actually misrepresents the Japanese meaning, not because grammatical forms differ.

Flag only substantive user-facing problems: wrong or misleading French, unnatural French that misleads,
or a genuinely missing core/common Japanese meaning. A correct secondary gloss does not excuse an absent
central meaning. Conversely, another English synonym, grammatical realization, illustrative context,
rare/technical extension, or stylistic improvement does not by itself establish a missing core meaning.
Useful optional expansion is not an error and does not warrant needs_review.

For each missing_core_meaning flag, supply missing_meaning_evidence identifying the Japanese concept,
why it is core/common, the closest existing French gloss(es), and why those glosses do not already cover
it. Merely saying an English gloss is not explicitly listed in French is insufficient. Do not assume
the first sense is always the only central one. If input_truncated is true, unseen glosses may cover
an apparent omission; do not assert a missing meaning on the basis of the incomplete input.

Use ok when core meanings are adequately covered, even if expansion would be useful. Use needs_review
only for a plausible substantive defect with unresolved evidence, and explain that uncertainty.
Use likely_error only for a well-supported substantial defect. Confidence is confidence in this
classification, not a calibrated probability. Do not automatically promote uncertain findings.

Suggestions must address the demonstrated defect, preserve valid existing glosses, and be grounded in
Japanese usage rather than mechanically translating every English gloss. For ok entries, return no
issues or suggested changes, an empty missing_meaning_evidence array, and keep_existing.
For other entries, missing_meaning_evidence must be nonempty exactly when missing_core_meaning is flagged.
Return exactly one result per input entry. Keep explanations and suggestions concise.
"""

AUDIT_INSTRUCTIONS = [
    "Audit the French glosses collectively against Japanese meaning, using English as supporting evidence.",
    "Return one result for every input entry.",
    "Use likely_error only for substantial user-facing problems supported by evidence.",
    "Use needs_review only for an uncertain substantive defect, not an optional addition.",
    "Use ok for adequate concise coverage, including normal noun renderings of suru nouns and na-adjectives.",
]

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["results"],
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "ent_seq",
                    "status",
                    "confidence",
                    "issue_types",
                    "problem_summary",
                    "core_meanings_french",
                    "suggested_french_senses",
                    "suggested_action",
                    "missing_meaning_evidence",
                ],
                "properties": {
                    "ent_seq": {"type": "integer"},
                    "status": {
                        "type": "string",
                        "enum": ["ok", "needs_review", "likely_error"],
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "issue_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "missing_core_meaning",
                                "wrong_translation",
                                "unnatural_french",
                                "part_of_speech_mismatch",
                                "rare_sense_overrepresented",
                                "other",
                            ],
                        },
                    },
                    "problem_summary": {"type": "string"},
                    "missing_meaning_evidence": {
                        "type": "array",
                        "description": "Evidence for each missing core meaning; empty if no missing_core_meaning issue.",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["japanese_meaning", "why_core", "closest_existing_french", "coverage_gap"],
                            "properties": {
                                "japanese_meaning": {"type": "string"},
                                "why_core": {"type": "string"},
                                "closest_existing_french": {"type": "string"},
                                "coverage_gap": {"type": "string"},
                            },
                        },
                    },
                    "core_meanings_french": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "suggested_french_senses": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "suggested_action": {
                        "type": "string",
                        "enum": [
                            "keep_existing",
                            "add_missing_sense",
                            "rewrite_existing",
                            "remove_french_senses",
                            "manual_review",
                        ],
                    },
                },
            },
        }
    },
}


@dataclass
class SenseData:
    index: int
    pos: list[str]
    misc: list[str]
    info: list[str]
    english: list[str]
    french: list[str]


@dataclass
class EntryData:
    ent_seq: int
    headword: str
    kanji: list[str]
    readings: list[str]
    senses: list[SenseData]
    current_french_senses: list[str]
    current_english_senses: list[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = clean_text(value)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def entry_sha256(context: dict[str, Any]) -> str:
    """Hash the complete parsed source entry, before request truncation or formatting."""
    return sha256_json({key: context[key] for key in ("ent_seq", "kanji", "readings", "senses")})


def parse_snapshot_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("Use YYYY-MM-DD")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Snapshot date must be YYYY-MM-DD.") from exc
    return value


def report_provenance(manifest: dict[str, Any]) -> dict[str, str]:
    """Use saved run metadata only; never attribute today's prompt or date to old results."""
    snapshot_date = manifest.get("dictionary_snapshot_date", "")
    date_source = manifest.get("dictionary_snapshot_date_source", "")
    metadata = manifest.get("download_metadata") or {}
    if not snapshot_date and manifest.get("xml_sha256") and metadata.get("xml_sha256") == manifest["xml_sha256"]:
        headers = {key.lower(): value for key, value in (metadata.get("headers") or {}).items()}
        last_modified = headers.get("last-modified")
        if last_modified:
            try:
                modified_at = parsedate_to_datetime(last_modified)
                if modified_at.tzinfo is None:
                    raise ValueError("Snapshot timestamps need an explicit timezone.")
                snapshot_date = modified_at.astimezone(timezone.utc).date().isoformat()
                date_source = "download_last_modified"
            except (TypeError, ValueError, OverflowError):
                pass
    return {
        "audit_prepared_at_utc": manifest.get("created_at_utc", ""),
        "dictionary_snapshot_date": snapshot_date,
        "dictionary_snapshot_date_source": date_source,
        "dictionary_xml_sha256": manifest.get("xml_sha256", ""),
        "audit_model": manifest.get("model", ""),
        "audit_reasoning_effort": manifest.get("reasoning_effort") or ("model_default" if manifest.get("model") else ""),
        "audit_prompt_version": manifest.get("audit_prompt_version", "unversioned"),
        "audit_prompt_sha256": manifest.get("audit_prompt_sha256", ""),
    }


def load_env(env_file: Path | None, api_key_file: Path | None) -> None:
    if env_file and env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)

    if not os.environ.get("OPENAI_API_KEY") and api_key_file and api_key_file.exists():
        os.environ["OPENAI_API_KEY"] = api_key_file.read_text(encoding="utf-8").strip()


def get_openai_client(args: argparse.Namespace):
    load_env(Path(args.env_file) if args.env_file else None, Path(args.api_key_file).expanduser())
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install dependencies first: python3 -m pip install -r requirements.txt") from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set and --api-key-file was not readable.")
    return OpenAI()


def download_jmdict(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gz_path = output_dir / args.gz_name
    xml_path = output_dir / args.xml_name
    metadata_path = output_dir / "JMdict.download-metadata.json"

    req = urllib.request.Request(args.url, headers={"User-Agent": "jmdict-french-audit/1.0"})
    with urllib.request.urlopen(req, timeout=args.timeout_seconds) as resp:
        headers = dict(resp.headers.items())
        data = resp.read()

    gz_path.write_bytes(data)
    xml_path.write_bytes(gzip.decompress(data))
    metadata = {
        "downloaded_at_utc": utc_now(),
        "url": args.url,
        "headers": headers,
        "gz_path": str(gz_path),
        "xml_path": str(xml_path),
        "gz_sha256": sha256_file(gz_path),
        "xml_sha256": sha256_file(xml_path),
        "gz_size_bytes": gz_path.stat().st_size,
        "xml_size_bytes": xml_path.stat().st_size,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {xml_path}")
    print(f"Wrote {gz_path}")
    print(f"Wrote {metadata_path}")
    if headers.get("Last-Modified"):
        print(f"Last-Modified: {headers['Last-Modified']}")
    print(f"XML SHA256: {metadata['xml_sha256']}")
    return 0


def glosses_for_language(sense_elem: ET.Element, language: str | None) -> list[str]:
    values: list[str] = []
    for gloss in sense_elem.findall("gloss"):
        lang = gloss.attrib.get(XML_LANG)
        if language is None:
            if lang is None or lang == "eng":
                values.append(clean_text(gloss.text))
        elif lang == language:
            values.append(clean_text(gloss.text))
    return unique_preserve_order(values)


def parse_entry(elem: ET.Element) -> EntryData | None:
    ent_seq_text = clean_text(elem.findtext("ent_seq"))
    if not ent_seq_text:
        return None

    kanji = unique_preserve_order(k.findtext("keb") for k in elem.findall("k_ele"))
    readings = unique_preserve_order(r.findtext("reb") for r in elem.findall("r_ele"))
    headword = kanji[0] if kanji else readings[0] if readings else ent_seq_text

    senses: list[SenseData] = []
    current_french_senses: list[str] = []
    current_english_senses: list[str] = []

    for index, sense_elem in enumerate(elem.findall("sense"), start=1):
        pos = unique_preserve_order(p.text for p in sense_elem.findall("pos"))
        misc = unique_preserve_order(m.text for m in sense_elem.findall("misc"))
        info = unique_preserve_order(i.text for i in sense_elem.findall("s_inf"))
        english = glosses_for_language(sense_elem, None)
        french = glosses_for_language(sense_elem, "fre")
        sense = SenseData(
            index=index,
            pos=pos,
            misc=misc,
            info=info,
            english=english,
            french=french,
        )
        senses.append(sense)

        if english:
            current_english_senses.append(format_sense(index, pos, english))
        if french:
            current_french_senses.append(format_sense(index, pos, french))

    if not current_french_senses:
        return None

    return EntryData(
        ent_seq=int(ent_seq_text),
        headword=headword,
        kanji=kanji,
        readings=readings,
        senses=senses,
        current_french_senses=current_french_senses,
        current_english_senses=current_english_senses,
    )


def format_sense(index: int, pos: list[str], glosses: list[str]) -> str:
    pos_text = ", ".join(pos) if pos else "pos unspecified"
    return f"{index}[{pos_text}]: {'; '.join(glosses)}"


def iter_french_entries(
    xml_path: Path,
    *,
    max_entries: int | None = None,
    ent_seq_filter: set[int] | None = None,
) -> Iterable[EntryData]:
    yielded = 0
    for _event, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "entry":
            continue
        entry = parse_entry(elem)
        elem.clear()
        if entry is None:
            continue
        if ent_seq_filter is not None and entry.ent_seq not in ent_seq_filter:
            continue
        yield entry
        yielded += 1
        if max_entries is not None and yielded >= max_entries:
            return


def compact_entry(entry: EntryData, max_senses: int, max_glosses: int) -> dict[str, Any]:
    senses = []
    for sense in entry.senses[:max_senses]:
        senses.append(
            {
                "sense_index": sense.index,
                "pos": sense.pos,
                "misc": sense.misc,
                "info": sense.info,
                "english": sense.english[:max_glosses],
                "french": sense.french[:max_glosses],
            }
        )
    return {
        "ent_seq": entry.ent_seq,
        "headword": entry.headword,
        "kanji": entry.kanji,
        "readings": entry.readings,
        "senses": senses,
        "input_truncated": len(entry.senses) > max_senses or any(
            len(sense.english) > max_glosses or len(sense.french) > max_glosses
            for sense in entry.senses[:max_senses]
        ),
    }


def build_user_prompt(entries: list[EntryData], max_senses: int, max_glosses: int) -> str:
    payload = {
        "instructions": AUDIT_INSTRUCTIONS,
        "entries": [compact_entry(e, max_senses=max_senses, max_glosses=max_glosses) for e in entries],
    }
    return json.dumps(payload, ensure_ascii=False)


def response_body_for_batch(
    *,
    model: str,
    entries: list[EntryData],
    max_senses: int,
    max_glosses: int,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_user_prompt(entries, max_senses=max_senses, max_glosses=max_glosses),
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "jmdict_french_audit_batch",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            }
        },
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    return body


def batched(values: list[EntryData], size: int) -> Iterable[list[EntryData]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def parse_ent_seq_filter(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


@contextmanager
def output_directory_lock(output_dir: Path):
    lock_path = output_dir / ".audit_run.lock"
    try:
        with lock_path.open("x", encoding="utf-8") as lock:
            lock.write(f"{os.getpid()}\n")
    except FileExistsError as exc:
        raise SystemExit("Another command may be using this directory. If it crashed, confirm it stopped before removing .audit_run.lock.") from exc
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def prepare(args: argparse.Namespace) -> int:
    xml_path = Path(args.xml_path)
    if not xml_path.exists():
        raise SystemExit(f"XML path does not exist: {xml_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_directory_lock(output_dir):
        return prepare_requests(args, xml_path, output_dir)


def prepare_requests(args: argparse.Namespace, xml_path: Path, output_dir: Path) -> int:
    context_path = output_dir / "entry_context.jsonl"
    requests_path = output_dir / "batch_requests.jsonl"
    manifest_path = output_dir / "manifest.json"
    previous = load_manifest(output_dir) if manifest_path.exists() else {}
    saved_outputs = (
        "analysis_results.jsonl", "direct_output.jsonl", "batch_output.jsonl", "suspect_entries.csv",
        "summary.md", "likely_error_missing_core_meaning.txt", "direct_failure.json", "batch_errors.jsonl",
    )
    if previous.get("execution_mode") or previous.get("batch_id") or any(
        (output_dir / name).exists() for name in saved_outputs
    ):
        raise SystemExit(
            "This directory contains audit execution state or saved results. Prepare a new directory; "
            "use run --resume or summarize to continue this run."
        )

    entries = list(
        iter_french_entries(
            xml_path,
            max_entries=args.max_entries,
            ent_seq_filter=parse_ent_seq_filter(args.ent_seq),
        )
    )
    if not entries:
        raise SystemExit("No entries with French glosses matched the filters.")

    with context_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    request_count = 0
    with requests_path.open("w", encoding="utf-8") as f:
        for request_count, group in enumerate(batched(entries, args.entries_per_request), start=1):
            first = group[0].ent_seq
            last = group[-1].ent_seq
            request = {
                "custom_id": f"jmdict-fre-{request_count:05d}-{first}-{last}",
                "method": "POST",
                "url": "/v1/responses",
                "body": response_body_for_batch(
                    model=args.model,
                    entries=group,
                    max_senses=args.max_senses_per_entry,
                    max_glosses=args.max_glosses_per_sense,
                    max_output_tokens=args.max_output_tokens,
                    reasoning_effort=args.reasoning_effort,
                ),
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    manifest = {
        "created_at_utc": utc_now(),
        "xml_path": str(xml_path),
        "xml_size_bytes": xml_path.stat().st_size,
        "xml_sha256": sha256_file(xml_path),
        "model": args.model,
        "audit_prompt_version": AUDIT_PROMPT_VERSION,
        "audit_prompt_sha256": sha256_json({
            "system": SYSTEM_PROMPT, "instructions": AUDIT_INSTRUCTIONS, "response_schema": RESPONSE_SCHEMA,
        }),
        "dictionary_snapshot_date": args.snapshot_date or "",
        "dictionary_snapshot_date_source": "user_supplied" if args.snapshot_date else "",
        "entries_per_request": args.entries_per_request,
        "max_senses_per_entry": args.max_senses_per_entry,
        "max_glosses_per_sense": args.max_glosses_per_sense,
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "entry_count": len(entries),
        "request_count": request_count,
        "context_path": str(context_path),
        "requests_path": str(requests_path),
    }
    metadata_candidate = xml_path.parent / "JMdict.download-metadata.json"
    if metadata_candidate.exists():
        metadata = json.loads(metadata_candidate.read_text(encoding="utf-8"))
        if metadata.get("xml_sha256") == manifest["xml_sha256"]:
            manifest["download_metadata"] = metadata
        else:
            print("Warning: download metadata does not match this XML; its snapshot date will not be used.")
    provenance = report_provenance(manifest)
    manifest["dictionary_snapshot_date"] = provenance["dictionary_snapshot_date"]
    manifest["dictionary_snapshot_date_source"] = provenance["dictionary_snapshot_date_source"]
    save_manifest(output_dir, manifest)

    print(f"Wrote {context_path}")
    print(f"Wrote {requests_path}")
    print(f"Wrote {manifest_path}")
    print(f"Entries: {len(entries)}")
    print(f"API requests: {request_count}")
    return 0


def load_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    temporary_path = output_dir / ".manifest.json.tmp"
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_dir / "manifest.json")


def require_batch_directory(output_dir: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("execution_mode") == "direct" or any(
        (output_dir / name).exists() for name in ("direct_output.jsonl", "direct_failure.json")
    ):
        raise SystemExit("This is a direct API run. Prepare a new directory for a Batch job.")


def submit(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    load_manifest(output_dir)
    with output_directory_lock(output_dir):
        return submit_prepared(args, output_dir)


def submit_prepared(args: argparse.Namespace, output_dir: Path) -> int:
    manifest = load_manifest(output_dir)
    require_batch_directory(output_dir, manifest)
    requests_path = Path(manifest["requests_path"])
    if not requests_path.exists():
        raise SystemExit(f"Batch request file not found: {requests_path}")

    client = get_openai_client(args)
    with requests_path.open("rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/responses",
        completion_window=args.completion_window,
        metadata={"project": "jmdict-french-audit", "created_by": "codex"},
    )

    manifest["submitted_at_utc"] = utc_now()
    manifest["input_file_id"] = uploaded_file.id
    manifest["batch_id"] = batch.id
    manifest["completion_window"] = args.completion_window
    save_manifest(output_dir, manifest)

    print(f"Uploaded file: {uploaded_file.id}")
    print(f"Batch: {batch.id}")
    print(f"Status: {getattr(batch, 'status', '(unknown)')}")
    return 0


def batch_id_from_args(args: argparse.Namespace, manifest: dict[str, Any]) -> str:
    batch_id = args.batch_id or manifest.get("batch_id")
    if not batch_id:
        raise SystemExit("No batch id provided and manifest has no batch_id.")
    return batch_id


def status(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    manifest = load_manifest(output_dir)
    client = get_openai_client(args)
    batch = client.batches.retrieve(batch_id_from_args(args, manifest))
    data = batch.model_dump() if hasattr(batch, "model_dump") else dict(batch)
    keys = [
        "id",
        "status",
        "request_counts",
        "input_file_id",
        "output_file_id",
        "error_file_id",
        "created_at",
        "in_progress_at",
        "completed_at",
        "failed_at",
        "expired_at",
        "cancelling_at",
        "cancelled_at",
    ]
    print(json.dumps({k: data.get(k) for k in keys if data.get(k) is not None}, indent=2))
    return 0


def read_file_content(client: Any, file_id: str) -> bytes:
    content = client.files.content(file_id)
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8")
    if hasattr(content, "read"):
        data = content.read()
        return data if isinstance(data, bytes) else str(data).encode("utf-8")
    if hasattr(content, "text"):
        return str(content.text).encode("utf-8")
    return str(content).encode("utf-8")


def extract_output_text(response_body: dict[str, Any]) -> str:
    if response_body.get("output_text"):
        return response_body["output_text"]
    parts: list[str] = []
    for item in response_body.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                parts.append(content["text"])
    return "\n".join(parts).strip()


def load_context(context_path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with context_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["ent_seq"])] = row
    return out


def normalize_result(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return {
        "ent_seq": int(row["ent_seq"]),
        "headword": context.get("headword", ""),
        "kanji": context.get("kanji", []),
        "readings": context.get("readings", []),
        "status": row.get("status", "needs_review"),
        "confidence": row.get("confidence", 0),
        "issue_types": row.get("issue_types", []) or [],
        "problem_summary": row.get("problem_summary", ""),
        "core_meanings_french": row.get("core_meanings_french", []) or [],
        "suggested_french_senses": row.get("suggested_french_senses", []) or [],
        "suggested_action": row.get("suggested_action", "manual_review"),
        "missing_meaning_evidence": row.get("missing_meaning_evidence", []) or [],
        "entry_sha256": entry_sha256(context),
        "current_french_senses": context.get("current_french_senses", []),
        "current_english_senses": context.get("current_english_senses", []),
    }


def parse_batch_output(
    raw_output_path: Path,
    context_by_seq: dict[int, dict[str, Any]],
    *,
    require_evidence: bool = False,
) -> tuple[list[dict[str, Any]], Counter]:
    rows: list[dict[str, Any]] = []
    usage = Counter()
    errors = 0
    with raw_output_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            envelope = json.loads(line)
            response = envelope.get("response") or {}
            if response.get("status_code") != 200:
                errors += 1
                continue
            body = response.get("body") or {}
            body_usage = body.get("usage") or {}
            for key in ["input_tokens", "output_tokens", "total_tokens"]:
                if isinstance(body_usage.get(key), int):
                    usage[key] += body_usage[key]

            output_text = extract_output_text(body)
            if not output_text:
                raise ValueError(f"No output text in batch output line {line_number}")
            payload = json.loads(output_text)
            for result in payload.get("results", []):
                if require_evidence or "missing_meaning_evidence" in result:
                    validate_audit_evidence(result)
                seq = int(result["ent_seq"])
                context = context_by_seq.get(seq)
                if context is None:
                    raise ValueError(f"Result ent_seq {seq} not found in context")
                rows.append(normalize_result(result, context))

    if errors:
        usage["non_200_responses"] = errors
    rows.sort(key=lambda r: r["ent_seq"])
    return rows, usage


def direct_request_sequences(request: dict[str, Any]) -> list[int]:
    if request.get("method") != "POST" or request.get("url") != "/v1/responses":
        raise ValueError("Prepared requests must target POST /v1/responses.")
    prompts = [
        content["text"]
        for item in request["body"]["input"]
        if item.get("role") == "user"
        for content in item.get("content", [])
        if content.get("type") == "input_text"
    ]
    if len(prompts) != 1:
        raise ValueError("Expected one prepared entry prompt per request.")
    entries = json.loads(prompts[0])["entries"]
    sequences = [entry["ent_seq"] for entry in entries]
    if not sequences or any(type(seq) is not int for seq in sequences):
        raise ValueError("Prepared entries must have integer sequence numbers.")
    if len(set(sequences)) != len(sequences):
        raise ValueError("Prepared request contains duplicate entries.")
    return sequences


def validate_direct_response(body: dict[str, Any], expected: list[int], result_schema: dict[str, Any] | None = None) -> None:
    if body.get("status") != "completed" or body.get("error"):
        raise ValueError(f"Response did not complete: {body.get('status')}; {body.get('incomplete_details')}")
    text = extract_output_text(body)
    if not text:
        raise ValueError("Response has no audit output (it may contain a refusal).")
    results = json.loads(text).get("results")
    if not isinstance(results, list):
        raise ValueError("Response has no results list.")
    schema = result_schema if result_schema is not None else RESPONSE_SCHEMA["properties"]["results"]["items"]
    required = set(schema["required"])
    if any(not isinstance(row, dict) or not required.issubset(row) for row in results):
        raise ValueError("Response contains incomplete result objects.")
    sequences = [row["ent_seq"] for row in results]
    if any(type(seq) is not int for seq in sequences):
        raise ValueError("Response sequence numbers must be integers.")
    if len(sequences) != len(expected) or set(sequences) != set(expected):
        raise ValueError("Response contains missing, duplicate, or unexpected entries.")
    if "missing_meaning_evidence" in required:
        for row in results:
            validate_audit_evidence(row)


def validate_audit_evidence(row: dict[str, Any]) -> None:
    evidence = row.get("missing_meaning_evidence")
    issues = row.get("issue_types")
    if not isinstance(issues, list) or not isinstance(evidence, list):
        raise ValueError("Audit issues and missing-meaning evidence must be lists.")
    if bool(evidence) != ("missing_core_meaning" in issues):
        raise ValueError("A missing-core-meaning issue requires evidence, and other issues must not carry it.")
    for item in evidence:
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in ("japanese_meaning", "why_core", "closest_existing_french", "coverage_gap")
        ):
            raise ValueError("Missing-meaning evidence must explain the concept, importance, existing French, and coverage gap.")
    if row.get("status") == "ok" and (
        issues or row.get("suggested_french_senses") or row.get("suggested_action") != "keep_existing"
    ):
        raise ValueError("An ok entry must not contain issues or suggested changes.")


def run_direct(args: argparse.Namespace, output_dir: Path) -> int:
    manifest = load_manifest(output_dir)
    if manifest.get("batch_id"):
        raise SystemExit("This directory already has a Batch job. Prepare a new directory for direct execution.")
    requests_path = Path(manifest["requests_path"])
    context_path = Path(manifest["context_path"])
    raw_path = output_dir / "direct_output.jsonl"
    fingerprints = {
        "direct_requests_sha256": sha256_file(requests_path),
        "direct_context_sha256": sha256_file(context_path),
    }
    if args.resume:
        if manifest.get("execution_mode") != "direct" or any(
            manifest.get(key) != value for key, value in fingerprints.items()
        ):
            raise SystemExit("Cannot resume: this is not the same prepared direct run.")
    elif manifest.get("execution_mode") or raw_path.exists() or any(
        (output_dir / name).exists() for name in (
            "analysis_results.jsonl", "suspect_entries.csv", "summary.md", "likely_error_missing_core_meaning.txt"
        )
    ):
        raise SystemExit("Run output already exists. Use run --resume or prepare a new directory.")

    requests = [json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    context = load_context(context_path)
    expected_by_request: dict[str, list[int]] = {}
    schema_by_request: dict[str, dict[str, Any]] = {}
    for request in requests:
        custom_id = request.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id or custom_id in expected_by_request:
            raise ValueError("Prepared requests need unique custom IDs.")
        if request["body"].get("model") != manifest["model"]:
            raise ValueError("Prepared request model differs from the manifest.")
        expected_by_request[custom_id] = direct_request_sequences(request)
        schema_by_request[custom_id] = request["body"]["text"]["format"]["schema"]["properties"]["results"]["items"]
    all_sequences = [seq for sequences in expected_by_request.values() for seq in sequences]
    if (len(requests) != manifest["request_count"] or not requests
            or len(all_sequences) != manifest["entry_count"]
            or len(context) != manifest["entry_count"]
            or len(set(all_sequences)) != len(all_sequences) or set(all_sequences) != set(context)):
        raise ValueError("Prepared requests, context, and manifest disagree about the audit entries.")

    completed: set[str] = set()
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            custom_id = record.get("custom_id")
            if custom_id not in expected_by_request or custom_id in completed:
                raise ValueError("Direct checkpoint has an unknown or duplicate request.")
            response = record["response"]
            if response.get("status_code") != 200:
                raise ValueError("Direct checkpoint contains an unsuccessful request.")
            validate_direct_response(response["body"], expected_by_request[custom_id], schema_by_request[custom_id])
            completed.add(custom_id)
    if len(completed) < manifest.get("direct_completed_requests", 0):
        raise ValueError("Saved direct responses are missing. Restore the checkpoint before resuming.")

    pending = [request for request in requests if request["custom_id"] not in completed]
    client = get_openai_client(args) if pending else None
    manifest.update(fingerprints)
    manifest.update(execution_mode="direct", direct_status="running")
    manifest.setdefault("direct_started_at_utc", utc_now())
    manifest.pop("direct_error", None)
    save_manifest(output_dir, manifest)

    def save_reports(status: str) -> None:
        rows, usage = parse_batch_output(raw_path, context) if raw_path.exists() else ([], Counter())
        write_artifacts(rows, output_dir, include_needs_review=args.include_needs_review, usage=usage, manifest=manifest)
        manifest.update(direct_status=status, direct_completed_requests=len(completed),
                        collected_result_count=len(rows), usage=dict(usage))
        if status == "completed":
            manifest["collected_at_utc"] = utc_now()
        save_manifest(output_dir, manifest)

    current_id: str | None = None
    body: dict[str, Any] | None = None
    try:
        for request in pending:
            current_id = request["custom_id"]
            body = None
            expected = expected_by_request[current_id]
            print(f"Request {len(completed) + 1}/{len(requests)}: auditing {len(expected)} entries...", flush=True)
            if client is None:
                raise RuntimeError("No API client is available for pending requests.")
            response = client.responses.create(**request["body"])
            body = response.model_dump() if hasattr(response, "model_dump") else dict(response)
            validate_direct_response(body, expected, schema_by_request[current_id])
            record = {"custom_id": current_id, "response": {"status_code": 200, "body": body}}
            with raw_path.open("a", encoding="utf-8") as checkpoint:
                checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
                checkpoint.flush()
                os.fsync(checkpoint.fileno())
            completed.add(current_id)
            manifest["direct_completed_requests"] = len(completed)
            save_manifest(output_dir, manifest)
            print(f"Saved {len(completed)}/{len(requests)} requests.", flush=True)
        save_reports("completed")
    except (Exception, KeyboardInterrupt) as exc:
        message = str(exc) or type(exc).__name__
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            message = message.replace(key, "[REDACTED]")
        manifest["direct_error"] = message
        failure = {"custom_id": current_id, "error": message, "response": body}
        (output_dir / "direct_failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        save_reports("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        if isinstance(exc, KeyboardInterrupt):
            raise SystemExit(130) from exc
        raise SystemExit(f"Direct run failed: {message}. Saved results are intact; use run --resume.") from exc
    print(f"Direct audit completed: {manifest['collected_result_count']} entries.")
    return 0


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    load_manifest(output_dir)
    with output_directory_lock(output_dir):
        return run_direct(args, output_dir)


def should_write_suspect(row: dict[str, Any], include_needs_review: bool) -> bool:
    if row.get("status") == "likely_error":
        return True
    return include_needs_review and row.get("status") == "needs_review"


def write_artifacts(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    include_needs_review: bool,
    usage: Counter | None = None,
    manifest: dict[str, Any] | None = None,
) -> None:
    results_path = output_dir / "analysis_results.jsonl"
    csv_path = output_dir / "suspect_entries.csv"
    summary_path = output_dir / "summary.md"
    missing_core_path = output_dir / "likely_error_missing_core_meaning.txt"
    provenance = report_provenance(manifest or {})

    with results_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_fields = [
        "ent_seq",
        "headword",
        "status",
        "confidence",
        "issue_types",
        "problem_summary",
        "core_meanings_french",
        "suggested_french_senses",
        "suggested_action",
        "current_french_senses",
        "current_english_senses",
        "missing_meaning_evidence",
        "entry_sha256",
        *provenance,
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            if not should_write_suspect(row, include_needs_review=include_needs_review):
                continue
            writer.writerow(
                {
                    "ent_seq": row["ent_seq"],
                    "headword": row["headword"],
                    "status": row["status"],
                    "confidence": row["confidence"],
                    "issue_types": ";".join(row.get("issue_types", [])),
                    "problem_summary": row.get("problem_summary", ""),
                    "core_meanings_french": " | ".join(row.get("core_meanings_french", [])),
                    "suggested_french_senses": " | ".join(row.get("suggested_french_senses", [])),
                    "suggested_action": row.get("suggested_action", ""),
                    "current_french_senses": " | ".join(row.get("current_french_senses", [])),
                    "current_english_senses": " | ".join(row.get("current_english_senses", [])),
                    "missing_meaning_evidence": json.dumps(row.get("missing_meaning_evidence", []), ensure_ascii=False),
                    "entry_sha256": row.get("entry_sha256", ""),
                    **provenance,
                }
            )

    removed = sorted(
        {
            int(row["ent_seq"])
            for row in rows
            if row.get("status") == "likely_error"
            and "missing_core_meaning" in (row.get("issue_types") or [])
        }
    )
    missing_core_path.write_text(",".join(str(v) for v in removed) + "\n", encoding="utf-8")

    status_counts = Counter(row.get("status", "") for row in rows)
    issue_counts: Counter[str] = Counter()
    issue_by_status: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        status = row.get("status", "")
        for issue in row.get("issue_types", []) or []:
            issue_counts[issue] += 1
            issue_by_status[status][issue] += 1

    lines = [
        "# JMdict French Audit Summary",
        "",
        f"- Generated (UTC): {utc_now()}",
        f"- Entries processed: {len(rows)}",
        f"- Audit prepared (UTC): {provenance['audit_prepared_at_utc'] or 'unknown'}",
        f"- Dictionary snapshot date: {provenance['dictionary_snapshot_date'] or 'unknown'}",
        f"- Snapshot date source: {provenance['dictionary_snapshot_date_source'] or 'unknown'}",
        f"- Dictionary XML SHA-256: {provenance['dictionary_xml_sha256'] or 'unknown'}",
        f"- Model: {provenance['audit_model'] or 'unknown'}",
        f"- Reasoning effort: {provenance['audit_reasoning_effort'] or 'unknown'}",
        f"- Audit prompt version: {provenance['audit_prompt_version']}",
        f"- Audit prompt SHA-256: {provenance['audit_prompt_sha256'] or 'unknown'}",
        "",
        "## Status Counts",
    ]
    for status in ["likely_error", "needs_review", "ok"]:
        count = status_counts.get(status, 0)
        pct = (count / len(rows) * 100) if rows else 0
        lines.append(f"- {status}: {count} ({pct:.1f}%)")

    lines.extend(["", "## Issue Counts"])
    for issue, count in issue_counts.most_common():
        lines.append(f"- {issue}: {count}")
    if not issue_counts:
        lines.append("- (none)")

    lines.extend(["", "## Issue Counts By Status"])
    for status in ["likely_error", "needs_review", "ok"]:
        parts = [f"{issue}={count}" for issue, count in issue_by_status[status].most_common()]
        lines.append(f"- {status}: {', '.join(parts) if parts else '(none)'}")

    lines.extend(["", "## Removal Candidate Set"])
    lines.append(f"- likely_error + missing_core_meaning: {len(removed)}")
    lines.append(f"- Sequence list: {missing_core_path.name}")

    if usage:
        lines.extend(["", "## Token Usage"])
        for key in ["input_tokens", "output_tokens", "total_tokens", "non_200_responses"]:
            if usage.get(key):
                lines.append(f"- {key}: {usage[key]}")

    lines.extend(["", "## Likely Error Examples"])
    examples = [row for row in rows if row.get("status") == "likely_error"]
    examples.sort(key=lambda r: (-(r.get("confidence") or 0), r["ent_seq"]))
    for row in examples[:20]:
        lines.append(
            f"- {row['ent_seq']} {row['headword']} conf={row.get('confidence')}: "
            f"{';'.join(row.get('issue_types', []))} | {row.get('problem_summary', '')}"
        )
    if not examples:
        lines.append("- (none)")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {results_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {missing_core_path}")


def collect(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    load_manifest(output_dir)
    with output_directory_lock(output_dir):
        return collect_batch(args, output_dir)


def collect_batch(args: argparse.Namespace, output_dir: Path) -> int:
    manifest = load_manifest(output_dir)
    require_batch_directory(output_dir, manifest)
    client = get_openai_client(args)
    batch_id = batch_id_from_args(args, manifest)
    batch = client.batches.retrieve(batch_id)
    batch_data = batch.model_dump() if hasattr(batch, "model_dump") else dict(batch)
    batch_status = batch_data.get("status")
    if batch_status != "completed" and not args.allow_partial:
        raise SystemExit(f"Batch status is {batch_status}; use status or --allow-partial before collecting.")

    output_file_id = batch_data.get("output_file_id")
    if not output_file_id:
        raise SystemExit("Batch has no output_file_id.")

    raw_output_path = output_dir / "batch_output.jsonl"
    raw_output_path.write_bytes(read_file_content(client, output_file_id))
    print(f"Wrote {raw_output_path}")

    error_file_id = batch_data.get("error_file_id")
    if error_file_id:
        error_path = output_dir / "batch_errors.jsonl"
        error_path.write_bytes(read_file_content(client, error_file_id))
        print(f"Wrote {error_path}")

    context_by_seq = load_context(Path(manifest["context_path"]))
    rows, usage = parse_batch_output(
        raw_output_path, context_by_seq, require_evidence=manifest.get("audit_prompt_version") == "2",
    )
    write_artifacts(rows, output_dir, include_needs_review=args.include_needs_review, usage=usage, manifest=manifest)

    manifest["collected_at_utc"] = utc_now()
    manifest["output_file_id"] = output_file_id
    manifest["error_file_id"] = error_file_id
    manifest["usage"] = dict(usage)
    manifest["collected_result_count"] = len(rows)
    save_manifest(output_dir, manifest)
    return 0


def summarize(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    if not (output_dir / "analysis_results.jsonl").exists():
        raise SystemExit(f"Results file not found: {output_dir / 'analysis_results.jsonl'}")
    with output_directory_lock(output_dir):
        return summarize_results(args, output_dir)


def summarize_results(args: argparse.Namespace, output_dir: Path) -> int:
    results_path = output_dir / "analysis_results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    usage = Counter()
    manifest: dict[str, Any] = {}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        usage.update(manifest.get("usage") or {})
    write_artifacts(rows, output_dir, include_needs_review=args.include_needs_review, usage=usage, manifest=manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download-jmdict", help="Download and decompress official JMdict.gz")
    dl.add_argument("--url", default=DEFAULT_JMDICT_URL)
    dl.add_argument("--output-dir", default="data")
    dl.add_argument("--gz-name", default="JMdict.gz")
    dl.add_argument("--xml-name", default="JMdict.xml")
    dl.add_argument("--timeout-seconds", type=int, default=120)
    dl.set_defaults(func=download_jmdict)

    prep = sub.add_parser("prepare", help="Prepare API requests from JMdict XML")
    prep.add_argument("--xml-path", default="data/JMdict.xml")
    prep.add_argument("--snapshot-date", type=parse_snapshot_date, help="Dictionary snapshot date (YYYY-MM-DD), if not available from matching download metadata")
    prep.add_argument("--output-dir", default="analysis_output_latest")
    prep.add_argument("--model", default="gpt-6-luna")
    prep.add_argument("--entries-per-request", type=int, default=20)
    prep.add_argument("--max-senses-per-entry", type=int, default=40)
    prep.add_argument("--max-glosses-per-sense", type=int, default=12)
    prep.add_argument("--max-output-tokens", type=int, default=20000)
    prep.add_argument("--reasoning-effort", default="")
    prep.add_argument("--max-entries", type=int)
    prep.add_argument("--ent-seq", help="Comma-separated sequence numbers for a targeted run")
    prep.set_defaults(func=prepare)

    direct = sub.add_parser("run", help="Audit prepared entries directly through the Responses API")
    direct.add_argument("--output-dir", default="analysis_output_latest")
    direct.add_argument("--env-file", default=".env")
    direct.add_argument("--api-key-file", default="~/openai-translation-key.txt")
    direct.add_argument("--resume", action="store_true", help="Continue the same run, skipping saved successful requests")
    direct.add_argument("--include-needs-review", action="store_true")
    direct.set_defaults(func=run)

    for name, help_text in [
        ("submit", "Upload request file and create an OpenAI Batch job"),
        ("status", "Print OpenAI Batch job status"),
        ("collect", "Download and parse completed OpenAI Batch output"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--output-dir", default="analysis_output_latest")
        p.add_argument("--env-file", default=".env")
        p.add_argument("--api-key-file", default="~/openai-translation-key.txt")
        p.add_argument("--batch-id")
        if name == "submit":
            p.add_argument("--completion-window", default="24h")
            p.set_defaults(func=submit)
        elif name == "status":
            p.set_defaults(func=status)
        else:
            p.add_argument("--allow-partial", action="store_true")
            p.add_argument("--include-needs-review", action="store_true")
            p.set_defaults(func=collect)

    summ = sub.add_parser("summarize", help="Regenerate CSV/summary from analysis_results.jsonl")
    summ.add_argument("--output-dir", default="analysis_output_latest")
    summ.add_argument("--include-needs-review", action="store_true")
    summ.set_defaults(func=summarize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
