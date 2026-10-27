import contextlib
import csv
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import jmdict_french_audit as audit


def result_for(sequence):
    status = {1: "ok", 2: "likely_error", 3: "needs_review"}[sequence]
    return {
        "ent_seq": sequence,
        "status": status,
        "confidence": 0.9,
        "issue_types": [] if status == "ok" else ["missing_core_meaning"],
        "problem_summary": "" if status == "ok" else "Review this translation.",
        "core_meanings_french": [],
        "suggested_french_senses": [],
        "suggested_action": "keep_existing" if status == "ok" else "manual_review",
    }


def response_for(body):
    sequences = audit.direct_request_sequences({"method": "POST", "url": "/v1/responses", "body": body})
    return {
        "status": "completed",
        "model": body["model"],
        "output": [{"type": "message", "content": [{
            "type": "output_text", "text": json.dumps({"results": [result_for(seq) for seq in sequences]})
        }]}],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


class FakeResponses:
    def __init__(self, handler=response_for):
        self.handler = handler
        self.calls = []

    def create(self, **body):
        self.calls.append(body)
        response = self.handler(body)
        return SimpleNamespace(model_dump=lambda: response)


def batch_client(calls, upload=None):
    return SimpleNamespace(
        files=SimpleNamespace(create=upload or (lambda **kwargs: (
            calls.append("files.create") or SimpleNamespace(id="file_created")))),
        batches=SimpleNamespace(create=lambda **kwargs: (
            calls.append("batches.create") or SimpleNamespace(id="batch_created", status="validating"))),
    )


class DirectRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "audit"
        self.xml = self.root / "JMdict.xml"
        self.xml.write_text(
            '<JMdict>' + ''.join(
                f'<entry><ent_seq>{seq}</ent_seq><r_ele><reb>word{seq}</reb></r_ele>'
                f'<sense><gloss>meaning {seq}</gloss><gloss xml:lang="fre">sens {seq}</gloss></sense></entry>'
                for seq in (1, 2, 3)
            ) + '</JMdict>', encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            audit.main(["prepare", "--xml-path", str(self.xml), "--output-dir", str(self.output),
                        "--entries-per-request", "2"])

    def manifest(self):
        return json.loads((self.output / "manifest.json").read_text())

    def run_direct(self, responses=None, *extra):
        client = SimpleNamespace(responses=responses or FakeResponses())
        with patch.object(audit, "get_openai_client", return_value=client) as factory:
            with contextlib.redirect_stdout(io.StringIO()):
                code = audit.main(["run", "--output-dir", str(self.output), *extra])
        return code, client.responses, factory

    def test_direct_cli_writes_complete_reports_and_usage_without_batch_client(self):
        code, responses, _ = self.run_direct()
        self.assertEqual(code, 0)
        self.assertEqual(len(responses.calls), 2)
        self.assertEqual(responses.calls[0]["model"], "gpt-6-luna")
        rows = [json.loads(line) for line in (self.output / "analysis_results.jsonl").read_text().splitlines()]
        self.assertEqual([row["ent_seq"] for row in rows], [1, 2, 3])
        with (self.output / "suspect_entries.csv").open() as f:
            self.assertEqual([row["ent_seq"] for row in csv.DictReader(f)], ["2"])
        self.assertEqual((self.output / "likely_error_missing_core_meaning.txt").read_text().strip(), "2")
        summary = (self.output / "summary.md").read_text()
        self.assertIn("Entries processed: 3", summary)
        self.assertIn("likely_error: 1", summary)
        manifest = self.manifest()
        self.assertEqual(manifest["direct_status"], "completed")
        self.assertEqual(manifest["collected_result_count"], 3)
        self.assertEqual(manifest["usage"], {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30})
        self.assertNotIn("batch_id", manifest)
        self.assertFalse((self.output / ".audit_run.lock").exists())

    def test_resume_skips_paid_successes_after_request_failure(self):
        def fail_second(body):
            if len(audit.direct_request_sequences({"method": "POST", "url": "/v1/responses", "body": body})) == 1:
                raise RuntimeError("temporary failure")
            return response_for(body)

        first = FakeResponses(fail_second)
        with self.assertRaises(SystemExit):
            self.run_direct(first)
        self.assertEqual(self.manifest()["direct_status"], "failed")
        self.assertEqual(self.manifest()["collected_result_count"], 2)
        checkpoint = (self.output / "direct_output.jsonl").read_text()
        with patch.object(audit, "get_openai_client") as factory:
            with self.assertRaises(SystemExit):
                audit.main(["run", "--output-dir", str(self.output)])
            factory.assert_not_called()
        code, resumed, _ = self.run_direct(None, "--resume", "--include-needs-review")
        self.assertEqual(code, 0)
        self.assertEqual(len(resumed.calls), 1)
        self.assertTrue((self.output / "direct_output.jsonl").read_text().startswith(checkpoint))
        self.assertEqual(self.manifest()["collected_result_count"], 3)
        self.assertEqual(self.manifest()["usage"]["total_tokens"], 30)
        with (self.output / "suspect_entries.csv").open() as f:
            self.assertEqual([row["ent_seq"] for row in csv.DictReader(f)], ["2", "3"])

    def test_finished_resume_regenerates_reports_without_api_access(self):
        self.run_direct()
        (self.output / "summary.md").unlink()
        _, _, factory = self.run_direct(None, "--resume")
        factory.assert_not_called()
        self.assertTrue((self.output / "summary.md").exists())

    def test_invalid_or_incomplete_response_is_not_checkpointed(self):
        for kind in ("missing", "duplicate", "unexpected", "incomplete", "refusal"):
            with self.subTest(kind=kind):
                def bad_response(body):
                    response = response_for(body)
                    if kind == "incomplete":
                        response["status"] = "incomplete"
                    elif kind == "refusal":
                        response["output"] = [{"content": [{"type": "refusal", "refusal": "No."}]}]
                    else:
                        sequences = {"missing": [1], "duplicate": [1, 1], "unexpected": [1, 3]}[kind]
                        response["output"][0]["content"][0]["text"] = json.dumps(
                            {"results": [result_for(seq) for seq in sequences]})
                    return response

                if (self.output / "manifest.json").exists() and self.manifest().get("execution_mode"):
                    extra = ("--resume",)
                else:
                    extra = ()
                with self.assertRaises(SystemExit):
                    self.run_direct(FakeResponses(bad_response), *extra)
                self.assertFalse((self.output / "direct_output.jsonl").exists())
                self.assertEqual(self.manifest()["collected_result_count"], 0)
                failure = json.loads((self.output / "direct_failure.json").read_text())
                self.assertIsNotNone(failure["response"])
                self.assertFalse((self.output / ".audit_run.lock").exists())
        self.run_direct(None, "--resume")
        self.assertEqual(self.manifest()["direct_status"], "completed")

    def test_changed_inputs_or_missing_checkpoint_refuse_resume_before_api_access(self):
        self.run_direct()
        raw = self.output / "direct_output.jsonl"
        original = raw.read_text()
        raw.unlink()
        with patch.object(audit, "get_openai_client") as factory:
            with self.assertRaises(ValueError):
                audit.main(["run", "--output-dir", str(self.output), "--resume"])
            factory.assert_not_called()
        raw.write_text(original)
        requests = Path(self.manifest()["requests_path"])
        requests.write_text(requests.read_text() + "\n")
        with patch.object(audit, "get_openai_client") as factory:
            with self.assertRaises(SystemExit):
                audit.main(["run", "--output-dir", str(self.output), "--resume"])
            factory.assert_not_called()

    def test_submitted_batch_cannot_be_run_directly(self):
        manifest = self.manifest()
        manifest["batch_id"] = "batch_existing"
        audit.save_manifest(self.output, manifest)
        with patch.object(audit, "get_openai_client") as factory:
            with self.assertRaises(SystemExit):
                audit.main(["run", "--output-dir", str(self.output)])
            factory.assert_not_called()

    def test_direct_run_cannot_be_submitted_as_batch(self):
        self.run_direct()
        with patch.object(audit, "get_openai_client") as factory:
            with self.assertRaises(SystemExit):
                audit.main(["submit", "--output-dir", str(self.output)])
            factory.assert_not_called()

    def test_existing_lock_prevents_concurrent_paid_execution(self):
        (self.output / ".audit_run.lock").write_text("123\n")
        with patch.object(audit, "get_openai_client") as factory:
            with self.assertRaises(SystemExit):
                audit.main(["run", "--output-dir", str(self.output)])
            factory.assert_not_called()
        self.assertTrue((self.output / ".audit_run.lock").exists())

    def test_existing_report_is_not_overwritten_by_a_fresh_direct_run(self):
        report = self.output / "suspect_entries.csv"
        report.write_text("keep this report\n")
        with patch.object(audit, "get_openai_client") as factory:
            with self.assertRaises(SystemExit):
                audit.main(["run", "--output-dir", str(self.output)])
            factory.assert_not_called()
        self.assertEqual(report.read_text(), "keep this report\n")

    def test_reprepared_manifest_does_not_allow_direct_checkpoints_to_be_submitted(self):
        self.run_direct()
        manifest = self.manifest()
        manifest.pop("execution_mode")
        audit.save_manifest(self.output, manifest)
        with patch.object(audit, "get_openai_client") as factory:
            with self.assertRaises(SystemExit):
                audit.main(["submit", "--output-dir", str(self.output)])
            factory.assert_not_called()

    def test_batch_submission_is_excluded_during_direct_client_creation(self):
        calls = []
        direct = SimpleNamespace(responses=FakeResponses())
        batch = batch_client(calls)

        def factory(args):
            if args.command == "run":
                with self.assertRaises(SystemExit):
                    audit.main(["submit", "--output-dir", str(self.output)])
                return direct
            return batch

        with patch.object(audit, "get_openai_client", side_effect=factory):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(["run", "--output-dir", str(self.output)]), 0)
        self.assertEqual(calls, [])
        self.assertEqual(len(direct.responses.calls), 2)
        self.assertEqual(self.manifest()["execution_mode"], "direct")
        self.assertNotIn("batch_id", self.manifest())

    def test_direct_execution_is_excluded_during_batch_upload(self):
        calls = []
        direct = SimpleNamespace(responses=FakeResponses())

        def upload(**kwargs):
            calls.append("files.create")
            with self.assertRaises(SystemExit):
                audit.main(["run", "--output-dir", str(self.output)])
            return SimpleNamespace(id="file_created")

        batch = batch_client(calls, upload)
        with patch.object(audit, "get_openai_client", side_effect=lambda args: direct if args.command == "run" else batch):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(["submit", "--output-dir", str(self.output)]), 0)
        self.assertEqual(calls, ["files.create", "batches.create"])
        self.assertEqual(direct.responses.calls, [])
        self.assertEqual(self.manifest()["batch_id"], "batch_created")
        self.assertFalse((self.output / "direct_output.jsonl").exists())

    def test_batch_collection_cannot_replace_direct_reports(self):
        self.run_direct()
        for remove_mode in (False, True):
            if remove_mode:
                manifest = self.manifest()
                manifest.pop("execution_mode")
                audit.save_manifest(self.output, manifest)
            before = {p.name: p.read_bytes() for p in self.output.iterdir()}
            with patch.object(audit, "get_openai_client") as factory:
                with self.assertRaises(SystemExit):
                    audit.main(["collect", "--output-dir", str(self.output), "--batch-id", "batch_external"])
                factory.assert_not_called()
            self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_all_directory_writers_share_the_operation_lock(self):
        (self.output / "analysis_results.jsonl").write_text("")
        (self.output / ".audit_run.lock").write_text("123\n")
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        commands = [
            ["prepare", "--xml-path", str(self.xml)],
            ["submit"], ["collect", "--batch-id", "batch_external"], ["summarize"],
        ]
        for command in commands:
            with self.subTest(command=command[0]):
                with patch.object(audit, "get_openai_client") as factory:
                    with self.assertRaises(SystemExit):
                        audit.main([*command, "--output-dir", str(self.output)])
                    factory.assert_not_called()
                self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_legacy_batch_collection_still_writes_reports(self):
        manifest = self.manifest()
        manifest["batch_id"] = "batch_existing"
        audit.save_manifest(self.output, manifest)
        requests = [json.loads(line) for line in Path(manifest["requests_path"]).read_text().splitlines()]
        raw = '\n'.join(json.dumps({"custom_id": r["custom_id"], "response": {
            "status_code": 200, "body": response_for(r["body"])
        }}) for r in requests)
        client = SimpleNamespace(
            batches=SimpleNamespace(retrieve=lambda batch_id: {"status": "completed", "output_file_id": "file_result"}),
            files=SimpleNamespace(content=lambda file_id: raw.encode()),
        )
        with patch.object(audit, "get_openai_client", return_value=client):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(["collect", "--output-dir", str(self.output)]), 0)
        self.assertEqual(self.manifest()["collected_result_count"], 3)
        self.assertFalse((self.output / ".audit_run.lock").exists())

    def test_failure_message_redacts_the_configured_key(self):
        secret = "test-private-value"

        def fail(body):
            raise RuntimeError("Rejected credential: " + secret)

        with patch.dict(os.environ, {"OPENAI_API_KEY": secret}):
            with self.assertRaises(SystemExit) as error:
                self.run_direct(FakeResponses(fail))
        self.assertNotIn(secret, str(error.exception))
        self.assertNotIn(secret, (self.output / "direct_failure.json").read_text())
        self.assertNotIn(secret, (self.output / "manifest.json").read_text())


if __name__ == "__main__":
    unittest.main()
