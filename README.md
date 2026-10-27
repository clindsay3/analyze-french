# JMdict French Audit

Audit French translations in [JMdict](https://www.edrdg.org/wiki/index.php/JMdict-EDICT_Dictionary_Project) against the Japanese forms and English meanings, using the regular [OpenAI Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses). The [Batch API](https://developers.openai.com/api/docs/guides/batch) is also available as an optional workflow.

The Python script selects entries that already have French glosses and flags missing core meanings, misleading translations, unnatural French, and part-of-speech mismatches. It produces a CSV for review, detailed JSONL results, and a Markdown summary. It does not modify the dictionary.

Results are model judgments, not verified dictionary corrections. Review flagged entries before changing or removing translations. English glosses are used as the reference, so errors or differences in their coverage can affect the findings. Entries without any French glosses are outside this audit's scope.

## Requirements and setup

- Python 3.10 or newer, with `venv` and `pip`.
- Internet access to download JMdict and use the API.
- Your own OpenAI API key, API billing, and access to the selected model. API usage has a separate cost from a ChatGPT subscription; consult [OpenAI pricing](https://developers.openai.com/api/docs/pricing) before a full run.

Clone the repository and install the dependency:

```bash
git clone https://github.com/clindsay3/analyze-french.git
cd analyze-french
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Commands below use macOS/Linux paths. On Windows, create the environment with `py -3 -m venv .venv` and replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

For API commands, set `OPENAI_API_KEY` in your environment, or create a local `.env` file in the repository root with:

```dotenv
OPENAI_API_KEY=your-api-key
```

Replace the placeholder with your key. `.env` and `.env.*` are ignored by Git. The script checks the environment first, then `.env`, then the optional legacy key file `~/openai-translation-key.txt`. API commands also accept `--env-file PATH` and `--api-key-file PATH`.

Run commands from the repository root. Keep each run's output directory intact: its manifest, prepared requests, and context files are needed to run, resume, and interpret the audit.

## Start with a small audit

### 1. Download JMdict

```bash
.venv/bin/python jmdict_french_audit.py download-jmdict --output-dir data
```

This downloads the official multilingual `JMdict.gz`, decompresses it to `data/JMdict.xml`, and records download metadata and SHA-256 checksums. You can instead supply an existing multilingual JMdict XML file to `prepare`; an English-only file has no French entries to audit.

### 2. Prepare three entries locally

```bash
.venv/bin/python jmdict_french_audit.py prepare \
  --xml-path data/JMdict.xml \
  --output-dir analysis_output_smoke \
  --max-entries 3
```

`prepare` creates the request file, entry context, and manifest locally. It does not contact OpenAI or incur API charges. The default model is `gpt-6-luna`; see [Model selection](#model-selection) for alternatives.

To select particular JMdict sequence numbers instead, replace `--max-entries 3` with `--ent-seq 1372800,1424150,1597040`.

### 3. Run the audit directly

```bash
.venv/bin/python jmdict_french_audit.py run --output-dir analysis_output_smoke
```

This sends the prepared Japanese, English, and French entry data to OpenAI through ordinary Responses API requests. It processes request groups sequentially, prints progress, and writes the reports before exiting. It does not upload a Batch input file or create a Batch job, so no separate `status` or `collect` step is needed.

API usage is paid at regular API rates. The default grouping remains 20 entries per request; selecting direct execution does not change the audit prompt, response schema, model, or grouping chosen during preparation.

### 4. Inspect the results

Open `analysis_output_smoke/summary.md` and `analysis_output_smoke/suspect_entries.csv`. An empty suspect CSV, apart from its header, is possible if none of the sample entries is classified as a likely error.

The direct runner checks that every request returns exactly one result for each expected entry. Missing, duplicate, unexpected, refused, or incomplete results stop the run with a diagnostic instead of being treated as a complete audit. The manifest's `direct_status` is `completed` only after every prepared request has succeeded and the reports have been written.

## Run the full audit

After checking the sample, prepare all entries with French glosses in a new output directory:

```bash
.venv/bin/python jmdict_french_audit.py prepare \
  --xml-path data/JMdict.xml \
  --output-dir analysis_output_full
.venv/bin/python jmdict_french_audit.py run --output-dir analysis_output_full
```

Use a different `analysis_output_*` directory for each new run. Re-running `prepare` in an existing directory overwrites its manifest and prepared files; it is not how you resume a direct audit. Full runs can process thousands of entries; cost depends on the model, dictionary snapshot, and generated output.

## Resume a direct run

Each successful request is saved to `direct_output.jsonl` before the next request begins. Handled errors and Ctrl-C produce partial reports from the saved results. After a crash or forced termination, saved requests remain available for resuming. Continue that same run with:

```bash
.venv/bin/python jmdict_french_audit.py run --output-dir analysis_output_full --resume
```

`--resume` validates the saved responses and skips successfully checkpointed requests. It also checks that the prepared request and context files have not changed. A resume of an already completed run regenerates its reports without API calls. Changing the model or preparation parameters requires a new run directory. An interrupted in-flight request may already have incurred a charge even if its response was not saved.

Report token totals cover successfully saved responses. Failed responses and interrupted in-flight requests may incur additional charges; use your OpenAI usage dashboard for billing totals.

Run only one process per output directory. An exclusive `.direct_run.lock` prevents concurrent direct execution. It is removed on normal exit and handled errors; after a crash or forced termination, remove that lock file only after confirming the previous process has stopped. A malformed or missing checkpoint must be restored or inspected before resuming; it is not silently discarded.

## Model selection

The default is **`gpt-6-luna`**, OpenAI's current model for focused, high-volume workloads. These API model IDs support Batch, the Responses API, and structured outputs according to their documentation (checked October 27, 2026):

| Model | Intended role |
| --- | --- |
| [`gpt-6-luna`](https://developers.openai.com/api/docs/models/gpt-6-luna) | Default for an economical full audit. |
| [`gpt-6.1-terra`](https://developers.openai.com/api/docs/models/gpt-6.1-terra) | Alternative for drafting and revising text. |
| [`gpt-6.1-sol`](https://developers.openai.com/api/docs/models/gpt-6.1-sol) | Alternative for more complex reasoning and difficult entries. |

These roles follow the model documentation; they are not a measured ranking of French dictionary accuracy. Check model access and current pricing for your API project, and review a small audit before selecting a model for a full run. `gpt-5-mini` remains available in the [API catalog](https://developers.openai.com/api/docs/models/gpt-5-mini), but is no longer this script's default. API availability can differ from ChatGPT and Codex model menus.

Select the model when preparing a new run. For example, to audit three entries with Sol:

```bash
.venv/bin/python jmdict_french_audit.py prepare \
  --xml-path data/JMdict.xml \
  --output-dir analysis_output_sol_smoke \
  --max-entries 3 \
  --model gpt-6.1-sol \
  --reasoning-effort medium
```

The script leaves reasoning effort to the API unless `--reasoning-effort` is supplied. Luna, Terra, and Sol currently default to `medium`; supported effort values depend on the model. Use the same output directory for the subsequent `run` command. Prepared request files retain their selected model: updating the script does not change an existing run or a previously submitted batch.

## Optional Batch API workflow

For offline runs where a longer wait is acceptable, prepare a separate directory and submit it through Batch:

```bash
.venv/bin/python jmdict_french_audit.py prepare \
  --xml-path data/JMdict.xml \
  --output-dir analysis_output_batch
.venv/bin/python jmdict_french_audit.py submit --output-dir analysis_output_batch
.venv/bin/python jmdict_french_audit.py status --output-dir analysis_output_batch
```

Batch processing is asynchronous, with a `24h` completion window. `status` checks once and exits; run it again later until it reports `completed`, then collect:

```bash
.venv/bin/python jmdict_french_audit.py collect --output-dir analysis_output_batch
```

Check `request_counts` for failed requests and inspect any `batch_errors.jsonl`. Unlike direct execution, the Batch collector does not enforce one unique result per prepared entry: compare `entry_count`, `collected_result_count`, and the sequence numbers before treating a Batch run as complete.

Submit once per intended run: repeating `submit` creates another job and can incur additional charges. Use separate directories for direct and Batch execution; the script refuses to submit a direct run as a Batch job or run an already submitted Batch directory directly. See [API pricing](https://developers.openai.com/api/docs/pricing) for the processing-tier price difference.

## Output files

All files for a run live in its output directory:

| File | Contents |
| --- | --- |
| `entry_context.jsonl` | Original entry data used to interpret the results. |
| `batch_requests.jsonl` | Prepared API requests, including the audit instructions; shared by direct and Batch execution. |
| `manifest.json` | Input checksum, model and parameters, entry/request counts, direct progress or Batch ID, and result metadata. |
| `direct_output.jsonl` | Successfully validated direct responses, saved per request for resuming. |
| `direct_failure.json` | The last failed direct response or error, when present; retained as a diagnostic after a later successful resume. |
| `batch_output.jsonl` | Raw Batch API output downloaded by `collect`. |
| `batch_errors.jsonl` | API errors, when the Batch job provides an error file. |
| `analysis_results.jsonl` | Parsed entry results, including status, issues, explanations, and suggested translations. |
| `suspect_entries.csv` | Entries classified as `likely_error`; optionally includes `needs_review`. |
| `summary.md` | Counts, issue breakdowns, token usage, and example findings. |
| `likely_error_missing_core_meaning.txt` | Comma-separated sequence numbers classified as both `likely_error` and `missing_core_meaning`, for manual review. |

The three result statuses are `ok`, `needs_review`, and `likely_error`. A model's confidence score is not a calibrated probability. The sequence-number list is a candidate list; generating it does not remove dictionary content.

To include ambiguous entries in the CSV, pass `--include-needs-review` to `run` (or Batch `collect`), or regenerate the reports from saved results without another API call:

```bash
.venv/bin/python jmdict_french_audit.py summarize \
  --output-dir analysis_output_full \
  --include-needs-review
```

## Options and troubleshooting

Use `--help` for the command list or a command's options:

```bash
.venv/bin/python jmdict_french_audit.py --help
.venv/bin/python jmdict_french_audit.py prepare --help
.venv/bin/python jmdict_french_audit.py run --help
.venv/bin/python jmdict_french_audit.py collect --help
```

`prepare` also accepts `--entries-per-request` (default 20), `--max-senses-per-entry` (40), `--max-glosses-per-sense` (12), `--max-output-tokens` (20000), and `--reasoning-effort`. Sense/gloss limits truncate the data sent to the model; keep them in mind when interpreting an audit. Supported reasoning settings depend on the selected model.

- **No French entries matched:** use the multilingual JMdict file and check any sequence-number filters.
- **No API key or authentication failure:** check your environment or local `.env`, key permissions, billing, and model access.
- **Direct run failed or was interrupted:** inspect `direct_failure.json` and `manifest.json`, then use `run --resume` to retry the remaining requests with unchanged preparation settings. Saved successful requests are skipped.
- **Batch still processing:** check `status` later. If it reports `failed`, `expired`, or `cancelled`, inspect the status output and the batch in your OpenAI project.
- **Partial or truncated results:** inspect the raw output and error file, compare sequence numbers with `entry_context.jsonl`, and prepare affected entries in a new directory if you need to change request settings. Fewer entries per request or a larger supported output limit may help with truncation. Do not resubmit a full Batch job just to check its status.

`collect --allow-partial` can retrieve available output from an unfinished or unsuccessful batch if an output file exists. Treat that report as partial and collect again after completion when possible.

The local tests use fake API responses and incur no API charges:

```bash
.venv/bin/python -m unittest -q
```

## Reproducibility and licensing

Preserve the exact dictionary snapshot and the run's manifest, requests, context, and raw output when comparing audits. A fresh download or a different model can produce different findings; identical inputs can also vary between model runs. Download directories (`data/`, `data_*/`), analysis directories (`analysis_output*/`, `recovered_analysis_output/`), and local credentials are excluded from Git.

The audit code and this documentation are available under the [MIT License](LICENSE), copyright Chris Lindsay.

JMdict is a separate dictionary resource maintained by the Electronic Dictionary Research and Development Group (EDRDG). Its Japanese and English components are covered by EDRDG's [dictionary licence statement](https://www.edrdg.org/edrdg/licence.html), which specifies Creative Commons Attribution-ShareAlike 4.0; non-English translations, including French, have separate compiler copyrights. Dictionary downloads and audit outputs containing dictionary material are not covered by this repository's MIT license. Consult the dictionary's documentation and applicable terms before redistributing them. This repository does not bundle dictionary data or previous audit outputs.
