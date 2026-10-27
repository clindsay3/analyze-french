# JMdict French Audit

Audit French translations in [JMdict](https://www.edrdg.org/wiki/index.php/JMdict-EDICT_Dictionary_Project) against the Japanese forms and English meanings, using the [OpenAI Batch API](https://developers.openai.com/api/docs/guides/batch).

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

Run commands from the repository root. Keep each run's output directory intact: its manifest and context files are needed to collect and interpret the results.

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

### 3. Submit the paid batch

```bash
.venv/bin/python jmdict_french_audit.py submit --output-dir analysis_output_smoke
```

This uploads the prepared Japanese, English, and French entry data to OpenAI and creates a batch. The batch ID is saved in `manifest.json`. Submit once per intended run: repeating `submit` creates another job and can incur additional charges.

### 4. Wait for completion

```bash
.venv/bin/python jmdict_french_audit.py status --output-dir analysis_output_smoke
```

Batch processing is asynchronous, with a `24h` completion window. Run `status` again later until it reports `completed`. The command checks once and exits; it does not wait or poll automatically. Check `request_counts` for failed requests as well as the batch status.

### 5. Collect and inspect the results

```bash
.venv/bin/python jmdict_french_audit.py collect --output-dir analysis_output_smoke
```

Open `analysis_output_smoke/summary.md` and `analysis_output_smoke/suspect_entries.csv`. An empty suspect CSV, apart from its header, is possible if none of the sample entries is classified as a likely error.

Compare the manifest's `entry_count` and `collected_result_count` and check for missing or duplicate sequence numbers in `analysis_results.jsonl`. A completed batch can still have failed requests or incomplete model results; the script does not enforce one unique result per prepared entry. Inspect any `batch_errors.jsonl` before treating a run as complete.

## Run the full audit

After checking the sample, prepare all entries with French glosses in a new output directory:

```bash
.venv/bin/python jmdict_french_audit.py prepare \
  --xml-path data/JMdict.xml \
  --output-dir analysis_output_full
.venv/bin/python jmdict_french_audit.py submit --output-dir analysis_output_full
.venv/bin/python jmdict_french_audit.py status --output-dir analysis_output_full
```

Once `status` reports `completed`, collect the results:

```bash
.venv/bin/python jmdict_french_audit.py collect --output-dir analysis_output_full
```

Use a different `analysis_output_*` directory for each new run. Re-running `prepare` in an existing directory overwrites its manifest and prepared files, including the saved batch ID. Full runs can process thousands of entries; cost depends on the model, dictionary snapshot, and generated output.

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

The script leaves reasoning effort to the API unless `--reasoning-effort` is supplied. Luna, Terra, and Sol currently default to `medium`; supported effort values depend on the model. Use the same output directory for subsequent `submit`, `status`, and `collect` commands. Prepared request files retain their selected model: updating the script does not change an existing run or a previously submitted batch.

## Output files

All files for a run live in its output directory:

| File | Contents |
| --- | --- |
| `entry_context.jsonl` | Original entry data used to interpret the results. |
| `batch_requests.jsonl` | Prepared Batch API requests, including the audit instructions. |
| `manifest.json` | Input checksum, requested model and parameters, entry/request counts, batch ID, and collection metadata. |
| `batch_output.jsonl` | Raw API output downloaded by `collect`. |
| `batch_errors.jsonl` | API errors, when the batch provides an error file. |
| `analysis_results.jsonl` | Parsed entry results, including status, issues, explanations, and suggested translations. |
| `suspect_entries.csv` | Entries classified as `likely_error`; optionally includes `needs_review`. |
| `summary.md` | Counts, issue breakdowns, token usage, and example findings. |
| `likely_error_missing_core_meaning.txt` | Comma-separated sequence numbers classified as both `likely_error` and `missing_core_meaning`, for manual review. |

The three result statuses are `ok`, `needs_review`, and `likely_error`. A model's confidence score is not a calibrated probability. The sequence-number list is a candidate list; generating it does not remove dictionary content.

To include ambiguous entries in the CSV, pass `--include-needs-review` to `collect`, or regenerate the reports from already collected results without another API call:

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
.venv/bin/python jmdict_french_audit.py collect --help
```

`prepare` also accepts `--entries-per-request` (default 20), `--max-senses-per-entry` (40), `--max-glosses-per-sense` (12), `--max-output-tokens` (20000), and `--reasoning-effort`. Sense/gloss limits truncate the data sent to the model; keep them in mind when interpreting an audit. Supported reasoning settings depend on the selected model.

- **No French entries matched:** use the multilingual JMdict file and check any sequence-number filters.
- **No API key or authentication failure:** check your environment or local `.env`, key permissions, billing, and model access.
- **Batch still processing:** check `status` later. If it reports `failed`, `expired`, or `cancelled`, inspect the status output and the batch in your OpenAI project.
- **Partial or truncated results:** inspect the raw output and error file, compare sequence numbers with `entry_context.jsonl`, and prepare only missing entries in a new directory. Fewer entries per request or a larger supported output limit may help with truncation. Do not resubmit a full batch just to check its status.

`collect --allow-partial` can retrieve available output from an unfinished or unsuccessful batch if an output file exists. Treat that report as partial and collect again after completion when possible.

## Reproducibility and licensing

Preserve the exact dictionary snapshot and the run's manifest, requests, context, and raw output when comparing audits. A fresh download or a different model can produce different findings; identical inputs can also vary between model runs. Download directories (`data/`, `data_*/`), analysis directories (`analysis_output*/`, `recovered_analysis_output/`), and local credentials are excluded from Git.

The audit code and this documentation are available under the [MIT License](LICENSE), copyright Chris Lindsay.

JMdict is a separate dictionary resource maintained by the Electronic Dictionary Research and Development Group (EDRDG). Its Japanese and English components are covered by EDRDG's [dictionary licence statement](https://www.edrdg.org/edrdg/licence.html), which specifies Creative Commons Attribution-ShareAlike 4.0; non-English translations, including French, have separate compiler copyrights. Dictionary downloads and audit outputs containing dictionary material are not covered by this repository's MIT license. Consult the dictionary's documentation and applicable terms before redistributing them. This repository does not bundle dictionary data or previous audit outputs.
