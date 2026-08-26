"""Self-contained HTML reporting for two-model comparisons."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DIMENSIONS = [
    ("code_planning", "Code Planning"),
    ("coding", "Coding"),
    ("analysis_documentation", "Code Analysis / Documentation"),
    ("terminal_bash", "Terminal / Bash"),
    ("orchestration", "Orchestration"),
]

COLORS = ["#6c8cff", "#ff8a5b"]


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: float | int | None, decimals: int = 1) -> str:
    return "—" if value is None else f"{value:.{decimals}f}"


def _quality_chart(key: str, title: str, models: list[dict[str, Any]]) -> str:
    rows = []
    for index, model in enumerate(models):
        value = float(model["summary"]["dimension_scores"].get(key, 0.0)) * 100
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{_esc(model['label'])}</div>
              <div class="track"><div class="bar" style="width:{value:.2f}%;background:{COLORS[index]}"></div></div>
              <div class="bar-value">{value:.1f}</div>
            </div>"""
        )
    return f"<section class='chart'><h3>{_esc(title)}</h3>{''.join(rows)}</section>"


def _speed_chart(
    title: str,
    field: str,
    models: list[dict[str, Any]],
    *,
    unit: str,
    lower_is_better: bool = False,
) -> str:
    values = [float(model["summary"].get(field) or 0.0) for model in models]
    maximum = max(values) if max(values, default=0.0) > 0 else 1.0
    rows = []
    for index, (model, value) in enumerate(zip(models, values)):
        width = value / maximum * 100
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{_esc(model['label'])}</div>
              <div class="track"><div class="bar" style="width:{width:.2f}%;background:{COLORS[index]}"></div></div>
              <div class="bar-value wide">{value:.1f} {_esc(unit)}</div>
            </div>"""
        )
    direction = " · lower is better" if lower_is_better else " · higher is better"
    return f"<section class='chart'><h3>{_esc(title)}<small>{direction}</small></h3>{''.join(rows)}</section>"


def _summary_table(models: list[dict[str, Any]]) -> str:
    rows = []
    for model in models:
        summary = model["summary"]
        rows.append(
            "<tr>"
            f"<td><strong>{_esc(model['label'])}</strong><br><span>{_esc(model['model_id'])}</span></td>"
            f"<td>{summary['overall_score'] * 100:.1f}</td>"
            f"<td>{summary['passed_cases']}/{summary['total_cases']}</td>"
            f"<td>{summary['error_cases']}</td>"
            f"<td>{_number(summary.get('median_ttft_ms'), 0)} ms</td>"
            f"<td>{_number(summary.get('median_e2e_ms'), 0)} ms</td>"
            f"<td>{_number(summary.get('decode_tokens_per_second'))}</td>"
            f"<td>{_number(summary.get('effective_tokens_per_second'))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _case_rows(models: list[dict[str, Any]]) -> str:
    by_model = [{item["case_id"]: item for item in model["cases"]} for model in models]
    rows = []
    for key, title in DIMENSIONS:
        case_ids = sorted(
            {
                item["case_id"]
                for model in models
                for item in model["cases"]
                if item["dimension"] == key
            }
        )
        for case_id in case_ids:
            first = next((mapping.get(case_id) for mapping in by_model if mapping.get(case_id)), None)
            score_cells = []
            for mapping in by_model:
                item = mapping.get(case_id)
                if not item:
                    score_cells.append("<td>—</td>")
                    continue
                klass = "pass" if item["score"] >= 0.999 else "fail"
                detail = _esc(item.get("detail", ""))
                error = _esc(item.get("error", ""))
                score_cells.append(
                    f"<td class='{klass}'><strong>{item['score'] * 100:.1f}</strong>"
                    f"<details><summary>details</summary><p>{detail}</p><p>{error}</p></details></td>"
                )
            rows.append(
                "<tr>"
                f"<td>{_esc(title)}</td><td><code>{_esc(case_id)}</code></td>"
                f"<td>{_esc(first['source'] if first else '')}</td>{''.join(score_cells)}</tr>"
            )
    return "".join(rows)


def render_html(report: dict[str, Any], output: Path) -> None:
    models = report["models"]
    quality_charts = "".join(_quality_chart(key, title, models) for key, title in DIMENSIONS)
    speed_charts = "".join(
        [
            _speed_chart("Decode throughput", "decode_tokens_per_second", models, unit="tok/s"),
            _speed_chart(
                "Effective full-run throughput",
                "effective_tokens_per_second",
                models,
                unit="tok/s",
            ),
            _speed_chart("Median time to first token", "median_ttft_ms", models, unit="ms", lower_is_better=True),
            _speed_chart("Median end-to-end latency", "median_e2e_ms", models, unit="ms", lower_is_better=True),
        ]
    )
    warnings = report.get("warnings") or []
    warning_html = "".join(f"<li>{_esc(item)}</li>" for item in warnings)
    model_headers = "".join(f"<th>{_esc(model['label'])}</th>" for model in models)
    created = datetime.fromisoformat(report["created_at"]).astimezone(timezone.utc)
    methodology = report["methodology"]
    configured_cap = methodology.get("max_tokens_per_request")
    cap_text = "no client-side cap" if configured_cap is None else f"{configured_cap} tokens"
    configured_models = (report.get("configuration") or {}).get("models") or []
    request_profiles = "; ".join(
        f"{item['label']}: {item.get('request_body') or 'endpoint defaults'}"
        for item in configured_models
    ) or "Endpoint defaults"
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(report['title'])}</title>
<style>
:root{{--ink:#202638;--muted:#667085;--panel:#fff;--line:#e6e9f0;--bg:#f5f7fb;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:42px 24px 70px}} h1{{font-size:34px;margin:0 0 6px}} h2{{margin:38px 0 16px}} h3{{margin:0 0 18px;font-size:17px}} h3 small{{color:var(--muted);font-weight:400}}
.subtitle,.meta,td span{{color:var(--muted)}} .hero,.chart,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:0 5px 22px rgba(26,36,59,.045)}}
.hero{{padding:26px 28px;margin-bottom:28px}} .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}} .chart{{padding:22px 22px 19px}}
.bar-row{{display:grid;grid-template-columns:150px 1fr 58px;gap:12px;align-items:center;margin:13px 0}} .bar-label{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}} .track{{height:17px;background:#eef1f6;border-radius:20px;overflow:hidden}} .bar{{height:100%;border-radius:20px;min-width:1px}} .bar-value{{font-variant-numeric:tabular-nums;text-align:right;font-weight:700}} .bar-value.wide{{font-size:13px}}
.panel{{padding:18px;overflow:auto}} table{{width:100%;border-collapse:collapse;min-width:900px}} th,td{{text-align:left;border-bottom:1px solid var(--line);padding:12px 10px;vertical-align:top}} th{{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}} td.pass>strong{{color:#087a55}} td.fail>strong{{color:#b5473f}} code{{font-size:12px}} details{{font-size:12px;color:var(--muted)}} summary{{cursor:pointer}}
.notice{{background:#fff7e7;border:1px solid #f0d28b;border-radius:10px;padding:12px 17px}} .method{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}} .method div{{padding:14px;background:#fafbfe;border:1px solid var(--line);border-radius:9px}} footer{{color:var(--muted);margin-top:30px;font-size:13px}}
@media(max-width:760px){{.grid,.method{{grid-template-columns:1fr}}.bar-row{{grid-template-columns:95px 1fr 55px}}main{{padding:24px 14px}}}}
@media print{{body{{background:#fff}}main{{max-width:none;padding:12mm}}.hero,.chart,.panel{{box-shadow:none;break-inside:avoid}}details{{display:none}}}}
</style>
</head>
<body><main>
<section class="hero"><h1>{_esc(report['title'])}</h1>
<p class="subtitle">Same prompts, immutable benchmark revisions, deterministic sample seed {report['seed']}.</p>
<p class="meta">Created {created.strftime('%Y-%m-%d %H:%M UTC')} · comparison wall time {_number(report.get('comparison_wall_seconds'))} s · tool mode {_esc(methodology['tool_mode'])}</p></section>
<h2>Summary</h2><section class="panel"><table><thead><tr><th>Model</th><th>Overall / 100</th><th>Perfect cases</th><th>Errors</th><th>Median TTFT</th><th>Median E2E</th><th>Decode tok/s</th><th>Full-run tok/s</th></tr></thead><tbody>{_summary_table(models)}</tbody></table></section>
<h2>Quality dimensions</h2><div class="grid">{quality_charts}</div>
<h2>Speed and throughput</h2><div class="grid">{speed_charts}</div>
<h2>Case-level results</h2><section class="panel"><table><thead><tr><th>Dimension</th><th>Case</th><th>Source</th>{model_headers}</tr></thead><tbody>{_case_rows(models)}</tbody></table></section>
<h2>Methodology</h2><section class="panel method">
<div><strong>Quality aggregate</strong><br>Unweighted mean of the five requested dimension scores. Every scheduled case counts; API and evaluation errors score zero.</div>
<div><strong>Coding</strong><br>HumanEval+ generations run in a networkless, read-only Docker container with CPU, memory, process, and time limits.</div>
<div><strong>Analysis / documentation</strong><br>CRUXEval-O exact output prediction plus transparent custom documentation concept coverage.</div>
<div><strong>Terminal / Bash</strong><br>NL2Bash inputs with an adapted score: utility match (40%), token F1 (35%), and token-sequence similarity (25%). Commands are not executed.</div>
<div><strong>Orchestration</strong><br>BFCL v4 simple, selection, and parallel cases. Calls are compared order-independently against accepted argument values.</div>
<div><strong>Throughput</strong><br>Decode tok/s excludes the first streamed token and divides the remainder by measured post-first-token time. Full-run tok/s divides measured case output tokens by the complete per-model run, including warmup and scoring.</div>
<div><strong>Output budget</strong><br>{_esc(cap_text)} per scored request. Endpoint-side limits may still apply.</div>
<div><strong>Reasoning request fields</strong><br>{_esc(request_profiles)}</div>
</section>
{f'<h2>Warnings</h2><section class="notice"><ul>{warning_html}</ul></section>' if warnings else ''}
<footer>Raw, auditable case results and benchmark artifact SHA-256 hashes are in the adjacent JSON file. API keys and header values are never written.</footer>
</main></body></html>"""
    output.write_text(document, encoding="utf-8")
