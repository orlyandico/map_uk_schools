#!/usr/bin/env bash
# Build the state-secondary ethnicity report PDF from its Markdown.
#
# Pipeline: cmark-gfm (GitHub Markdown) -> HTML with inline CSS -> wkhtmltopdf,
# rendered A4 landscape with auto-sized table columns so the wide group-by-group
# table fits without stretching the School column.
#
# Usage:
#   ./build_grammar_pdf.sh [input.md] [output.pdf] [font_px]
# Defaults: grammar_ethnicity_report.md  grammar_ethnicity_report.pdf  11
set -euo pipefail

IN="${1:-grammar_ethnicity_report.md}"
OUT="${2:-grammar_ethnicity_report.pdf}"
FONT="${3:-11}"

if [[ ! -f "$IN" ]]; then
  echo "Input Markdown not found: $IN" >&2
  exit 1
fi

HTML="$(mktemp --suffix=.html)"
trap 'rm -f "$HTML"' EXIT

cmark-gfm --extension table --extension autolink --extension strikethrough "$IN" | \
  awk -v font="$FONT" 'BEGIN{
    print "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>";
    print "@page{size:A4 landscape;margin:10mm;}";
    print "body{font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",Helvetica,Arial,sans-serif;font-size:" font "px;line-height:1.4;color:#24292f;}";
    print "h1{font-size:1.7em;border-bottom:1px solid #d1d9e0;padding-bottom:0.2em;}";
    print "h2{font-size:1.3em;border-bottom:1px solid #d1d9e0;padding-bottom:0.2em;margin-top:16px;}";
    print "p{margin:0 0 8px 0;} em{color:#59636e;}";
    print "table{border-collapse:collapse;width:auto;max-width:90%;margin:0 0 12px 0;}";
    print "th,td{border:1px solid #d1d9e0;padding:3px 7px;text-align:right;}";
    print "td:nth-child(2),th:nth-child(2){text-align:left;white-space:nowrap;}";
    print "th:first-child,td:first-child{text-align:left;}";
    print "th{background:#f6f8fa;font-weight:600;}";
    print "tr:nth-child(2n){background:#f6f8fa;}";
    print "blockquote{padding:0 1em;color:#59636e;border-left:0.25em solid #d1d9e0;margin:0 0 12px 0;max-width:90%;}";
    print "</style></head><body>";
  } {print} END{print "</body></html>"}' > "$HTML"

wkhtmltopdf --enable-local-file-access \
  --orientation Landscape --page-size A4 \
  --margin-top 10mm --margin-bottom 10mm --margin-left 10mm --margin-right 10mm \
  "$HTML" "$OUT"

echo "Wrote $OUT ($(du -h "$OUT" | cut -f1), font ${FONT}px)"
