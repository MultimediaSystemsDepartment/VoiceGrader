#!/usr/bin/env sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
output=${1:-"$(dirname "$root")/gpu-stress-tester.zip"}

cd "$(dirname "$root")"
rm -f "$output"
zip -r "$output" "$(basename "$root")" \
  -x 'gpu-stress-tester/data/*' \
     'gpu-stress-tester/outputs/*' \
     'gpu-stress-tester/model-cache/*' \
     'gpu-stress-tester/.pytest_cache/*' \
     'gpu-stress-tester/**/__pycache__/*' \
     'gpu-stress-tester/.env'
printf 'Created %s\n' "$output"
