#!/usr/bin/env bash

# PYTHONUNBUFFERED=1 => shows logs

cd "$(dirname "$0")"
set -e
docker build -t educ244-final-proj .
docker run \
  --rm \
  --name educ244-final-proj-container \
  --mount type=bind,src="$(pwd)"/app/assets,dst=/app/assets \
  -e PY_ENV='production' \
  -e PYTHONUNBUFFERED=1 \
  -p 3000:3000 \
  -d educ244-final-proj