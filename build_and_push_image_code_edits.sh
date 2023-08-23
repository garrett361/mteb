#!/usr/bin/env bash
docker build -f Dockerfile_code_edits --tag garrettgoon/mteb_det .
docker push garrettgoon/mteb_det

