#!/usr/bin/env bash
docker build -f Dockerfile_base_reqs  --tag garrettgoon/mteb_det_base_reqs .
docker push garrettgoon/mteb_det_base_reqs

