#!/usr/bin/env bash
docker compose -f docker-compose.yml up --scale worker=2 --build
