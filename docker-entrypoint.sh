#!/bin/sh

set -e

. ./.venv/bin/activate

exec gunicorn --bind 0.0.0.0:5000 --access-logfile - --error-logfile - wsgi:app
