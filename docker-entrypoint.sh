#!/bin/sh
set -e

# Set umask so host-mounted output files in /data/processed and /models are accessible
umask 0002

# If first arg is a flag, assume uvicorn
if [ "${1#-}" != "$1" ]; then
    set -- python -m uvicorn backend.app.main:app "$@"
fi

exec "$@"
