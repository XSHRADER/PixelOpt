#!/usr/bin/env sh
# Set up and start PixelOpt on macOS or Linux. All the work is in launch.py.
#   ./launch.sh            ./launch.sh --check            ./launch.sh --once --no-browser
cd "$(dirname "$0")" || exit 1
exec python3 launch.py "$@"
