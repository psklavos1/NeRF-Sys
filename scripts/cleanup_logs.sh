#!/bin/bash

LOG_DIR="logs"

if [ ! -d "$LOG_DIR" ]; then
  echo "Log directory $LOG_DIR does not exist."
  exit 1
fi

echo "Deleting all files and directories in $LOG_DIR..."

# Remove everything inside logs/
rm -rf "$LOG_DIR"/* "$LOG_DIR"/.* 2>/dev/null

echo "Done."