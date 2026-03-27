#!/usr/bin/env bash
# Build and launch the Rust sidecar service for Polymarket orderbook streaming.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDECAR_DIR="$SCRIPT_DIR/../sidecar"

echo "Building polyfill-sidecar (release)..."
cd "$SIDECAR_DIR"
cargo build --release

echo "Starting sidecar on port ${SIDECAR_PORT:-8080}..."
RUST_LOG="${RUST_LOG:-polyfill_sidecar=info,polyfill_rs=warn}" \
    ./target/release/polyfill-sidecar
