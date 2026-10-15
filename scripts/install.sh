#!/usr/bin/env bash
# Install the verified release without changing services or model selectors.
set -euo pipefail
if [[ $# != 2 ]]; then
  printf 'Usage: bash %s DOWNLOAD_DIRECTORY NEW_RUNTIME_DIRECTORY\n' "$0" >&2
  exit 2
fi
download=$(realpath -- "$1")
destination=$(realpath -m -- "$2")
foundation=/opt/ciru/glm53-iu4
for command in python3 tar zstd; do
  command -v "$command" >/dev/null || { printf 'Missing command: %s\n' "$command" >&2; exit 2; }
done
[[ ! -e "$destination" ]] || { printf 'Refusing to overwrite: %s\n' "$destination" >&2; exit 2; }

python3 - "$download" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
release = json.loads((root / 'release.json').read_text())
for name, item in release['archives'].items():
    p = root / name
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    if p.stat().st_size != item['bytes'] or h.hexdigest() != item['sha256']:
        raise SystemExit('Archive checksum mismatch: ' + name)
print('Release archive checksums verified.')
PY

work=$(mktemp -d "${TMPDIR:-/tmp}/minicpm-ultra-install.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
tar --zstd -xf "$download/runtime/minicpm-ultra-runtime.tar.zst" -C "$work"

if [[ ! -e "$foundation" ]]; then
  printf 'Installing the pinned shared foundation at %s. Administrator access is required.\n' "$foundation"
  sudo mkdir -p /opt/ciru
  # Extract into a unique directory; never unpack over an existing installation.
  foundation_stage=$(sudo mktemp -d /opt/ciru/.minicpm-foundation.XXXXXXXX)
  sudo tar --zstd --no-same-owner -xf "$download/runtime/ciru-rocm10-gfx1151-foundation.tar.zst" -C "$foundation_stage"
  if [[ -e "$foundation" ]]; then
    printf 'Foundation appeared during installation; keeping extracted files at %s\n' "$foundation_stage" >&2
    exit 2
  fi
  sudo mv -T -- "$foundation_stage/foundation" "$foundation"
  sudo rmdir -- "$foundation_stage"
fi

# This checks the foundation against the original validated byte identities.
# An incompatible pre-existing foundation is reported, never overwritten.
python3 "$work/INSTALL.py" --verify-only
python3 "$work/INSTALL.py" --prefix "$destination" --model "$download/weights" --verify-weights
printf '\nInstalled. Start as your normal user:\n  bash %q/serve.sh --host 127.0.0.1 --port 8095\n' "$destination"
