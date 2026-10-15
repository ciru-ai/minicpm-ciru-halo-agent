# Install MiniCPM 4.7 Ultra

## Hardware and platform

- AMD Ryzen AI Max+ 395 / Radeon 8060S, `gfx1151`, with **128 GB unified memory** is the measured configuration.
- Linux with a working AMD GPU driver and permission to access `/dev/kfd` and `/dev/dri/renderD*`.
- About **75 GB free disk space**, including the model, compressed downloads, and installed runtime. Keep the `weights` directory after installation: the server reads it directly.
- A normal systemd user session. The launcher checks for an already active `qwen-main.service` and uses GPU ownership locks; it does not stop existing services for you.

The release contains a pinned Python 3.14 / ROCm 10 stack. Do not replace individual components with a generic `pip install vllm` or upgrade Torch independently. Those changes select a different, unmeasured runtime.

## Ubuntu 24.04 prerequisites

```bash
sudo apt-get update
sudo apt-get install -y \
  git python3 python3-venv ca-certificates zstd \
  build-essential pkg-config xxd libgomp1 libnuma1 libdrm2 libelf1 \
  libssl3t64 libstdc++6 zlib1g
```

Your AMD driver must already support the hardware. Confirm GPU device access as the account that will run inference. GPU group membership changes may require logging out and back in.

**Testing scope:** end-to-end GPU serving was validated on NixOS. Ubuntu 24.04 userspace installation checks are recorded in `release.json`; they do not substitute for a full Ubuntu GPU-serving qualification. Keep that distinction when comparing performance or reporting installation problems.

## Download and install

```bash
git clone https://github.com/ciru-ai/minicpm-ciru-halo-agent.git
cd minicpm-ciru-halo-agent
python3 -m venv .download-env
. .download-env/bin/activate
python -m pip install 'huggingface_hub>=1.0,<2'

python scripts/download.py --destination "$HOME/models/minicpm-ultra"
bash scripts/install.sh "$HOME/models/minicpm-ultra" "$HOME/.local/opt/minicpm-ultra"
```

The downloader resolves an immutable Hugging Face commit, downloads that snapshot, and verifies both runtime archives. The installer verifies all target-weight hashes and refuses to overwrite an existing installation.

The shared foundation goes in `/opt/ciru/glm53-iu4` because the validated Python environment uses that location. **This does not load or require GLM weights.** The MiniCPM model-specific engine, kernels, and drafter install separately at the directory you choose. An existing matching foundation is reused; an incompatible one is reported without alteration.

The default temporary directory must hold the extracted model-runtime archive, roughly 2–3 GB. Set `TMPDIR` to a disk-backed directory if `/tmp` is too small. Foundation extraction happens directly under `/opt/ciru` and needs approximately 12 GB there.

## Start

```bash
bash "$HOME/.local/opt/minicpm-ultra/serve.sh" --host 127.0.0.1 --port 8095
```

In a second terminal:

```bash
curl --fail http://127.0.0.1:8095/health
curl --fail http://127.0.0.1:8095/v1/models
```

Use model ID `minicpm-q8`. First startup can compile kernels; wait for the health endpoint. Compilation caches are retained for subsequent starts. Stop with Ctrl-C; the launcher shuts down only its own server process group. It does not install, enable, mask, or restore a system service.

## NixOS

Enable `programs.nix-ld` and provide host libraries for the C++ runtime, zlib, libdrm, elfutils, numactl, and OpenSSL. GCC, Git, pkg-config, and xxd are needed for JIT builds. Use your system's current store paths through `CIRU_HOST_LIBRARY_PATH`; the download deliberately does not carry the build machine's Nix store paths.

An example development shell is included as `shell.nix`:

```bash
nix-shell shell.nix
bash "$HOME/.local/opt/minicpm-ultra/serve.sh" --host 127.0.0.1 --port 8095
```

## Configuration and limits

The default profile enables native vision, tool parsing, prefix/image caching, 262,144 maximum context, eight request slots, a 24 GiB KV budget, and adaptive draft depth. The eight slots share memory; they are not eight independent full-context allocations.

Thinking is controlled per request with `chat_template_kwargs.enable_thinking`. See [API examples](API.md). Preserve the model's completion budget in agent clients so long reasoning and tool calls can finish naturally.

The original test model and all source/kernel identities are retained in the release manifests. Source directory/module names reflect development history; they are not separate downloadable models or selectable performance profiles.
