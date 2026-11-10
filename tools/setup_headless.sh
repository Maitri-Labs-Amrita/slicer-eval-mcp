#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Set up a headless Linux server (Ubuntu/Debian, no sudo) to run slicer-mcp.

  tools/setup_headless.sh [--shared DIR] [--prefix DIR] [--slicer-dir DIR] [--slicer-tarball FILE]

1. Xvfb: the system one if on PATH, else extracted from its .deb into <prefix>/root (apt-get download; dpkg -x).
2. Slicer 5.10.0: reused if <slicer-dir>/Slicer-5.10.0-linux-amd64 exists, else from --slicer-tarball (no network),
   else downloaded from download.slicer.org; the tarball is checked against a pinned sha512.
3. The system libraries Slicer needs and the server lacks (found with ldd), extracted the same way.
4. config/runtime.local.json (gitignored): runtime.json with this user's paths. A rerun keeps every other setting
   already in the file (port, state file, ...).
5. A check: Slicer starts under Xvfb and gets an OpenGL context.

--shared DIR sets --prefix and --slicer-dir to one group folder, so a team on one cluster installs once and every
member only writes their own config.

Run it again at any time: every step skips what is already there. Then run everything under the virtual display:
  SLICER_MCP_RUNTIME=config/runtime.local.json uv run python tools/xvfb_session.py -- <command>
USAGE
}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="$HOME/.local/slicer-mcp-headless"
SLICER_DIR="$HOME/Applications"
SLICER_TARBALL=""
SLICER_VERSION="5.10.0"
SLICER_NAME="Slicer-$SLICER_VERSION-linux-amd64"
SLICER_URL="https://download.slicer.org/bitstream/6911b598ac7b1c95e7934427"
SLICER_SHA512="2ea56b6f0c027fa73c832b23c34948e69b1b5124edf337a35f6a062f5cb78e7feb792c11bc02a4986f26e458ddfd954b00255953018bf6cc7d73834aba9f0267"

while [ $# -gt 0 ]; do
  case "$1" in
    --shared) PREFIX="$2"; SLICER_DIR="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --slicer-dir) SLICER_DIR="$2"; shift 2 ;;
    --slicer-tarball) SLICER_TARBALL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

ROOT="$PREFIX/root"
DEBS="$PREFIX/debs"
LIBDIR="$ROOT/usr/lib/x86_64-linux-gnu"
PULSELIBDIR="$LIBDIR/pulseaudio"
SLICER="$SLICER_DIR/$SLICER_NAME/Slicer"
mkdir -p "$ROOT" "$DEBS" "$SLICER_DIR"

say() { printf '\n== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

extract_pkg() {
  local pkg="$1"
  (cd "$DEBS" && ls "${pkg}"_*.deb >/dev/null 2>&1 || apt-get download "$pkg" >/dev/null) \
    || die "apt-get download $pkg failed"
  dpkg -x "$(ls "$DEBS/${pkg}"_*.deb | head -1)" "$ROOT"
  echo "  extracted $pkg"
}

say "0/5 prerequisites"
command -v apt-get >/dev/null && command -v dpkg >/dev/null || die "needs apt-get and dpkg (Ubuntu/Debian)"
[ "$(uname -m)" = "x86_64" ] || die "Slicer's Linux build is x86_64 only (this is $(uname -m))"
command -v python3 >/dev/null || die "python3 not found"
if ! command -v uv >/dev/null; then
  [ -x "$HOME/.local/bin/uv" ] && die "uv is in ~/.local/bin but not on PATH: add 'export PATH=\$HOME/.local/bin:\$PATH' to ~/.bashrc (the client starts uv by name)"
  die "uv not found: install it (https://docs.astral.sh/uv/) and put it on PATH"
fi
echo "  ok: apt-get, dpkg, python3, uv"

say "1/5 Xvfb"
if command -v Xvfb >/dev/null; then
  XVFB="$(command -v Xvfb)"
  echo "  system Xvfb: $XVFB"
else
  XVFB="$ROOT/usr/bin/Xvfb"
  [ -x "$XVFB" ] || extract_pkg xvfb
  echo "  user Xvfb: $XVFB"
fi

say "2/5 Slicer $SLICER_VERSION"
if [ -x "$SLICER" ]; then
  echo "  present: $SLICER"
else
  TARBALL="$SLICER_TARBALL"
  if [ -z "$TARBALL" ]; then
    TARBALL="$SLICER_DIR/$SLICER_NAME.tar.gz"
    echo "  downloading $SLICER_URL (430 MB; on a slow link this takes long) -> $TARBALL"
    curl -fL --retry 5 -C - -o "$TARBALL" "$SLICER_URL" \
      || die "download failed (run again to resume, or pass --slicer-tarball FILE)"
  fi
  [ -f "$TARBALL" ] || die "no such tarball: $TARBALL"
  echo "  checking sha512"
  echo "$SLICER_SHA512  $TARBALL" | sha512sum -c --quiet - \
    || die "sha512 mismatch for $TARBALL (not Slicer $SLICER_VERSION, or an incomplete download: run again to resume)"
  tar -xzf "$TARBALL" -C "$SLICER_DIR"
  [ -x "$SLICER" ] || die "after extraction $SLICER is missing"
  echo "  installed: $SLICER"
fi

say "3/5 system libraries"
declare -A PKG=(
  [libGLU.so.1]=libglu1-mesa [libOpenGL.so.0]=libopengl0 [libxcb-xinerama.so.0]=libxcb-xinerama0
  [libpulse-mainloop-glib.so.0]=libpulse-mainloop-glib0 [libpulse.so.0]=libpulse0
  [libasyncns.so.0]=libasyncns0 [libsndfile.so.1]=libsndfile1
  [libFLAC.so.12]=libflac12t64 [libmp3lame.so.0]=libmp3lame0 [libmpg123.so.0]=libmpg123-0t64
  [libogg.so.0]=libogg0 [libopus.so.0]=libopus0 [libvorbis.so.0]=libvorbis0a [libvorbisenc.so.2]=libvorbisenc2
  [libpcre2-16.so.0]=libpcre2-16-0 [libxkbcommon-x11.so.0]=libxkbcommon-x11-0 [libxkbcommon.so.0]=libxkbcommon0
  [libxcb-icccm.so.4]=libxcb-icccm4 [libxcb-image.so.0]=libxcb-image0 [libxcb-keysyms.so.1]=libxcb-keysyms1
  [libxcb-render-util.so.0]=libxcb-render-util0 [libxcb-xkb.so.1]=libxcb-xkb1 [libxcb-randr.so.0]=libxcb-randr0
  [libxcb-shape.so.0]=libxcb-shape0 [libxcb-xfixes.so.0]=libxcb-xfixes0 [libxcb-sync.so.1]=libxcb-sync1
  [libxcb-util.so.1]=libxcb-util1 [libxcb-cursor.so.0]=libxcb-cursor0 [libnss3.so]=libnss3
  [libGL.so.1]=libgl1 [libEGL.so.1]=libegl1 [libGLX.so.0]=libglx0 [libglapi.so.0]=libglapi-mesa
  [libXt.so.6]=libxt6 [libSM.so.6]=libsm6 [libICE.so.6]=libice6 [libfontconfig.so.1]=libfontconfig1
  [libasound.so.2]=libasound2t64 [libdbus-1.so.3]=libdbus-1-3 [libxshmfence.so.1]=libxshmfence1
)
IGNORE='^(libodbc\.so\.2|libpq\.so\.5)$'
missing_libs() {
  local sd="$SLICER_DIR/$SLICER_NAME"
  LD_LIBRARY_PATH="$LIBDIR:$PULSELIBDIR:$(find "$sd/lib" -type d | tr '\n' ':')" bash -c '
    for f in "$0/Slicer" "$0/bin/SlicerApp-real" "$0"/lib/QtPlugins/*/*.so "$1"; do ldd "$f" 2>/dev/null; done' "$sd" "$XVFB" \
    | awk '/not found/ {print $1}' | sort -u | grep -Ev "$IGNORE" || true
}
for pass in 1 2 3; do
  MISSING="$(missing_libs)"
  [ -z "$MISSING" ] && break
  for lib in $MISSING; do
    pkg="${PKG[$lib]:-}"
    [ -n "$pkg" ] || die "missing $lib and no package is known for it: add it to PKG in $0 (apt-file search $lib)"
    extract_pkg "$pkg"
  done
done
MISSING="$(missing_libs)"
[ -z "$MISSING" ] || die "still missing after 3 passes: $MISSING"
echo "  ok: nothing missing (extra libraries in $LIBDIR)"

say "4/5 config/runtime.local.json"
SLICER="$SLICER" XVFB="$XVFB" LIBDIR="$LIBDIR" PULSELIBDIR="$PULSELIBDIR" REPO="$REPO" python3 - <<'EOF'
import json, os
repo = os.environ["REPO"]
p = os.path.join(repo, "config", "runtime.local.json")
old = json.load(open(p)) if os.path.exists(p) else {}
rt = json.load(open(os.path.join(repo, "config", "runtime.json")))
for k, v in old.items():
    rt[k] = {**rt[k], **v} if isinstance(v, dict) and isinstance(rt.get(k), dict) else v
rt["_comment"] = ("Written by tools/setup_headless.sh for this user and machine (gitignored). Use via "
                  "SLICER_MCP_RUNTIME=config/runtime.local.json under tools/xvfb_session.py. Deployment settings, NOT part "
                  "of the task protocol and not hashed into traces.")
rt["slicer_executable"] = os.environ["SLICER"]
rt["xvfb_executable"] = os.environ["XVFB"]
rt["extra_lib_dirs"] = [p for p in (os.environ["LIBDIR"], os.environ["PULSELIBDIR"]) if os.path.isdir(p)]
rt["eval_launch"]["caffeinate"] = False
changed = sorted(k for k in rt if k in old and old[k] != rt[k] and not k.startswith("_"))
if changed:
    print(f"  updated: {', '.join(changed)} (everything else kept)")
open(p, "w").write(json.dumps(rt, indent=2, ensure_ascii=False) + "\n")
print(f"  wrote {p}")
EOF

say "5/5 check: Slicer under Xvfb with OpenGL"
cd "$REPO"
uv sync -q
OUT="$(SLICER_MCP_RUNTIME=config/runtime.local.json timeout 300 uv run python tools/xvfb_session.py -- "$SLICER" \
  --no-splash --no-main-window --disable-settings --ignore-slicerrc --python-code \
  'import vtk; rw = vtk.vtkRenderWindow(); rw.SetOffScreenRendering(1); rw.Render(); print("SLICER_MCP_GL_OK", rw.GetClassName(), bool(rw.SupportsOpenGL())); exit()' 2>&1)" || true
if printf '%s' "$OUT" | grep -q "SLICER_MCP_GL_OK .* True"; then
  echo "  ok: $(printf '%s' "$OUT" | grep SLICER_MCP_GL_OK)"
else
  printf '%s\n' "$OUT" | tail -20
  die "Slicer did not get an OpenGL context under Xvfb (output above)"
fi

cat <<EOF

Done. Next (from $REPO):
  SLICER_MCP_RUNTIME=config/runtime.local.json uv run python tools/xvfb_session.py -- bash
  uv run slicer-mcp launch --runtime config/runtime.local.json --no-main-window
  uv run slicer-mcp case prepare --runtime config/runtime.local.json --volume <ct file>   # prints case_id, lease
  uv run python examples/scripted_episode.py <out dir> <case_id> slicer_mcp/ -- \\
      uv run slicer-mcp serve --runtime config/runtime.local.json
  uv run slicer-mcp launch --stop --runtime config/runtime.local.json
EOF
