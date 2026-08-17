#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TVM_HOME="${TVM_HOME:-${HOME}/tvm}"
TVM_VENV="${TVM_VENV:-${HOME}/tvm_venv}"
TVM_REPOSITORY="https://github.com/DLiao0311/tvm.git"
TVM_BRANCH="pr-13770"
TVM_COMMIT="735c39a665887fdd9a1c5700a66b7904943d4d3e"

log() {
  printf '\n==> %s\n' "$*"
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

if [ ! -r /etc/os-release ]; then
  fail "cannot identify the operating system"
fi

# shellcheck disable=SC1091
source /etc/os-release
if [ "${ID:-}" != "ubuntu" ] || [ "${VERSION_ID:-}" != "20.04" ]; then
  fail "this setup is validated only for Ubuntu 20.04; found ${PRETTY_NAME:-unknown}"
fi

if [ "$(id -u)" -eq 0 ]; then
  SUDO=()
else
  command -v sudo >/dev/null 2>&1 || fail "sudo is required to install system packages"
  SUDO=(sudo)
fi

missing_packages() {
  local package
  for package in "$@"; do
    if ! dpkg-query -W -f='${db:Status-Abbrev}' "${package}" 2>/dev/null | grep -q '^ii'; then
      printf '%s\n' "${package}"
    fi
  done
}

BASE_PACKAGES=(
  build-essential
  ca-certificates
  git
  gnupg
  python3-dev
  python3-pip
  python3-venv
  wget
  gcc-riscv64-linux-gnu
  g++-riscv64-linux-gnu
)

mapfile -t BASE_MISSING < <(missing_packages "${BASE_PACKAGES[@]}")
if [ "${#BASE_MISSING[@]}" -gt 0 ]; then
  log "Installing missing Ubuntu packages: ${BASE_MISSING[*]}"
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y --no-install-recommends "${BASE_MISSING[@]}"
else
  log "Required base Ubuntu packages are already installed"
fi

LLVM_PACKAGES=(llvm-14 llvm-14-dev clang-14)
mapfile -t LLVM_MISSING < <(missing_packages "${LLVM_PACKAGES[@]}")
if [ "${#LLVM_MISSING[@]}" -gt 0 ]; then
  log "Configuring the LLVM 14 package repository"
  LLVM_KEY_TEMP="$(mktemp)"
  trap 'rm -f "${LLVM_KEY_TEMP:-}"' EXIT
  wget -qO "${LLVM_KEY_TEMP}" https://apt.llvm.org/llvm-snapshot.gpg.key
  "${SUDO[@]}" install -m 0644 "${LLVM_KEY_TEMP}" /etc/apt/trusted.gpg.d/apt.llvm.org.asc
  printf '%s\n' \
    'deb https://apt.llvm.org/focal/ llvm-toolchain-focal-14 main' \
    | "${SUDO[@]}" tee /etc/apt/sources.list.d/llvm14.list >/dev/null
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y --no-install-recommends "${LLVM_MISSING[@]}"
  rm -f "${LLVM_KEY_TEMP}"
  trap - EXIT
fi

command -v llvm-config-14 >/dev/null 2>&1 || fail "llvm-config-14 is not available in PATH"
LLVM_VERSION="$(llvm-config-14 --version)"
case "${LLVM_VERSION}" in
  14.*) ;;
  *) fail "LLVM 14 is required, but llvm-config-14 reports ${LLVM_VERSION}" ;;
esac

log "Creating or updating the Python environment at ${TVM_VENV}"
if [ ! -e "${TVM_VENV}" ]; then
  python3 -m venv "${TVM_VENV}"
elif [ ! -x "${TVM_VENV}/bin/python" ]; then
  fail "${TVM_VENV} exists but is not a usable Python virtual environment"
fi

VENV_PYTHON_VERSION="$("${TVM_VENV}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "${VENV_PYTHON_VERSION}" != "3.8" ]; then
  fail "${TVM_VENV} uses Python ${VENV_PYTHON_VERSION}; this setup requires Python 3.8"
fi

"${TVM_VENV}/bin/python" -m pip install \
  pip==25.0.1 \
  setuptools==75.3.4 \
  wheel==0.45.1 \
  cmake==4.4.2
"${TVM_VENV}/bin/python" -m pip install -r "${PROJECT_ROOT}/requirements.txt"

log "Preparing the pinned TVM fork at ${TVM_HOME}"
if [ ! -e "${TVM_HOME}" ]; then
  git clone --recursive --branch "${TVM_BRANCH}" "${TVM_REPOSITORY}" "${TVM_HOME}"
elif [ ! -d "${TVM_HOME}/.git" ]; then
  fail "${TVM_HOME} already exists but is not a Git checkout"
fi

TVM_ORIGIN="$(git -C "${TVM_HOME}" remote get-url origin 2>/dev/null || true)"
case "${TVM_ORIGIN}" in
  http://github.com/DLiao0311/tvm.git|https://github.com/DLiao0311/tvm.git|git@github.com:DLiao0311/tvm.git) ;;
  *) fail "${TVM_HOME} does not use the expected TVM fork: ${TVM_ORIGIN:-no origin}" ;;
esac

TVM_LOCAL_EXCLUDE="${TVM_HOME}/.git/info/exclude"
if ! grep -qxF '/build_host/' "${TVM_LOCAL_EXCLUDE}"; then
  printf '/build_host/\n' >> "${TVM_LOCAL_EXCLUDE}"
fi

CURRENT_TVM_COMMIT="$(git -C "${TVM_HOME}" rev-parse HEAD)"
if [ "${CURRENT_TVM_COMMIT}" != "${TVM_COMMIT}" ]; then
  if [ -n "$(git -C "${TVM_HOME}" status --porcelain --untracked-files=no)" ]; then
    fail "${TVM_HOME} has tracked local changes; preserve them before switching TVM revisions"
  fi
  git -C "${TVM_HOME}" fetch origin "${TVM_COMMIT}"
  git -C "${TVM_HOME}" checkout --detach "${TVM_COMMIT}"
fi
git -C "${TVM_HOME}" submodule update --init --recursive

log "Building TVM in ${TVM_HOME}/build_host"
export PATH="${TVM_VENV}/bin:${PATH}"
if [ -e "${TVM_HOME}/build_host/config.cmake" ]; then
  fail "${TVM_HOME}/build_host/config.cmake overrides the pinned root config.cmake; preserve and remove it before retrying"
fi
cmake -S "${TVM_HOME}" -B "${TVM_HOME}/build_host"
grep -q '^LLVM_DIR:PATH=.*/llvm-14/' "${TVM_HOME}/build_host/CMakeCache.txt" \
  || fail "CMake did not select LLVM 14; inspect ${TVM_HOME}/build_host/CMakeCache.txt"
cmake --build "${TVM_HOME}/build_host" --parallel "$(nproc)"

test -f "${TVM_HOME}/build_host/libtvm.so" || fail "TVM build did not produce libtvm.so"
test -f "${TVM_HOME}/build_host/libtvm_runtime.so" || fail "TVM build did not produce libtvm_runtime.so"

export TVM_LIBRARY_PATH="${TVM_HOME}/build_host"
export PYTHONPATH="${TVM_HOME}/python${PYTHONPATH:+:${PYTHONPATH}}"
export TVM_HOME
"${TVM_VENV}/bin/python" -m pip install -e "${TVM_HOME}/python"

ENVIRONMENT_LOCAL="${PROJECT_ROOT}/configs/environment.local"
if [ ! -e "${ENVIRONMENT_LOCAL}" ]; then
  log "Creating ${ENVIRONMENT_LOCAL}"
  {
    printf 'export TVM_HOME=%q\n' "${TVM_HOME}"
    printf 'export TVM_LIBRARY_PATH="${TVM_HOME}/build_host"\n'
    printf 'export PYTHONPATH="${TVM_HOME}/python${PYTHONPATH:+:${PYTHONPATH}}"\n'
    printf '\n'
    printf 'export DIM=16\n'
    printf 'export SCRATCHPAD_KB=512\n'
    printf 'export ACCUMULATOR_KB=256\n'
  } > "${ENVIRONMENT_LOCAL}"
else
  log "Keeping existing ${ENVIRONMENT_LOCAL}"
fi

VENV_ACTIVATE="${TVM_VENV}/bin/activate"
ACTIVATE_MARKER="# tvm-gemmini-fpga-deployment environment"
if ! grep -qxF "${ACTIVATE_MARKER}" "${VENV_ACTIVATE}"; then
  log "Linking the project environment to ${VENV_ACTIVATE}"
  {
    printf '\n%s\n' "${ACTIVATE_MARKER}"
    printf 'if [ -f %q ]; then\n' "${ENVIRONMENT_LOCAL}"
    printf '    . %q\n' "${ENVIRONMENT_LOCAL}"
    printf 'fi\n'
  } >> "${VENV_ACTIVATE}"
fi

log "Verifying the completed environment"
command -v riscv64-linux-gnu-gcc >/dev/null 2>&1 \
  || fail "riscv64-linux-gnu-gcc is not available in PATH"

"${TVM_VENV}/bin/python" - <<'PY'
import os
from pathlib import Path

import tvm
from tvm._ffi.base import _LIB

expected_source = Path(os.environ["TVM_HOME"]).resolve() / "python" / "tvm"
loaded_source = Path(tvm.__file__).resolve().parent
loaded_library = Path(_LIB._name).resolve()
expected_library = Path(os.environ["TVM_LIBRARY_PATH"]).resolve() / "libtvm.so"

if loaded_source != expected_source:
    raise RuntimeError(f"loaded TVM Python package from {loaded_source}, expected {expected_source}")
if loaded_library != expected_library:
    raise RuntimeError(f"loaded TVM library from {loaded_library}, expected {expected_library}")

print(f"TVM Python: {loaded_source}")
print(f"TVM library: {loaded_library}")
PY

printf '\nSetup complete. Load the environment with:\n'
printf '  source %q\n' "${TVM_VENV}/bin/activate"
