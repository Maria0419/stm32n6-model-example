#!/usr/bin/env bash

set -euo pipefail

resolve_path() {
  local source=$1
  local dir

  while [[ -h "$source" ]]; do
    dir=$(cd -P -- "$(dirname -- "$source")" && pwd)
    source=$(readlink -- "$source")
    [[ "$source" != /* ]] && source=$dir/$source
  done

  if [[ "$source" != /* ]]; then
    source=$(cd -P -- "$(dirname -- "$source")" && pwd)/$(basename -- "$source")
  fi

  printf '%s\n' "$source"
}

SCRIPT_PATH=$(resolve_path "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd -P -- "$(dirname -- "$SCRIPT_PATH")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)

FSBL_FLASH_ADDR_DEFAULT="0x70000000"
APP_FLASH_ADDR_DEFAULT="0x70100000"
WORK_DIR_DEFAULT="$PROJECT_DIR/out/deploy"
WORK_DIR="${WORK_DIR:-$WORK_DIR_DEFAULT}"
FSBL_FLASH_ADDR="${FSBL_FLASH_ADDR:-$FSBL_FLASH_ADDR_DEFAULT}"
APP_FLASH_ADDR="${APP_FLASH_ADDR:-$APP_FLASH_ADDR_DEFAULT}"
CONNECT_ARGS="${CONNECT_ARGS:-port=SWD mode=HOTPLUG}"
PROGRAMMER_CLI="${PROGRAMMER_CLI:-}"
SIGNING_TOOL_CLI="${SIGNING_TOOL_CLI:-}"
EXTERNAL_LOADER="${EXTERNAL_LOADER:-}"
FSBL_TRUSTED_BIN="${FSBL_TRUSTED_BIN:-$WORK_DIR/FSBL-trusted.bin}"
APP_TRUSTED_BIN="${APP_TRUSTED_BIN:-$WORK_DIR/Appli-trusted.bin}"

usage() {
  cat <<'EOF'
Usage:
  deploy_signed.sh

Flow used by this command:
  1. Find the latest FSBL and Appli .bin artifacts.
  2. Add the STM32N6 trusted header with STM32_SigningTool_CLI.
  3. Program both images into XSPI with the STM32N6570-DK external loader.
  4. Reset the target.

Requirements:
  - Board in DEV boot mode while programming.
  - BOOT1 HIGH before flashing.
  - External loader for MX66UW1G45G available.

Environment overrides:
  FSBL_BIN          Explicit FSBL BIN path.
  APP_BIN           Explicit Appli BIN path.
  FSBL_TRUSTED_BIN  Explicit output path for trusted FSBL.
  APP_TRUSTED_BIN   Explicit output path for trusted Appli.
  PROGRAMMER_CLI    Explicit STM32_Programmer_CLI path.
  SIGNING_TOOL_CLI  Explicit STM32_SigningTool_CLI path.
  EXTERNAL_LOADER   Explicit .stldr path.
  CONNECT_ARGS      STM32_Programmer_CLI connect args.
  FSBL_FLASH_ADDR   FSBL flash address. Default: 0x70000000
  APP_FLASH_ADDR    Appli flash address. Default: 0x70100000
  WORK_DIR          Output directory for generated trusted binaries.
EOF
}

log() {
  printf '[deploy] %s\n' "$*"
}

die() {
  printf '[deploy] error: %s\n' "$*" >&2
  exit 1
}

resolve_programmer_cli() {
  local candidate

  if [[ -n "$PROGRAMMER_CLI" ]]; then
    [[ -x "$PROGRAMMER_CLI" ]] || die "PROGRAMMER_CLI is not executable: $PROGRAMMER_CLI"
    return
  fi

  if candidate=$(command -v STM32_Programmer_CLI 2>/dev/null); then
    PROGRAMMER_CLI="$candidate"
    return
  fi

  for candidate in \
    /home/maria/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI \
    /usr/local/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI \
    /opt/st/stm32cubeprogrammer/bin/STM32_Programmer_CLI
  do
    if [[ -x "$candidate" ]]; then
      PROGRAMMER_CLI="$candidate"
      return
    fi
  done

  die "STM32_Programmer_CLI not found; set PROGRAMMER_CLI or add it to PATH"
}

resolve_signing_tool_cli() {
  local candidate

  if [[ -n "$SIGNING_TOOL_CLI" ]]; then
    [[ -x "$SIGNING_TOOL_CLI" ]] || die "SIGNING_TOOL_CLI is not executable: $SIGNING_TOOL_CLI"
    return
  fi

  for candidate in \
    /home/maria/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_SigningTool_CLI \
    /usr/local/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_SigningTool_CLI
  do
    if [[ -x "$candidate" ]]; then
      SIGNING_TOOL_CLI="$candidate"
      return
    fi
  done

  die "STM32_SigningTool_CLI not found; set SIGNING_TOOL_CLI"
}

resolve_external_loader() {
  local candidate

  if [[ -n "$EXTERNAL_LOADER" ]]; then
    [[ -f "$EXTERNAL_LOADER" ]] || die "EXTERNAL_LOADER not found: $EXTERNAL_LOADER"
    return
  fi

  for candidate in \
    "$PROJECT_DIR/MX66UW1G45G_STM32N6570-DK.stldr" \
    /home/maria/STMicroelectronics/STM32Cube/STM32CubeProgrammer/api/lib/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr
  do
    if [[ -f "$candidate" ]]; then
      EXTERNAL_LOADER="$candidate"
      return
    fi
  done

  die "STM32N6570-DK external loader not found; set EXTERNAL_LOADER"
}

find_latest_file() {
  local pattern=$1

  find "$PROJECT_DIR" -type f -path "$pattern" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n1 \
    | cut -d' ' -f2-
}

resolve_artifacts() {
  FSBL_BIN="${FSBL_BIN:-$(find_latest_file '*/STM32CubeIDE/FSBL/*/*FSBL.bin')}"
  APP_BIN="${APP_BIN:-$(find_latest_file '*/STM32CubeIDE/Appli/*/*Appli.bin')}"

  [[ -n "$FSBL_BIN" && -f "$FSBL_BIN" ]] || die "FSBL artifact not found. Set FSBL_BIN."
  [[ -n "$APP_BIN" && -f "$APP_BIN" ]] || die "Appli artifact not found. Set APP_BIN."
}

ensure_work_dir() {
  local requested_work_dir=$WORK_DIR
  local fallback_work_dir="${TMPDIR:-/tmp}/stm32n6-deploy-${USER:-user}"

  if mkdir -p -- "$WORK_DIR" 2>/dev/null && [[ -w "$WORK_DIR" ]]; then
    return
  fi

  WORK_DIR="$fallback_work_dir"
  mkdir -p -- "$WORK_DIR"

  if [[ "$FSBL_TRUSTED_BIN" == "$requested_work_dir/FSBL-trusted.bin" ]]; then
    FSBL_TRUSTED_BIN="$WORK_DIR/FSBL-trusted.bin"
  fi
  if [[ "$APP_TRUSTED_BIN" == "$requested_work_dir/Appli-trusted.bin" ]]; then
    APP_TRUSTED_BIN="$WORK_DIR/Appli-trusted.bin"
  fi

  log "WORK_DIR not writable, using $WORK_DIR"
}

make_trusted_image() {
  local input_bin=$1
  local output_bin=$2
  local label=$3

  log "Adding STM32N6 header to $label"
  "$SIGNING_TOOL_CLI" \
    -bin "$input_bin" \
    -nk \
    -of 0x80000000 \
    -t fsbl \
    -o "$output_bin" \
    -hv 2.3 \
    -align

  [[ -f "$output_bin" ]] || die "trusted image was not created: $output_bin"
}

run_programmer() {
  if [[ ${EUID:-$(id -u)} -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
    sudo -E "$PROGRAMMER_CLI" "$@"
    return
  fi

  "$PROGRAMMER_CLI" "$@"
}

flash_image() {
  local image=$1
  local addr=$2
  local label=$3
  local -a connect_args=()

  read -r -a connect_args <<< "$CONNECT_ARGS"

  log "Flashing $label at $addr"
  run_programmer -c "${connect_args[@]}" -el "$EXTERNAL_LOADER" -w "$image" "$addr" -v
}

case "${1:-}" in
  "" )
    ;;
  --help|-h)
    usage
    exit 0
    ;;
  *)
    die "deploy_signed.sh does not accept options; run it with no arguments"
    ;;
esac

resolve_artifacts
ensure_work_dir
resolve_signing_tool_cli
resolve_external_loader
resolve_programmer_cli

make_trusted_image "$FSBL_BIN" "$FSBL_TRUSTED_BIN" "FSBL"
make_trusted_image "$APP_BIN" "$APP_TRUSTED_BIN" "Appli"

flash_image "$FSBL_TRUSTED_BIN" "$FSBL_FLASH_ADDR" "FSBL"
flash_image "$APP_TRUSTED_BIN" "$APP_FLASH_ADDR" "Appli"

log "Resetting target"
read -r -a connect_args <<< "$CONNECT_ARGS"
run_programmer -c "${connect_args[@]}" -rst
log "Done"
