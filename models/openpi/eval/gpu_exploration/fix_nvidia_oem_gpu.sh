#!/usr/bin/env bash
set -euo pipefail

# Repair the NVIDIA driver/kernel-module pairing for the current Ubuntu OEM kernel.
# Intended for the OpenPI control machine that is currently falling back to CPU.
#
# Examples:
#   bash models/openpi/run_scripts/fix_nvidia_oem_gpu.sh --dry-run
#   bash models/openpi/run_scripts/fix_nvidia_oem_gpu.sh
#   bash models/openpi/run_scripts/fix_nvidia_oem_gpu.sh --driver-pkg nvidia-driver-580-open

DRY_RUN=0
DRIVER_PKG=""

usage() {
    cat <<'EOF'
Usage:
  bash models/openpi/run_scripts/fix_nvidia_oem_gpu.sh [--dry-run] [--driver-pkg <pkg>]

Options:
  --dry-run             Show the apt install plan without changing the system.
  --driver-pkg <pkg>    Override the NVIDIA driver metapackage.
                        Default: recommended package from `ubuntu-drivers devices`,
                        else `nvidia-driver-595-open`.
  -h, --help            Show this help text.

This script:
  1. Detects the running kernel, for example `6.17.0-1020-oem`.
  2. Selects an NVIDIA driver package.
  3. Installs the matching prebuilt kernel module package for the running kernel.
  4. Prints the required reboot and validation steps.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --driver-pkg)
            DRIVER_PKG="${2:-}"
            if [[ -z "$DRIVER_PKG" ]]; then
                echo "ERROR: --driver-pkg requires a value" >&2
                exit 2
            fi
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v apt-cache >/dev/null 2>&1; then
    echo "ERROR: apt-cache not found; this script expects Ubuntu/Debian apt." >&2
    exit 1
fi

KERNEL="$(uname -r)"

pick_recommended_driver() {
    local recommended
    if command -v ubuntu-drivers >/dev/null 2>&1; then
        recommended="$(ubuntu-drivers devices 2>/dev/null | awk '/recommended/ {print $3; exit}')"
        if [[ -n "$recommended" ]]; then
            echo "$recommended"
            return 0
        fi
    fi
    echo "nvidia-driver-595-open"
}

detect_oem_track() {
    local pkg
    for pkg in \
        $(dpkg-query -W -f='${Package}\n' 'linux-image-oem-*' 2>/dev/null) \
        $(dpkg-query -W -f='${Package}\n' 'linux-headers-oem-*' 2>/dev/null)
    do
        case "$pkg" in
            linux-image-oem-*)
                echo "${pkg#linux-image-}"
                return 0
                ;;
            linux-headers-oem-*)
                echo "${pkg#linux-headers-}"
                return 0
                ;;
        esac
    done
    return 1
}

if [[ -z "$DRIVER_PKG" ]]; then
    DRIVER_PKG="$(pick_recommended_driver)"
fi

DRIVER_SUFFIX="${DRIVER_PKG#nvidia-driver-}"
if [[ "$DRIVER_SUFFIX" == "$DRIVER_PKG" ]]; then
    echo "ERROR: unsupported driver package name: $DRIVER_PKG" >&2
    exit 1
fi

MODULE_PKG="linux-modules-nvidia-${DRIVER_SUFFIX}-${KERNEL}"
TRACK_KIND="kernel"

if OEM_TRACK="$(detect_oem_track)"; then
    MODULE_PKG="linux-modules-nvidia-${DRIVER_SUFFIX}-${OEM_TRACK}"
    TRACK_KIND="oem-meta"
fi

policy_candidate() {
    apt-cache policy "$1" | awk '/Candidate:/ {print $2; exit}'
}

echo "kernel:       $KERNEL"
echo "driver_pkg:   $DRIVER_PKG"
echo "track_kind:   $TRACK_KIND"
echo "module_pkg:   $MODULE_PKG"
echo

echo "Current NVIDIA state:"
nvidia-smi -L || true
ls /dev/nvidia* 2>/dev/null || true
echo

echo "Refreshing apt metadata..."
sudo apt-get update
echo

DRIVER_CANDIDATE="$(policy_candidate "$DRIVER_PKG")"
MODULE_CANDIDATE="$(policy_candidate "$MODULE_PKG")"

if [[ -z "$DRIVER_CANDIDATE" || "$DRIVER_CANDIDATE" == "(none)" ]]; then
    echo "ERROR: apt does not know package $DRIVER_PKG" >&2
    exit 1
fi
if [[ -z "$MODULE_CANDIDATE" || "$MODULE_CANDIDATE" == "(none)" ]]; then
    echo "ERROR: apt does not know package $MODULE_PKG" >&2
    echo "Try overriding the driver package or check that the running kernel has a matching NVIDIA module package." >&2
    exit 1
fi

echo "APT policy:"
apt-cache policy "$DRIVER_PKG" "$MODULE_PKG"
echo

APT_INSTALL=(sudo apt-get install -y "$DRIVER_PKG" "$MODULE_PKG")
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run:"
    sudo apt-get -s install "$DRIVER_PKG" "$MODULE_PKG"
else
    echo "Installing:"
    "${APT_INSTALL[@]}"
fi

echo
echo "Next steps:"
echo "  1. Reboot the machine."
echo "  2. After reboot, verify:"
echo "       nvidia-smi -L"
echo "       ls /dev/nvidia*"
echo "       \$OPENPI_REPO/.venv/bin/python models/openpi/eval/diagnose_openpi_runtime.py"
echo "  3. Re-run the PI 0.5 benchmark once JAX reports GPU."
