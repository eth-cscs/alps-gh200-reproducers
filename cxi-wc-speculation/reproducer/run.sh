#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-only
# Build wcprobe and run the reproducer phases, one NIC per phase, printing the
# kernel messages each phase produced.
#
# Usage (as root, on an idle Grace node with CXI NICs):
#   ./run.sh                      # phases: write spec spectgt read
#   PHASES="write spec" ./run.sh  # also available: prfm (NIC of write),
#                                 # specpage (NIC of spectgt, whole 64K page)
#   NICS="0 1 2 3" ./run.sh       # NIC used for write spec spectgt read
#   FORCE_AVOID_WC=1 ./run.sh     # only with the patched cxi_user (../patch)
#
# Root is needed for dmesg and /dev/kmsg markers; wcprobe itself is not
# privileged. Each phase runs on its own NIC so that an auto-masked error bit
# on one NIC does not hide the next phase.
set -u
D=$(cd "$(dirname "$0")" && pwd)
PHASES=${PHASES:-write spec spectgt read}
read -r -a N <<< "${NICS:-0 1 2 3}"
nic_of() {
	case $1 in
	write | prfm) echo "${N[0]}" ;;
	spec) echo "${N[1]}" ;;
	spectgt | specpage) echo "${N[2]}" ;;
	read) echo "${N[3]}" ;;
	esac
}

sec() { printf '\n== %s\n' "$*"; }
die() { echo "error: $*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "run as root (dmesg, /dev/kmsg)"

sec "system"
echo "host:   $(hostname)  kernel: $(uname -r)  page size: $(getconf PAGESIZE)"
echo "cxi_user srcversion: $(cat /sys/module/cxi_user/srcversion 2>/dev/null)"
rpm -q cray-cxi-driver-dkms cray-libcxi 2>/dev/null
P=/sys/module/cxi_user/parameters/force_avoid_wc
if [ -n "${FORCE_AVOID_WC:-}" ]; then
	[ -w $P ] || die "FORCE_AVOID_WC set but cxi_user has no force_avoid_wc (stock module)"
	echo "$FORCE_AVOID_WC" > $P
fi
[ -r $P ] && echo "force_avoid_wc: $(cat $P)" || echo "force_avoid_wc: n/a (stock cxi_user)"
for d in /sys/class/cxi/cxi*; do
	echo "${d##*/}: pci $(basename "$(readlink -f "$d/device")") numa $(cat "$d/device/numa_node") bar0 $(cut -d' ' -f1 < "$d/device/resource" | head -1) pi_ext mask $(cat "$d/device/err_flgs_irqa/pi_ext/mask")"
done
# /dev/cxi* users other than the per-NIC retry handlers (cxi_rh) would mix
# their own launch-page traffic into the results
others=0
for p in /proc/[0-9]*; do
	for f in "$p"/fd/*; do
		case $(readlink "$f" 2>/dev/null) in
		/dev/cxi[0-9]*)
			comm=$(cat "$p/comm" 2>/dev/null)
			echo "/dev/cxi* user: pid ${p#/proc/} $comm"
			[ "$comm" = cxi_rh ] || others=$((others + 1))
			break ;;
		esac
	done
done
[ "$others" = 0 ] || echo "warning: $others process(es) other than cxi_rh use /dev/cxi*; results may include their accesses"

sec "build"
cflags=$(pkg-config --cflags libcxi 2>/dev/null)
libs=$(pkg-config --libs libcxi 2>/dev/null)
if [ -z "$cflags" ]; then	# libcxi.h is in /usr/include/libcxi on SHS installs
	h=$(find /usr/include /usr/local/include /opt/cray -name libcxi.h 2>/dev/null | head -1)
	[ -n "$h" ] || die "libcxi.h not found (install cray-libcxi-devel)"
	cflags="-I$(dirname "$h")"
fi
[ -n "$libs" ] || libs=-lcxi
echo "cflags: $cflags  libs: $libs"
gcc -O2 -g -Wall -o "$D/wcprobe" "$D/wcprobe.c" $cflags $libs || die "build failed"
echo "ok"

run_phase() {	# name nic wcprobe-args...
	local name=$1 nic=$2; shift 2
	local dev=/sys/class/cxi/cxi$nic/device
	[ -d "$dev" ] || { sec "$name: no cxi$nic, skipped"; return; }
	local mask; mask=$(cat "$dev/err_flgs_irqa/pi_ext/mask")
	sec "$name on cxi$nic: wcprobe --dev $nic $*"
	if (( 0x${mask##*,} & 0x1000 )); then
		echo "skipped: pri_mb_xlat_pf_addr_error (bit 12) already masked on cxi$nic (reboot to clear)"
		return
	fi
	local bind=()
	command -v numactl > /dev/null &&
		bind=(numactl --cpunodebind="$(cat "$dev/numa_node")" --membind="$(cat "$dev/numa_node")")
	local before; before=$(dmesg | wc -l)
	echo "wcprobe: BEGIN $name cxi$nic" > /dev/kmsg
	timeout 300 "${bind[@]}" "$D/wcprobe" --dev "$nic" "$@"
	echo "wcprobe exit: $?"
	sleep 3
	echo "wcprobe: END $name cxi$nic" > /dev/kmsg
	echo "-- kernel messages:"
	dmesg | tail -n +"$((before + 1))" | grep -v 'wcprobe: '
	local n; n=$(dmesg | tail -n +"$((before + 1))" | grep -c 'pri_mb_xlat_pf_addr_error')
	echo "-- pri_mb_xlat_pf_addr_error: $n"
	RESULTS+=("$name cxi$nic: $n")
}

RESULTS=()
for ph in $PHASES; do
	nic=$(nic_of "$ph")
	case $ph in
	write)   run_phase write   "$nic" --mode write --iters 200000 ;;
	spec)    run_phase spec    "$nic" --mode spec --iters 100000 --lines 16 ;;
	spectgt) run_phase spectgt "$nic" --mode spec --iters 100000 --lines 16 --tgt ;;
	specpage) run_phase specpage "$nic" --mode spec --iters 100000 --lines 1024 ;;
	prfm)    run_phase prfm    "$nic" --mode prfm --iters 2000 --lines 1024 ;;
	read)    run_phase read    "$nic" --mode read --width 8 --iters 1 --lines 1 --offset 0x40 ;;
	*) echo "unknown phase: $ph" ;;
	esac
done

sec "summary (pri_mb_xlat_pf_addr_error per phase; logged counts are rate limited)"
printf '%s\n' "${RESULTS[@]}"
