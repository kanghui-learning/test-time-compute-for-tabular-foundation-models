#!/usr/bin/env bash
# Install the three patched backbones (tabpfn, tabicl, tabfm) and the pinned TabArena
# benchmark packages into the CURRENT Python environment.
#
# For each backbone this script:
#   1. clones the public upstream repository into $THIRD_PARTY_DIR/<pkg>,
#   2. checks out the exact upstream base commit the paper used,
#   3. applies our patch series from patches/<pkg>/ with `git am` (authorship kept),
#   4. verifies the resulting source tree is bit-identical to the tree we ran
#      (git tree-hash check), and
#   5. runs `pip install -e`.
# TabArena (autogluon/tabarena) is checked out at a pinned commit and its two
# subpackages (packages/bencheval, packages/tabarena) are installed editable, unpatched.
#
# Usage:
#   bash scripts/install_forks.sh [--dir DIR] [--only "tabpfn tabicl tabfm tabarena"] [--no-pip]
#   THIRD_PARTY_DIR=/path PIP="python -m pip" bash scripts/install_forks.sh
#
# Idempotent: re-running on an already-installed checkout verifies the tree hash and
# re-runs `pip install -e` (cheap). A checkout that is dirty or on an unexpected tree
# is NOT modified; the script stops with an error so you can inspect it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="$REPO_ROOT/patches"
THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-$REPO_ROOT/third_party}"
PIP="${PIP:-python -m pip}"
ONLY="tabpfn tabicl tabfm tabarena"
DO_PIP=1

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) THIRD_PARTY_DIR="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --no-pip) DO_PIP=0; shift ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[install_forks] $*"; }

command -v git >/dev/null || die "git not found"
mkdir -p "$THIRD_PARTY_DIR"
THIRD_PARTY_DIR="$(cd "$THIRD_PARTY_DIR" && pwd)"

# name | upstream URL | base commit | expected tree after patch ("-" = no patch) | pip targets
#   The expected tree hash is the git tree of the exact fork commit used for the paper:
#     tabpfn  fork commit bf9b2c99590bdfe9f32bdd8ede77095ec5c5bb2c
#     tabicl  fork commit efda645f3dd620906a55a743d27a9b664299024c
#     tabfm   fork commit 78beef73fa74a60808e3ae9a0e90f5d1cad2edf6
SPECS=(
    "tabpfn|https://github.com/PriorLabs/TabPFN.git|95ff6772152eadc43e97b2a9d27899d44551eba8|f2094982e7858328c1cf7f7c39b9d8a3f8f0de88|."
    "tabicl|https://github.com/soda-inria/tabicl.git|46b91961db4f8873dd049ec09990698a435e1e29|2df5bffc050befbd0c77dae34f5aa4fab11d08be|."
    "tabfm|https://github.com/google-research/tabfm.git|2fb67e86477e0d2d3466baed00c2fc43053b0973|7abb81574e911a38affe36a4ce4902fd223218e9|.[pytorch]"
    "tabarena|https://github.com/autogluon/tabarena.git|abd24c7f06e4294e86efe069fe1cd057293a17d1|-|packages/bencheval packages/tabarena"
)

# Committer identity for `git am` (authorship comes from the patches themselves).
# Fixed dates make the resulting commit hashes reproducible across machines.
GIT_AM=(git -c user.name="tfm-ttc-release" -c user.email="tfm-ttc-release@localhost"
        -c core.autocrlf=false am --committer-date-is-author-date --keep-cr)

head_tree() { git -C "$1" rev-parse 'HEAD^{tree}'; }

prepare_checkout() {
    local name="$1" url="$2" base="$3" want_tree="$4"
    local dir="$THIRD_PARTY_DIR/$name"

    if [ -d "$dir/.git" ]; then
        log "$name: existing checkout at $dir"
        [ -z "$(git -C "$dir" status --porcelain --untracked-files=no)" ] \
            || die "$name: $dir has uncommitted changes; refusing to touch it"
        git -C "$dir" cat-file -e "$base^{commit}" 2>/dev/null \
            || git -C "$dir" fetch --quiet origin "$base" \
            || die "$name: base commit $base not available in $dir"
    elif [ -e "$dir" ]; then
        die "$name: $dir exists but is not a git checkout"
    else
        log "$name: cloning $url"
        git clone --quiet --filter=blob:none --no-checkout "$url" "$dir" \
            || die "$name: clone failed"
        git -C "$dir" cat-file -e "$base^{commit}" 2>/dev/null \
            || git -C "$dir" fetch --quiet origin "$base" \
            || die "$name: base commit $base not found upstream"
    fi

    if [ "$want_tree" = "-" ]; then
        # Unpatched package: just sit on the base commit.
        if [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)" != "$base" ]; then
            git -C "$dir" checkout --quiet --detach "$base" || die "$name: checkout $base failed"
        fi
        [ "$(git -C "$dir" rev-parse HEAD)" = "$base" ] || die "$name: HEAD is not $base"
        log "$name: at $base (no patch)"
        return
    fi

    # Already patched? (tree identical to the paper's fork commit)
    if git -C "$dir" rev-parse --verify --quiet HEAD >/dev/null \
        && [ "$(head_tree "$dir")" = "$want_tree" ]; then
        log "$name: already patched (tree $want_tree)"
        return
    fi

    local series=("$PATCH_DIR/$name"/*.patch)
    [ -e "${series[0]}" ] || die "$name: no patches found in $PATCH_DIR/$name"

    log "$name: checking out base $base and applying ${#series[@]} patches"
    git -C "$dir" checkout --quiet -B ttc-release "$base" || die "$name: checkout $base failed"
    (cd "$dir" && "${GIT_AM[@]}" --quiet "${series[@]}") || {
        git -C "$dir" am --abort >/dev/null 2>&1 || true
        die "$name: git am failed (patches do not apply to $base)"
    }
    local got
    got="$(head_tree "$dir")"
    [ "$got" = "$want_tree" ] \
        || die "$name: patched tree $got != expected $want_tree (source differs from the paper's fork)"
    log "$name: patched OK (tree $got)"
}

pip_install() {
    local name="$1" targets="$2"
    local dir="$THIRD_PARTY_DIR/$name" t args=()
    for t in $targets; do
        if [ "$t" = "." ] || [[ "$t" == .\[* ]]; then
            args+=(-e "$dir${t#.}")
        else
            args+=(-e "$dir/$t")
        fi
    done
    log "$name: $PIP install ${args[*]}"
    $PIP install "${args[@]}" || die "$name: pip install failed"
}

for spec in "${SPECS[@]}"; do
    IFS='|' read -r name url base want_tree targets <<<"$spec"
    case " $ONLY " in *" $name "*) ;; *) continue ;; esac
    prepare_checkout "$name" "$url" "$base" "$want_tree"
    [ "$DO_PIP" = 1 ] && pip_install "$name" "$targets"
done

log "done (third-party dir: $THIRD_PARTY_DIR)"
