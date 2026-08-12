#!/usr/bin/env bash
# Install the pre-commit hook that keeps restricted data out of git.
#
#   bash coding_bench/scripts/install_hooks.sh
#
# The same check runs in CI, so the hook is a convenience rather than the
# defence. Anyone can skip a local hook with --no-verify, nobody can skip CI.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
hooks_dir="$(git rev-parse --git-path hooks)"
hook_path="${hooks_dir}/pre-commit"

if [ -f "${hook_path}" ] && ! grep -q "check_restricted.py" "${hook_path}"; then
    echo "A pre-commit hook already exists at ${hook_path} and does not call the"
    echo "restricted data check. Merge them by hand rather than losing one."
    exit 1
fi

mkdir -p "${hooks_dir}"
cat > "${hook_path}" <<'HOOK'
#!/usr/bin/env bash
# Refuse to commit restricted data. Installed by coding_bench/scripts/install_hooks.sh
set -euo pipefail
repo_root="$(git rev-parse --show-toplevel)"
python3 "${repo_root}/coding_bench/scripts/check_restricted.py" --staged
HOOK

chmod +x "${hook_path}"
echo "Installed ${hook_path}"
python3 "${repo_root}/coding_bench/scripts/check_restricted.py" --staged || true
