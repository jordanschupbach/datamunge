#!/usr/bin/env bash
# Commit a generated documentation site to origin/gh-pages without switching branches.
set -euo pipefail

site=$(cd "${1:?Usage: publish_docs_branch.sh SITE_DIRECTORY}" && pwd -P)
test -s "$site/index.html"
touch "$site/.nojekyll"
git_dir=$(git rev-parse --absolute-git-dir)
source_revision=$(git rev-parse HEAD)
branch=refs/heads/gh-pages

parent=()
remote_ref=$(git ls-remote --refs origin "$branch")
if [[ -n "$remote_ref" ]]; then
  git fetch --no-tags origin "$branch"
  parent=(-p "$(git rev-parse FETCH_HEAD)")
fi

scratch=$(mktemp -d)
cleanup() {
  if [[ -f "$scratch/index" ]]; then unlink "$scratch/index"; fi
  rmdir "$scratch"
}
trap cleanup EXIT
tree=$(
  cd "$site"
  export GIT_INDEX_FILE="$scratch/index"
  git --git-dir="$git_dir" --work-tree="$site" add --force --all -- .
  git --git-dir="$git_dir" write-tree
)

if (( ${#parent[@]} )) && [[ $(git rev-parse "${parent[1]}^{tree}") == "$tree" ]]; then
  echo 'gh-pages already contains this site.'
  exit 0
fi

commit=$(git commit-tree "$tree" "${parent[@]}" -m "Publish documentation from $source_revision")
git push origin "$commit:$branch"
