# Match just's completion script to the executable on PATH, including when
# direnv changes it after compinit. Homebrew's dynamic completion loader can
# execute the default recipe when used with older Nix-provided just versions.
# Source after compinit; this file is safe to source more than once.
if (( ! $+functions[compdef] )); then
  return 0
fi

_datamunge_just_complete() {
  local executable=${commands[just]}
  [[ -n $executable ]] || return 1
  if [[ ${_datamunge_just_completion_binary-} != $executable ]]; then
    local script
    script=$("$executable" --completions zsh) || return 1
    # Both static and dynamic scripts register their real handler with compdef.
    # Run outside a function named _just so the script only installs itself.
    eval "$script"
    typeset -g _datamunge_just_completion_delegate=${_comps[just]}
    typeset -g _datamunge_just_completion_binary=$executable
    compdef _datamunge_just_complete just
  fi
  "$_datamunge_just_completion_delegate" "$@"
}

compdef _datamunge_just_complete just
