# Sourced by EAT before anything runs (and by eat/eval.sh).
export EAT_GPUS="{{GPU}}"
export KB4090_RIG="{{RIG}}"
# Shorter official eval for the in-loop score (fewer seeds and timing rounds).
# The final, authoritative score comes from `kb4090 eat-final` without this overlay.
export KB4090_RIG_OVERLAY="{{OVERLAY}}"
# Activate the Python environment where kb4090 is installed, e.g.:
# source "$HOME/venvs/kb4090/bin/activate"
