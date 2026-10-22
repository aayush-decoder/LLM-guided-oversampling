"""Shared engine used by every method and version. Contains NO method logic:
data protocol, serialisation, the LLM switch, baselines, evaluation, run
bookkeeping and figures. Any v1-vs-v2 difference therefore comes from the
method, never from how data was split or scored."""
