"""
Provider / model registry — the easy way to add an endpoint, model id and context
window once, then refer to it by a short alias instead of repeating
--base-url / --model / --api-key on every run.

Config lives in `providers.json` next to this file (override the path with the
$BENCH_PROVIDERS env var). It has two maps:

  "providers": alias -> { base_url, api_key_env?, api_key? }
  "models":    alias -> { provider, model, context_window?, max_tokens? }

  * api_key_env names an environment variable to read the key from.
  * api_key is a literal fallback (handy for local servers that want a dummy key).
  * context_window / max_tokens are metadata; max_tokens, if set, becomes the
    default completion budget for that model.

Nothing here is mandatory: `run --base-url ... --model <raw id>` still works with
no config file at all, and a raw model id with no provider falls back to a local
Ollama endpoint (the previous default).
"""

import json
import os
import urllib.parse

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "providers.json")


def config_path() -> str:
    return os.environ.get("BENCH_PROVIDERS", _DEFAULT_PATH)


def load(path=None) -> dict:
    """Load the registry; returns {'providers': {...}, 'models': {...}} (empty if no file)."""
    path = path or config_path()
    try:
        with open(path) as f:
            reg = json.load(f)
    except FileNotFoundError:
        reg = {}
    except json.JSONDecodeError as e:
        raise SystemExit(f"providers config {path} is not valid JSON: {e}")
    reg.setdefault("providers", {})
    reg.setdefault("models", {})
    return reg


def _resolve_key(prov: dict, cli_key) -> str:
    if cli_key:
        return cli_key
    env = prov.get("api_key_env")
    if env and os.environ.get(env):
        return os.environ[env]
    if prov.get("api_key"):
        return prov["api_key"]
    return os.environ.get("OPENAI_API_KEY", "")


def resolve(reg, model_arg, provider_arg=None, base_url_arg=None,
            api_key_arg=None, default_provider=None) -> dict:
    """
    Turn CLI args into a concrete endpoint.

    A --model that matches a model alias supplies the model id, its provider and
    context window. --provider / --base-url / --api-key always override the
    registry. A raw model id with neither provider nor base_url falls back to
    `default_provider`.

    Returns {base_url, model, api_key, context_window, max_tokens,
             provider, capabilities}.
    """
    models = reg.get("models", {})
    providers = reg.get("providers", {})

    if not model_arg:
        raise ValueError("specify --model: an alias from `cli.py models`, or a raw "
                         "model id together with --provider or --base-url.")

    entry = models.get(model_arg, {})            # {} when model_arg is a raw id
    model_id = entry.get("model", model_arg)
    context_window = entry.get("context_window")
    max_tokens = entry.get("max_tokens")

    prov_name = provider_arg or entry.get("provider")
    if not prov_name and not base_url_arg and default_provider:
        prov_name = default_provider
    prov = {}
    if prov_name:
        prov = providers.get(prov_name)
        if prov is None:
            known = ", ".join(providers) or "(none configured)"
            raise ValueError(f"unknown provider '{prov_name}'. Known providers: {known}")

    base_url = base_url_arg or prov.get("base_url")
    if not base_url:
        raise ValueError(
            f"no endpoint for model '{model_arg}'. Add it to "
            f"{os.path.basename(config_path())}, or pass --provider / --base-url.")

    capabilities = list(prov.get("capabilities", []))
    return {"base_url": base_url, "model": model_id,
            "api_key": _resolve_key(prov, api_key_arg),
            "context_window": context_window, "max_tokens": max_tokens,
            "provider": prov_name, "capabilities": capabilities}


def provider_name_from_url(base_url: str) -> str:
    """Derive a short, readable provider alias from an endpoint URL.

    https://api.openai.com/v1 -> 'openai', https://openrouter.ai/api/v1 ->
    'openrouter', http://localhost:11434/v1 -> 'local'. Cosmetic only — the
    user mostly refers to the model alias, not the provider.
    """
    host = urllib.parse.urlparse(base_url).hostname or ""
    if not host or host == "localhost" or all(c.isdigit() or c == "." for c in host):
        return "local"
    labels = [l for l in host.split(".") if l not in ("api", "www")]
    if len(labels) > 1 and len(labels[-1]) <= 3:          # drop a trailing .com/.ai/.io
        labels = labels[:-1]
    name = (labels[-1] if labels else host).replace("-", "_")
    return name or "provider"


def register_model(reg, *, alias, base_url, model_id, api_key=None,
                   api_key_env=None, context_window=None, max_tokens=None):
    """Add (or update) a provider + model in the registry dict.

    Reuses an existing provider that already points at the same base_url so
    repeatedly adding models from one endpoint doesn't pile up duplicate
    providers; otherwise creates a provider named after the endpoint host.
    Returns (reg, provider_name). Pure aside from mutating `reg`.
    """
    reg.setdefault("providers", {})
    reg.setdefault("models", {})

    prov_name = next((n for n, p in reg["providers"].items()
                      if p.get("base_url") == base_url), None)
    if prov_name is None:
        prov_name = _unique_provider_name(provider_name_from_url(base_url), reg["providers"])

    prov = reg["providers"].setdefault(prov_name, {})
    prov["base_url"] = base_url
    if api_key_env:                                       # env reference wins; drop any literal
        prov["api_key_env"] = api_key_env
        prov.pop("api_key", None)
    elif api_key:
        prov["api_key"] = api_key
        prov.pop("api_key_env", None)

    model = {"provider": prov_name, "model": model_id}
    if context_window:
        model["context_window"] = int(context_window)
    if max_tokens:
        model["max_tokens"] = int(max_tokens)
    reg["models"][alias] = model
    return reg, prov_name


def _unique_provider_name(base, existing) -> str:
    if base not in existing:
        return base
    i = 2
    while f"{base}{i}" in existing:
        i += 1
    return f"{base}{i}"


def save(reg, path=None) -> str:
    """Write the registry to disk as pretty JSON. Returns the path written."""
    path = path or config_path()
    with open(path, "w") as f:
        json.dump(reg, f, indent=2)
        f.write("\n")
    return path
