"""
Model runner. Talks to any OpenAI-compatible /chat/completions endpoint, which
covers hosted APIs and every common local server (Ollama, vLLM, LM Studio,
llama.cpp, TGI). Set base_url + model + api_key.

An optional native Anthropic Messages API streaming path is triggered when the
resolved provider carries the `native_anthropic` capability flag. It captures
per-phase wall-clock timings from stream events and stores them alongside the
response; the OpenAI-compatible path is otherwise unchanged. Add the `oauth`
capability to authenticate via a logged-in OAuth profile (`ant auth login` /
Claude Code) instead of an API key.

Samples are drawn one request at a time (client-side), not via the server's `n`
parameter, because many local servers silently ignore `n` and return a single
completion — which would make pass@k vanish without warning.

A `mock` mode synthesizes answers without a server so you can test the whole
pipeline (and see that the metrics light up) before pointing at a real model.
Mock answers are deterministic per (item, sample) so test runs are reproducible.
"""

import sys
import json
import math
import time
import random
import hashlib
import subprocess
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

import grading
import storage

try:
    import anthropic
except Exception:                               # optional dependency
    anthropic = None

SYSTEM = "You are a careful reasoning assistant. Work through each problem step by step."

# OAuth on /v1/messages requires this beta header; the SDK does not add it itself.
_OAUTH_BETA = "oauth-2025-04-20"

# raw/parsed/correct/confidence/latency/prompt_tokens/completion_tokens
_ERROR_ROW = (storage.ERROR_MARKER, None, None, None, 0, None, None, None)

# errors worth retrying: transport problems and malformed/oddly-shaped responses.
# The native Anthropic path raises the SDK's own exception types (not urllib's), so
# fold in its transient ones — rate limits (429), overloaded/5xx, dropped connections
# (APITimeoutError subclasses APIConnectionError) — but never 4xx like BadRequest,
# which would only fail again.
_RETRYABLE_ANTHROPIC = (
    (anthropic.RateLimitError, anthropic.InternalServerError,
     anthropic.APIConnectionError) if anthropic is not None else ())
_RETRYABLE = (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
              OSError, ValueError, json.JSONDecodeError) + _RETRYABLE_ANTHROPIC


def _retry_backoff(err, attempt):
    """Seconds to wait before the next retry. Honor a server-sent Retry-After header
    when present (rate limits / overload give a real reset hint), capped so a stuck
    window can't stall the run; otherwise fall back to capped exponential backoff."""
    resp = getattr(err, "response", None)
    if resp is not None:
        try:
            return min(float(resp.headers.get("retry-after")), 60)
        except (TypeError, ValueError):
            pass
    return min(2 ** attempt, 8)


def _provider_has_capability(cfg, name):
    caps = cfg.get("capabilities") or []
    return name in caps



def _format_instructions(item, ask_confidence: bool) -> str:
    fmt = ("Show your reasoning, then end with a line in exactly this format:\n"
           "ANSWER: <your final answer>")
    if item["answer_type"] == "choice" and item.get("choices"):
        opts = ", ".join(item["choices"].split("|"))
        fmt += f"\nYour answer must be exactly one of: {opts}."
    if ask_confidence:
        fmt += "\nThen on the next line:\nCONFIDENCE: <an integer 0-100 for how certain you are>"
    return fmt


def build_messages(item, ask_confidence: bool):
    fmt = _format_instructions(item, ask_confidence)

    system_msg = {"role": "system", "content": SYSTEM}
    turns = item.get("turns")
    if turns and isinstance(turns, str):
        turns = turns.split("|") if turns else []
    if turns:
        messages = [system_msg]
        for idx, turn in enumerate(turns):
            content = turn
            messages.append({"role": "user", "content": content})
            # Insert an assistant turn between user turns so the conversation
            # alternates (system, user, assistant, user, ...) rather than
            # collapsing two consecutive user messages. It is a neutral
            # acknowledgement on purpose: the probe tests whether the model
            # re-reads turn-1's facts across the turn boundary, so echoing the
            # concrete state here would hand it the answer and defeat the probe.
            if idx < len(turns) - 1:
                messages.append({"role": "assistant",
                                 "content": "Understood — ready for the next instruction."})
        # Append the format instructions to the last user turn.
        messages[-1]["content"] = f"{messages[-1]['content']}\n\n{fmt}"
        return messages
    user = f"{item['prompt']}\n\n{fmt}"
    return [system_msg, {"role": "user", "content": user}]

def _to_anthropic_messages(messages):
    """Convert OpenAI-style message list to Anthropic format.

    Anthropic uses a top-level `system` parameter instead of a system message.
    """
    system = ""
    anthropic_messages = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            system = content
            continue
        if role == "user":
            anthropic_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            anthropic_messages.append({"role": "assistant", "content": content})
    return system, anthropic_messages


def _parse_anthropic_stream(stream):
    """
    Consume an Anthropic SDK message stream and return
    (text, prompt_tokens, completion_tokens, timings, think_tokens).

    `timings` is a dict of per-phase wall-clock times (seconds since request
    start): ttft, first_reasoning, answer_wall, total.
    """
    t0 = time.time()
    text_parts = []
    think_parts = []
    prompt_tokens = None
    completion_tokens = None
    think_tokens = 0

    ttft = None
    first_reasoning = None
    answer_started = False
    answer_wall = None

    for event in stream:
        now = time.time()
        if ttft is None:
            ttft = now - t0

        if event.type == "message_start":
            usage = event.message.usage
            prompt_tokens = getattr(usage, "input_tokens", None)

        elif event.type == "content_block_start":
            block = event.content_block
            if getattr(block, "type", None) == "thinking":
                if first_reasoning is None:
                    first_reasoning = now - t0
            elif getattr(block, "type", None) == "text":
                if not answer_started:
                    answer_started = True
                    answer_wall = now - t0

        elif event.type == "content_block_delta":
            delta = event.delta
            d_type = getattr(delta, "type", None)
            if d_type == "text_delta":
                if not answer_started:
                    answer_started = True
                    answer_wall = now - t0
                text_parts.append(delta.text)
            elif d_type == "thinking_delta":
                if first_reasoning is None:
                    first_reasoning = now - t0
                think_parts.append(delta.thinking)

        elif event.type == "message_delta":
            usage = event.usage
            completion_tokens = getattr(usage, "output_tokens", None)
            details = getattr(usage, "output_tokens_details", None)
            if details is not None:
                think_tokens = getattr(details, "thinking_tokens", 0) or 0

    total = time.time() - t0
    text = "".join(text_parts)
    timings = {
        "ttft": ttft,
        "first_reasoning": first_reasoning,
        "answer_wall": answer_wall,
        "total": total,
    }
    return text, prompt_tokens, completion_tokens, timings, think_tokens


def call_anthropic(api_key, model, messages, temperature, max_tokens, timeout,
                   oauth=False):
    """One native Anthropic streaming completion.

    Returns (text, prompt_tokens, completion_tokens, timings, think_tokens).
    Raises ValueError if the response is unusable.

    With oauth=True the client is built with no API key, so the SDK resolves the
    logged-in OAuth profile (`ant auth login` / Claude Code) and refreshes it
    automatically; the `anthropic-beta: oauth-2025-04-20` header that OAuth
    requires on /v1/messages is added explicitly. A set $ANTHROPIC_API_KEY in the
    environment outranks the profile in the SDK — unset it to use OAuth.
    """
    if anthropic is None:
        raise ValueError("anthropic package is required for the native Anthropic path")

    system, anthropic_messages = _to_anthropic_messages(messages)
    if oauth:
        try:
            client = anthropic.Anthropic(
                timeout=timeout, default_headers={"anthropic-beta": _OAUTH_BETA})
        except anthropic.AnthropicError as e:    # no resolvable OAuth credential
            raise ValueError(
                f"no Anthropic OAuth credential found -- run `ant auth login` "
                f"(or set $ANTHROPIC_AUTH_TOKEN): {e}")
    else:
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    # messages.stream() streams implicitly (no `stream` kwarg), and adaptive-thinking
    # models (Opus 4.8, Sonnet 4.6) reject `temperature` with a 400 — so neither is sent.
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
        "thinking": {"type": "adaptive", "display": "omitted"},
    }
    if system:
        params["system"] = system

    with client.messages.stream(**params) as stream:
        text, pt, ct, timings, think_tokens = _parse_anthropic_stream(stream)

    if not text:
        raise ValueError("Anthropic stream produced no text content")
    return text, pt, ct, timings, think_tokens


# A model invocation that fails partway (rate limit, overload) but is worth retrying.
# Subclasses ValueError so it lands in _RETRYABLE alongside the SDK/transport errors.
class ClaudeCliRetryable(ValueError):
    pass


def call_claude_cli(model, messages, timeout):
    """One completion via the Claude Code CLI (`claude -p`) instead of the HTTP API.

    Routes through the CLI's own subscription/OAuth path, which the raw /v1/messages
    OAuth token is rate-limited far more aggressively than. Returns
    (text, prompt_tokens, completion_tokens, timings, think_tokens) like call_anthropic.

    The bench's own system prompt replaces Claude Code's, and the dynamic
    (workspace/tool) system-prompt sections are excluded so the model sees a clean
    reasoning prompt rather than the coding-agent harness. Thinking-token counts are
    not exposed on this path, so think_tokens is 0.
    """
    system, amsgs = _to_anthropic_messages(messages)
    # claude -p takes a single prompt string. Single-turn items have one user message;
    # multi-turn conversations are flattened with role labels so history survives.
    if len(amsgs) == 1:
        prompt = amsgs[0]["content"]
    else:
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in amsgs)

    cmd = ["claude", "-p", prompt, "--model", model,
           "--output-format", "json", "--exclude-dynamic-system-prompt-sections"]
    if system:
        cmd += ["--system-prompt", system]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"claude -p timed out after {timeout}s") from e
    total = time.time() - t0

    if proc.returncode != 0:
        raise ClaudeCliRetryable(
            f"claude -p exited {proc.returncode}: {(proc.stderr or '').strip()[:300]}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise ValueError(f"claude -p returned non-JSON output: {proc.stdout[:300]}") from e

    if data.get("is_error") or data.get("api_error_status"):
        raise ClaudeCliRetryable(
            f"claude -p API error: {data.get('api_error_status') or data.get('subtype')}")

    text = (data.get("result") or "").strip()
    if not text:
        raise ValueError("claude -p produced no result text")

    usage = data.get("usage") or {}
    pt = usage.get("input_tokens")
    ct = usage.get("output_tokens")
    ttft = (data.get("ttft_ms") or 0) / 1000 or None
    timings = {"ttft": ttft, "first_reasoning": ttft, "answer_wall": ttft, "total": total}
    return text, pt, ct, timings, 0


def token_entropy_stats(content):
    """Per-token entropy stats from OpenAI-style logprobs `content` -- a list of
    {token, logprob, top_logprobs:[{token, logprob}, ...]}. This is the one Phase-3
    'latent friction' signal that IS observable, and only on a provider that returns
    logprobs (open-weights via vLLM/TGI; not hosted Anthropic). Entropy H(X) is taken
    over the renormalized top-k distribution per token. Returns None if no usable tokens.

    friction_transitions  : tokens whose entropy exceeds mean+2*std (high-friction
                            calc vs smooth memorized generation).
    logprob_divergence_spikes : tokens whose chosen-token logprob < -2.0 (the model
                            committed to a low-probability token -- a hard turn)."""
    entropies, divergence = [], 0
    for tok in content or []:
        tl = tok.get("top_logprobs") or []
        ps = [math.exp(e["logprob"]) for e in tl if e.get("logprob") is not None]
        z = sum(ps)
        if z <= 0:
            continue
        ps = [p / z for p in ps]
        entropies.append(-sum(p * math.log2(p) for p in ps if p > 0))
        if tok.get("logprob") is not None and tok["logprob"] < -2.0:
            divergence += 1
    if not entropies:
        return None
    mean = sum(entropies) / len(entropies)
    var = sum((h - mean) ** 2 for h in entropies) / len(entropies)
    sd = math.sqrt(var)
    friction = sum(1 for h in entropies if sd > 0 and h > mean + 2 * sd)
    return {
        "token_entropy_mean": mean,
        "token_entropy_max": max(entropies),
        "friction_transitions": friction,
        "logprob_divergence_spikes": divergence,
        "n_tokens_scored": len(entropies),
    }


def call_api(base_url, api_key, model, messages, temperature, max_tokens, timeout,
             want_logprobs=False):
    """One completion. Returns
    (text, prompt_tokens, completion_tokens, reasoning_tokens, logprob_stats).
    reasoning_tokens is the provider's own hidden-reasoning count from
    usage.completion_tokens_details.reasoning_tokens (o1/o3/DeepSeek-style), or
    None when the provider doesn't expose it. logprob_stats is None unless
    want_logprobs and the server returns logprobs. Raises ValueError if the
    response is not shaped like a chat completion."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    if want_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 5
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    # Some servers return HTTP 200 with an error/empty body; don't let that crash the run.
    choices = body.get("choices") if isinstance(body, dict) else None
    if not choices:
        raise ValueError(f"response has no choices: {str(body)[:200]}")
    text = (choices[0].get("message") or {}).get("content")
    if text is None:
        # Reasoning models spend completion tokens on hidden reasoning first; if
        # max_tokens runs out before any content, finish_reason is "length" and
        # content is null. Name the real cause so the fix (raise max_tokens) is clear.
        if choices[0].get("finish_reason") == "length":
            raise ValueError(
                "response truncated at the token cap before any content "
                f"(finish_reason=length, max_tokens={max_tokens}); raise --max-tokens "
                "— reasoning models need more budget to emit an answer.")
        raise ValueError("response choice missing message.content")
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens")
    logprob_stats = None
    if want_logprobs:
        content = (choices[0].get("logprobs") or {}).get("content")
        logprob_stats = token_entropy_stats(content)
    return (text, usage.get("prompt_tokens"), usage.get("completion_tokens"),
            reasoning_tokens, logprob_stats)


def _mock_seed(item_id, sample_idx):
    return int(hashlib.sha1(f"{item_id}|{sample_idx}".encode()).hexdigest(), 16) % (2 ** 32)


def _mock_text(item, target, rnd, mode):
    """Synthetic answer text aiming at `target`. 'noisy' degrades with difficulty."""
    if mode == "perfect":
        ok = True
    elif mode == "random":
        ok = rnd.random() < 0.5
    else:  # noisy
        p = max(0.05, 0.95 - 0.11 * item["difficulty"] - (0.18 if item["has_distractor"] else 0))
        ok = rnd.random() < p
    if ok:
        ans = target
    elif item["answer_type"] in ("int", "justified_choice"):
        ans = str(int(target) + rnd.choice([-3, -2, -1, 1, 2, 3]))
    else:
        opts = item["choices"].split("|")
        ans = rnd.choice([o for o in opts if o != target] or [target])
    conf = rnd.randint(55, 95) if ok else rnd.randint(30, 80)
    return f"Reasoning omitted in mock.\nANSWER: {ans}\nCONFIDENCE: {conf}"


def mock_one(item, sample_idx, mode):
    """One deterministic synthetic answer. 'noisy' degrades with difficulty/distractor."""
    rnd = random.Random(_mock_seed(item["item_id"], sample_idx))
    return _mock_text(item, item["gold"], rnd, mode), None, None


def mock_turn(item, sample_idx, turn_idx, mode):
    """Synthetic per-turn answer: turn 0 of a sequential item targets the committed
    subgold; later turns target the final gold."""
    target = item["subgold"] if (turn_idx == 0 and item.get("subgold")) else item["gold"]
    rnd = random.Random(_mock_seed(item["item_id"], sample_idx * 100 + turn_idx + 1))
    return _mock_text(item, target, rnd, mode)


def _turns_of(item):
    turns = item.get("turns")
    if isinstance(turns, str):
        return turns.split("|") if turns else []
    return turns or []


def _sequential_messages(item, replies, ask_confidence):
    """Messages for the NEXT turn (index len(replies)). The real prior assistant
    replies are in place, so each turn is answered before the next is revealed --
    the model commits turn 1 without seeing the turn-2 pivot (E3/H3)."""
    turns = _turns_of(item)
    fmt = _format_instructions(item, ask_confidence)
    msgs = [{"role": "system", "content": SYSTEM}]
    for idx in range(len(replies) + 1):
        msgs.append({"role": "user", "content": f"{turns[idx]}\n\n{fmt}"})
        if idx < len(replies):
            msgs.append({"role": "assistant", "content": replies[idx]})
    return msgs


def _call_turn(messages, item, cfg, sample_idx, turn_idx):
    """One model call for a single turn. Returns (text, ptok, ctok, telemetry)."""
    if cfg.get("mock"):
        return mock_turn(item, sample_idx, turn_idx, cfg["mock"]), None, None, None
    if _provider_has_capability(cfg, "claude_cli"):
        text, pt, ct, timings, think = call_claude_cli(
            cfg["model"], messages, cfg["timeout"])
        return text, pt, ct, _build_telemetry(cfg, pt, ct, timings, think, text)
    if _provider_has_capability(cfg, "native_anthropic"):
        text, pt, ct, timings, think = call_anthropic(
            cfg["api_key"], cfg["model"], messages,
            cfg["temperature"], cfg["max_tokens"], cfg["timeout"],
            oauth=_provider_has_capability(cfg, "oauth"))
        return text, pt, ct, _build_telemetry(cfg, pt, ct, timings, think, text)
    text, pt, ct, rt, lp = call_api(
        cfg["base_url"], cfg["api_key"], cfg["model"], messages,
        cfg["temperature"], cfg["max_tokens"], cfg["timeout"],
        want_logprobs=_provider_has_capability(cfg, "logprobs"))
    telemetry = (_build_telemetry(cfg, pt, ct, None, rt or 0, text, logprob_stats=lp)
                 if (lp or rt) else None)
    return text, pt, ct, telemetry


def _sequential_completion(item, cfg, sample_idx):
    """Genuine multi-turn: answer each turn before revealing the next. Grades the
    FINAL turn as primary; the committed turn-1 reply is returned via `extra` for
    sub-gold grading. Returns (final_text, ptok, ctok, telemetry, err, extra)."""
    last_err = None
    for attempt in range(cfg["retries"] + 1):
        try:
            replies, pt_sum, ct_sum, telem = [], 0, 0, None
            for ti in range(len(_turns_of(item))):
                msgs = _sequential_messages(item, replies, cfg["ask_confidence"])
                text, pt, ct, tlm = _call_turn(msgs, item, cfg, sample_idx, ti)
                if not text:
                    raise ValueError("empty turn reply")
                replies.append(text)
                pt_sum += pt or 0
                ct_sum += ct or 0
                telem = tlm or telem            # keep the latest turn's telemetry
            extra = {"intermediate_text": replies[0]}
            return replies[-1], pt_sum or None, ct_sum or None, telem, None, extra
        except _RETRYABLE as e:
            last_err = e
            if attempt < cfg["retries"]:
                time.sleep(_retry_backoff(e, attempt))
    return None, None, None, None, last_err, None


def _one_completion(item, cfg, sample_idx):
    """Return (text, ptok, ctok, telemetry, err, extra). err is None on success.
    `extra` carries the committed turn-1 reply for genuine multi-turn items."""
    if item.get("subgold"):                     # genuine sequential multi-turn (E3/H3)
        return _sequential_completion(item, cfg, sample_idx)
    last_err = None
    for attempt in range(cfg["retries"] + 1):
        try:
            if cfg.get("mock"):
                text, pt, ct = mock_one(item, sample_idx, cfg["mock"])
                telemetry = _build_telemetry(cfg, pt, ct, None, 0, text)
                return text, pt, ct, telemetry, None, None
            elif _provider_has_capability(cfg, "claude_cli"):
                text, pt, ct, timings, think_tokens = call_claude_cli(
                    cfg["model"], build_messages(item, cfg["ask_confidence"]),
                    cfg["timeout"])
                telemetry = _build_telemetry(cfg, pt, ct, timings, think_tokens, text)
                return text, pt, ct, telemetry, None, None
            elif _provider_has_capability(cfg, "native_anthropic"):
                text, pt, ct, timings, think_tokens = call_anthropic(
                    cfg["api_key"], cfg["model"],
                    build_messages(item, cfg["ask_confidence"]),
                    cfg["temperature"], cfg["max_tokens"], cfg["timeout"],
                    oauth=_provider_has_capability(cfg, "oauth"))
                telemetry = _build_telemetry(cfg, pt, ct, timings, think_tokens, text)
                return text, pt, ct, telemetry, None, None
            else:
                text, pt, ct, rt, lp = call_api(
                    cfg["base_url"], cfg["api_key"], cfg["model"],
                    build_messages(item, cfg["ask_confidence"]),
                    cfg["temperature"], cfg["max_tokens"], cfg["timeout"],
                    want_logprobs=_provider_has_capability(cfg, "logprobs"))
                # OpenAI-compat: no stream timings; reasoning-token count and
                # logprob entropy are captured when the provider exposes them.
                telemetry = (_build_telemetry(cfg, pt, ct, None, rt or 0, text,
                                              logprob_stats=lp) if (lp or rt) else None)
                return text, pt, ct, telemetry, None, None
        except _RETRYABLE as e:
            last_err = e
            if attempt < cfg["retries"]:
                time.sleep(_retry_backoff(e, attempt))
    return None, None, None, None, last_err, None


def _build_telemetry(cfg, prompt_tokens, completion_tokens, timings, think_tokens, text,
                     logprob_stats=None):
    """Build an honest telemetry payload. Token-entropy fields are populated only
    when the provider returned logprobs (open-weights path); otherwise they are
    annotated as unobservable rather than reported as zero."""
    capabilities = list(cfg.get("capabilities", []))
    unobservable = {}

    # Anthropic Opus 4.8 exposes no thinking-token breakdown; be honest about it.
    if think_tokens:
        reasoning_token_source = "native_usage"
        reasoning_tokens = int(think_tokens)
    else:
        reasoning_token_source = "unavailable"
        reasoning_tokens = 0
        unobservable["reasoning_tokens"] = "not_exposed_by_provider"

    # Reasoning density proxy: only meaningful when we have a non-zero completion
    # and the provider exposes a reasoning-token count.
    if reasoning_tokens and completion_tokens:
        answer_estimate = max(completion_tokens - reasoning_tokens, 1)
        reasoning_density_proxy = (completion_tokens - answer_estimate) / answer_estimate
    else:
        reasoning_density_proxy = None
        if reasoning_token_source == "unavailable":
            unobservable["reasoning_density_proxy"] = "reasoning_tokens_unavailable"

    # Timing fields.
    timings = timings or {}
    ttft_ms = int(round(timings.get("ttft", 0) * 1000)) if timings.get("ttft") is not None else None
    first_reasoning_ms = int(round(timings.get("first_reasoning", 0) * 1000)) if timings.get("first_reasoning") is not None else None
    answer_wall_ms = int(round(timings.get("answer_wall", 0) * 1000)) if timings.get("answer_wall") is not None else None

    # first_reasoning == ttft when display:omitted, so it is ops-only.
    if first_reasoning_ms is not None:
        unobservable["first_reasoning_ms"] = "ops_only_equals_ttft_when_display_omitted"

    # reasoning_wall_ms defensible proxy: time from stream start to the
    # end of reasoning / start of answer (answer_wall - ttft). When the
    # provider uses display:omitted, the raw CoT is never returned, so
    # this is the best available lower bound on wall-clock reasoning time.
    if answer_wall_ms is not None and ttft_ms is not None:
        reasoning_wall_ms = max(answer_wall_ms - ttft_ms, 0)
    else:
        reasoning_wall_ms = None
        unobservable["reasoning_wall_ms"] = "ttft_or_answer_wall_unavailable"

    # Token entropy / logprob divergence: observable ONLY with provider logprobs.
    if logprob_stats:
        token_entropy_mean = logprob_stats["token_entropy_mean"]
        token_entropy_max = logprob_stats["token_entropy_max"]
        friction_transitions = logprob_stats["friction_transitions"]
        logprob_divergence_spikes = logprob_stats["logprob_divergence_spikes"]
    else:
        token_entropy_mean = token_entropy_max = None
        friction_transitions = logprob_divergence_spikes = None
        unobservable["token_entropy"] = "requires_provider_logprobs"

    # These remain unobservable: hosted CoT is omitted, so there is no token stream
    # to map a Tree-of-Thought over, and no intra-reasoning token rate.
    unobservable["thinking_tps"] = "unobservable"
    unobservable["tot_branch_map"] = "unobservable_without_raw_cot"

    return {
        "capabilities": capabilities,
        "reasoning_token_source": reasoning_token_source,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "reasoning_density_proxy": reasoning_density_proxy,
        "ttft_ms": ttft_ms,
        "first_reasoning_ms": first_reasoning_ms,
        "reasoning_wall_ms": reasoning_wall_ms,
        "answer_wall_ms": answer_wall_ms,
        "token_entropy_mean": token_entropy_mean,
        "token_entropy_max": token_entropy_max,
        "friction_transitions": friction_transitions,
        "logprob_divergence_spikes": logprob_divergence_spikes,
        "unobservable_fields": unobservable,
    }

def _process(item, cfg):
    """Run all samples for one item.

    Returns (item_id, rows, telemetry_list, err_summary, solved). `solved` is
    True when no sample errored and a majority of the item's samples graded
    correct (for the common n=1 case, simply whether the one answer was right).
    """
    rows, telemetry_list, last_err = [], [], None
    n_correct = 0
    choices = item["choices"].split("|") if item.get("choices") else None
    for s in range(cfg["n"]):
        t0 = time.time()
        text, ptok, ctok, telemetry, err, extra = _one_completion(item, cfg, s)
        latency = int((time.time() - t0) * 1000)
        if err is not None:
            last_err = err
            rows.append((storage.ERROR_MARKER, None, None, None, latency, None, None, None))
            continue
        if telemetry is not None:
            telemetry_list.append((item["item_id"], s, telemetry))
        parsed, correct, conf, parse_source = grading.grade(
            text, item["answer_type"], item["gold"], choices)
        n_correct += 1 if correct else 0
        metadata = {"parse_source": parse_source}
        # Genuine multi-turn: grade the committed turn-1 reply against the subgold so
        # metrics can separate "got the pre-pivot value" from "revised correctly".
        if extra and item.get("subgold"):
            iparsed, icorrect, _, _ = grading.grade(
                extra["intermediate_text"], item["answer_type"], item["subgold"], choices)
            metadata["intermediate_correct"] = bool(icorrect)
            metadata["intermediate_parsed"] = iparsed
        rows.append((text, parsed, correct, conf, latency, ptok, ctok, metadata))
    solved = last_err is None and n_correct * 2 > cfg["n"]
    return item["item_id"], rows, telemetry_list, (str(last_err) if last_err else None), solved




def _fmt_dur(seconds):
    """Compact H:MM:SS / M:SS duration for the progress line."""
    s = int(round(max(seconds, 0)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


class _Progress:
    """Dependency-free live progress for a run.

    On a TTY it repaints one line in place (carriage return) with a bar,
    percentage, solved/wrong/error counts, a live accuracy estimate,
    throughput and ETA. When output is redirected (no TTY) it instead emits
    a plain line every ~5% so logs stay readable.
    """

    def __init__(self, total, stream=None, width=28):
        self.total = max(total, 1)
        self.width = width
        self.stream = stream if stream is not None else sys.stdout
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.start = time.time()
        self._last_paint = 0.0
        self._step = max(1, self.total // 20)            # ~5% cadence when redirected

    def _c(self, text, code):
        """Wrap `text` in an ANSI color on a TTY; return it unchanged otherwise."""
        return f"\033[{code}m{text}\033[0m" if self.tty else text

    def _counts(self, done, solved, err):
        """Solved/wrong/error counts plus a live accuracy estimate (errors excluded)."""
        wrong = max(done - solved - err, 0)
        answered = solved + wrong
        acc = f"{solved / answered * 100:.0f}%" if answered else "--"
        err_s = self._c(f"err={err}", 31) if err else f"err={err}"
        return f"{self._c(f'✓{solved}', 32)} {self._c(f'✗{wrong}', 33)} {err_s}  acc={acc}"

    def update(self, done, solved, err):
        now = time.time()
        final = done >= self.total
        if self.tty:
            if not final and now - self._last_paint < 0.1:   # throttle repaints
                return
        elif not final and done % self._step:                # ~5% cadence in logs
            return
        self._last_paint = now

        elapsed = now - self.start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 and not final else 0.0
        frac = done / self.total
        stats = (f"{done}/{self.total}  {self._counts(done, solved, err)}  "
                 f"{rate:4.1f} it/s  ETA {_fmt_dur(eta)}")
        if self.tty:
            filled = int(self.width * frac)
            bar = "█" * filled + "░" * (self.width - filled)
            self.stream.write(f"\r  [{bar}] {frac * 100:5.1f}%  {stats}  ")
        else:
            self.stream.write(f"  {frac * 100:5.1f}%  {stats}\n")
        self.stream.flush()

    def finish(self, solved, err):
        elapsed = time.time() - self.start
        rate = self.total / elapsed if elapsed > 0 else 0.0
        if self.tty:
            self.stream.write("\n")
        wrong = max(self.total - solved - err, 0)
        answered = solved + wrong
        acc = f"{solved / answered * 100:.1f}%" if answered else "--"
        self.stream.write(
            f"done in {_fmt_dur(elapsed)}  ({self.total} items: {solved} solved, "
            f"{wrong} wrong, {err} errored)  acc={acc}  {rate:.1f} it/s\n")
        self.stream.flush()


def run(con, run_id, items, cfg):
    storage.new_run(con, run_id, cfg.get("model", "mock"), cfg.get("base_url", ""), cfg)
    done = storage.done_items(con, run_id) if cfg.get("resume") else set()
    todo = [it for it in items if it["item_id"] not in done]
    total = len(todo)
    print(f"running {total} items (skipping {len(done)} done)  "
          f"model={cfg.get('model', 'mock')}  workers={cfg['workers']}  n={cfg['n']}")
    if total == 0:
        print("nothing to do — every item already has a stored response (use --resume off "
              "to re-run, or change --run-id).")
        return

    n_done = n_solved = n_err = 0
    prog = _Progress(total)
    prog.update(0, 0, 0)
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
        futs = {ex.submit(_process, it, cfg): it for it in todo}
        for fut in as_completed(futs):
            try:
                item_id, rows, telemetry_list, err, solved = fut.result()
            except Exception as e:                       # never let one item abort the run
                item_id, rows, telemetry_list, err, solved = futs[fut]["item_id"], [_ERROR_ROW], [], str(e), False
            for item_id_t, s, telemetry in telemetry_list:
                storage.save_telemetry(con, run_id, item_id_t, s, **telemetry)
            for i, (raw, parsed, correct, conf, lat, pt, ct, meta) in enumerate(rows):
                storage.save_response(con, run_id, item_id, i, raw, parsed, correct, conf, lat, pt, ct,
                                    metadata=meta)
            n_done += 1
            if err is not None:
                n_err += 1
            elif solved:
                n_solved += 1
            if n_done % 25 == 0 or n_done == total:
                con.commit()
            prog.update(n_done, n_solved, n_err)
    con.commit()
    prog.finish(n_solved, n_err)
