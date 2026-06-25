"""
Model runner. Talks to any OpenAI-compatible /chat/completions endpoint, which
covers hosted APIs and every common local server (Ollama, vLLM, LM Studio,
llama.cpp, TGI). Set base_url + model + api_key.

An optional native Anthropic Messages API streaming path is triggered when the
resolved provider carries the `native_anthropic` capability flag. It captures
per-phase wall-clock timings from stream events and stores them alongside the
response; the OpenAI-compatible path is otherwise unchanged.

Samples are drawn one request at a time (client-side), not via the server's `n`
parameter, because many local servers silently ignore `n` and return a single
completion — which would make pass@k vanish without warning.

A `mock` mode synthesizes answers without a server so you can test the whole
pipeline (and see that the metrics light up) before pointing at a real model.
Mock answers are deterministic per (item, sample) so test runs are reproducible.
"""

import sys
import json
import time
import random
import hashlib
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

# raw/parsed/correct/confidence/latency/prompt_tokens/completion_tokens
_ERROR_ROW = (storage.ERROR_MARKER, None, None, None, 0, None, None, None)

# errors worth retrying: transport problems and malformed/oddly-shaped responses
_RETRYABLE = (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
              OSError, ValueError, json.JSONDecodeError)


def _provider_has_capability(cfg, name):
    caps = cfg.get("capabilities") or []
    return name in caps



def build_messages(item, ask_confidence: bool):
    fmt = ("Show your reasoning, then end with a line in exactly this format:\n"
           "ANSWER: <your final answer>")
    if item["answer_type"] == "choice" and item.get("choices"):
        opts = ", ".join(item["choices"].split("|"))
        fmt += f"\nYour answer must be exactly one of: {opts}."
    if ask_confidence:
        fmt += "\nThen on the next line:\nCONFIDENCE: <an integer 0-100 for how certain you are>"

    system_msg = {"role": "system", "content": SYSTEM}
    turns = item.get("turns")
    if turns and isinstance(turns, str):
        turns = turns.split("|") if turns else []
    if turns:
        messages = [system_msg]
        for idx, turn in enumerate(turns):
            content = turn
            messages.append({"role": "user", "content": content})
            # Insert a genuine assistant turn between user turns so the
            # conversation alternates (system, user, assistant, user, ...)
            # rather than collapsing two consecutive user messages into
            # one turn. The assistant echoes the prior user state so the
            # model has a concrete carry-over to build on.
            if idx < len(turns) - 1:
                messages.append({"role": "assistant",
                                 "content": f"Understood. Current state noted."})
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


def call_anthropic(api_key, model, messages, temperature, max_tokens, timeout):
    """One native Anthropic streaming completion.

    Returns (text, prompt_tokens, completion_tokens, timings, think_tokens).
    Raises ValueError if the response is unusable.
    """
    if anthropic is None:
        raise ValueError("anthropic package is required for the native Anthropic path")

    system, anthropic_messages = _to_anthropic_messages(messages)
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
        "temperature": temperature,
        "stream": True,
        "thinking": {"type": "adaptive", "display": "omitted"},
    }
    if system:
        params["system"] = system

    with client.messages.stream(**params) as stream:
        text, pt, ct, timings, think_tokens = _parse_anthropic_stream(stream)

    if not text:
        raise ValueError("Anthropic stream produced no text content")
    return text, pt, ct, timings, think_tokens


def call_api(base_url, api_key, model, messages, temperature, max_tokens, timeout):
    """One completion. Returns (text, prompt_tokens, completion_tokens).
    Raises ValueError if the response is not shaped like a chat completion."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
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
        raise ValueError("response choice missing message.content")
    usage = body.get("usage") or {}
    return text, usage.get("prompt_tokens"), usage.get("completion_tokens")


def _mock_seed(item_id, sample_idx):
    return int(hashlib.sha1(f"{item_id}|{sample_idx}".encode()).hexdigest(), 16) % (2 ** 32)


def mock_one(item, sample_idx, mode):
    """One deterministic synthetic answer. 'noisy' degrades with difficulty/distractor."""
    rnd = random.Random(_mock_seed(item["item_id"], sample_idx))
    gold = item["gold"]
    if mode == "perfect":
        ok = True
    elif mode == "random":
        ok = rnd.random() < 0.5
    else:  # noisy
        p = max(0.05, 0.95 - 0.11 * item["difficulty"] - (0.18 if item["has_distractor"] else 0))
        ok = rnd.random() < p
    if ok:
        ans = gold
    elif item["answer_type"] == "int":
        ans = str(int(gold) + rnd.choice([-3, -2, -1, 1, 2, 3]))
    else:
        opts = item["choices"].split("|")
        ans = rnd.choice([o for o in opts if o != gold] or [gold])
    conf = rnd.randint(55, 95) if ok else rnd.randint(30, 80)
    return f"Reasoning omitted in mock.\nANSWER: {ans}\nCONFIDENCE: {conf}", None, None


def _one_completion(item, cfg, sample_idx):
    """Return (text, ptok, ctok, telemetry, err). err is None on success."""
    last_err = None
    for attempt in range(cfg["retries"] + 1):
        try:
            if cfg.get("mock"):
                text, pt, ct = mock_one(item, sample_idx, cfg["mock"])
                telemetry = _build_telemetry(cfg, pt, ct, None, 0, text)
                return text, pt, ct, telemetry, None
            elif _provider_has_capability(cfg, "native_anthropic"):
                text, pt, ct, timings, think_tokens = call_anthropic(
                    cfg["api_key"], cfg["model"],
                    build_messages(item, cfg["ask_confidence"]),
                    cfg["temperature"], cfg["max_tokens"], cfg["timeout"])
                telemetry = _build_telemetry(cfg, pt, ct, timings, think_tokens, text)
                return text, pt, ct, telemetry, None
            else:
                text, pt, ct = call_api(
                    cfg["base_url"], cfg["api_key"], cfg["model"],
                    build_messages(item, cfg["ask_confidence"]),
                    cfg["temperature"], cfg["max_tokens"], cfg["timeout"])
                # OpenAI-compat: no stream timings available.
                return text, pt, ct, None, None
        except _RETRYABLE as e:
            last_err = e
            if attempt < cfg["retries"]:
                time.sleep(min(2 ** attempt, 8))
    return None, None, None, None, last_err


def _build_telemetry(cfg, prompt_tokens, completion_tokens, timings, think_tokens, text):
    """Build an honest telemetry payload for the native Anthropic path."""
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

    # These columns are unobservable on Anthropic / current stream parser.
    unobservable["token_entropy"] = "unobservable"
    unobservable["thinking_tps"] = "unobservable"
    unobservable["tot_branch_map"] = "unobservable"

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
        "unobservable_fields": unobservable,
    }

def _process(item, cfg):
    """Run all samples for one item. Returns (item_id, rows, telemetry_list, err_summary)."""
    rows, telemetry_list, last_err = [], [], None
    for s in range(cfg["n"]):
        t0 = time.time()
        text, ptok, ctok, telemetry, err = _one_completion(item, cfg, s)
        latency = int((time.time() - t0) * 1000)
        if err is not None:
            last_err = err
            rows.append((storage.ERROR_MARKER, None, None, None, latency, None, None, None))
            continue
        if telemetry is not None:
            telemetry_list.append((item["item_id"], s, telemetry))
        parsed, correct, conf, parse_source = grading.grade(
            text, item["answer_type"], item["gold"],
            item["choices"].split("|") if item.get("choices") else None)
        rows.append((text, parsed, correct, conf, latency, ptok, ctok, parse_source))
    return item["item_id"], rows, telemetry_list, (str(last_err) if last_err else None)




def _fmt_dur(seconds):
    """Compact H:MM:SS / M:SS duration for the progress line."""
    s = int(round(max(seconds, 0)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


class _Progress:
    """Dependency-free live progress for a run.

    On a TTY it repaints one line in place (carriage return) with a bar,
    percentage, item counts, throughput and ETA. When output is redirected
    (no TTY) it instead emits a plain line every ~5% so logs stay readable.
    """

    def __init__(self, total, stream=None, width=28):
        self.total = max(total, 1)
        self.width = width
        self.stream = stream if stream is not None else sys.stdout
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.start = time.time()
        self._last_paint = 0.0
        self._step = max(1, self.total // 20)            # ~5% cadence when redirected

    def update(self, done, ok, err):
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
        stats = (f"{done}/{self.total}  ok={ok} err={err}  "
                 f"{rate:4.1f} it/s  ETA {_fmt_dur(eta)}")
        if self.tty:
            filled = int(self.width * frac)
            bar = "█" * filled + "░" * (self.width - filled)
            self.stream.write(f"\r  [{bar}] {frac * 100:5.1f}%  {stats}  ")
        else:
            self.stream.write(f"  {frac * 100:5.1f}%  {stats}\n")
        self.stream.flush()

    def finish(self, ok, err):
        elapsed = time.time() - self.start
        rate = self.total / elapsed if elapsed > 0 else 0.0
        if self.tty:
            self.stream.write("\n")
        tail = f", {err} errored" if err else ""
        self.stream.write(
            f"done in {_fmt_dur(elapsed)}  ({self.total} items, {ok} ok{tail}, "
            f"{rate:.1f} it/s)\n")
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

    n_done = n_ok = n_err = 0
    prog = _Progress(total)
    prog.update(0, 0, 0)
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
        futs = {ex.submit(_process, it, cfg): it for it in todo}
        for fut in as_completed(futs):
            try:
                item_id, rows, telemetry_list, err = fut.result()
            except Exception as e:                       # never let one item abort the run
                item_id, rows, telemetry_list, err = futs[fut]["item_id"], [_ERROR_ROW], [], str(e)
            for item_id_t, s, telemetry in telemetry_list:
                storage.save_telemetry(con, run_id, item_id_t, s, **telemetry)
            for i, (raw, parsed, correct, conf, lat, pt, ct, parse_source) in enumerate(rows):
                storage.save_response(con, run_id, item_id, i, raw, parsed, correct, conf, lat, pt, ct,
                                    metadata={"parse_source": parse_source})
            n_done += 1
            n_err += err is not None
            n_ok += err is None
            if n_done % 25 == 0 or n_done == total:
                con.commit()
            prog.update(n_done, n_ok, n_err)
    con.commit()
    prog.finish(n_ok, n_err)
