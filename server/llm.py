"""LLM helpers.

expand_action takes a casual action description (any language) and turns it
into one short English sentence, roughly the ~20-word style the model was
trained on. write_report summarizes the recent action log for a caregiver.

We use OpenAI if OPENAI_API_KEY is set, else Gemini if GEMINI_API_KEY is set.
With no key both fall back to a plain non-LLM version so the service still runs.
The "source" we return says which path was used.

Env: OPENAI_API_KEY / OPENAI_MODEL (gpt-4o-mini), GEMINI_API_KEY / GEMINI_MODEL
(gemini-2.5-flash).
"""

import json
import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("vpoclip.llm")

ADD_ACTION_INSTRUCTION = (
    "You are an expert at writing text descriptions for CLIP-based zero-shot "
    "action recognition. Given an action name and an optional casual "
    "description (possibly not in English), produce a JSON object: "
    '{"canonical_name": str, "description": str}. '
    "canonical_name is a short English label. description is ONE English "
    "sentence of about 20 words describing what a camera would visually observe "
    "when a person performs this action - third-person, present tense, concrete "
    "and visual, mentioning the setting, relevant objects and body posture, "
    'starting with "A person ...". Reply ONLY with JSON.'
)

REPORT_INSTRUCTION = (
    "You are an assistant for caregivers of elderly people. You get a timeline "
    "of actions recognized by a camera in the person's home. Write a short "
    "summary (2-4 sentences) of what the person did, and mention anything "
    "notable such as long inactivity or unusual patterns. Do not invent events "
    "that are not in the timeline. Answer in the language requested by the user."
)

_client = None


def _provider():
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    return None


def llm_available():
    return _provider() is not None


def _call_openai(system_instruction, contents, want_json):
    global _client
    if _client is None or not hasattr(_client, "chat"):
        from openai import OpenAI

        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    kwargs = {"response_format": {"type": "json_object"}} if want_json else {}
    response = _client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": contents},
        ],
        **kwargs,
    )
    return response.choices[0].message.content


def _call_gemini(system_instruction, contents, want_json):
    global _client
    from google import genai
    from google.genai import types

    if _client is None or not hasattr(_client, "models"):
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json" if want_json else None,
    )
    response = _client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=contents,
        config=config,
    )
    return response.text


def _generate(system_instruction, contents, want_json=False):
    provider = _provider()
    call = _call_openai if provider == "openai" else _call_gemini
    # one retry, these calls fail every now and then
    last_error = None
    for attempt in range(2):
        try:
            text = call(system_instruction, contents, want_json)
            return json.loads(text) if want_json else text.strip()
        except Exception as exc:
            last_error = exc
            log.warning("%s call failed (attempt %d): %s", provider, attempt + 1, exc)
    raise RuntimeError(f"{provider} request failed: {last_error}")


def expand_action(name, casual_description=None):
    """Returns (canonical_name, prompts, source)."""
    provider = _provider()
    if provider is None:
        log.warning("No LLM key set, using template prompts for %r", name)
        return name.strip(), _template_prompts(name), "template"

    request = {"action_name": name}
    if casual_description:
        request["casual_description"] = casual_description
    data = _generate(ADD_ACTION_INSTRUCTION, json.dumps(request, ensure_ascii=False), want_json=True)

    canonical = str(data.get("canonical_name", "")).strip() or name.strip()
    description = str(data.get("description", "")).strip()
    if not description:
        raise RuntimeError(f"{provider} returned no usable description: {data}")
    return canonical, [description], provider


def _template_prompts(name):
    name = name.strip().rstrip(".")
    return [f"A person is {name} indoors, clearly visible from a third-person camera view."]


def write_report(events, minutes, language="en"):
    """Returns (report_text, source). events come from ActionLog.recent()."""
    if not events:
        return f"No activity was recognized in the last {minutes} minutes.", "local"

    lines = []
    for e in events:
        stamp = time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))
        lines.append(f"{stamp} {e['action']} ({e['confidence']:.2f})")
    timeline = "\n".join(lines)

    provider = _provider()
    if provider is None:
        log.warning("No LLM key set, returning a plain summary")
        return _plain_summary(events, minutes), "local"

    contents = (
        f"Timeline of the last {minutes} minutes (time, action, confidence):\n"
        f"{timeline}\n\nWrite the summary in this language: {language}"
    )
    return _generate(REPORT_INSTRUCTION, contents), provider


def _plain_summary(events, minutes):
    counts = {}
    for e in events:
        counts[e["action"]] = counts.get(e["action"], 0) + 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    listing = ", ".join(f"{action} ({count}x)" for action, count in ranked[:5])
    return (
        f"In the last {minutes} minutes, {len(events)} actions were recognized. "
        f"Most frequent: {listing}."
    )
