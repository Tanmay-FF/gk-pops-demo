# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Multi-backend VLM analyzer for POPS case reports.

Default backend is Qwen3-VL-2B (local) — sized to fit alongside the
detection + classification models within 8 GB VRAM once the detection
stack's GPU memory is released before case-report generation (see
TrackingEngine._run_case_report). Claude API and other open-source local
models (Moondream2, InternVL2-2B) remain available as alternate backends.
"""
import base64
import time
import traceback
from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image

from .cancellation import RunCancelled
from .config import (
    MOONDREAM2_MODEL_ID, QWEN3_VL_MODEL_ID, INTERNVL2_MODEL_ID,
    VLM_MAX_TOKENS_PER_FRAME, VLM_MAX_TOKENS_SUMMARY,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FrameAnalysis:
    frame_idx: int
    timestamp: float
    trigger: str
    description: str
    cart_id: object = None  # int or None


@dataclass
class CaseReportData:
    vlm_available: bool = True
    vlm_backend: str = ""
    executive_summary: str = ""
    frame_analyses: list = field(default_factory=list)
    risk_assessment: str = ""
    risk_level: str = "UNKNOWN"
    confidence: float = 0.0
    suspect_profile: str = ""
    actionable_insights_lp: str = ""
    actionable_insights_leo: str = ""
    generation_time_seconds: float = 0.0
    error_message: str = None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a retail loss prevention AI analyst for Gatekeeper Systems.  You are
analysing security camera footage processed by the POPS (Push-Out Probability
Score) system.

POPS scores range 0-100:
  0-30  LOW      - Normal shopping behaviour
  31-70 MEDIUM   - Suspicious activity, needs verification
  71-100 HIGH    - Likely theft / push-out

Key behaviours:
  - Person pushing a shopping cart toward the exit (OUTBOUND) = potential push-out
  - Unbagged merchandise in cart = higher risk
  - Person abandoning cart near exit = grab-and-run pattern
  - Person-cart link = the system identified who owns the cart

Describe what you see objectively and professionally.  Focus on people, carts,
merchandise visibility, movement direction, and behaviour relevant to loss
prevention."""


def _frame_prompt(capture: dict) -> str:
    ctx = capture.get("pops_context", {})
    score = ctx.get("score", "N/A")
    event = ctx.get("event", "N/A")
    fill = ctx.get("fill", "?")
    bag = ctx.get("bag", "?")
    direction = ctx.get("direction", "?")
    trigger = capture.get("trigger", "unknown")
    ts = capture.get("timestamp", 0)

    # Higher POPS score = pack more suspect-identifying detail; lower = brief context.
    try:
        is_high = isinstance(score, (int, float)) and score >= HIGH_POPS_THRESHOLD
    except Exception:
        is_high = False

    if is_high:
        focus = (
            "Pack the maximum suspect-identifying detail - clothing colors and "
            "style, build, hair, accessories, distinguishing marks, cart contents, "
            "movement direction."
        )
    else:
        focus = (
            "Briefly state who is in frame, the cart state, and what they are "
            "doing. No suspect framing."
        )

    return (
        f"POPS context - t={ts:.1f}s, trigger={trigger}, score={score}, "
        f"event={event}, fill={fill}, bag={bag}, direction={direction}.\n\n"
        f"{focus}\n\n"
        f"HARD CONSTRAINTS - your reply MUST follow all of these:\n"
        f"  - Maximum 3 lines, around 60 words total.\n"
        f"  - Plain prose only. Single paragraph.\n"
        f"  - No markdown headers, no tables, no bullet points, no numbered lists.\n"
        f"  - No section labels like 'Person:', 'Cart:', 'Behaviour:'.\n"
        f"  - Never use em dashes. Use commas, colons or full stops instead.\n"
        f"  - No preamble, no closing remarks. Start with the description directly."
    )


HIGH_POPS_THRESHOLD = 71


def _find_top_suspect(pops_summary: dict) -> tuple[str | None, float]:
    """Scan the FINAL POPS table; return (key, max_score). The key is the
    `pops_summary` key as-is (e.g. "C3"); pair it with capture['cart_id']
    via `_is_suspect_capture` since captures store the integer display id."""
    best_key, best_score = None, -1.0
    for key, info in pops_summary.items():
        try:
            ms = float(info.get("max_score", 0))
        except (TypeError, ValueError):
            continue
        if ms > best_score:
            best_score = ms
            best_key = key
    return best_key, best_score


def _is_suspect_capture(capture: dict, suspect_key: str | None) -> bool:
    if suspect_key is None:
        return False
    cid = capture.get("cart_id")
    if cid is None:
        return False
    # pops_summary keys are "C{id}"; capture['cart_id'] is the integer id.
    return f"C{cid}" == suspect_key or str(cid) == str(suspect_key)


def _format_pops_table(pops_summary: dict, suspect_key: str | None = None) -> str:
    lines = []
    for key, info in pops_summary.items():
        marker = " *** SUSPECT ***" if key == suspect_key else ""
        lines.append(
            f"  {key}: peak score {info.get('max_score', '?')}, "
            f"peak event {info.get('peak_event', '?')}{marker}"
        )
    return "\n".join(lines) or "  (no carts detected)"


def _format_analytics_context(analytics_result) -> str:
    """Compact analytics summary for the VLM prompt — queue spikes, top
    high-dwell zones, and the auto-generated insight string. Returns "" when
    nothing useful is available so the caller can omit the section."""
    if analytics_result is None:
        return ""
    spikes = list(getattr(analytics_result, "queue_spikes", None) or [])
    dwell  = list(getattr(analytics_result, "dwell_summary", None) or [])
    insight = (getattr(analytics_result, "insight_text", "") or "").strip()

    if not spikes and not dwell and not insight:
        return ""

    lines = ["=== STORE ANALYTICS CONTEXT ==="]
    if insight:
        lines.append(f"Auto-insight: {insight}")

    severe = [s for s in spikes
              if getattr(s, "severity", "NORMAL") in ("QUEUE_FORMING", "BACKED_UP")]
    if severe:
        lines.append("Queue spikes:")
        for s in severe[:5]:
            reasons = ", ".join((s.reasons or [])[:3])
            extra = f", reasons=[{reasons}]" if reasons else ""
            lines.append(
                f"  - {s.zone_name!r}: severity={s.severity}, "
                f"avg_dwell={s.avg_dwell_s:.1f}s, "
                f"peak_occupancy={s.peak_occupancy}{extra}"
            )
    else:
        lines.append("Queue spikes: None detected (zone occupancy within normal limits).")

    if dwell:
        ranked = sorted(dwell, key=lambda d: d.get("avg_dwell_s", 0.0), reverse=True)[:3]
        lines.append("Top dwell zones (avg, visits, p95):")
        for d in ranked:
            lines.append(
                f"  - {d.get('zone_name','?')!r}: "
                f"avg={float(d.get('avg_dwell_s', 0.0)):.1f}s, "
                f"visits={int(d.get('n_visits', 0))}, "
                f"p95={float(d.get('p95_s', 0.0)):.1f}s"
            )

    return "\n".join(lines)


def _build_deterministic_normal_summary(pops_summary: dict,
                                        video_info: dict,
                                        max_score: int,
                                        video_summary: dict | None,
                                        analytics_result) -> dict:
    """Compose the normal-case report from structured data WITHOUT a VLM call.

    Reasoning: small local VLMs (Moondream2) tend to hallucinate runaway
    token loops when asked to summarise structured tabular data text-only.
    For the LOW-risk path every datum we want is already available in the
    POPS table + analytics result + tracker summary, so we build the report
    deterministically. The VLM stays in the loop for HIGH-POPS frames
    (suspect description from images) where it actually adds value.
    """
    # ---- counts ----
    vs = video_summary or {}
    total_people = vs.get("total_people_seen")
    total_carts  = vs.get("total_carts_seen")
    total_links  = vs.get("total_links_established")

    # ---- analytics derivatives ----
    dwell = list(getattr(analytics_result, "dwell_summary", None) or [])
    dwell = [d for d in dwell if d.get("n_visits", 0) > 0]
    top_dwell = max(dwell, key=lambda d: d.get("avg_dwell_s", 0.0)) if dwell else None
    spikes = list(getattr(analytics_result, "queue_spikes", None) or [])
    severe = [s for s in spikes
              if getattr(s, "severity", "") in ("QUEUE_FORMING", "BACKED_UP")]
    crowd = [s for s in spikes
             if str(getattr(s, "zone_id", "")).startswith("crowd_")]
    backed_up = [s for s in severe if getattr(s, "severity", "") == "BACKED_UP"]
    queue_forming = [s for s in severe if getattr(s, "severity", "") == "QUEUE_FORMING"]

    # ---- executive summary ----
    es: list[str] = []
    if total_people is not None and total_carts is not None:
        es.append(
            f"The video shows {total_people} unique people and {total_carts} "
            f"carts (links established: {total_links})."
        )
    es.append(
        f"Peak POPS score across all carts was {max_score} "
        f"(HIGH threshold {HIGH_POPS_THRESHOLD}); no security incidents detected."
    )
    if top_dwell and top_dwell.get("avg_dwell_s", 0.0) >= 5.0:
        es.append(
            f"Highest-dwell zone: {top_dwell.get('zone_name', '?')} - "
            f"avg {top_dwell.get('avg_dwell_s', 0.0):.1f}s, "
            f"p95 {top_dwell.get('p95_s', 0.0):.1f}s, "
            f"{int(top_dwell.get('n_visits', 0))} visits."
        )
    if backed_up:
        names = ", ".join(s.zone_name for s in backed_up[:3])
        es.append(f"Queue BACKED UP at: {names}.")
    elif queue_forming:
        names = ", ".join(s.zone_name for s in queue_forming[:3])
        es.append(f"Queue forming at: {names}.")
    elif crowd:
        es.append(
            f"{len(crowd)} crowd cluster(s) detected (no zone-based queue spikes)."
        )
    else:
        es.append("No queue spikes or crowd clusters flagged.")

    # ---- risk assessment ----
    ra = [
        f"Risk level: LOW. Peak POPS {max_score} is below the HIGH threshold "
        f"of {HIGH_POPS_THRESHOLD}, so no push-out or abandonment was inferred."
    ]
    if severe or crowd:
        ra.append(
            "Some crowding was observed but it does not by itself indicate "
            "loss; review the LP recommendations below for staffing follow-up."
        )
    else:
        ra.append("Store activity appeared routine throughout the clip.")

    # ---- LP recommendations ----
    lp: list[str] = ["Continue routine monitoring."]
    for s in backed_up[:3]:
        lp.append(f"Investigate sustained backup at {s.zone_name} "
                  f"(peak occupancy {s.peak_occupancy}, "
                  f"avg dwell {s.avg_dwell_s:.1f}s).")
    for s in queue_forming[:3]:
        lp.append(f"Watch {s.zone_name} for further build-up "
                  f"(peak occupancy {s.peak_occupancy}).")
    if top_dwell and top_dwell.get("avg_dwell_s", 0.0) >= 30.0:
        lp.append(
            f"Consider staffing review for {top_dwell.get('zone_name', '?')} "
            f"(avg dwell {top_dwell.get('avg_dwell_s', 0.0):.1f}s)."
        )
    if not crowd and not severe and (top_dwell is None
                                     or top_dwell.get("avg_dwell_s", 0.0) < 30.0):
        lp.append("No staffing or layout adjustments indicated by this clip.")

    return {
        "executive_summary": " ".join(es),
        "risk_level": "LOW",
        "confidence": 1.0,
        "risk_assessment": " ".join(ra),
        "suspect_profile": "",  # case_report_builder hides empty profiles
        "actionable_insights_lp": "\n".join(f"- {line}" for line in lp),
        "actionable_insights_leo": "- Not applicable - no incident detected.",
    }


def _high_incident_prompt(frame_descs: list[FrameAnalysis],
                          suspect_key: str, max_score: int,
                          pops_summary: dict,
                          analytics_block: str = "") -> str:
    """Suspect-focused prompt. Frames are pre-filtered to the suspect cart so
    every detail in the SUSPECT PROFILE traces back to the correct person."""
    lines = []
    for fa in frame_descs:
        lines.append(f"[{fa.timestamp:.1f}s | {fa.trigger}] {fa.description}")
    frame_text = "\n".join(lines) or "  (no suspect frames available)"
    pops_text = _format_pops_table(pops_summary, suspect_key)
    analytics_section = f"\n{analytics_block}\n" if analytics_block else ""

    return f"""\
The POPS system has flagged a HIGH-risk incident based on the FINAL score
table below. The suspect is the person linked to Cart {suspect_key}, which
reached a peak POPS score of {max_score}. Build the case report
EXCLUSIVELY around this suspect - every clothing, accessory, and behavioural
detail must come from the suspect frame descriptions below.

=== SUSPECT FRAME DESCRIPTIONS ===
{frame_text}

=== FINAL POPS SCORE TABLE ===
{pops_text}
{analytics_section}
Fill in the template below. Replace each <…> placeholder with details
GROUNDED IN THE SUSPECT FRAME DESCRIPTIONS ABOVE. Do NOT carry placeholder
text, bracket markers, or example values into your final reply. If an
attribute isn't visible in any frame, write "not visible" (do not guess).

EXECUTIVE SUMMARY:
<2-3 sentences describing what the suspect did, referencing Cart {suspect_key} and the peak POPS score {max_score}. If STORE ANALYTICS CONTEXT shows queue spikes or high-dwell zones, mention them by name; otherwise omit.>

RISK LEVEL: HIGH
CONFIDENCE: <a single number between 0.5 and 1.0 - how clearly the suspect was visible>

RISK ASSESSMENT:
<2-3 sentences on the threat, referencing the peak POPS score and the specific behaviour observed.>

SUSPECT PROFILE:
- Age & gender: <e.g. "man, 30s" or "not visible">
- Build & height: <e.g. "slim, average height" or "not visible">
- Hair & headwear: <color/length + cap/beanie/hood/none>
- Top: <garment + colour + any logo; include outerwear like jacket/hoodie>
- Bottom & footwear: <pants/shorts + shoe type/colour>
- Bags & marks: <backpack/bag colour + visible tattoos or prints>
- Cart at peak: <fill level, bagged/unbagged, visible merchandise>
- Behaviour: <gait, gaze, urgency, any interaction with staff>

LP TEAM RECOMMENDATIONS:
- <bullet 1: actionable LP step grounded in this incident>
- <bullet 2>
- <bullet 3>

LAW ENFORCEMENT SUMMARY:
- <bullet 1: factual evidence point referencing Cart {suspect_key} and the peak POPS score>
- <bullet 2>
- <bullet 3>

HARD CONSTRAINTS - your reply MUST follow ALL of these:
- Replace every <placeholder> with concrete details from the frames above.
- Do NOT include the angle brackets or the word "placeholder" in your reply.
- Do NOT mention carts, persons, or zones not in the POPS table / analytics block above.
- EXECUTIVE SUMMARY and RISK ASSESSMENT: maximum 80 words each.
- SUSPECT PROFILE: exactly 8 bullets, each on its own single line.
- LP TEAM RECOMMENDATIONS: 3-5 bullets. LAW ENFORCEMENT SUMMARY: 3-5 bullets.
- Each header on its OWN line. Do NOT collapse the report into one paragraph.
- Never use em dashes. Use commas, colons or full stops instead.
- Do NOT repeat any sentence. Stop as soon as LAW ENFORCEMENT SUMMARY is complete."""


def _normal_situation_prompt(frame_descs: list[FrameAnalysis],
                             pops_summary: dict, video_info: dict,
                             max_score: int,
                             analytics_block: str = "") -> str:
    """Benign situation summary — no suspect framing. Used when no cart in
    the final POPS table reached the HIGH threshold."""
    lines = []
    for fa in frame_descs:
        lines.append(f"[{fa.timestamp:.1f}s | {fa.trigger}] {fa.description}")
    frame_text = "\n".join(lines) or "  (no representative frames)"
    pops_text = _format_pops_table(pops_summary)
    vid = video_info.get("video_name", "the video")
    analytics_section = f"\n{analytics_block}\n" if analytics_block else ""

    return f"""\
The POPS system reviewed {vid} and found NO high-risk incidents. The peak
POPS score across all carts was {max_score} (the HIGH threshold is
{HIGH_POPS_THRESHOLD}). Provide a neutral, non-incident situation summary
based on the representative frames and final POPS table below - there is no
suspect to profile.

=== REPRESENTATIVE FRAMES ===
{frame_text}

=== FINAL POPS SCORE TABLE ===
{pops_text}
{analytics_section}
Fill in the template below. Replace each <…> placeholder with details
GROUNDED IN THE REPRESENTATIVE FRAMES AND POPS TABLE ABOVE. Do NOT carry
placeholder text, bracket markers, or example values into your final reply.

EXECUTIVE SUMMARY:
<2-3 sentences describing what is visible - general shopping activity, store conditions. If STORE ANALYTICS CONTEXT lists queue spikes or high-dwell zones, call them out by zone name. Otherwise state that store activity appeared normal. Do NOT frame anyone as a suspect.>

RISK LEVEL: LOW
CONFIDENCE: 0.9

RISK ASSESSMENT:
<1-2 sentences confirming no security incidents were detected. Reference the peak POPS score {max_score} and what it indicates - e.g. routine browsing.>

SUSPECT PROFILE:
Not applicable - no high-risk subject.

LP TEAM RECOMMENDATIONS:
- Continue routine monitoring.
- <one general operational note if anything is worth a passing mention; otherwise omit this line>

LAW ENFORCEMENT SUMMARY:
- Not applicable - no incident detected.

HARD CONSTRAINTS - your reply MUST follow ALL of these:
- Replace every <placeholder> with concrete details from the frames / analytics above.
- Do NOT include the angle brackets or the word "placeholder" in your reply.
- Headers must be on their own lines (do NOT collapse into one paragraph).
- Do NOT mention carts, persons, or zones absent from the POPS table or analytics block.
- Never use em dashes. Use commas, colons or full stops instead.
- Do NOT repeat any sentence. Stop as soon as LAW ENFORCEMENT SUMMARY is complete."""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class VLMAnalyzer:
    """Multi-backend VLM analyzer for POPS case reports."""

    def __init__(self, backend: str = "Claude (API)",
                 api_key: str = "", device: str = "cuda",
                 should_cancel=None):
        self._backend = backend
        self._api_key = api_key
        self._device = device
        #: Polled between model calls, and per generated token on the local
        #: backends. A plain callable rather than the engine's cancel token: the
        #: analyzer has no business knowing about run sequences, and this is the
        #: shape a transformers StoppingCriteria wants. Defaults to "never
        #: cancelled" so every existing caller keeps working unchanged.
        self._should_cancel = should_cancel or (lambda: False)
        self._local_model = None
        self._local_processor = None
        self._local_tokenizer = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def analyze_incident(self, captures, pops_data,
                         event_log, peak_snapshots, video_info,
                         analytics_result=None):
        """Run VLM analysis driven by the FINAL POPS score table.

        - If any cart's final score >= HIGH_POPS_THRESHOLD: describe only
          the suspect cart's captures and produce a structured suspect-
          focused incident report.
        - Otherwise: pick a couple of representative frames and produce a
          benign situation summary with no suspect framing.

        analytics_result (optional AnalyticsResult): when provided, queue
        spikes / high-dwell zones / auto-insight are folded into the prompt
        so the executive summary can reference them.
        """
        t0 = time.perf_counter()
        result = CaseReportData(vlm_backend=self._backend)

        try:
            pops_summary = pops_data.get("pops_summary", {}) or {}
            suspect_key, max_score = _find_top_suspect(pops_summary)
            max_score_int = int(max_score) if max_score >= 0 else 0
            analytics_block = _format_analytics_context(analytics_result)

            if max_score >= HIGH_POPS_THRESHOLD and suspect_key is not None:
                self._analyze_high_incident(
                    captures, pops_summary, suspect_key, max_score_int, result,
                    analytics_block=analytics_block,
                )
            else:
                self._analyze_normal_situation(
                    captures, pops_summary, video_info, max_score_int, result,
                    analytics_block=analytics_block,
                    analytics_result=analytics_result,
                    video_summary=(pops_data or {}).get("summary"),
                )

            result.vlm_available = True

        except RunCancelled:
            # Deliberately NOT folded into the result. Every other failure here
            # becomes an inline "the VLM could not run" report, which is right —
            # a case report is worth degrading rather than losing. A cancel is
            # not a failure: the user asked for the work to stop, so it has to
            # reach the caller and clear the panel instead of rendering as an
            # error nobody caused.
            raise
        except Exception as e:
            traceback.print_exc()
            result.vlm_available = False
            result.error_message = str(e)

        result.generation_time_seconds = round(time.perf_counter() - t0, 2)
        return result

    # ------------------------------------------------------------------
    # Branch handlers
    # ------------------------------------------------------------------
    def _analyze_high_incident(self, captures, pops_summary, suspect_key,
                               max_score, result, analytics_block: str = ""):
        """Describe ONLY the suspect cart's captures, then build the
        structured incident report."""
        # The risk level is NOT the VLM's to decide. POPS already scored this
        # cart at or above HIGH_POPS_THRESHOLD — that is the only reason this
        # branch runs — and the prompt hands the model "RISK LEVEL: HIGH" as a
        # fixed line, not a question. Set FIRST, before any model call, so that
        # neither a header the parser fails to recognise nor an outright VLM
        # failure can leave a genuine push-out rendering as UNKNOWN.
        result.risk_level = "HIGH"

        suspect_caps = [c for c in captures if _is_suspect_capture(c, suspect_key)]
        # If trigger labelling missed the suspect (rare), fall back to all
        # captures so we still produce a report rather than a blank one.
        if not suspect_caps:
            suspect_caps = list(captures)

        frame_analyses = []
        for cap in suspect_caps:
            desc = self._describe_frame(cap)
            frame_analyses.append(FrameAnalysis(
                frame_idx=cap["frame_idx"],
                timestamp=cap["timestamp"],
                trigger=cap["trigger"],
                description=desc,
                cart_id=cap.get("cart_id"),
            ))
        result.frame_analyses = frame_analyses

        prompt = _high_incident_prompt(
            frame_analyses, suspect_key, max_score, pops_summary,
            analytics_block=analytics_block,
        )
        summary_text = self._call_vlm(None, prompt, VLM_MAX_TOKENS_SUMMARY)
        self._parse_summary(summary_text, result)
        if result.risk_level in ("UNKNOWN", "LOW", "MEDIUM"):
            # The model contradicted (or lost) the level it was given. POPS is
            # the scorer of record, so clamp back rather than let a 2B model
            # downgrade an incident the deterministic pipeline already flagged.
            # CRITICAL is left alone — that is the model escalating, not
            # disagreeing.
            result.risk_level = "HIGH"

    def _analyze_normal_situation(self, captures, pops_summary, video_info,
                                  max_score, result, analytics_block: str = "",
                                  analytics_result=None, video_summary=None):
        """No-suspect path — deterministic report from analytics + POPS table.

        We deliberately skip the VLM here. Small local models (Moondream2)
        hallucinate runaway token loops when asked to summarise structured
        text, and Claude on a benign clip just paraphrases data we already
        have. The report fields are built from the POPS table, dwell summary,
        queue-spike list, and tracker counts."""
        result.frame_analyses = []   # no per-frame VLM either — keeps run fast

        summary_data = _build_deterministic_normal_summary(
            pops_summary=pops_summary,
            video_info=video_info,
            max_score=max_score,
            video_summary=video_summary,
            analytics_result=analytics_result,
        )
        result.executive_summary    = summary_data["executive_summary"]
        result.risk_level           = summary_data["risk_level"]
        result.confidence           = summary_data["confidence"]
        result.risk_assessment      = summary_data["risk_assessment"]
        result.suspect_profile      = summary_data["suspect_profile"]
        result.actionable_insights_lp  = summary_data["actionable_insights_lp"]
        result.actionable_insights_leo = summary_data["actionable_insights_leo"]

    def unload_model(self):
        """Free local model VRAM."""
        if self._local_model is not None:
            import gc
            import torch
            del self._local_model
            del self._local_processor
            del self._local_tokenizer
            self._local_model = None
            self._local_processor = None
            self._local_tokenizer = None
            # gc.collect() before empty_cache(), not after. On the error path
            # this runs while an exception is still being handled, and its
            # traceback frames hold references to the activations inside
            # generate(); those frames are reference cycles, so only a
            # collection releases them. Without it empty_cache() has nothing to
            # give back and the caller's move of the detection stack back onto
            # the GPU OOMs.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Per-frame description
    # ------------------------------------------------------------------
    def _describe_frame(self, capture: dict) -> str:
        prompt = _frame_prompt(capture)
        image_bytes = capture["image_bytes"]
        return self._call_vlm(image_bytes, prompt, VLM_MAX_TOKENS_PER_FRAME)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def _call_vlm(self, image_bytes, prompt, max_tokens):
        # The one chokepoint every backend and every call goes through — the
        # per-frame descriptions and the final summary alike — so one check here
        # covers all of them, including the Claude API path where an in-flight
        # HTTPS request cannot be interrupted at all. Granularity is therefore
        # "one model call": a cancel lands between frames rather than mid-frame,
        # except on the local backends, which also stop per token (see
        # _cancel_criteria).
        if self._should_cancel():
            raise RunCancelled("cancelled before a VLM call")
        if "Claude" in self._backend:
            return self._call_claude(image_bytes, prompt, max_tokens)
        elif "Moondream" in self._backend:
            return self._call_moondream(image_bytes, prompt, max_tokens)
        elif "Qwen" in self._backend:
            return self._call_qwen3vl(image_bytes, prompt, max_tokens)
        elif "InternVL" in self._backend:
            return self._call_internvl(image_bytes, prompt, max_tokens)
        else:
            raise ValueError(f"Unknown VLM backend: {self._backend}")

    # ------------------------------------------------------------------
    # Claude API
    # ------------------------------------------------------------------
    def _call_claude(self, image_bytes, prompt, max_tokens):
        """Call Claude via claude-agent-sdk (uses OAuth token via
        CLAUDE_CODE_OAUTH_TOKEN env var).  Falls back to the anthropic
        SDK if a regular API key is provided.
        """
        import os
        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        # Regular API key → use anthropic SDK directly
        if api_key and api_key.startswith("sk-ant-api"):
            return self._call_claude_sdk(image_bytes, prompt, max_tokens, api_key)

        # Otherwise → use claude-agent-sdk (OAuth token)
        return self._call_claude_agent_sdk(image_bytes, prompt, max_tokens)

    def _call_claude_sdk(self, image_bytes, prompt, max_tokens, api_key):
        """Call Claude via the anthropic Python SDK (requires API key)."""
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        content = []
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": b64},
            })
        content.append({"type": "text", "text": prompt})

        msg = client.messages.create(
            model="claude-sonnet-4-6-20250415",
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        return msg.content[0].text

    def _call_claude_agent_sdk(self, image_bytes, prompt, max_tokens):
        """Call Claude via claude-agent-sdk (uses OAuth token automatically).

        Follows the same pattern as the annotation_tool reference project.
        """
        import asyncio
        import os
        import tempfile
        from claude_agent_sdk import query, ClaudeAgentOptions

        full_prompt = f"{_SYSTEM_PROMPT}\n\n"

        # If image provided, save to temp file and ask Claude to read it
        if image_bytes:
            img_path = os.path.join(tempfile.gettempdir(), "pops_vlm_frame.jpg")
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            img_path_fwd = img_path.replace("\\", "/")
            full_prompt += f"Read the image at {img_path_fwd} and then answer:\n\n"

        full_prompt += prompt

        async def _run():
            result_text = ""
            async for message in query(
                prompt=full_prompt,
                options=ClaudeAgentOptions(allowed_tools=["Read"]),
            ):
                if hasattr(message, "result") and message.result:
                    result_text = message.result
                    break
                if hasattr(message, "content"):
                    for block in message.content:
                        if hasattr(block, "text"):
                            result_text = block.text
                            break
                    if result_text:
                        break
            return result_text

        try:
            loop = asyncio.new_event_loop()
            return loop.run_until_complete(_run())
        finally:
            loop.close()

    # ------------------------------------------------------------------
    # Moondream2 (local, ~2 GB VRAM)
    # ------------------------------------------------------------------
    def _load_moondream(self):
        if self._local_model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[VLM] Loading {MOONDREAM2_MODEL_ID} ...")
        self._local_tokenizer = AutoTokenizer.from_pretrained(
            MOONDREAM2_MODEL_ID, trust_remote_code=True)
        self._local_model = AutoModelForCausalLM.from_pretrained(
            MOONDREAM2_MODEL_ID, trust_remote_code=True,
            device_map={"": self._device})
        self._local_model.eval()
        print("[VLM] Moondream2 loaded.")

    def _cancel_criteria(self):
        """A transformers StoppingCriteriaList that ends generation when the
        run is cancelled, or None when there is nothing to cancel.

        Imported here rather than at module scope: transformers is a heavy
        import and the Claude API backend never needs it.
        """
        from transformers import StoppingCriteria, StoppingCriteriaList

        should_cancel = self._should_cancel

        class _Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return should_cancel()

        return StoppingCriteriaList([_Cancelled()])

    def _call_moondream(self, image_bytes, prompt, max_tokens):
        self._load_moondream()
        if image_bytes:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            # Newer HfMoondream.generate() requires image_embeds positionally,
            # so the text-only summary path uses a blank placeholder image and
            # the same encode_image + answer_question API as the per-frame path.
            # The prompt carries the real content; Moondream effectively ignores
            # a uniform image when answering a text-heavy question.
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        enc = self._local_model.encode_image(image)
        return self._local_model.answer_question(enc, prompt,
                                                  self._local_tokenizer)

    # ------------------------------------------------------------------
    # Qwen3-VL-2B (local, ~4 GB VRAM)
    # ------------------------------------------------------------------
    def _load_qwen3vl(self):
        if self._local_model is not None:
            return
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        except ImportError as e:
            raise RuntimeError(
                "Qwen3-VL requires transformers>=4.57.0. "
                "Run: pip install 'transformers>=4.57.0'"
            ) from e
        print(f"[VLM] Loading {QWEN3_VL_MODEL_ID} ...")
        self._local_model = Qwen3VLForConditionalGeneration.from_pretrained(
            QWEN3_VL_MODEL_ID, device_map="auto",
            dtype="auto")
        self._local_processor = AutoProcessor.from_pretrained(QWEN3_VL_MODEL_ID)
        self._local_model.eval()
        print("[VLM] Qwen3-VL loaded.")

    def _call_qwen3vl(self, image_bytes, prompt, max_tokens):
        self._load_qwen3vl()
        messages = [{"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]}]
        user_content = []
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            user_content.append({"type": "image", "image": f"data:image/jpeg;base64,{b64}"})
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        text = self._local_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        if image_bytes:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            inputs = self._local_processor(
                text=[text], images=[image], return_tensors="pt",
                padding=True).to(self._local_model.device)
        else:
            inputs = self._local_processor(
                text=[text], return_tensors="pt",
                padding=True).to(self._local_model.device)
        import torch
        with torch.no_grad():
            # Small VLMs (Qwen3-VL-2B in particular) loop badly with greedy
            # defaults — the same sentence repeats until max_tokens runs out.
            # Repetition penalty + n-gram block + EOS gives clean truncation.
            out = self._local_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                pad_token_id=self._local_processor.tokenizer.eos_token_id,
                # Cancellation, checked per generated token. Without it a Cancel
                # pressed during a summary pass waits out up to
                # VLM_MAX_TOKENS_SUMMARY tokens on a 2B model — tens of seconds
                # — because generate() is one uninterruptible call.
                stopping_criteria=self._cancel_criteria(),
            )
        if self._should_cancel():
            # The criteria stopped generation, so `out` is a truncated
            # half-sentence. Raising rather than returning it keeps a cancelled
            # report from being parsed and rendered as a real finding.
            raise RunCancelled("cancelled mid-generation")
        return self._local_processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True)[0]

    # ------------------------------------------------------------------
    # InternVL2-2B (local, ~4 GB VRAM)
    # ------------------------------------------------------------------
    def _load_internvl(self):
        if self._local_model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        print(f"[VLM] Loading {INTERNVL2_MODEL_ID} ...")
        self._local_model = AutoModel.from_pretrained(
            INTERNVL2_MODEL_ID, trust_remote_code=True,
            device_map="auto", torch_dtype="auto")
        self._local_tokenizer = AutoTokenizer.from_pretrained(
            INTERNVL2_MODEL_ID, trust_remote_code=True)
        self._local_model.eval()
        print("[VLM] InternVL2 loaded.")

    def _call_internvl(self, image_bytes, prompt, max_tokens):
        self._load_internvl()
        import torch
        if image_bytes:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            query = f"<image>\n{_SYSTEM_PROMPT}\n\n{prompt}"
            with torch.no_grad():
                response = self._local_model.chat(
                    self._local_tokenizer, image, query,
                    generation_config={"max_new_tokens": max_tokens})
            return response
        else:
            with torch.no_grad():
                response = self._local_model.chat(
                    self._local_tokenizer, None, prompt,
                    generation_config={"max_new_tokens": max_tokens})
            return response

    # ------------------------------------------------------------------
    # Parse summary response
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_summary(text: str, result: CaseReportData):
        """Parse the structured VLM summary into CaseReportData fields.

        Models often violate the requested format (collapse the whole reply
        into one paragraph, skip the line breaks, lowercase the headers,
        etc.). We therefore locate each known header anywhere in the text
        with a permissive regex and slice the response by header positions.

        After header-based slicing, we also salvage RISK LEVEL / CONFIDENCE
        from any field that still carries an inline mention, so a verbose
        executive-summary prose like "the risk level is HIGH, confidence
        moderate" still populates the structured fields.
        """
        import re

        # Header spellings are matched with a WIDE net on purpose. Qwen3-VL-2B
        # reliably produces the right seven sections but renames them as it
        # goes — observed in one run: "RISK ASSESSENT" (typo), "SUSPECT
        # PROFILES" (plural), "LAW ENFORCER SUMMARY" (wrong word). Each miss
        # silently drops a whole section, so `\w*` stems beat exact spellings.
        SECTION_KEYS = [
            ("executive_summary",     r"executive\s+summ\w*"),
            ("risk_level_marker",     r"risk\s+level"),
            ("confidence_marker",     r"confidence"),
            ("risk_assessment",       r"risk\s+assess\w*"),
            ("suspect_profile",       r"suspect\s+profiles?"),
            ("actionable_insights_lp",
                r"(?:lp\s+team\s+recommendations?|loss\s+prevention(?:\s+team)?\s+recommendations?)"),
            ("actionable_insights_leo",
                r"law\s+enforce\w*\s+(?:summary|recommendations?)"),
        ]
        # Match each header optionally on its own line; allow ":" or "-" or
        # whitespace as the body separator.
        #
        # The trailing bold group is load-bearing. Models write the header as
        # `**RISK LEVEL:** HIGH` — bold CLOSES after the colon, not before it.
        # Without consuming that `**`, every section body began with a literal
        # "**", which was merely ugly for prose fields but broke risk level
        # outright: the old extractor took value.split()[0] == "**", stripped
        # non-alpha to "", and its `if token` guard then left a genuine
        # push-out reading UNKNOWN.
        header_re = re.compile(
            r"(?im)^\s*(?:[#*\-•–—]{0,3}\s*)?"   # leading bullet/dash/#'s
            r"(?:\*\*|__)?"                      # markdown bold open
            r"(" + "|".join(rx for _, rx in SECTION_KEYS) + r")"
            r"(?:\*\*|__)?"                      # bold close BEFORE separator
            r"\s*[:–—\-]\s*"                     # separator
            r"(?:\*\*|__)?[ \t]*",               # bold close AFTER separator
            re.IGNORECASE,
        )

        matches = list(header_re.finditer(text))
        if not matches:
            # No headers detected at all — dump the whole thing into the
            # executive summary so something is shown rather than nothing.
            # Fall through to the salvage block so risk_level / confidence /
            # risk_assessment can still be extracted from inline prose.
            result.executive_summary = text.strip()

        # Slice [match.end → next match.start] into each field.
        sliced: dict[str, str] = {}
        for i, m in enumerate(matches):
            attr = next((a for a, rx in SECTION_KEYS
                         if re.fullmatch(rx, m.group(1), re.IGNORECASE)), None)
            if attr is None:
                continue
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk = text[start:end].strip()
            # First-write-wins so later duplicates don't clobber.
            sliced.setdefault(attr, chunk)

        # Apply the sliced strings to the result, then post-process the
        # risk_level / confidence markers (single tokens, not free text).
        for attr, value in sliced.items():
            if attr == "risk_level_marker":
                # Search the whole slice for a known level rather than trusting
                # the first whitespace-delimited token: "**HIGH**", "HIGH —
                # push-out", and "Level: HIGH" all have to work.
                m_lvl = re.search(
                    r"\b(LOW|MEDIUM|MODERATE|HIGH|CRITICAL|UNKNOWN)\b",
                    value, re.IGNORECASE)
                if m_lvl:
                    token = m_lvl.group(1).upper()
                    result.risk_level = "MEDIUM" if token == "MODERATE" else token
                else:
                    # Unrecognised wording — keep the old behaviour so a level
                    # we simply don't have a name for still reaches the report.
                    first = value.split()[0] if value.split() else ""
                    token = re.sub(r"[^A-Za-z]+", "", first).upper()
                    if token:
                        result.risk_level = token
            elif attr == "confidence_marker":
                num_match = re.search(r"\d+(?:\.\d+)?", value)
                if num_match:
                    try:
                        c = float(num_match.group(0))
                        # Some models output "70%" — normalise to 0..1
                        if c > 1.0:
                            c = c / 100.0
                        result.confidence = max(0.0, min(1.0, c))
                    except ValueError:
                        pass
            else:
                setattr(result, attr, value)

        # Last-resort salvage: if the headed fields didn't populate risk_level,
        # confidence, or risk_assessment, scan the executive summary for inline
        # mentions in either word order: "risk level is HIGH" OR "HIGH risk".
        # The gap between the label and the value has to absorb BOTH shapes,
        # and they are not the same thing:
        #   prose    — "risk level is medium"      (a connector word)
        #   markdown — "**RISK LEVEL:** HIGH"      (`:`, `*`, `*`, space)
        # When this block runs at all it is because header slicing failed,
        # which usually means raw markdown is still sitting inline. `\W{0,3}`
        # on either side of an optional connector covers both; either half
        # alone silently misses the other case.
        _GAP = r"\W{0,3}\s*(?:is|are|of|=|:)?\W{0,3}\s*"
        if result.risk_level == "UNKNOWN" and result.executive_summary:
            patterns = [
                r"risk\s+level" + _GAP + r"(LOW|MEDIUM|MODERATE|HIGH|CRITICAL|UNKNOWN)",
                r"\b(LOW|MEDIUM|MODERATE|HIGH|CRITICAL)\s+risk\b",
                r"indicat\w+\s+(?:a|an)?\s*(LOW|MEDIUM|MODERATE|HIGH|CRITICAL)\s+(?:risk|threat|priority|likelihood)",
            ]
            for pat in patterns:
                m = re.search(pat, result.executive_summary, re.IGNORECASE)
                if m:
                    token = m.group(1).upper()
                    result.risk_level = "MEDIUM" if token == "MODERATE" else token
                    break

        if result.confidence == 0.0 and result.executive_summary:
            m = re.search(r"confidence" + _GAP + r"(\d+(?:\.\d+)?)\s*(%?)",
                          result.executive_summary, re.IGNORECASE)
            if m:
                try:
                    c = float(m.group(1))
                    if m.group(2) == "%" or c > 1.0:
                        c = c / 100.0
                    result.confidence = max(0.0, min(1.0, c))
                except ValueError:
                    pass

        if not result.risk_assessment and result.executive_summary:
            # Pull a sentence that actually talks about risk so the section
            # isn't empty when the model collapsed everything into one para.
            sentences = re.split(r"(?<=[.!?])\s+", result.executive_summary)
            risk_lines = [s for s in sentences
                          if re.search(r"risk|threat|push-?out|theft|grab", s, re.IGNORECASE)]
            if risk_lines:
                result.risk_assessment = " ".join(risk_lines[:3])

    # ------------------------------------------------------------------
    @staticmethod
    def fallback_result(backend: str, error: str) -> CaseReportData:
        return CaseReportData(
            vlm_available=False, vlm_backend=backend,
            error_message=error,
            risk_level="UNKNOWN", confidence=0.0,
        )


# ---------------------------------------------------------------------------
# Module-level helper used by scene_detector for crop labeling
# ---------------------------------------------------------------------------
_md_singleton = None   # lazy-loaded Moondream2 model for scene labeling

_SCENE_KEYWORDS = ["aisle", "shelf", "checkout", "entrance", "exit",
                   "display", "atm", "counter", "floor", "rack", "refrigerator"]


def caption_crop_moondream(crop_bgr) -> str | None:
    """
    Ask Moondream2 to describe a region crop and return the first matching
    store keyword, or None if not recognized.
    Only loads the model on first call; subsequent calls reuse the singleton.
    """
    global _md_singleton
    try:
        import cv2 as _cv2
        from PIL import Image as _Image
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if _md_singleton is None:
            print(f"[SCENE] Loading {MOONDREAM2_MODEL_ID} for region labeling …")
            _tok = AutoTokenizer.from_pretrained(
                MOONDREAM2_MODEL_ID, trust_remote_code=True)
            _mdl = AutoModelForCausalLM.from_pretrained(
                MOONDREAM2_MODEL_ID, trust_remote_code=True, device_map="auto")
            _mdl.eval()
            _md_singleton = (_mdl, _tok)

        mdl, tok = _md_singleton
        rgb = _cv2.cvtColor(crop_bgr, _cv2.COLOR_BGR2RGB)
        img = _Image.fromarray(rgb)
        enc = mdl.encode_image(img)
        answer = mdl.answer_question(enc, "What retail store area or object is this?", tok)
        answer_lc = answer.lower()
        for kw in _SCENE_KEYWORDS:
            if kw in answer_lc:
                return kw
        return None
    except Exception as e:
        print(f"[SCENE] Moondream labeling failed: {e}")
        return None
