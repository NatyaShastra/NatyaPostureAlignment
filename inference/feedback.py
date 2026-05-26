"""
inference/feedback.py — LLM coaching feedback via Groq, with template fallback.

Flow:
  1. build_coaching_prompt()  — construct a structured prompt
  2. call_groq()              — call Groq API (llama-3.3-70b-versatile)
  3. get_llm_feedback()       — orchestrates 1+2 and falls back to templates
"""

from __future__ import annotations
import os
from groq import Groq

# ---------------------------------------------------------------------------
# Expert templates (fallback when API is unavailable)
# ---------------------------------------------------------------------------

ADAVU_FEEDBACK: dict[str, dict] = {
    "Nattadavu": {
        "description": "Nattadavu is a foundational adavu combining a strong aramandi stance with coordinated arm and foot strikes.",
        "key_focus": [
            "Maintain a deep, stable aramandi — knees wide, thighs near parallel to the floor",
            "Strike the floor firmly and rhythmically on each beat",
            "Keep your spine upright — do not lean forward",
        ],
        "common_errors": [
            "Shallow aramandi — the single most common fault in beginners",
            "Loss of foot strike clarity on the second beat",
            "Arms drifting out of the prescribed position",
        ],
        "encouragement": "Nattadavu is the foundation of your entire practice. Master this and everything else becomes accessible.",
    },
    "Thattadavu": {
        "description": "Thattadavu is the first adavu taught. It focuses purely on rhythmic foot-stamping in aramandi.",
        "key_focus": [
            "Each foot strike must be flat, firm, and exactly on the beat",
            "Maintain equal weight distribution — do not favour one side",
            "Your torso must remain still — only the legs are working",
        ],
        "common_errors": [
            "Tip-toe strikes instead of full flat-foot contact",
            "Upper body swaying with each stamp",
            "Losing the tala rhythm after the third repetition",
        ],
        "encouragement": "Thattadavu is your metronome. Every great Bharatanatyam dancer has spent hundreds of hours on exactly this.",
    },
    "ThattimettuAdavu": {
        "description": "Thattimettu combines a flat foot strike with a heel raise in alternation, building rhythmic footwork.",
        "key_focus": [
            "The transition between thatta and mettu must be sharp and clean",
            "The heel raise is controlled — not a full rise to tiptoe",
            "Arms in nayika position must stay level during footwork",
        ],
        "common_errors": [
            "Blurring the distinction between the flat and heel beats",
            "Aramandi height reducing during the mettu phase",
            "Rushing the mettu — give it its full rhythmic value",
        ],
        "encouragement": "This adavu is where footwork starts to get musical. Listen to the mridangam and let your feet respond.",
    },
    "ParavalAdavu": {
        "description": "Paraval Adavu introduces jumping and fast travelling movements, testing stamina and precision.",
        "key_focus": [
            "Land softly — toes first, then heel — to protect the knees",
            "Maintain aramandi even on landing",
            "Arms must complete their movement before the jump, not during it",
        ],
        "common_errors": [
            "Heavy flat-footed landings — absorb impact through calf and ankle",
            "Collapsing the aramandi on each landing",
            "Losing direction — maintain a straight travel line",
        ],
        "encouragement": "Paraval is where the dance gets powerful and alive. Channel that energy with precision.",
    },
    "KudithuMettuAdavu": {
        "description": "Kudithu Mettu involves jumping with heel raises, one of the more physically demanding adavus.",
        "key_focus": [
            "The jump and mettu must synchronise — heel rises as you land, not after",
            "Both knees should track over the second toe",
            "Breath rhythm should match tala — do not hold your breath",
        ],
        "common_errors": [
            "Separating the jump and heel raise into two distinct movements",
            "Knees caving inward on landing",
            "Progressive reduction in aramandi depth due to fatigue",
        ],
        "encouragement": "This adavu builds the stamina that long performances demand. Embrace the difficulty — it is progress.",
    },
    "MandiAdavu": {
        "description": "Mandi Adavu brings the dancer to a seated position on one knee, requiring deep flexibility.",
        "key_focus": [
            "Descend slowly and with full control",
            "The raised knee must stay in line with the shoulder",
            "Hasta must remain precise throughout the descent",
        ],
        "common_errors": [
            "Dropping into mandi rather than lowering with control",
            "The grounded knee lifting off the floor prematurely",
            "Losing hasta formation while focusing on the leg movement",
        ],
        "encouragement": "Mandi is a test of patience and flexibility equally. The slowness is the skill.",
    },
    "SarukkalAdavu": {
        "description": "Sarukkal Adavu involves sliding footwork across the floor, creating a smooth gliding quality.",
        "key_focus": [
            "The sliding foot should barely leave the floor — maximum 1–2 cm clearance",
            "Maintain continuous aramandi throughout the slide",
            "Decelerate smoothly at the end — no abrupt stops",
        ],
        "common_errors": [
            "Lifting the foot too high, turning a slide into a step",
            "Losing aramandi during the slide",
            "Abrupt stops that break the gliding quality",
        ],
        "encouragement": "Sarukkal gives Bharatanatyam its moments of fluidity. Think of the floor as water — skim it.",
    },
    "ThaThaiThamAdavu": {
        "description": "Named after its rhythmic syllables, each syllable corresponds to a distinct footwork movement.",
        "key_focus": [
            "Say the sollukattu aloud while practising until fully automatic",
            "Each syllable is a separate physical event — do not blend them",
            "Arm movements should anticipate the foot by a half-beat",
        ],
        "common_errors": [
            "Rushing the Tham — it is longer than the two preceding beats",
            "Arms and feet becoming desynchronised after the first repetition",
            "Forgetting to return to start position cleanly",
        ],
        "encouragement": "This adavu teaches you to think in syllables rather than counts. That is a fundamental shift.",
    },
    "ThaiThaiThaThamAdavu": {
        "description": "A flowing sequence that challenges the dancer to maintain grace across a longer phrase.",
        "key_focus": [
            "Think of the phrase as a single breath of movement, not individual steps",
            "Transitions between positions should be smooth, not jerky",
            "The final Tham is a moment of resolution — give it full weight",
        ],
        "common_errors": [
            "Breaking the phrase into staccato individual steps",
            "Body weight shifting laterally causing instability",
            "Rushing to complete the phrase — trust the tala",
        ],
        "encouragement": "This adavu is where technique starts to become art. Let the phrase breathe.",
    },
    "ThathaiThaha": {
        "description": "Related to ThaThaiTham, this adavu adds a syllable creating a slightly longer rhythmic phrase.",
        "key_focus": [
            "The added syllable must have its own distinct articulation",
            "Ensure you return to origin after each phrase — spatial drift is common",
            "The head movement (shirobheda) at the final syllable must be deliberate",
        ],
        "common_errors": [
            "Treating the extra syllable as an afterthought and rushing it",
            "Drifting out of position across repetitions",
            "Neglecting facial expression (abhinaya) while focusing on footwork",
        ],
        "encouragement": "As phrases get longer, so does your capacity to hold focus. Every extra syllable is an opportunity.",
    },
    "ThaiyaThaihi": {
        "description": "Characterised by a swaying, lyrical quality, this adavu develops expressive movement.",
        "key_focus": [
            "The lateral sway comes from the hips, not the shoulders",
            "Maintain your aramandi baseline as the torso sways",
            "Eyes (drishti) should travel with the movement, not stay fixed forward",
        ],
        "common_errors": [
            "Swaying from the shoulders creating a top-heavy appearance",
            "Aramandi flattening during the sway",
            "Mechanical eye movements that feel choreographed rather than expressive",
        ],
        "encouragement": "This adavu is where you discover that Bharatanatyam is not just about precision — it is about presence.",
    },
    "ThaHathaJhamTharithamAdavu": {
        "description": "A complex adavu with a longer sollukattu, challenging memory, coordination, and rhythm simultaneously.",
        "key_focus": [
            "Break the phrase into two halves and practise each independently before combining",
            "The Jam syllable is the pivot of the phrase — get it exactly right",
            "Arm positions must be memorised as firmly as the footwork",
        ],
        "common_errors": [
            "Memory lapses causing hesitation mid-phrase",
            "The two halves being performed at different tempos",
            "Arm and foot sequences correct independently but misaligned when combined",
        ],
        "encouragement": "Complex adavus like this one will expand your capacity as a dancer far beyond this step alone.",
    },
    "TheermanaAdavu": {
        "description": "Theerumanam adavus are concluding movements, used to mark the end of a sequence with rhythmic finality.",
        "key_focus": [
            "The final beat must land with full conviction — this is the punctuation",
            "Maintain eye contact with the audience at the conclusion",
            "The stillness after the final beat is part of the adavu — hold it",
        ],
        "common_errors": [
            "Fading out before the final beat rather than landing strong",
            "Moving on before the concluding stillness has been held",
            "A flat expression — the conclusion should radiate completion",
        ],
        "encouragement": "Endings define performances. Every Theerumanam is a chance to make the audience feel something final.",
    },
    "KarthariAdavu": {
        "description": "Named after the kartharimukha hasta (scissors gesture), this adavu integrates the gesture into footwork.",
        "key_focus": [
            "The kartharimukha hasta must be precise — fingers form a clean scissors shape",
            "The hand gesture and the step must arrive simultaneously",
            "Keep the wrists relaxed — tension destroys the elegance of this hasta",
        ],
        "common_errors": [
            "Fingers not fully extended, creating a vague rather than sharp scissors shape",
            "The hasta arriving late, after the foot has already landed",
            "Wrist stiffness making the gesture look angular and harsh",
        ],
        "encouragement": "Karthari is one of the most visually striking adavus when done well. The hand tells a story.",
    },
    "ThaTheiTheiTha": {
        "description": "Features a characteristic rhythmic pattern with alternating weight shifts and arm gestures.",
        "key_focus": [
            "Each weight shift must be crisp and land exactly on the beat",
            "Arm gestures should flow naturally from the shoulder, not be forced",
            "Keep the torso centred — do not over-rotate on the shifts",
        ],
        "common_errors": [
            "Blurring the weight shifts into a continuous sway",
            "Arms lagging behind the footwork",
            "Torso over-rotating on each shift",
        ],
        "encouragement": "This adavu is all about rhythm and balance working together — trust your centre.",
    },
    "UtplavanadaAdavu": {
        "description": "Utplavana involves leaping movements that test both power and precision of landing.",
        "key_focus": [
            "Generate power from a strong aramandi push-off",
            "Land with soft knees — absorb through calf, not the joint",
            "Maintain arm formation through the air, not just on the ground",
        ],
        "common_errors": [
            "Jumping before completing the arm setup",
            "Hard landings that compress the knee",
            "Arms collapsing during the jump",
        ],
        "encouragement": "Every leap is a moment of flight. The height is secondary — the form in the air is everything.",
    },
    "UthsangaAdavu": {
        "description": "Uthsanga adavus involve embracing or encircling arm movements combined with footwork.",
        "key_focus": [
            "The arm arc must be full and deliberate — not a quick afterthought",
            "Footwork should remain grounded and steady while arms move",
            "The combination should feel integrated, not like two separate actions",
        ],
        "common_errors": [
            "Rushing the arm arc to keep up with the feet",
            "Feet losing tala alignment while the arms move",
            "The movement looking like two disconnected actions",
        ],
        "encouragement": "Uthsanga teaches the integration of arms and legs. When they speak the same language, the adavu sings.",
    },
    "KorvaiAdavu": {
        "description": "Korvai adavus are concluding sequences used at cadence points, requiring rhythmic precision.",
        "key_focus": [
            "The korvai must resolve exactly on the sam (first beat)",
            "Build intensity progressively through the sequence",
            "Maintain a strong aramandi right through to the final moment",
        ],
        "common_errors": [
            "Arriving at the sam early or late — destroying the rhythmic resolution",
            "Energy dropping before the final beat",
            "Aramandi collapsing at the resolution point",
        ],
        "encouragement": "A well-executed korvai is one of the most satisfying moments in Bharatanatyam. Own it.",
    },
}

DEFAULT_FEEDBACK: dict = {
    "description": "This adavu is a foundational movement combining footwork, hand gestures, and rhythmic precision.",
    "key_focus": [
        "Maintain a deep and stable aramandi throughout",
        "Ensure your hasta is precise and held with intention",
        "Stay connected to the tala — let rhythm guide every movement",
    ],
    "common_errors": [
        "Losing aramandi depth as fatigue sets in",
        "Hasta formation becoming imprecise",
        "Disconnecting from the tala and rushing or dragging steps",
    ],
    "encouragement": "Every adavu is a conversation between your body and the rhythm. Keep listening and keep refining.",
}


# ---------------------------------------------------------------------------
# Groq client (initialised lazily)
# ---------------------------------------------------------------------------

_groq_client: Groq | None = None


def init_groq_client(api_key: str | None = None) -> None:
    """
    Initialise the module-level Groq client.
    Call once at startup. api_key defaults to GROQ_API_KEY env var.
    """
    global _groq_client
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if key:
        _groq_client = Groq(api_key=key)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_coaching_prompt(
    adavu_class: str,
    score_result: dict,
    flagged_joints: list[dict],
    confidence: float,
) -> str:
    if flagged_joints:
        lines = []
        for j in sorted(flagged_joints, key=lambda x: -x["deviation"]):
            name      = j["joint"].replace("_", " ")
            actual    = j["measured"]
            ref       = j["reference"]
            dev       = j["deviation_deg"]
            direction = "too open" if actual > ref else "too closed"
            lines.append(f"  - {name}: measured {actual:.1f}°, reference {ref:.1f}° ({direction} by {dev:.1f}°)")
        flagged_text = "\n".join(lines)
    else:
        flagged_text = "  None — all joint angles within acceptable range."

    region_lines = "\n".join(
        f"  - {r.capitalize()}: {score_result['region_scores'].get(r, 100):.1f}%"
        for r in ["legs", "arms", "torso"]
    )

    return f"""You are an expert Bharatanatyam dance teacher giving feedback to a student.

The student just performed: {adavu_class}
Detection confidence: {confidence:.1%}

SCORE SUMMARY:
{region_lines}
Overall score: {score_result['overall_score']:.1f}%
Result: {"PASSED" if score_result['passed'] else "NOT PASSED"} (threshold: {score_result['pass_threshold']:.0f}%)

FLAGGED JOINT ANGLES (deviating from reference by more than 1.5 standard deviations):
{flagged_text}

Write a coaching response with exactly these three sections:

1. WHAT YOU DID WELL (2–3 sentences — genuine specific praise based on the score)
2. CORRECTIONS NEEDED (one clear sentence per flagged joint, explaining physically what to fix — be specific about body position, not just the angle number)
3. NEXT PRACTICE FOCUS (1–2 sentences — one concrete thing to drill in the next session)

Be warm but direct. Use the vocabulary of Bharatanatyam where appropriate (aramandi, hasta, tala, etc.).
Do not repeat the numbers from the score report — translate them into physical, actionable language.
Keep the total response under 200 words."""


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_groq(prompt: str, max_tokens: int = 400) -> str | None:
    """Call the Groq API. Returns response text or None on failure."""
    if _groq_client is None:
        return None
    try:
        response = _groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Bharatanatyam dance teacher. Give concise, specific, warm coaching feedback.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  Groq API call failed: {e}")
        return None


def get_llm_feedback(
    adavu_class: str,
    score_result: dict,
    flagged_joints: list[dict],
    confidence: float,
) -> tuple[str, str]:
    """
    Get personalised coaching feedback from Groq, falling back to templates.

    Returns:
        (feedback_text, source)  where source is "llm" or "template"
    """
    prompt    = build_coaching_prompt(adavu_class, score_result, flagged_joints, confidence)
    llm_text  = call_groq(prompt)

    if llm_text:
        return llm_text, "llm"

    # Template fallback
    fb = ADAVU_FEEDBACK.get(adavu_class, DEFAULT_FEEDBACK)
    fallback = (
        f"Key focus: {fb['key_focus'][0]}\n"
        f"Watch for: {fb['common_errors'][0]}\n"
        f"{fb['encouragement']}"
    )
    return fallback, "template"