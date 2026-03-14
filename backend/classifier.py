"""
Gaslight Guard — NLP Classifier
================================
Uses a fine-tuned DistilBERT binary sarcasm detector as the core model,
combined with rule-based pattern matching to distinguish four categories:

Labels: Sincere | Sarcastic | Passive-Aggressive | Gaslighting
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Model loader — fine-tuned DistilBERT (binary sarcasm detection)
# ---------------------------------------------------------------------------
_sarcasm_pipeline = None

_LOCAL_MODEL_DIR = Path(__file__).parent / "model" / "sarcasm-distilbert"
_LABEL_NAMES = ["Sincere", "Sarcastic", "Passive-Aggressive", "Gaslighting"]


def _model_is_available() -> bool:
    return (_LOCAL_MODEL_DIR / "config.json").exists()


def _load_sarcasm_model():
    """Load the fine-tuned DistilBERT binary sarcasm classifier."""
    global _sarcasm_pipeline
    if _sarcasm_pipeline is None:
        from transformers import pipeline as hf_pipeline
        print(f"[classifier] Loading DistilBERT sarcasm model from {_LOCAL_MODEL_DIR}")
        _sarcasm_pipeline = hf_pipeline(
            "text-classification",
            model=str(_LOCAL_MODEL_DIR),
            tokenizer=str(_LOCAL_MODEL_DIR),
            device=-1,
            top_k=None,
        )
    return _sarcasm_pipeline


def _get_scores_from_model(text: str) -> Dict[str, float]:
    """
    Uses the binary DistilBERT sarcasm score combined with rule-based pattern
    matching to produce scores for all 4 labels.
    """
    pipe = _load_sarcasm_model()
    results = pipe(text, truncation=True, max_length=128)
    score_map = {r["label"]: r["score"] for r in results}
    sarcasm_prob = score_map.get("LABEL_1", score_map.get("sarcastic", 0.0))
    sincere_prob = 1.0 - sarcasm_prob

    pscores = _pattern_scores(text)
    gl_pattern = pscores["Gaslighting"]
    pa_pattern = pscores["Passive-Aggressive"]

    if gl_pattern > 0.15:
        return {
            "Gaslighting": 0.5 + gl_pattern * 0.3 + sarcasm_prob * 0.1,
            "Sarcastic": sarcasm_prob * 0.15,
            "Passive-Aggressive": pa_pattern * 0.2,
            "Sincere": sincere_prob * 0.15,
        }
    elif pa_pattern > 0.10:
        return {
            "Passive-Aggressive": 0.4 + pa_pattern * 0.3 + sarcasm_prob * 0.15,
            "Sarcastic": sarcasm_prob * 0.2,
            "Gaslighting": gl_pattern * 0.1,
            "Sincere": sincere_prob * 0.15,
        }
    else:
        return {
            "Sarcastic": sarcasm_prob * 0.85,
            "Sincere": sincere_prob * 0.85,
            "Passive-Aggressive": pa_pattern * 0.1,
            "Gaslighting": gl_pattern * 0.05,
        }


# ---------------------------------------------------------------------------
# Gaslighting & passive-aggression linguistic pattern library
# ---------------------------------------------------------------------------
GASLIGHTING_PATTERNS = [
    r"\byou('re| are) (too |being )?(so |very |really )?(sensitive|dramatic|overreacting|paranoid|crazy|insane|delusional)\b",
    r"\bi never (said|did) that\b",
    r"\bthat('s| is) not what (happened|i said)\b",
    r"\byou('re| are) (imagining|making up|twisting)\b",
    r"\bno one (else|other) (believes|thinks|agrees)\b",
    r"\byou always (do this|act like|make everything)\b",
    r"\bit('s| is) (always|all) (your fault|about you)\b",
    r"\bstop (being so|acting so) (emotional|dramatic|sensitive)\b",
    r"\byou need (help|therapy|to calm down)\b",
    r"\bi('m| am) (just|only) (joking|kidding|teasing)\b",
    r"\bcan('t| not) you (take|handle) a joke\b",
    r"\byou('re| are) (remembering|misremembering) it wrong\b",
    r"\beveryone (thinks|agrees|knows) you('re| are)\b",
]

PASSIVE_AGGRESSIVE_PATTERNS = [
    r"\bwhatever\b",
    r"\bfine[,.]?\s*(then)?\b",
    r"\bif you (say|think) so\b",
    r"\bsure[,.]?\s*(sure)?\b",
    r"\bno (problem|worries)[,.]?\s*(at all)?\b",
    r"\bi('m| am) (just|not) (saying|sure)\b",
    r"\bdo (whatever|what) you want\b",
    r"\bi('ll| will) (just|probably) (leave|go|disappear)\b",
    r"\bmust be (nice|great|wonderful)\b",
    r"\bnot (that|like) (it matters|anyone cares|you care)\b",
    r"\boh[,.]?\s*(so)?\s*(now|suddenly)\b",
    r"\bif (that('s| is) what) you (want|need|think)\b",
    r"\bdon('t| not) (worry|mind) (about|me)\b",
    r"\bi('m| am) (fine|okay)[,.]?\s*really\b",
]

SARCASM_CUES = [
    r"\b(oh\s+)?wow[,.]?\s+(just\s+)?wow\b",
    r"\boh\s+(how\s+)?(absolutely\s+)?(wonderful|amazing|great|fantastic|brilliant|lovely)\b",
    r"\b(yeah|yep|sure)[,.]?\s+(right|totally|obviously)\b",
    r"\b(clearly|obviously)\s+(you('re| are)|that('s| is))\b",
    r"\b(so\s+)?(clever|smart|brilliant)\s+(of\s+you|move|choice)\b",
    r"\b(because\s+)?that('s| is)\s+(definitely|totally|obviously)\s+(not|wrong|right)\b",
    r"\boh\s+(please|come on)\b",
    r"\b(as\s+)?if\s+(that|this|you)\b",
    r"\b(what|how)\s+(a|an)\s+(great|wonderful|brilliant|excellent)\s+(idea|choice|plan)\b",
]

# Sanity check templates keyed by dominant label
SANITY_CHECKS = {
    "Gaslighting": (
        "What you're experiencing is real. This message contains language patterns "
        "commonly used to make someone doubt their own memory or perception. "
        "You are NOT being overly sensitive — gaslighting is a form of psychological "
        "manipulation, and recognizing it is the first step toward protecting yourself."
    ),
    "Sarcastic": (
        "This message uses sarcasm to deliver a message that feels hurtful despite "
        "appearing superficially positive. Your emotional response to it is valid. "
        "Repeated sarcastic communication can erode self-esteem over time."
    ),
    "Passive-Aggressive": (
        "This message carries indirect hostility behind polite or neutral-sounding words. "
        "The conflict is real, even if it's never stated openly. Passive-aggression is a "
        "communication style that avoids direct honesty and can be emotionally exhausting."
    ),
    "Sincere": (
        "This message reads as genuine and straightforward. No hidden hostility detected."
    ),
}

COPING_STRATEGIES = {
    "Gaslighting": [
        "Write down the actual events in a private journal while they're fresh.",
        "Reach out to a trusted friend or counselor to get an outside perspective.",
        "Set a clear boundary: 'I remember it differently and that's okay — we can agree to disagree.'",
        "Consider whether this is a recurring pattern in the relationship.",
        "Contact the Crisis Text Line: Text HOME to 741741 (free, 24/7).",
    ],
    "Sarcastic": [
        "Name it calmly: 'That sounded sarcastic to me — did you mean something else?'",
        "Decide if this deserves a response — sometimes silence is more powerful.",
        "Reflect on whether this person consistently uses sarcasm as a weapon.",
        "Practice self-affirmations: your feelings about this are legitimate.",
    ],
    "Passive-Aggressive": [
        "Invite direct conversation: 'It seems like something is bothering you. Can we talk about it openly?'",
        "Avoid playing the guessing game — ask for clarity directly.",
        "Protect your energy by limiting exposure if the pattern is consistent.",
        "Practice assertive communication in your own responses.",
    ],
    "Sincere": [
        "This looks like healthy communication — no action needed.",
        "If something still feels off, trust your gut and bring it up directly.",
    ],
}

RESPONSE_SUGGESTIONS = {
    "Gaslighting": [
        '"I understand we remember it differently. I trust my own memory."',
        '"I don\'t think this conversation is productive right now. I\'ll revisit it later."',
        '"My feelings are valid regardless of your interpretation of events."',
    ],
    "Sarcastic": [
        '"Could you say what you mean directly? I want to understand you."',
        '"I noticed that was sarcastic. If you\'re upset, let\'s talk about it."',
        '"I\'d prefer we communicate more directly with each other."',
    ],
    "Passive-Aggressive": [
        '"It sounds like you might be frustrated. What\'s really going on?"',
        '"I\'d rather we be honest with each other than talk around the issue."',
        '"I\'m open to hearing what you\'re actually feeling."',
    ],
    "Sincere": [
        '"Thanks for being straightforward — I appreciate that."',
        '"I hear you. Let\'s keep this kind of open dialogue going."',
    ],
}


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

GASLIGHTING_KEYWORDS = [
    "you're imagining", "never said", "never happened", "too sensitive",
    "overreacting", "you're crazy", "you're paranoid", "making things up",
    "remember it wrong", "misremembering", "you always do this",
    "your fault", "brought this on yourself", "everyone agrees",
    "no one believes you", "you're delusional", "stop being dramatic",
    "need therapy", "just joking", "can't take a joke", "you started this",
    "your friends lied", "you're the problem", "you're unstable",
    "twisted my words", "rewriting history", "literally in your head",
]

PASSIVE_AGGRESSIVE_KEYWORDS = [
    "fine whatever", "do whatever you want", "don't worry about me",
    "i'll just", "like always", "as usual", "no worries at all",
    "must be nice", "i'm fine really", "forget i said", "if you say so",
    "i guess", "not that it matters", "oh so now", "suddenly you",
    "you remembered", "only waited", "handle it alone", "no one asked",
    "i'll just sit here", "go have fun", "i'll manage",
]

SARCASM_KEYWORDS = [
    "oh wow", "oh great", "oh brilliant", "oh fantastic", "oh sure",
    "oh absolutely", "oh perfect", "what a surprise", "totally makes sense",
    "clearly the best", "genius idea", "wonderful plan", "love how",
    "great job breaking", "couldn't have seen", "mark the calendar",
    "yeah right", "sure sure", "obviously", "groundbreaking",
    "revolutionary", "who could have seen this coming", "shocker",
    "color me surprised", "how convenient",
]

SINCERE_KEYWORDS = [
    "i appreciate", "thank you", "i'm sorry", "i understand",
    "i hear you", "that must be", "i care", "how are you",
    "i'm here for you", "let's talk", "i want to help",
    "genuinely", "honestly", "i value", "i respect",
]


def _keyword_scores(text: str) -> Dict[str, float]:
    """Enhanced keyword-based scoring when no model is available."""
    t = text.lower()

    def count_hits(keywords):
        return sum(1 for kw in keywords if kw in t)

    gl = count_hits(GASLIGHTING_KEYWORDS)
    pa = count_hits(PASSIVE_AGGRESSIVE_KEYWORDS)
    sa = count_hits(SARCASM_KEYWORDS)
    si = count_hits(SINCERE_KEYWORDS)

    # Also include pattern scores
    pscores = _pattern_scores(text)
    gl += pscores["Gaslighting"] * 5
    pa += pscores["Passive-Aggressive"] * 5
    sa += pscores["Sarcastic"] * 5

    # Base scores with slight Sincere bias when nothing detected
    scores = {
        "Sincere":            max(0.1, si * 0.4 + (0.5 if (gl + pa + sa) == 0 else 0)),
        "Sarcastic":          sa * 0.4,
        "Passive-Aggressive": pa * 0.4,
        "Gaslighting":        gl * 0.4,
    }
    total = sum(scores.values()) or 1
    return {k: v / total for k, v in scores.items()}


def _pattern_scores(text: str) -> Dict[str, float]:
    """Return raw pattern-match scores (0-1) for each toxic label."""
    t = text.lower()
    gl = sum(1 for p in GASLIGHTING_PATTERNS if re.search(p, t)) / len(GASLIGHTING_PATTERNS)
    pa = sum(1 for p in PASSIVE_AGGRESSIVE_PATTERNS if re.search(p, t)) / len(PASSIVE_AGGRESSIVE_PATTERNS)
    sa = sum(1 for p in SARCASM_CUES if re.search(p, t)) / len(SARCASM_CUES)
    return {"Gaslighting": gl, "Passive-Aggressive": pa, "Sarcastic": sa}


@dataclass
class AnalysisResult:
    label: str
    confidence: float
    all_scores: Dict[str, float]
    sanity_check: str
    coping_strategies: List[str]
    suggested_responses: List[str]
    severity: str  # "low" | "medium" | "high"
    pattern_highlights: List[str]


def _severity(conf: float, label: str) -> str:
    if label == "Sincere":
        return "low"
    if conf >= 0.70:
        return "high"
    if conf >= 0.45:
        return "medium"
    return "low"


def _find_highlights(text: str) -> List[str]:
    """Return substrings that matched toxic patterns."""
    t = text.lower()
    hits = []
    for pat in GASLIGHTING_PATTERNS + PASSIVE_AGGRESSIVE_PATTERNS + SARCASM_CUES:
        m = re.search(pat, t)
        if m:
            hits.append(m.group(0))
    return list(set(hits))[:5]


def _model_info() -> Dict:
    """Return metadata about which model is active."""
    if _model_is_available():
        return {"type": "fine-tuned", "base_model": "distilbert-base-uncased"}
    return {"type": "rule-only", "base_model": "none"}


def analyze_single(text: str) -> AnalysisResult:
    """Analyze a single message using DistilBERT + rule-based pattern matching."""
    try:
        model_scores = _get_scores_from_model(text)
    except Exception as e:
        print(f"[classifier] Model unavailable ({e}) — using keyword fallback")
        model_scores = _keyword_scores(text)

    total = sum(model_scores.values()) or 1
    blended = {k: round(v / total, 4) for k, v in model_scores.items()}

    top_label = max(blended, key=blended.__getitem__)
    conf = blended[top_label]

    return AnalysisResult(
        label=top_label,
        confidence=conf,
        all_scores=blended,
        sanity_check=SANITY_CHECKS[top_label],
        coping_strategies=COPING_STRATEGIES[top_label],
        suggested_responses=RESPONSE_SUGGESTIONS[top_label],
        severity=_severity(conf, top_label),
        pattern_highlights=_find_highlights(text),
    )


def analyze_conversation(messages: List[str]) -> Dict:
    """
    Contextual Consistency Audit over a conversation window.
    Looks for escalating toxic patterns across ≤5 messages.
    """
    window = messages[-5:]  # last 5 messages
    individual = [analyze_single(m) for m in window]

    label_counts: Dict[str, int] = {"Sincere": 0, "Sarcastic": 0,
                                     "Passive-Aggressive": 0, "Gaslighting": 0}
    for r in individual:
        label_counts[r.label] += 1

    toxic_count = sum(v for k, v in label_counts.items() if k != "Sincere")
    pattern_detected = toxic_count >= 2

    # Dominant toxic pattern
    toxic_labels = {k: v for k, v in label_counts.items() if k != "Sincere"}
    dominant = max(toxic_labels, key=toxic_labels.__getitem__) if any(toxic_labels.values()) else "Sincere"

    escalation = False
    if len(individual) >= 3:
        recent_toxic = sum(1 for r in individual[-3:] if r.label != "Sincere")
        escalation = recent_toxic >= 2

    severity_map = {"low": 1, "medium": 2, "high": 3}
    avg_sev = sum(severity_map[r.severity] for r in individual) / len(individual)
    overall_severity = "high" if avg_sev >= 2.5 else "medium" if avg_sev >= 1.5 else "low"

    if pattern_detected:
        context_note = (
            f"Over this {len(window)}-message window, a recurring pattern of "
            f"**{dominant}** communication was detected ({toxic_count}/{len(window)} messages). "
        )
        if escalation:
            context_note += (
                "The pattern appears to be **escalating** in recent messages. "
                "This is a significant warning sign."
            )
        else:
            context_note += "This persistent pattern — not a one-off remark — is what makes it harmful."
    else:
        context_note = "No persistent toxic pattern detected across this conversation window."

    return {
        "individual_results": [
            {
                "message": window[i],
                "label": r.label,
                "confidence": r.confidence,
                "all_scores": r.all_scores,
                "severity": r.severity,
                "pattern_highlights": r.pattern_highlights,
            }
            for i, r in enumerate(individual)
        ],
        "label_counts": label_counts,
        "pattern_detected": pattern_detected,
        "dominant_pattern": dominant,
        "escalation_detected": escalation,
        "overall_severity": overall_severity,
        "context_note": context_note,
        "sanity_check": SANITY_CHECKS[dominant],
        "coping_strategies": COPING_STRATEGIES[dominant],
        "suggested_responses": RESPONSE_SUGGESTIONS[dominant],
    }


# ---------------------------------------------------------------------------
# Sarcastic Sentiment Flip stress test (Evaluation Plan)
# ---------------------------------------------------------------------------
STRESS_TEST_SAMPLES = [
    ("Oh wow, you're absolutely brilliant for doing that.", "Sarcastic"),
    ("What a fantastic idea, I'm sure it'll work out great.", "Sarcastic"),
    ("You're so smart, I'm amazed you managed to tie your shoes today.", "Sarcastic"),
    ("Sure, because that totally makes sense.", "Sarcastic"),
    ("Oh, clearly you know best about everything.", "Sarcastic"),
    ("Great job breaking everything again.", "Sarcastic"),
    ("Yeah, right, I'm sure you didn't mean it.", "Sarcastic"),
    ("Oh how wonderful, another delay.", "Sarcastic"),
    ("Brilliant, just what we needed.", "Sarcastic"),
    ("You're such a great friend, always there when it's convenient for you.", "Sarcastic"),
]

def run_stress_test(additional_samples: List[Tuple[str, str]] = None) -> Dict:
    """Sarcastic Sentiment Flip Test — checks if sarcasm is mis-labelled as Sincere."""
    samples = STRESS_TEST_SAMPLES[:]
    if additional_samples:
        samples.extend(additional_samples)

    results = []
    correct = 0
    for text, expected in samples:
        r = analyze_single(text)
        passed = r.label == expected
        if passed:
            correct += 1
        results.append({
            "text": text,
            "expected": expected,
            "predicted": r.label,
            "confidence": r.confidence,
            "passed": passed,
        })

    accuracy = round(correct / len(samples), 4)
    return {
        "total": len(samples),
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
        "verdict": "PASS" if accuracy >= 0.70 else "FAIL",
    }
