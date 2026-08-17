"""
Grey-Swan Text Event Analyzer
==============================
Analyzes free-text descriptions of world events and estimates the probability
of a grey-swan (extreme) market regime transition.

Two-tier approach:
  1. FinBERT financial sentiment model  (primary, if transformers available)
  2. Keyword-based risk scanner          (always available, fallback)

Signals are combined and mapped to regime probabilities using the
same 5-class regime schema as the Grey-Swan model:
  0 = Normal, 1 = Elevated-Vol, 2 = Stress, 3 = Transition, 4 = Extreme

Usage:
    from text_analyzer import analyze_event
    result = analyze_event("Federal Reserve emergency rate cut amid bank failures")
"""

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Keyword dictionaries  (category -> list of patterns)
# ---------------------------------------------------------------------------

CREDIT_COLLAPSE = [
    "bank failure", "bank run", "bank collapse", "banking crisis",
    "bankruptcy", "insolvency", "default", "sovereign default", "debt default",
    "counterparty risk", "counterparty failure", "contagion",
    "credit freeze", "credit crunch", "credit crisis", "credit event",
    "bailout", "bail-in", "Too big to fail", "systemic risk",
    "systemic failure", "banking collapse", "savings and loan",
    "fractional reserve", "deposit flight", "capital flight",
    "credit default swap", "CDS blowout", "margin call cascade",
    "leverage unwind", "deleveraging", "margin call",
    "credit markets freeze", "credit market", "bond default",
    "debt crisis", "financial crisis", "bank failure",
]

LIQUIDITY = [
    "liquidity crisis", "liquidity crunch", "liquidity squeeze",
    "frozen markets", "market freeze", "seizure",
    "fire sale", "forced selling", "liquidation",
    "repo market", "repo spike", "interbank lending",
    "money market fund", "commercial paper",
    "dollar shortage", "funding gap", "cash crunch",
    "redemption wave", "run on the fund", "capital withdrawal",
    "markets freeze", "frozen", "liquidity",
]

GEOPOLITICAL = [
    "war", "military conflict", "invasion", "occupation",
    "sanctions", "trade war", "embargo", "blockade",
    "tariff shock", "trade restriction", "export ban",
    "nuclear threat", "missile launch", "military escalation",
    "regime change", "coup", "civil war", "insurgency",
    "cyberattack", "cyber warfare", "critical infrastructure attack",
    "territorial dispute", "border conflict", "proxy war",
    "energy embargo", "oil embargo", "supply chain disruption",
    "political crisis", "government shutdown", "sovereign crisis",
    "assassination", "terrorist attack", "mass casualty",
    "refugee crisis", "humanitarian crisis",
    "military invasion", "energy crisis",
]

MACRO = [
    "recession", "depression", "stagflation", "hyperinflation",
    "inflation surge", "inflation spike", "price shock",
    "yield curve inversion", "yield curve control",
    "rate hike", "emergency rate", "rate cut cycle",
    "unemployment surge", "jobless claims", "mass layoffs",
    "gdp contraction", "economic contraction", "growth collapse",
    "deflation", "debt deflation", "balance sheet recession",
    "fiscal crisis", "austerity", "monetary policy failure",
    "central bank error", "policy mistake", "wrong-footed",
    "dollar collapse", "currency crisis", "currency devaluation",
    "capital controls", "foreign reserve depletion",
    "supply shock", "demand shock", "cost-push inflation",
    "property crisis",
]

MARKET = [
    "flash crash", "circuit breaker", "trading halt",
    "volatility spike", "vix spike", "volmageddon",
    "market crash", "market collapse", "market meltdown",
    "black monday", "black swan", "grey swan",
    "panic selling", "panic", "mass selloff",
    "correction", "bear market", "capitulation",
    "short squeeze", "gamma squeeze", "naked short",
    "algorithmic crash", "liquidity vacuum", "air pocket",
    "limit down", "limit up", "gap down", "gap up",
    "portfolio insurance", "stop loss cascade",
    "options expiry", "quad witching",
    "wipes", "wipeout",
]

CONTAGION = [
    "contagion effect", "domino effect", "chain reaction",
    "systemic collapse", "cascading failure",
    "emerging market crisis", "developing market stress",
    "dollar milkshake", "risk-off", "flight to safety",
    "flight to quality", "safe haven", "treasury rally",
    "gold spike", "crypto crash", "crypto winter",
    "correlation spike", "correlation breakdown",
    "volatility contagion", "cross-asset selloff",
    "global recession", "synchronized downturn",
]

RISK_CATEGORIES = {
    "Credit/Collapse": CREDIT_COLLAPSE,
    "Liquidity": LIQUIDITY,
    "Geopolitical": GEOPOLITICAL,
    "Macro": MACRO,
    "Market": MARKET,
    "Contagion": CONTAGION,
}

# ---------------------------------------------------------------------------
# Financial sentiment lexicon
# ---------------------------------------------------------------------------

BEARISH_WORDS = {
    "crash": -2.0, "crashes": -2.0, "collapse": -2.0, "crisis": -1.8, "crises": -1.8,
    "panic": -1.8, "panicked": -1.8, "meltdown": -1.8,
    "plunge": -1.5, "plunges": -1.5, "crumble": -1.5, "freefall": -1.5,
    "tumble": -1.3, "tumbles": -1.3, "slump": -1.3, "slumps": -1.3,
    "selloff": -1.2, "sell-off": -1.2, "selloffs": -1.2,
    "decline": -0.8, "declines": -0.8, "drop": -0.8, "drops": -0.8,
    "fall": -0.7, "falls": -0.7, "weakness": -0.8,
    "recession": -1.5, "depression": -2.0, "default": -1.8, "defaults": -1.8,
    "bankruptcy": -1.8, "bankruptcies": -1.8, "insolvency": -1.7,
    "failure": -1.5, "failures": -1.5, "failed": -1.3,
    "fear": -1.2, "fears": -1.2, "uncertainty": -0.8, "risk": -0.5,
    "volatile": -0.7, "volatility": -0.7, "instability": -1.0,
    "turmoil": -1.5, "chaos": -1.5, "disaster": -1.8,
    "catastrophe": -1.8, "emergency": -1.0, "rescue": -1.2,
    "bailout": -1.3, "bailouts": -1.3,
    "loss": -0.8, "losses": -1.0, "damage": -0.8, "hit": -0.5,
    "collapse": -2.0, "collapses": -2.0, "plummet": -1.8, "plummets": -1.8,
    "evaporate": -1.5, "evaporates": -1.5,
    "wiped out": -2.0, "devastating": -1.8, "catastrophic": -2.0,
    "doom": -1.8, "gloom": -1.2, "bloodbath": -2.0,
    "dead": -1.0, "dying": -1.0, "unprecedented": -0.5,
    "wartime": -1.2, "devastation": -1.8,
    "freeze": -1.3, "freezes": -1.3, "freezing": -1.3,
    "worst": -1.5, "surge": 0.0,
}

BULLISH_WORDS = {
    "rally": 1.5, "surge": 1.3, "soar": 1.5, "boom": 1.5,
    "bull": 1.2, "growth": 0.8, "recovery": 1.0, "expansion": 1.0,
    "optimism": 1.0, "confidence": 0.8, "breakout": 1.0,
    "all-time high": 1.5, "record high": 1.5, "upbeat": 0.8,
    "strong": 0.6, "robust": 0.7, "resilient": 0.7,
    "stimulus": 0.5, "easing": 0.5, "support": 0.5,
    "rebound": 1.0, "bounce": 0.8, "momentum": 0.7,
    "beat expectations": 1.0, "outperform": 0.8,
}

# ---------------------------------------------------------------------------
# Intensity modifiers
# ---------------------------------------------------------------------------

INTENSIFIERS = {
    "very": 1.5, "extremely": 2.0, "severely": 1.8,
    "massive": 1.8, "huge": 1.5, "unprecedented": 2.0,
    "historic": 1.5, "record": 1.3, "sharp": 1.3,
    "sudden": 1.3, "abrupt": 1.3, "dramatic": 1.5,
    "catastrophic": 2.0, "devastating": 2.0, "worst": 2.0,
    "first": 0.8, "major": 1.3, "significant": 1.2,
}

NEGATORS = {"not", "no", "never", "neither", "nor", "barely", "hardly", "without", "avoid"}

# ---------------------------------------------------------------------------
# Regime mapping
# ---------------------------------------------------------------------------

REGIME_LABELS = ["Normal", "Elevated-Vol", "Stress", "Transition", "Extreme"]

# Base regime probabilities (market equilibrium)
BASE_PROBS = np.array([0.55, 0.20, 0.12, 0.08, 0.05])

# How each signal shifts the distribution
SIGNAL_SHIFTS = {
    "low":     np.array([ 0.05, -0.01, -0.02, -0.01, -0.01]),
    "mild":    np.array([-0.10,  0.10,  0.00,  0.00,  0.00]),
    "moderate":np.array([-0.25,  0.05,  0.10,  0.06,  0.04]),
    "high":    np.array([-0.40, -0.05,  0.15,  0.15,  0.15]),
    "extreme": np.array([-0.50, -0.10, -0.05,  0.15,  0.50]),
}


# ---------------------------------------------------------------------------
# Core analyzer class
# ---------------------------------------------------------------------------

class TextEventAnalyzer:
    """Analyzes free-text event descriptions for grey-swan risk signals."""

    def __init__(self):
        self._sentiment_pipeline = None
        self._model_name = None
        self._load_sentiment_model()

    def _load_sentiment_model(self):
        """Try to load FinBERT or DistilBERT sentiment model."""
        try:
            from transformers import pipeline as hf_pipeline
            # Try FinBERT first (financial domain), fall back to distilbert
            for name in [
                "ProsusAI/finbert",
                "distilbert-base-uncased-finetuned-sst-2-english",
            ]:
                try:
                    self._sentiment_pipeline = hf_pipeline(
                        "sentiment-analysis",
                        model=name,
                        truncation=True,
                        max_length=512,
                    )
                    self._model_name = name
                    break
                except Exception:
                    continue
        except ImportError:
            pass
        except Exception:
            pass

    @property
    def has_transformer(self) -> bool:
        return self._sentiment_pipeline is not None

    @property
    def model_info(self) -> str:
        if self._model_name:
            return f"transformer ({self._model_name})"
        return "keyword-only"

    # --- Sentiment -----------------------------------------------------------

    def _transformer_sentiment(self, text: str) -> float:
        """Returns sentiment in [-1, 1] using transformer model."""
        if not self._sentiment_pipeline:
            return 0.0
        try:
            result = self._sentiment_pipeline(text[:512])[0]
            label = result["label"].lower()
            score = result["score"]
            if "negative" in label or "neg" in label or label == "LABEL_0":
                return -score
            elif "positive" in label or "pos" in label or label == "LABEL_2":
                return score
            else:
                return 0.0
        except Exception:
            return 0.0

    def _lexicon_sentiment(self, text: str) -> float:
        """Rule-based financial sentiment from lexicon."""
        clean = re.sub(r'[^\w\s-]', ' ', text.lower())
        words = clean.split()
        total = 0.0
        count = 0
        i = 0
        while i < len(words):
            w = words[i]
            multiplier = 1.0
            # Check for intensifier before sentiment word
            if i > 0 and words[i - 1] in INTENSIFIERS:
                multiplier = INTENSIFIERS[words[i - 1]]
            # Check for negator
            negated = False
            if i > 0 and words[i - 1] in NEGATORS:
                negated = True

            # Check multi-word phrases first
            matched = False
            for phrase in list(BEARISH_WORDS.keys()) + list(BULLISH_WORDS.keys()):
                if " " in phrase and phrase in text.lower():
                    val = BEARISH_WORDS.get(phrase, BULLISH_WORDS.get(phrase, 0))
                    if negated:
                        val *= -1
                    total += val * multiplier * 2  # phrase match weighted higher
                    count += 1
                    matched = True
                    break

            if not matched:
                if w in BEARISH_WORDS:
                    val = BEARISH_WORDS[w] * multiplier
                    if negated:
                        val *= -1
                    total += val
                    count += 1
                elif w in BULLISH_WORDS:
                    val = BULLISH_WORDS[w] * multiplier
                    if negated:
                        val *= -1
                    total += val
                    count += 1
            i += 1

        if count == 0:
            return 0.0
        raw = total / count
        return max(-1.0, min(1.0, raw / 2.0))

    def compute_sentiment(self, text: str) -> Tuple[float, str]:
        """Returns (score in [-1,1], source)."""
        if self.has_transformer:
            s = self._transformer_sentiment(text)
            return s, self._model_name
        s = self._lexicon_sentiment(text)
        return s, "lexicon"

    # --- Keyword risk scanning -----------------------------------------------

    def scan_risk_keywords(self, text: str) -> Dict[str, List[str]]:
        """Detect financial risk keywords by category."""
        text_lower = text.lower()
        found = {}
        for cat, terms in RISK_CATEGORIES.items():
            hits = []
            for term in terms:
                if term.lower() in text_lower:
                    hits.append(term)
            if hits:
                found[cat] = hits
        return found

    def risk_score(self, keyword_hits: Dict[str, List[str]], sentiment: float) -> float:
        """Combine keyword hits + sentiment into a 0-1 risk score."""
        cat_weights = {
            "Credit/Collapse": 1.5,
            "Liquidity": 1.3,
            "Geopolitical": 1.1,
            "Macro": 1.0,
            "Market": 1.2,
            "Contagion": 1.4,
        }
        kw_score = 0.0
        total_hits = 0
        categories_hit = 0
        for cat, hits in keyword_hits.items():
            w = cat_weights.get(cat, 1.0)
            kw_score += len(hits) * w
            total_hits += len(hits)
            categories_hit += 1

        # Normalize keyword score with diminishing returns
        kw_norm = min(kw_score / 8.0, 1.0)

        # Multi-category bonus (cross-cutting crises are riskier)
        if categories_hit >= 3:
            kw_norm = min(kw_norm * 1.3, 1.0)
        elif categories_hit >= 2:
            kw_norm = min(kw_norm * 1.15, 1.0)

        # Sentiment contribution (negative = more risk)
        sent_risk = max(0, -sentiment)

        # Combined risk
        risk = 0.50 * kw_norm + 0.50 * sent_risk
        return min(max(risk, 0.0), 1.0)

    # --- Regime probability mapping ------------------------------------------

    def map_to_regimes(self, risk: float, sentiment: float) -> np.ndarray:
        """Map risk score + sentiment to regime probability distribution."""
        probs = BASE_PROBS.copy()

        if risk < 0.15:
            level = "low"
        elif risk < 0.35:
            level = "mild"
        elif risk < 0.55:
            level = "moderate"
        elif risk < 0.75:
            level = "high"
        else:
            level = "extreme"

        probs += SIGNAL_SHIFTS[level]

        # Additional sentiment-based fine-tuning
        if sentiment < -0.6:
            probs[4] += 0.08
            probs[3] += 0.05
            probs[0] -= 0.13
        elif sentiment < -0.3:
            probs[2] += 0.05
            probs[3] += 0.03
            probs[0] -= 0.08

        # Clamp and normalize
        probs = np.maximum(probs, 0.01)
        probs = probs / probs.sum()
        return probs

    def map_to_extreme_probs(self, risk: float, sentiment: float) -> Dict[str, float]:
        """Estimate extreme event probabilities at 5d/10d/20d horizons."""
        base_risk = risk * 0.7 + max(0, -sentiment) * 0.3

        p5 = min(base_risk * 0.6 + 0.05, 0.95)
        p10 = min(base_risk * 0.75 + 0.08, 0.95)
        p20 = min(base_risk * 0.65 + 0.10, 0.95)

        return {
            "extreme_5d_prob": round(p5, 4),
            "extreme_10d_prob": round(p10, 4),
            "extreme_20d_prob": round(p20, 4),
        }

    # --- Explanation generation -----------------------------------------------

    def generate_explanation(
        self,
        text: str,
        sentiment: float,
        sent_source: str,
        keyword_hits: Dict[str, List[str]],
        risk: float,
        regime_probs: np.ndarray,
    ) -> str:
        """Generate a human-readable explanation of the analysis."""
        parts = []

        # Sentiment
        if sentiment < -0.5:
            parts.append(f"Strongly negative sentiment detected ({sentiment:+.2f})")
        elif sentiment < -0.2:
            parts.append(f"Negative sentiment detected ({sentiment:+.2f})")
        elif sentiment < 0.2:
            parts.append(f"Neutral sentiment ({sentiment:+.2f})")
        elif sentiment < 0.5:
            parts.append(f"Moderately positive sentiment ({sentiment:+.2f})")
        else:
            parts.append(f"Strong positive sentiment ({sentiment:+.2f})")

        # Keywords
        if keyword_hits:
            total = sum(len(v) for v in keyword_hits.values())
            cats = ", ".join(keyword_hits.keys())
            parts.append(f"Detected {total} risk-associated terms across {cats}")
            top_cat = max(keyword_hits.items(), key=lambda x: len(x[1]))
            parts.append(
                f"Primary risk domain: {top_cat[0]} "
                f"({', '.join(top_cat[1][:3])})"
            )
        else:
            parts.append("No specific financial risk keywords detected")

        # Regime
        top_idx = int(np.argmax(regime_probs))
        top_prob = regime_probs[top_idx]
        parts.append(
            f"Most likely regime: {REGIME_LABELS[top_idx]} "
            f"({top_prob:.0%} probability)"
        )

        # Risk level
        if risk > 0.7:
            parts.append("HIGH grey-swan risk — elevated caution advised")
        elif risk > 0.4:
            parts.append("MODERATE grey-swan risk — monitoring recommended")
        elif risk > 0.2:
            parts.append("LOW-MODERATE risk — within normal parameters")
        else:
            parts.append("LOW risk — market conditions appear stable")

        return ". ".join(parts) + "."

    # --- Main entry point ----------------------------------------------------

    def analyze(self, text: str) -> Dict:
        """
        Analyze an event description and return grey-swan risk assessment.

        Returns dict with:
            sentiment_score, sentiment_source,
            risk_keywords (by category),
            risk_score (0-1),
            regime_probabilities,
            extreme_5d/10d/20d_prob,
            explanation,
        """
        text = text.strip()
        if not text:
            return {"error": "Empty input text"}

        sentiment, sent_source = self.compute_sentiment(text)
        keyword_hits = self.scan_risk_keywords(text)
        risk = self.risk_score(keyword_hits, sentiment)
        regime_probs = self.map_to_regimes(risk, sentiment)
        extreme_probs = self.map_to_extreme_probs(risk, sentiment)
        explanation = self.generate_explanation(
            text, sentiment, sent_source, keyword_hits, risk, regime_probs
        )

        return {
            "input_text": text[:200],
            "sentiment_score": round(float(sentiment), 4),
            "sentiment_source": sent_source,
            "risk_keywords": keyword_hits,
            "risk_keyword_count": sum(len(v) for v in keyword_hits.values()),
            "risk_score": round(float(risk), 4),
            "regime_probabilities": {
                REGIME_LABELS[i]: round(float(regime_probs[i]), 4)
                for i in range(5)
            },
            "most_likely_regime": REGIME_LABELS[int(np.argmax(regime_probs))],
            **extreme_probs,
            "explanation": explanation,
        }


# ---------------------------------------------------------------------------
# Singleton instance (loaded once at startup)
# ---------------------------------------------------------------------------

_analyzer: Optional[TextEventAnalyzer] = None


def get_analyzer() -> TextEventAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = TextEventAnalyzer()
    return _analyzer


def analyze_event(text: str) -> Dict:
    """Convenience function — analyze a text event description."""
    return get_analyzer().analyze(text)


if __name__ == "__main__":
    import json

    test_events = [
        "Federal Reserve announces emergency 50bp rate cut as bank failures mount",
        "Markets hit all-time high on strong earnings and dovish Fed",
        "Russia launches military invasion of Ukraine, triggering global sanctions and energy crisis",
        "Major US bank declares bankruptcy, credit markets freeze, contagion fears mount",
        "S&P 500 rises 0.3% on quiet trading day",
        "China Evergrande defaults on debt payments, sparking fear of broader property crisis and contagion",
    ]

    analyzer = get_analyzer()
    print(f"Model: {analyzer.model_info}\n")

    for event in test_events:
        result = analyzer.analyze(event)
        print(f"Event: {event[:80]}...")
        print(f"  Sentiment: {result['sentiment_score']:+.3f} ({result['sentiment_source']})")
        print(f"  Risk keywords: {result['risk_keyword_count']}")
        print(f"  Risk score: {result['risk_score']:.3f}")
        print(f"  Regime: {result['most_likely_regime']}")
        print(f"  Extreme 5d: {result['extreme_5d_prob']:.1%}")
        print(f"  {result['explanation']}")
        print()
