"""
PhishGuard AI - Feature Extraction & Preprocessing Module
---------------------------------------------------------
Shared preprocessing, URL analysis, keyword detection, and risk scoring
used identically by both model training and the Flask backend.
"""

import re
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd


# Known URL shorteners commonly exploited in phishing
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "tiny.cc", "shorte.st",
    "clck.ru", "v.gd", "soo.gd", "s.id"
}

# Suspicious or high-abuse top-level domains frequently seen in malicious campaigns
SUSPICIOUS_TLDS = {
    "xyz", "top", "buzz", "club", "work", "stream", "download",
    "gq", "cf", "tk", "ml", "ga", "click", "link", "loan", "cam",
    "live", "monster", "online", "rest", "support", "bid"
}

# Suspicious trigger keywords and phrases
SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "password", "account", "suspended",
    "click", "login", "bank", "update", "confirm", "winner",
    "free", "security", "immediately", "billing", "action required",
    "unauthorized", "credentials", "expire", "ssn", "invoice",
    "lottery", "prize", "reactivate", "compromised", "identity"
]

# Pre-compiled regex patterns for performance
URL_REGEX = re.compile(
    r'(?:https?://|www\.)[^\s<>"]+|(?:\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s<>"]*)?)',
    re.IGNORECASE
)
IP_IN_URL_REGEX = re.compile(r'https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/[^\s]*)?', re.IGNORECASE)
HTML_TAG_REGEX = re.compile(r'<[^>]+>', re.IGNORECASE)
WHITESPACE_REGEX = re.compile(r'\s+')

# Pre-compiled keyword regexes
KEYWORD_PATTERNS = {
    kw: re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE) if " " not in kw else None
    for kw in SUSPICIOUS_KEYWORDS
}


def clean_email_text(text: Any) -> str:
    """
    Cleans and normalizes email content:
    - Converts to string if null or malformed
    - Converts to lowercase
    - Strips HTML tags and entities
    - Normalizes extra whitespace
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    cleaned = HTML_TAG_REGEX.sub(" ", text)
    
    # Decode basic HTML entities
    cleaned = cleaned.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    
    # Lowercase
    cleaned = cleaned.lower()
    
    # Collapse multiple whitespaces
    cleaned = WHITESPACE_REGEX.sub(" ", cleaned).strip()
    
    return cleaned


def extract_urls(text: str) -> List[str]:
    """Finds all URLs and domain-like links in raw or cleaned text."""
    if not text:
        return []
    return URL_REGEX.findall(text)


def extract_url_features(text: str) -> Dict[str, float]:
    """
    Extracts structural URL features from email text:
    - url_count: total number of URLs
    - avg_url_length: average length of URLs found
    - has_ip_url: 1.0 if URL contains an IP address, 0.0 otherwise
    - max_url_dots: maximum dot count in any single URL
    - has_at_symbol: 1.0 if '@' is in any URL (often used for credential obfuscation)
    - has_shortener: 1.0 if known URL shortener is present
    - http_count: number of insecure HTTP links
    - has_suspicious_tld: 1.0 if suspicious TLD found
    """
    urls = extract_urls(text)
    
    if not urls:
        return {
            "url_count": 0.0,
            "avg_url_length": 0.0,
            "has_ip_url": 0.0,
            "max_url_dots": 0.0,
            "has_at_symbol": 0.0,
            "has_shortener": 0.0,
            "http_count": 0.0,
            "has_suspicious_tld": 0.0,
        }
    
    url_lengths = [len(u) for u in urls]
    has_ip = 1.0 if IP_IN_URL_REGEX.search(text) else 0.0
    
    max_dots = 0
    has_at = 0.0
    has_shortener = 0.0
    http_count = 0.0
    has_suspicious_tld = 0.0
    
    for u in urls:
        u_lower = u.lower()
        dots = u_lower.count(".")
        if dots > max_dots:
            max_dots = dots
        if "@" in u_lower:
            has_at = 1.0
        if u_lower.startswith("http://"):
            http_count += 1.0
        
        # Check shorteners
        for shortener in URL_SHORTENERS:
            if shortener in u_lower:
                has_shortener = 1.0
                break
        
        # Check suspicious TLDs
        for tld in SUSPICIOUS_TLDS:
            pattern = rf'\.{tld}(?:/|$|\?|#)'
            if re.search(pattern, u_lower):
                has_suspicious_tld = 1.0
                break
                
    return {
        "url_count": float(len(urls)),
        "avg_url_length": float(np.mean(url_lengths)) if url_lengths else 0.0,
        "has_ip_url": has_ip,
        "max_url_dots": float(max_dots),
        "has_at_symbol": has_at,
        "has_shortener": has_shortener,
        "http_count": http_count,
        "has_suspicious_tld": has_suspicious_tld,
    }


def extract_keyword_features(text: str) -> Dict[str, float]:
    """
    Extracts frequency features for known phishing trigger keywords.
    """
    cleaned = clean_email_text(text)
    features = {}
    total_keyword_hits = 0.0
    
    for kw in SUSPICIOUS_KEYWORDS:
        pat = KEYWORD_PATTERNS.get(kw)
        if pat is not None:
            hits = float(len(pat.findall(cleaned)))
        else:
            hits = 1.0 if kw in cleaned else 0.0
        features[f"kw_{kw.replace(' ', '_')}"] = hits
        total_keyword_hits += hits
        
    features["total_keyword_hits"] = total_keyword_hits
    return features


def extract_tabular_features(text: str) -> np.ndarray:
    """
    Combines URL and keyword numerical features into a 1D numpy array.
    """
    url_feats = extract_url_features(text)
    kw_feats = extract_keyword_features(text)
    
    # Deterministic order
    values = [url_feats[k] for k in sorted(url_feats.keys())] + [kw_feats[k] for k in sorted(kw_feats.keys())]
    return np.array(values, dtype=np.float32)


def extract_batch_tabular_features(texts: List[str]) -> np.ndarray:
    """Extracts tabular features for a list of texts into a 2D numpy array."""
    rows = [extract_tabular_features(t) for t in texts]
    return np.array(rows, dtype=np.float32)


def analyze_risk_signals(raw_text: str) -> Tuple[List[str], str, str, List[str]]:
    """
    Rule-based safety and threat inspection to extract human-readable
    red flags, risk level, explanation, and practical recommendations.
    
    Returns:
        red_flags: list of detected warning signals
        risk_level: 'Low', 'Moderate', 'High', or 'Critical'
        explanation: detailed summary
        recommendations: checklist of security recommendations
    """
    cleaned = clean_email_text(raw_text)
    urls = extract_urls(raw_text)
    url_feats = extract_url_features(raw_text)
    
    red_flags = []
    
    # 1. URL Inspection
    if url_feats["has_ip_url"] > 0:
        red_flags.append("Raw IP address used in link (bypasses legitimate DNS domain verification)")
        
    if url_feats["has_shortener"] > 0:
        red_flags.append("URL shortening service detected (masks real destination website)")
        
    if url_feats["has_at_symbol"] > 0:
        red_flags.append("Contains '@' symbol in URL (frequent indicator of credential spoofing or URL redirection)")
        
    if url_feats["max_url_dots"] >= 4:
        red_flags.append(f"Excessive domain nesting / subdomains detected ({int(url_feats['max_url_dots'])} dots in link)")
        
    if url_feats["has_suspicious_tld"] > 0:
        red_flags.append("Contains link with high-risk / low-cost Top-Level Domain commonly used in phishing")
        
    if url_feats["http_count"] > 0:
        red_flags.append(f"Insecure plain HTTP links found ({int(url_feats['http_count'])} unencrypted link(s))")
        
    if url_feats["url_count"] >= 4:
        red_flags.append(f"High link density ({int(url_feats['url_count'])} links embedded throughout email)")

    # 2. Urgent / Coercive Language
    urgency_terms = ["urgent", "immediately", "suspended", "action required", "expire", "compromised", "reactivate"]
    found_urgency = [w for w in urgency_terms if re.search(rf'\b{re.escape(w)}\b', cleaned)]
    if found_urgency:
        red_flags.append(f"High-pressure urgent wording detected: {', '.join(found_urgency[:3])}")

    # 3. Credential & Financial Harvesting
    cred_terms = ["password", "verify", "login", "credentials", "ssn", "billing", "bank", "unauthorized"]
    found_cred = [w for w in cred_terms if re.search(rf'\b{re.escape(w)}\b', cleaned)]
    if found_cred:
        red_flags.append(f"Credential/account verification requests: {', '.join(found_cred[:3])}")

    # 4. Bait / Financial lures
    lure_terms = ["winner", "lottery", "prize", "free", "invoice"]
    found_lures = [w for w in lure_terms if re.search(rf'\b{re.escape(w)}\b', cleaned)]
    if found_lures:
        red_flags.append(f"Financial lure / suspicious incentive detected: {', '.join(found_lures[:2])}")

    # Risk level determination
    score = len(red_flags)
    if url_feats["has_ip_url"] > 0 or url_feats["has_at_symbol"] > 0:
        score += 2
        
    if score >= 4:
        risk_level = "Critical"
        explanation = "Multiple high-severity phishing characteristics identified, including deceptive links and coercive credential harvesting cues."
    elif score >= 2:
        risk_level = "High"
        explanation = "Several warning signs detected that align with common email phishing patterns and social engineering tactics."
    elif score == 1:
        risk_level = "Moderate"
        explanation = "Suspicious keyword or link characteristics were detected. Caution is strongly advised before taking any action."
    else:
        risk_level = "Low"
        explanation = "No prominent heuristic phishing triggers found. However, always exercise caution with unexpected senders."

    # Practical recommendations
    recommendations = [
        "Do not click on links or download attachments directly from this message.",
        "Verify the sender's identity through an official secondary communication channel.",
        "Check the exact sender email address, looking for subtle typosquatting (e.g., paypa1.com).",
        "If unsure, forward the message to your organization's IT/Security team for review."
    ]
    
    return red_flags, risk_level, explanation, recommendations
