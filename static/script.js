/**
 * PhishGuard AI - Frontend Controller Script
 * -------------------------------------------
 * Handles interactive client-side logic, sample insertion, live character counting,
 * AJAX requests to Flask /predict, dynamic DOM rendering, and XSS safety escaping.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const form = document.getElementById("email-analysis-form");
    const textarea = document.getElementById("email-text-input");
    const charCountEl = document.getElementById("char-count");
    const clearBtn = document.getElementById("clear-input-btn");
    const samplePhishingBtn = document.getElementById("sample-phishing-btn");
    const sampleSafeBtn = document.getElementById("sample-safe-btn");
    const submitBtn = document.getElementById("analyze-submit-btn");

    const loadingState = document.getElementById("loading-state");
    const loadingStepText = document.getElementById("loading-step-text");
    const errorAlert = document.getElementById("error-alert");
    const errorMessageText = document.getElementById("error-message-text");

    const resultWrapper = document.getElementById("result-wrapper");
    const resultCard = document.getElementById("result-card");
    const resultIconBox = document.getElementById("result-icon-box");
    const resultLabel = document.getElementById("result-label");
    const resultRiskPill = document.getElementById("result-risk-pill");
    const resultConfidenceVal = document.getElementById("result-confidence-val");
    const confidenceBarFill = document.getElementById("confidence-bar-fill");
    const flagsList = document.getElementById("flags-list");
    const explanationText = document.getElementById("explanation-text");
    const recommendationsList = document.getElementById("recommendations-list");

    // Realistic Demonstration Samples
    const SAMPLES = {
        phishing: `Subject: URGENT: Your PayPal Account Has Been Temporarily Suspended!

Dear Valued Customer,

We detected unauthorized login attempts to your PayPal account from an unrecognized IP address (45.33.32.156). For your security, access to your funds and credit card billing has been suspended.

You must immediately verify your identity and confirm your account password within 24 hours to prevent permanent account termination:

>> http://192.168.1.1/paypal/verify-identity.php?ref=security-update

Failure to update your credentials will result in permanent closure of your account and legal reporting.

Sincerely,
PayPal Security & Fraud Prevention Division`,

        safe: `Subject: Team Sync: Q4 Roadmap & Sprint Planning (Thursday 2:00 PM)

Hi Team,

Please review the attached project agenda before our weekly sprint planning meeting this Thursday at 2:00 PM EST in Conference Room B. 

We will be reviewing our quarterly milestones, release schedule, and onboarding documentation for the new engineers. The updated slide deck is available on our internal wiki:
https://intranet.company.org/wiki/q4-roadmap

Let me know if you have any questions or items you would like added to the discussion agenda.

Best regards,
Sarah Jenkins
Senior Project Manager | Engineering Operations`
    };

    // Helper: Escape untrusted HTML characters
    function escapeHTML(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Live Character Counter
    function updateCharCount() {
        const len = textarea.value.length;
        charCountEl.textContent = len.toLocaleString();
        if (len > 45000) {
            charCountEl.style.color = "var(--phishing-red)";
        } else {
            charCountEl.style.color = "var(--text-muted)";
        }
    }

    textarea.addEventListener("input", updateCharCount);

    // Clear Button Handler
    clearBtn.addEventListener("click", () => {
        textarea.value = "";
        updateCharCount();
        hideError();
        resultWrapper.style.display = "none";
        textarea.focus();
    });

    // Sample Injection Handlers
    samplePhishingBtn.addEventListener("click", () => {
        textarea.value = SAMPLES.phishing;
        updateCharCount();
        hideError();
        resultWrapper.style.display = "none";
        textarea.focus();
    });

    sampleSafeBtn.addEventListener("click", () => {
        textarea.value = SAMPLES.safe;
        updateCharCount();
        hideError();
        resultWrapper.style.display = "none";
        textarea.focus();
    });

    // Error Alert Functions
    function showError(message) {
        errorMessageText.textContent = message;
        errorAlert.style.display = "flex";
        errorAlert.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function hideError() {
        errorAlert.style.display = "none";
        errorMessageText.textContent = "";
    }

    // Form Submit (Prediction Pipeline)
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        hideError();

        const emailText = textarea.value.trim();

        // Client-side Validation
        if (!emailText) {
            showError("Please enter or paste email content to analyze.");
            textarea.focus();
            return;
        }

        if (emailText.length < 5) {
            showError("Email content is too short for reliable analysis. Please provide more text.");
            return;
        }

        // Enter Loading State
        submitBtn.disabled = true;
        clearBtn.disabled = true;
        samplePhishingBtn.disabled = true;
        sampleSafeBtn.disabled = true;
        loadingState.style.display = "block";
        resultWrapper.style.display = "none";

        // Animated Status Text
        const loadingSteps = [
            "Sanitizing content & extracting URL features...",
            "Scanning for deception indicators & suspicious keywords...",
            "Computing TF-IDF vector & executing ML classifier..."
        ];
        let stepIdx = 0;
        const intervalId = setInterval(() => {
            stepIdx = (stepIdx + 1) % loadingSteps.length;
            loadingStepText.textContent = loadingSteps[stepIdx];
        }, 500);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify({ email_text: emailText })
            });

            clearInterval(intervalId);

            const data = await response.json();

            if (!response.ok) {
                const errMsg = data.error || `Server responded with status code ${response.status}.`;
                showError(errMsg);
                return;
            }

            // Render Result Card
            renderPredictionResult(data);

        } catch (err) {
            clearInterval(intervalId);
            console.error("Analysis Request Error:", err);
            showError("Unable to connect to the prediction server. Please verify the Flask backend is running on http://127.0.0.1:5000.");
        } finally {
            submitBtn.disabled = false;
            clearBtn.disabled = false;
            samplePhishingBtn.disabled = false;
            sampleSafeBtn.disabled = false;
            loadingState.style.display = "none";
        }
    });

    // Render Prediction Results Card
    function renderPredictionResult(data) {
        const isPhishing = data.label.toLowerCase() === "phishing";

        // Update Card Theme
        resultCard.className = isPhishing ? "result-card is-phishing" : "result-card is-safe";

        // Update Icon
        if (isPhishing) {
            resultIconBox.innerHTML = `
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
                    <line x1="12" y1="9" x2="12" y2="13"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
            `;
            resultLabel.textContent = "PHISHING DETECTED";
        } else {
            resultIconBox.innerHTML = `
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    <path d="m9 12 2 2 4-4"/>
                </svg>
            `;
            resultLabel.textContent = "LEGITIMATE / SAFE";
        }

        // Risk Level Pill
        const risk = (data.risk_level || "Low").toLowerCase();
        resultRiskPill.className = `risk-pill risk-${risk}`;
        resultRiskPill.textContent = `Risk Level: ${data.risk_level || "Low"}`;

        // Confidence Percentage & Bar
        const confVal = typeof data.confidence === "number" ? data.confidence : parseFloat(data.confidence) || 0;
        resultConfidenceVal.textContent = `${confVal.toFixed(1)}%`;
        
        // Trigger smooth bar fill transition
        confidenceBarFill.style.width = "0%";
        setTimeout(() => {
            confidenceBarFill.style.width = `${Math.min(100, Math.max(5, confVal))}%`;
        }, 80);

        // Populate Red Flags / Threat Indicators
        flagsList.innerHTML = "";
        const flags = Array.isArray(data.red_flags) ? data.red_flags : [];
        if (flags.length > 0) {
            flags.forEach(flag => {
                const li = document.createElement("li");
                li.className = "flag-item";
                li.innerHTML = `
                    <span class="flag-icon">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="12" y1="8" x2="12" y2="12"/>
                            <line x1="12" y1="16" x2="12.01" y2="16"/>
                        </svg>
                    </span>
                    <span>${escapeHTML(flag)}</span>
                `;
                flagsList.appendChild(li);
            });
        } else {
            const li = document.createElement("li");
            li.className = "no-flags-msg";
            li.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                    <polyline points="22 4 12 14.01 9 11.01"/>
                </svg>
                <span>No prominent structural URL or coercive urgency indicators identified.</span>
            `;
            flagsList.appendChild(li);
        }

        // Explanation
        explanationText.textContent = data.explanation || "No detailed assessment provided.";

        // Practical Recommendations
        recommendationsList.innerHTML = "";
        const recs = Array.isArray(data.recommendations) ? data.recommendations : [];
        recs.forEach(rec => {
            const li = document.createElement("li");
            li.className = "rec-item";
            li.innerHTML = `
                <span class="rec-icon">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="20 6 9 17 4 12"/>
                    </svg>
                </span>
                <span>${escapeHTML(rec)}</span>
            `;
            recommendationsList.appendChild(li);
        });

        // Show Card & Smooth Scroll
        resultWrapper.style.display = "block";
        resultWrapper.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // Fetch and populate model metrics if available
    async function loadMetrics() {
        try {
            const res = await fetch("/api/metrics");
            if (!res.ok) return;
            const data = await res.json();

            if (data.accuracy) {
                const accEl = document.getElementById("metric-accuracy");
                if (accEl) accEl.textContent = `${data.accuracy.toFixed(2)}%`;
            }
            if (data.precision) {
                const precEl = document.getElementById("metric-precision");
                if (precEl) precEl.textContent = `${data.precision.toFixed(2)}%`;
            }
            if (data.recall) {
                const recEl = document.getElementById("metric-recall");
                if (recEl) recEl.textContent = `${data.recall.toFixed(2)}%`;
            }
            if (data.f1_score) {
                const f1El = document.getElementById("metric-f1");
                if (f1El) f1El.textContent = `${data.f1_score.toFixed(2)}%`;
            }

            if (data.confusion_matrix) {
                const cm = data.confusion_matrix;
                const total = (cm.true_negatives || 0) + (cm.true_positives || 0) + (cm.false_positives || 0) + (cm.false_negatives || 0);
                
                const tnEl = document.getElementById("cm-tn-val");
                if (tnEl && cm.true_negatives !== undefined) {
                    const pct = total > 0 ? ((cm.true_negatives / total) * 100).toFixed(1) : 0;
                    tnEl.textContent = `${cm.true_negatives.toLocaleString()} (${pct}%)`;
                }

                const tpEl = document.getElementById("cm-tp-val");
                if (tpEl && cm.true_positives !== undefined) {
                    const pct = total > 0 ? ((cm.true_positives / total) * 100).toFixed(1) : 0;
                    tpEl.textContent = `${cm.true_positives.toLocaleString()} (${pct}%)`;
                }

                const fpEl = document.getElementById("cm-fp-val");
                if (fpEl && cm.false_positives !== undefined) {
                    const pct = total > 0 ? ((cm.false_positives / total) * 100).toFixed(1) : 0;
                    fpEl.textContent = `${cm.false_positives.toLocaleString()} (${pct}%)`;
                }

                const fnEl = document.getElementById("cm-fn-val");
                if (fnEl && cm.false_negatives !== undefined) {
                    const pct = total > 0 ? ((cm.false_negatives / total) * 100).toFixed(1) : 0;
                    fnEl.textContent = `${cm.false_negatives.toLocaleString()} (${pct}%)`;
                }
            }
        } catch (err) {
            console.warn("Could not load dynamic metrics:", err);
        }
    }

    // Call metrics loader
    loadMetrics();

    // Initialize char counter
    updateCharCount();
});
