"""
assurance/report.py

Assembles a structured assurance report from scan results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from assurance.dataset_scanner import ScanResult as DataScanResult
from assurance.model_fingerprint import FingerprintResult
from assurance.backdoor_scanner import BackdoorScanResult
from assurance.provenance import ChainVerificationResult


@dataclass
class Finding:
    severity: str        # HIGH / MEDIUM / LOW / PASS
    category: str        # e.g. "Backdoor behavior"
    description: str
    evidence: str        = ""
    recommendation: str  = ""


@dataclass
class AssuranceReport:
    overall_status: str                     = "UNKNOWN"
    data_integrity_score: int               = 100
    model_integrity_score: int              = 100
    provenance_status: str                  = "N/A"
    findings: List[Finding]                 = field(default_factory=list)
    recommended_disposition: str            = ""
    dataset_path: str                       = ""
    model_path: str                         = ""


LIMITATIONS = [
    "Behavioral trigger scanning is not universal — only visible patch-based triggers at candidate positions are tested.",
    "Label-consistency disagreement is NOT proof of poisoning — it only flags statistical anomalies.",
    "Model behavioral fingerprint may drift naturally after fine-tuning on legitimate data.",
    "SHA-256 detects any byte-level change but cannot identify what changed or whether the change is malicious.",
    "Only supported attack classes (duplicate flooding, label anomalies, known trigger patterns, file substitution, record tampering) are evaluated.",
    "Fully stealthy or semantic attacks outside the tested threat model are NOT guaranteed to be detected.",
    "This is a prototype implementation intended for demonstration purposes.",
]

SUPPORTED_ATTACKS = [
    "Duplicate image flooding",
    "Label-consistency anomalies",
    "Visible patch-based backdoor trigger behavior",
    "Model file substitution / modification (SHA-256)",
    "Inference-record tampering / chain corruption",
]

NOT_GUARANTEED = [
    "Arbitrary unknown backdoor architectures",
    "Fully stealthy attacks with minimal behavioral change",
    "Semantic or invisible perturbation attacks",
    "Multi-vector or adaptive attacks",
    "Attacks outside the tested threat model",
]


def build_report(
    data_result:      Optional[DataScanResult]        = None,
    fingerprint_result: Optional[FingerprintResult]   = None,
    backdoor_result:  Optional[BackdoorScanResult]    = None,
    chain_result:     Optional[ChainVerificationResult] = None,
    dataset_path: str = "",
    model_path:   str = "",
) -> AssuranceReport:
    """Build an AssuranceReport from the four scan results."""

    report = AssuranceReport(dataset_path=dataset_path, model_path=model_path)
    findings: List[Finding] = []

    # ── Data integrity ──────────────────────────────────────────────────────
    if data_result:
        report.data_integrity_score = data_result.integrity_score
        report.dataset_path = data_result.dataset_path

        if data_result.duplicate_count > 0:
            sev = "MEDIUM" if data_result.duplicate_count > 10 else "LOW"
            findings.append(Finding(
                severity=sev,
                category="Duplicate images",
                description=f"{data_result.duplicate_count} near-duplicate images detected.",
                evidence=f"{len(data_result.duplicate_groups)} duplicate groups found using perceptual hashing.",
                recommendation="Review and remove duplicate samples; investigate data pipeline.",
            ))

        if data_result.label_anomalies:
            n = len(data_result.label_anomalies)
            findings.append(Finding(
                severity="MEDIUM",
                category="Label-consistency anomalies",
                description=f"{n} samples where reference classifier strongly disagrees with dataset label.",
                evidence="Reference model prediction confidence ≥ 70% for a different class.",
                recommendation="REVIEW — Label disagreement is not proof of poisoning but warrants investigation.",
            ))

        # Contributor risk
        high_risk_contributors = [
            c for c in (data_result.contributor_stats or [])
            if c.risk_level == "HIGH"
        ]
        if high_risk_contributors:
            cnames = ", ".join(c.contributor_id for c in high_risk_contributors)
            findings.append(Finding(
                severity="HIGH",
                category="High-risk contributor",
                description=f"Contributors with elevated anomaly rates: {cnames}",
                evidence="Anomaly rate exceeds 30% of submitted samples.",
                recommendation="Quarantine and audit data from flagged contributors.",
            ))
    else:
        report.data_integrity_score = 0

    # ── Model fingerprint ───────────────────────────────────────────────────
    model_integrity_penalties = 0
    if fingerprint_result:
        if not fingerprint_result.binary_match:
            model_integrity_penalties += 40
            findings.append(Finding(
                severity="HIGH",
                category="Model substitution / modification",
                description="SHA-256 fingerprint does not match registered reference.",
                evidence=(
                    f"Expected: {fingerprint_result.reference_sha256[:16]}…\n"
                    f"Observed: {fingerprint_result.sha256[:16]}…"
                ),
                recommendation="DO NOT deploy — verify model provenance before use.",
            ))
        if fingerprint_result.behavioral_match_pct < 90.0:
            model_integrity_penalties += 30
            findings.append(Finding(
                severity="HIGH",
                category="Behavioral fingerprint mismatch",
                description=f"Canary-set agreement: {fingerprint_result.behavioral_match_pct:.1f}% (expected ≥ 90%).",
                evidence=f"{fingerprint_result.canary_mismatches}/{fingerprint_result.canary_total} canary predictions changed.",
                recommendation="Behavioral change detected — inspect for fine-tuning or model substitution.",
            ))
        report.model_integrity_score = max(0, 100 - model_integrity_penalties)
        report.model_path = fingerprint_result.model_path
    else:
        report.model_integrity_score = 50  # unknown

    # ── Backdoor / trigger scan ─────────────────────────────────────────────
    if backdoor_result:
        if backdoor_result.most_suspicious_asr > 0.5:
            findings.append(Finding(
                severity="HIGH",
                category="Backdoor-like trigger behavior",
                description=(
                    f"Suspicious trigger position: {backdoor_result.most_suspicious_position}  "
                    f"(ASR={backdoor_result.most_suspicious_asr*100:.0f}%)"
                ),
                evidence=(
                    f"Target class: {backdoor_result.target_class}.  "
                    f"Fraction of test images classified as target class when trigger applied: "
                    f"{backdoor_result.most_suspicious_asr*100:.0f}%."
                ),
                recommendation=(
                    "HIGH — Investigate training data for poisoned examples. "
                    "Do NOT deploy without further analysis."
                ),
            ))
            report.model_integrity_score = max(0, report.model_integrity_score - 20)
        elif backdoor_result.most_suspicious_asr > 0.25:
            findings.append(Finding(
                severity="MEDIUM",
                category="Moderate prediction shift under trigger",
                description=f"ASR={backdoor_result.most_suspicious_asr*100:.0f}% — below HIGH threshold but notable.",
                evidence=f"Position: {backdoor_result.most_suspicious_position}",
                recommendation="REVIEW — monitor for pattern at higher trigger intensities.",
            ))

    # ── Provenance chain ────────────────────────────────────────────────────
    if chain_result:
        if chain_result.is_intact:
            report.provenance_status = "VERIFIED"
            findings.append(Finding(
                severity="PASS",
                category="Inference provenance",
                description="All inference records verified. Chain intact.",
                evidence=f"{chain_result.valid_records}/{chain_result.total_records} records valid.",
                recommendation="Continue monitoring.",
            ))
        else:
            report.provenance_status = "FAILED"
            findings.append(Finding(
                severity="HIGH",
                category="Inference chain tampered",
                description=(
                    f"Chain broken at record {chain_result.first_broken} "
                    f"({chain_result.broken_record_id})."
                ),
                evidence="Record hash does not match recomputed hash from stored fields.",
                recommendation="Reject this inference audit trail — evidence of tampering.",
            ))
    else:
        report.provenance_status = "N/A"

    # ── Overall status ──────────────────────────────────────────────────────
    high_findings   = [f for f in findings if f.severity == "HIGH"]
    medium_findings = [f for f in findings if f.severity == "MEDIUM"]

    if high_findings:
        report.overall_status = "REVIEW REQUIRED — HIGH severity findings present"
        report.recommended_disposition = "DO NOT DEPLOY without thorough investigation."
    elif medium_findings:
        report.overall_status = "CAUTION — Medium severity findings present"
        report.recommended_disposition = "REVIEW before deployment. Investigate flagged samples."
    elif any(f.severity == "LOW" for f in findings):
        report.overall_status = "ACCEPTABLE — Minor issues detected"
        report.recommended_disposition = "Acceptable for deployment with monitoring."
    else:
        report.overall_status = "PASS — No significant issues detected"
        report.recommended_disposition = "Acceptable for deployment."

    report.findings = findings
    return report

