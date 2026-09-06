import re
import math
import numpy as np
import pandas as pd


class DatasetDSSAnalyzer:
    """
    Generic Dataset Suitability Score (DSS) analyzer.

    DSS evaluates how suitable a cybersecurity dataset is for
    research on alert prioritization using:

        1. Alert/Event suitability
        2. Context availability
        3. Threat-intelligence availability
        4. Exploitability information
        5. Historical/temporal suitability
        6. Ground-truth quality
        7. Scalability

    Score range:
        0 = unavailable
        1 = very poor
        2 = poor
        3 = moderate
        4 = good
        5 = excellent
    """

    DEFAULT_WEIGHTS = {
        "Alert/Event": 0.20,
        "Context": 0.20,
        "Threat Intelligence": 0.15,
        "Exploitability": 0.10,
        "Historical": 0.15,
        "Ground Truth": 0.10,
        "Scalability": 0.10,
    }

    # --------------------------------------------------------
    # Keyword dictionaries
    # --------------------------------------------------------

    ALERT_KEYWORDS = [
        "alert",
        "event",
        "alarm",
        "severity",
        "risk",
        "attack",
        "intrusion",
        "detection",
        "incident",
        "malicious",
        "anomaly",
    ]

    CONTEXT_KEYWORDS = [
        "host",
        "asset",
        "device",
        "machine",
        "user",
        "account",
        "role",
        "department",
        "organization",
        "network",
        "zone",
        "segment",
        "os",
        "operating",
        "application",
        "service",
        "process",
        "business",
        "criticality",
        "location",
    ]

    TI_KEYWORDS = [
        "ioc",
        "indicator",
        "threat",
        "intel",
        "intelligence",
        "malicious_ip",
        "malicious_domain",
        "domain",
        "url",
        "hash",
        "md5",
        "sha1",
        "sha256",
        "campaign",
        "actor",
        "apt",
        "malware",
        "botnet",
        "attack_type",
        "technique",
        "mitre",
        "ttp",
    ]

    EXPLOIT_KEYWORDS = [
        "cve",
        "cvss",
        "vulnerability",
        "vuln",
        "exploit",
        "patch",
        "patched",
        "unpatched",
        "kev",
        "known_exploited",
    ]

    HISTORY_KEYWORDS = [
        "timestamp",
        "time",
        "date",
        "duration",
        "flow_start",
        "flow_end",
        "start_time",
        "end_time",
        "session",
        "sequence",
        "connection",
    ]

    LABEL_KEYWORDS = [
        "label",
        "class",
        "target",
        "attack",
        "category",
        "type",
        "ground_truth",
        "groundtruth",
        "malicious",
        "benign",
        "normal",
    ]

    SOURCE_KEYWORDS = [
        "src_ip",
        "source_ip",
        "src_addr",
        "source_addr",
        "src_port",
        "source_port",
        "ipv4_src",
        "ipv6_src",
    ]

    DESTINATION_KEYWORDS = [
        "dst_ip",
        "destination_ip",
        "dst_addr",
        "destination_addr",
        "dst_port",
        "destination_port",
        "ipv4_dst",
        "ipv6_dst",
    ]

    def __init__(self, df, dataset_name=None, weights=None):
        self.df = df
        self.dataset_name = dataset_name or "Unknown"

        self.weights = (
            weights.copy()
            if weights is not None
            else self.DEFAULT_WEIGHTS.copy()
        )

        self.columns = [
            str(c).strip().lower()
            for c in df.columns
        ]

        self.original_columns = list(df.columns)

        self.results = {}

    # ========================================================
    # Utility functions
    # ========================================================

    def _matching_columns(self, keywords):
        """
        Return columns whose names contain one of the keywords.
        """

        matches = []

        for original, normalized in zip(
            self.original_columns,
            self.columns
        ):
            for keyword in keywords:

                keyword = keyword.lower()

                if keyword in normalized:
                    matches.append(original)
                    break

        return list(dict.fromkeys(matches))

    def _count_matches(self, keywords):
        return len(self._matching_columns(keywords))

    def _has(self, keywords):
        return self._count_matches(keywords) > 0

    def _nonnull_ratio(self, columns):
        """
        Average non-null ratio for matched columns.
        """

        if not columns:
            return 0.0

        ratios = []

        for col in columns:
            ratios.append(
                self.df[col].notna().mean()
            )

        return float(np.mean(ratios))

    # ========================================================
    # 1. ALERT / EVENT SUITABILITY
    # ========================================================

    def score_alert_event(self):

        score = 0
        evidence = []

        alert_cols = self._matching_columns(
            self.ALERT_KEYWORDS
        )

        if alert_cols:
            score += 2
            evidence.append(
                f"Alert/event-related fields: {alert_cols[:8]}"
            )

        time_cols = self._matching_columns(
            self.HISTORY_KEYWORDS
        )

        if time_cols:
            score += 1
            evidence.append(
                f"Temporal fields: {time_cols[:5]}"
            )

        label_cols = self._matching_columns(
            self.LABEL_KEYWORDS
        )

        if label_cols:
            score += 1
            evidence.append(
                f"Event/attack labels: {label_cols[:5]}"
            )

        network_fields = (
            self._matching_columns(self.SOURCE_KEYWORDS)
            + self._matching_columns(self.DESTINATION_KEYWORDS)
        )

        if network_fields:
            score += 1
            evidence.append(
                f"Network event identifiers: "
                f"{network_fields[:8]}"
            )

        return min(score, 5), evidence

    # ========================================================
    # 2. CONTEXT
    # ========================================================

    def score_context(self):

        score = 0
        evidence = []

        context_cols = self._matching_columns(
            self.CONTEXT_KEYWORDS
        )

        if len(context_cols) >= 2:
            score += 2
        elif len(context_cols) == 1:
            score += 1

        if context_cols:
            evidence.append(
                f"Context fields: {context_cols[:10]}"
            )

        # Check endpoint identity
        endpoint_cols = (
            self._matching_columns(self.SOURCE_KEYWORDS)
            + self._matching_columns(self.DESTINATION_KEYWORDS)
        )

        if endpoint_cols:
            score += 1
            evidence.append(
                "Source/destination identity available"
            )

        # User/host/device context
        identity_keywords = [
            "user",
            "account",
            "host",
            "device",
            "machine",
            "asset",
        ]

        identity_cols = self._matching_columns(
            identity_keywords
        )

        if identity_cols:
            score += 1
            evidence.append(
                f"Identity/asset fields: {identity_cols[:6]}"
            )

        # OS/application/process context
        operational_keywords = [
            "os",
            "application",
            "process",
            "service",
        ]

        operational_cols = self._matching_columns(
            operational_keywords
        )

        if operational_cols:
            score += 1
            evidence.append(
                f"Operational context: {operational_cols[:6]}"
            )

        return min(score, 5), evidence

    # ========================================================
    # 3. THREAT INTELLIGENCE
    # ========================================================

    def score_threat_intelligence(self):

        score = 0
        evidence = []

        ti_cols = self._matching_columns(
            self.TI_KEYWORDS
        )

        if ti_cols:
            score += 2
            evidence.append(
                f"Threat-intelligence related fields: "
                f"{ti_cols[:10]}"
            )

        # IP/domain/URL/hash are useful for external TI
        ioc_cols = []

        ioc_keywords = [
            "ip",
            "addr",
            "domain",
            "url",
            "hash",
            "md5",
            "sha1",
            "sha256",
        ]

        ioc_cols = self._matching_columns(ioc_keywords)

        if ioc_cols:
            score += 2
            evidence.append(
                f"IOC-enrichment fields: {ioc_cols[:10]}"
            )

        # Attack technique/category
        technique_cols = self._matching_columns(
            [
                "attack_type",
                "technique",
                "mitre",
                "ttp",
                "tactic",
                "malware",
                "botnet",
                "campaign",
            ]
        )

        if technique_cols:
            score += 1
            evidence.append(
                f"Threat classification fields: "
                f"{technique_cols[:8]}"
            )

        return min(score, 5), evidence

    # ========================================================
    # 4. EXPLOITABILITY
    # ========================================================

    def score_exploitability(self):

        score = 0
        evidence = []

        exploit_cols = self._matching_columns(
            self.EXPLOIT_KEYWORDS
        )

        if exploit_cols:
            score += 3
            evidence.append(
                f"Exploit/vulnerability fields: "
                f"{exploit_cols[:10]}"
            )

        cvss_cols = self._matching_columns(
            ["cvss"]
        )

        if cvss_cols:
            score += 1
            evidence.append(
                f"CVSS fields: {cvss_cols}"
            )

        patch_cols = self._matching_columns(
            [
                "patch",
                "patched",
                "unpatched",
            ]
        )

        if patch_cols:
            score += 1
            evidence.append(
                f"Patch-status fields: {patch_cols}"
            )

        return min(score, 5), evidence

    # ========================================================
    # 5. HISTORICAL / TEMPORAL
    # ========================================================

    def score_historical(self):

        score = 0
        evidence = []

        temporal_cols = self._matching_columns(
            self.HISTORY_KEYWORDS
        )

        if temporal_cols:
            score += 2
            evidence.append(
                f"Temporal fields: {temporal_cols[:10]}"
            )

        source_cols = self._matching_columns(
            self.SOURCE_KEYWORDS
        )

        destination_cols = self._matching_columns(
            self.DESTINATION_KEYWORDS
        )

        if source_cols and destination_cols:
            score += 1
            evidence.append(
                "Source/destination relationship available"
            )

        if self._has([
            "duration",
            "flow_duration",
            "session",
        ]):
            score += 1
            evidence.append(
                "Duration/session information available"
            )

        if self._has([
            "sequence",
            "connection",
            "session",
        ]):
            score += 1
            evidence.append(
                "Sequence/connection information available"
            )

        return min(score, 5), evidence

    # ========================================================
    # 6. GROUND TRUTH
    # ========================================================

    def score_ground_truth(self):

        score = 0
        evidence = []

        label_cols = self._matching_columns(
            self.LABEL_KEYWORDS
        )

        if label_cols:
            score += 3
            evidence.append(
                f"Label fields: {label_cols[:10]}"
            )

            ratio = self._nonnull_ratio(
                label_cols
            )

            if ratio >= 0.95:
                score += 1
                evidence.append(
                    f"High label completeness: "
                    f"{ratio:.2%}"
                )
            elif ratio >= 0.80:
                score += 0.5
                evidence.append(
                    f"Moderate label completeness: "
                    f"{ratio:.2%}"
                )

        # Multiple attack classes
        for col in label_cols:
            try:
                unique_values = (
                    self.df[col]
                    .dropna()
                    .nunique()
                )

                if unique_values >= 3:
                    score += 1
                    evidence.append(
                        f"{col}: {unique_values} classes"
                    )
                    break

            except Exception:
                pass

        return min(score, 5), evidence

    # ========================================================
    # 7. SCALABILITY
    # ========================================================

    def score_scalability(self):

        score = 0
        evidence = []

        n_rows = len(self.df)

        if n_rows >= 10_000_000:
            score = 5

        elif n_rows >= 1_000_000:
            score = 4

        elif n_rows >= 100_000:
            score = 3

        elif n_rows >= 10_000:
            score = 2

        elif n_rows >= 1_000:
            score = 1

        else:
            score = 0

        evidence.append(
            f"Rows available in analyzed dataframe: "
            f"{n_rows:,}"
        )

        if n_rows >= 1_000_000:
            evidence.append(
                "Suitable for high-volume experimentation"
            )

        return score, evidence

    # ========================================================
    # DSS
    # ========================================================

    def calculate(self):

        scoring_functions = {
            "Alert/Event": self.score_alert_event,
            "Context": self.score_context,
            "Threat Intelligence":
                self.score_threat_intelligence,
            "Exploitability":
                self.score_exploitability,
            "Historical":
                self.score_historical,
            "Ground Truth":
                self.score_ground_truth,
            "Scalability":
                self.score_scalability,
        }

        self.results = {}

        for criterion, function in scoring_functions.items():

            score, evidence = function()

            self.results[criterion] = {
                "score": float(score),
                "weight": self.weights[criterion],
                "weighted_score":
                    score * self.weights[criterion],
                "evidence": evidence,
            }

        dss = sum(
            item["weighted_score"]
            for item in self.results.values()
        )

        return dss

    # ========================================================
    # Detailed matrix
    # ========================================================

    def detailed_result(self):

        dss = self.calculate()

        result = {
            "Dataset": self.dataset_name,
        }

        for criterion, item in self.results.items():
            result[criterion] = item["score"]

        result["DSS"] = round(dss, 3)
        result["DSS (%)"] = round(
            (dss / 5) * 100,
            2
        )

        return result

    # ========================================================
    # Evidence table
    # ========================================================

    def evidence_table(self):

        self.calculate()

        rows = []

        for criterion, item in self.results.items():

            rows.append({
                "Dataset": self.dataset_name,
                "Criterion": criterion,
                "Score": item["score"],
                "Weight": item["weight"],
                "Weighted Score":
                    item["weighted_score"],
                "Evidence":
                    " | ".join(item["evidence"])
            })

        return pd.DataFrame(rows)