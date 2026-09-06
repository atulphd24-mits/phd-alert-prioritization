import numpy as np
import pandas as pd


class DatasetDSSAnalyzer:
    """
    Dataset Suitability Score (DSS) analyzer for alert-prioritization research.

    DSS evaluates:
        1. Alert/Event suitability
        2. Context availability
        3. Threat-intelligence availability/potential
        4. Exploitability information/potential
        5. Historical/temporal suitability
        6. Ground-truth quality
        7. Scalability

    Important methodological rule:
        ``len(df)`` may represent only a development sample.  Therefore,
        scalability must use ``total_rows`` (full-dataset metadata) whenever
        it is available.

    Scores:
        0 = unavailable/unsuitable
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

    ALERT_KEYWORDS = [
        "alert", "event", "alarm", "severity", "risk", "attack",
        "intrusion", "detection", "incident", "malicious", "anomaly"
    ]

    CONTEXT_KEYWORDS = [
        "host", "asset", "device", "machine", "user", "account", "role",
        "department", "organization", "network", "zone", "segment", "os",
        "operating", "application", "service", "process", "business",
        "criticality", "location"
    ]

    # Native TI fields only. Endpoint identifiers are handled separately as
    # "enrichment potential" and are not falsely treated as existing TI.
    TI_NATIVE_KEYWORDS = [
        "ioc", "indicator", "threat", "intel", "intelligence",
        "malicious_ip", "malicious_domain", "campaign", "actor", "apt",
        "malware", "botnet", "mitre", "ttp", "tactic", "technique"
    ]

    EXPLOIT_NATIVE_KEYWORDS = [
        "cve", "cvss", "vulnerability", "vuln", "exploit", "patch",
        "patched", "unpatched", "kev", "known_exploited"
    ]

    HISTORY_KEYWORDS = [
        "timestamp", "time", "date", "duration", "flow_start", "flow_end",
        "start_time", "end_time", "session", "sequence", "connection"
    ]

    LABEL_KEYWORDS = [
        "label", "class", "target", "attack", "category", "ground_truth",
        "groundtruth", "malicious", "benign", "normal"
    ]

    SOURCE_KEYWORDS = [
        "src_ip", "source_ip", "src_addr", "source_addr", "src_port",
        "source_port", "ipv4_src", "ipv6_src"
    ]

    DESTINATION_KEYWORDS = [
        "dst_ip", "destination_ip", "dst_addr", "destination_addr",
        "dst_port", "destination_port", "ipv4_dst", "ipv6_dst"
    ]

    def __init__(
        self,
        df,
        dataset_name=None,
        weights=None,
        total_rows=None,
        source_files=None,
        sample_size=None,
    ):
        self.df = df
        self.dataset_name = dataset_name or "Unknown"
        self.weights = (
            weights.copy() if weights is not None else self.DEFAULT_WEIGHTS.copy()
        )

        if set(self.weights) != set(self.DEFAULT_WEIGHTS):
            raise ValueError("weights must contain exactly the seven DSS criteria")
        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("DSS weights must sum to 1.0")

        self.columns = [str(c).strip().lower() for c in df.columns]
        self.original_columns = list(df.columns)
        self.total_rows = int(total_rows) if total_rows is not None else None
        self.sample_size = sample_size
        self.source_files = list(source_files or [])
        self.results = {}

    def _matching_columns(self, keywords):
        matches = []
        for original, normalized in zip(self.original_columns, self.columns):
            if any(keyword.lower() in normalized for keyword in keywords):
                matches.append(original)
        return list(dict.fromkeys(matches))

    def _has(self, keywords):
        return bool(self._matching_columns(keywords))

    def _nonnull_ratio(self, columns):
        if not columns:
            return 0.0
        return float(np.mean([self.df[col].notna().mean() for col in columns]))

    def score_alert_event(self):
        score, evidence = 0, []
        alert_cols = self._matching_columns(self.ALERT_KEYWORDS)
        if alert_cols:
            score += 2
            evidence.append(f"Alert/event fields: {alert_cols[:8]}")

        time_cols = self._matching_columns(self.HISTORY_KEYWORDS)
        if time_cols:
            score += 1
            evidence.append(f"Temporal fields: {time_cols[:5]}")

        label_cols = self._matching_columns(self.LABEL_KEYWORDS)
        if label_cols:
            score += 1
            evidence.append(f"Event/attack labels: {label_cols[:5]}")

        endpoints = self._matching_columns(self.SOURCE_KEYWORDS) + \
                    self._matching_columns(self.DESTINATION_KEYWORDS)
        if endpoints:
            score += 1
            evidence.append(f"Network event identifiers: {endpoints[:8]}")
        return min(score, 5), evidence

    def score_context(self):
        score, evidence = 0, []
        context_cols = self._matching_columns(self.CONTEXT_KEYWORDS)

        # Native organizational/endpoint context.
        if len(context_cols) >= 4:
            score += 2
        elif len(context_cols) >= 2:
            score += 1.5
        elif context_cols:
            score += 1
        if context_cols:
            evidence.append(f"Native context fields: {context_cols[:10]}")

        endpoints = self._matching_columns(self.SOURCE_KEYWORDS) + \
                    self._matching_columns(self.DESTINATION_KEYWORDS)
        if endpoints:
            score += 1
            evidence.append("Source/destination identifiers support asset-context enrichment")

        identity = self._matching_columns(
            ["user", "account", "host", "device", "machine", "asset"]
        )
        if identity:
            score += 1
            evidence.append(f"Identity/asset fields: {identity[:6]}")

        operational = self._matching_columns(
            ["os", "application", "process", "service", "department", "role"]
        )
        if operational:
            score += 1
            evidence.append(f"Operational context: {operational[:6]}")

        return min(score, 5), evidence

    def score_threat_intelligence(self):
        score, evidence = 0, []
        native = self._matching_columns(self.TI_NATIVE_KEYWORDS)

        if native:
            score += 3
            evidence.append(f"Native TI fields: {native[:10]}")

        endpoints = self._matching_columns(self.SOURCE_KEYWORDS) + \
                    self._matching_columns(self.DESTINATION_KEYWORDS)
        ti_enrichment = self._matching_columns(
            ["domain", "url", "hash", "md5", "sha1", "sha256"]
        )
        if endpoints or ti_enrichment:
            score += 1
            evidence.append(
                "External TI enrichment potential via network indicators"
            )

        technique = self._matching_columns(
            ["attack_type", "technique", "mitre", "ttp", "tactic",
             "malware", "botnet", "campaign", "actor"]
        )
        if technique:
            score += 1
            evidence.append(f"Threat classification fields: {technique[:8]}")

        return min(score, 5), evidence

    def score_exploitability(self):
        score, evidence = 0, []
        native = self._matching_columns(self.EXPLOIT_NATIVE_KEYWORDS)

        if native:
            score += 3
            evidence.append(f"Native vulnerability/exploit fields: {native[:10]}")

        if self._has(["cvss"]):
            score += 1
            evidence.append("CVSS information available")

        endpoints = self._matching_columns(self.SOURCE_KEYWORDS) + \
                    self._matching_columns(self.DESTINATION_KEYWORDS)
        if endpoints or self._has(["host", "asset", "device", "machine"]):
            score += 1
            evidence.append(
                "Asset mapping permits external CVE/CVSS/exploitability enrichment"
            )

        return min(score, 5), evidence

    def score_historical(self):
        score, evidence = 0, []
        temporal = self._matching_columns(self.HISTORY_KEYWORDS)
        if temporal:
            score += 2
            evidence.append(f"Temporal fields: {temporal[:10]}")

        src = self._matching_columns(self.SOURCE_KEYWORDS)
        dst = self._matching_columns(self.DESTINATION_KEYWORDS)
        if src and dst:
            score += 1
            evidence.append("Source/destination relationship available")

        if self._has(["duration", "flow_duration", "session"]):
            score += 1
            evidence.append("Duration/session information available")

        if self._has(["sequence", "connection", "session"]):
            score += 1
            evidence.append("Sequence/connection information available")

        return min(score, 5), evidence

    def score_ground_truth(self):
        score, evidence = 0, []
        labels = self._matching_columns(self.LABEL_KEYWORDS)

        if not labels:
            return 0, ["No ground-truth/label field detected"]

        score += 3
        evidence.append(f"Label fields: {labels[:10]}")
        ratio = self._nonnull_ratio(labels)

        if ratio >= 0.95:
            score += 1
            evidence.append(f"High label completeness: {ratio:.2%}")
        elif ratio >= 0.80:
            score += 0.5
            evidence.append(f"Moderate label completeness: {ratio:.2%}")

        for col in labels:
            try:
                unique_values = self.df[col].dropna().nunique()
                if unique_values >= 3:
                    score += 1
                    evidence.append(f"{col}: {unique_values} classes in analyzed sample")
                    break
            except Exception:
                continue

        return min(score, 5), evidence

    def score_scalability(self):
        """
        Score full-dataset scale, not the development sample.

        If total_rows is unavailable, the score is explicitly marked as
        sample-based and should not be reported as a final DSS result.
        """
        n_rows = self.total_rows if self.total_rows is not None else len(self.df)

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

        if self.total_rows is None:
            evidence = [
                f"Full dataset size unavailable; analyzed sample rows: {len(self.df):,}",
                "WARNING: scalability score is sample-based and should not be used as final DSS"
            ]
        else:
            evidence = [
                f"Full dataset rows (metadata): {self.total_rows:,}",
                f"Rows analyzed for schema/evidence: {len(self.df):,}"
            ]
            if self.sample_size is not None:
                evidence.append(f"Development sample size: {self.sample_size:,}")

        if n_rows >= 1_000_000:
            evidence.append("Suitable for high-volume experimentation")

        return score, evidence

    def calculate(self):
        functions = {
            "Alert/Event": self.score_alert_event,
            "Context": self.score_context,
            "Threat Intelligence": self.score_threat_intelligence,
            "Exploitability": self.score_exploitability,
            "Historical": self.score_historical,
            "Ground Truth": self.score_ground_truth,
            "Scalability": self.score_scalability,
        }

        self.results = {}
        for criterion, function in functions.items():
            score, evidence = function()
            self.results[criterion] = {
                "score": float(score),
                "weight": float(self.weights[criterion]),
                "weighted_score": float(score * self.weights[criterion]),
                "evidence": evidence,
            }

        return sum(item["weighted_score"] for item in self.results.values())

    def detailed_result(self):
        dss = self.calculate()
        result = {"Dataset": self.dataset_name}
        for criterion, item in self.results.items():
            result[criterion] = item["score"]
        result["DSS"] = round(dss, 3)
        result["DSS (%)"] = round((dss / 5) * 100, 2)
        result["Analyzed Rows"] = len(self.df)
        result["Total Rows"] = self.total_rows
        return result

    def evidence_table(self):
        self.calculate()
        return pd.DataFrame([
            {
                "Dataset": self.dataset_name,
                "Criterion": criterion,
                "Score": item["score"],
                "Weight": item["weight"],
                "Weighted Score": item["weighted_score"],
                "Evidence": " | ".join(item["evidence"]),
            }
            for criterion, item in self.results.items()
        ])
