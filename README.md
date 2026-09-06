# phd-alert-prioritization
To develop a novel alert prioritization mechanism using risk scoring, contextual information, and threat intelligence


## DSS methodology

The DSS analyzer supports development-time row sampling (for example, 1,000
or 10,000 rows) without using the sample size as a proxy for dataset
scalability. Full dataset row counts are obtained through `DatasetLoader.metadata()`
and passed to `DatasetDSSAnalyzer(total_rows=...)`.

Threat intelligence and exploitability are also separated conceptually:
native fields are scored as available evidence, while source/destination
identifiers and asset identifiers are treated as potential inputs for external
TI/vulnerability enrichment rather than as existing threat intelligence.

For final paper results, report the development sample size separately from
the full dataset size and retain the evidence table for auditability.
