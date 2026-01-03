"""
Analysis module for benchmark and config deviation analysis.

Pure functions for computing consistency metrics, finding deviations,
and analyzing benchmark results. No I/O operations - all data passed as parameters.

Usage:
    from deriva.modules.analysis import (
        compute_consistency_score,
        find_deviations,
        analyze_config_deviations,
        ConfigDeviation,
        DeviationReport,
    )
"""

from __future__ import annotations

from .consistency import (
    compare_object_sets,
    compute_consistency_score,
    find_inconsistencies,
    jaccard_similarity,
)
from .deviation import (
    analyze_config_deviations,
    analyze_from_object_types,
    build_deviation_report,
    compute_deviation_stats,
    extract_element_type,
    extract_node_type,
    generate_recommendations,
    group_objects_by_config,
)
from .types import (
    ConfigDeviation,
    DeviationReport,
    InconsistencyInfo,
    InterModelMetrics,
    IntraModelMetrics,
)

__all__ = [
    # Types
    "ConfigDeviation",
    "DeviationReport",
    "InconsistencyInfo",
    "IntraModelMetrics",
    "InterModelMetrics",
    # Consistency functions
    "compute_consistency_score",
    "find_inconsistencies",
    "compare_object_sets",
    "jaccard_similarity",
    # Deviation functions
    "analyze_config_deviations",
    "analyze_from_object_types",
    "group_objects_by_config",
    "compute_deviation_stats",
    "extract_node_type",
    "extract_element_type",
    "generate_recommendations",
    "build_deviation_report",
]
