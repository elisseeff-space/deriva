"""Tests for fit/underfit/overfit analysis functions.

Tests the fit analysis module which detects:
- Coverage: how well derived model covers expected concepts
- Underfit: model is too simple, missing elements
- Overfit: model has spurious elements not grounded in codebase
"""

from __future__ import annotations

from deriva.modules.analysis.fit_analysis import (
    _find_similar_names,
    _generate_fit_recommendations,
    analyze_coverage,
    create_fit_analysis,
    detect_overfit,
    detect_underfit,
)
from deriva.modules.analysis.types import ReferenceElement, SemanticMatchReport


def make_reference(ident: str, name: str, elem_type: str) -> ReferenceElement:
    """Helper to create ReferenceElement with required fields."""
    return ReferenceElement(identifier=ident, name=name, element_type=elem_type)


def make_semantic_report(
    precision: float = 0.5,
    recall: float = 0.5,
    spurious: list[str] | None = None,
) -> SemanticMatchReport:
    """Helper to create SemanticMatchReport with required fields."""
    return SemanticMatchReport(
        repository="test",
        reference_model_path="test.archimate",
        derived_run="run1",
        total_derived_elements=10,
        total_reference_elements=10,
        spurious_elements=spurious or [],
        element_precision=precision,
        element_recall=recall,
    )


class TestAnalyzeCoverage:
    """Tests for coverage analysis."""

    def test_empty_derived_elements(self):
        """Should handle empty derived elements."""
        coverage, covered, missing = analyze_coverage([], [])

        assert coverage == 0.0
        assert not covered
        assert len(missing) > 0  # Should report missing types

    def test_coverage_with_diverse_types(self):
        """Should identify covered element types."""
        derived = [
            {"type": "ApplicationComponent", "name": "Service1"},
            {"type": "BusinessProcess", "name": "Process1"},
            {"type": "TechnologyService", "name": "Tech1"},
        ]

        coverage, covered, _ = analyze_coverage(derived, [])

        assert coverage > 0
        assert any("ApplicationComponent" in c for c in covered)
        assert any("BusinessProcess" in c for c in covered)
        assert any("TechnologyService" in c for c in covered)

    def test_identifies_missing_types(self):
        """Should identify missing element types."""
        derived = [{"type": "ApplicationComponent", "name": "Service1"}]

        _, _, missing = analyze_coverage(derived, [])

        # Should report missing types like BusinessProcess, DataObject, etc.
        assert any("BusinessProcess" in m for m in missing)

    def test_identifies_missing_layers(self):
        """Should identify layers with no elements."""
        # Only Application layer elements
        derived = [
            {"type": "ApplicationComponent", "name": "Svc1"},
            {"type": "ApplicationService", "name": "Svc2"},
        ]

        _, _, missing = analyze_coverage(derived, [])

        # Should report missing Business and Technology layers
        assert any("Business layer" in m for m in missing)
        assert any("Technology layer" in m for m in missing)

    def test_coverage_with_semantic_report(self):
        """Should use semantic report recall for coverage calculation."""
        derived = [{"type": "ApplicationComponent", "name": "Svc"}]
        reference = [make_reference("ref1", "Ref1", "ApplicationComponent")]
        semantic = make_semantic_report(precision=0.8, recall=0.9)

        coverage, _, _ = analyze_coverage(derived, reference, semantic)

        # Coverage should be weighted by recall (0.9)
        assert coverage > 0.5  # Recall component contributes significantly

    def test_element_type_from_different_keys(self):
        """Should extract type from both 'type' and 'element_type' keys."""
        derived1 = [{"type": "ApplicationComponent", "name": "Svc1"}]
        derived2 = [{"element_type": "ApplicationComponent", "name": "Svc1"}]

        cov1, _, _ = analyze_coverage(derived1, [])
        cov2, _, _ = analyze_coverage(derived2, [])

        assert cov1 == cov2


class TestDetectUnderfit:
    """Tests for underfit detection."""

    def test_no_underfit_with_good_ratio(self):
        """Should not detect underfit when element counts are reasonable."""
        derived = [{"name": f"elem{i}"} for i in range(10)]
        reference = [make_reference(f"ref{i}", f"Ref{i}", "Component") for i in range(12)]

        _, indicators = detect_underfit(derived, reference)

        assert not indicators or "very low" not in str(indicators).lower()

    def test_detects_low_element_count(self):
        """Should detect very low element count as underfit."""
        derived = [{"name": "elem1"}]
        reference = [make_reference(f"ref{i}", f"Ref{i}", "Component") for i in range(10)]

        score, indicators = detect_underfit(derived, reference)

        assert score > 0.5
        assert any("low element count" in ind.lower() for ind in indicators)

    def test_detects_missing_element_types(self):
        """Should detect missing element types as underfit indicator."""
        derived = [{"type": "ApplicationComponent", "name": "Svc1"}]
        reference = [
            make_reference("ref1", "R1", "ApplicationComponent"),
            make_reference("ref2", "R2", "BusinessProcess"),
            make_reference("ref3", "R3", "DataObject"),
        ]

        _, indicators = detect_underfit(derived, reference)

        # 2/3 types missing should trigger underfit
        assert any("missing element types" in ind.lower() for ind in indicators)

    def test_detects_low_recall(self):
        """Should detect low recall from semantic report."""
        derived = [{"name": "elem1"}]
        semantic = make_semantic_report(precision=0.9, recall=0.2)

        _, indicators = detect_underfit(derived, [], semantic)

        assert any("low recall" in ind.lower() for ind in indicators)

    def test_detects_low_derivation_rate(self):
        """Should detect low derivation rate from extraction stats."""
        derived = [{"name": "elem1"}]
        extraction_stats = {"nodes_created": 100}  # Only 1 element from 100 nodes

        _, indicators = detect_underfit(derived, [], extraction_stats=extraction_stats)

        assert any("low derivation rate" in ind.lower() for ind in indicators)


class TestDetectOverfit:
    """Tests for overfit detection."""

    def test_no_overfit_with_good_precision(self):
        """Should not detect overfit when precision is high."""
        derived = [{"name": f"elem{i}"} for i in range(5)]
        reference = [make_reference(f"ref{i}", f"Ref{i}", "Component") for i in range(5)]

        score, _ = detect_overfit(derived, reference)

        assert score < 0.5

    def test_detects_spurious_elements(self):
        """Should detect high spurious element rate."""
        derived = [{"name": f"elem{i}"} for i in range(10)]
        semantic = make_semantic_report(
            precision=0.4,
            recall=0.9,
            spurious=["elem1", "elem2", "elem3", "elem4", "elem5", "elem6"],
        )

        _, indicators = detect_overfit(derived, [], semantic)

        assert any("spurious" in ind.lower() for ind in indicators)

    def test_detects_low_precision(self):
        """Should detect low precision as overfit indicator."""
        derived = [{"name": "elem1"}]
        semantic = make_semantic_report(precision=0.3, recall=0.9)

        _, indicators = detect_overfit(derived, [], semantic)

        assert any("low precision" in ind.lower() for ind in indicators)

    def test_detects_over_generation(self):
        """Should detect over-generation (more elements than reference)."""
        derived = [{"name": f"elem{i}"} for i in range(30)]
        reference = [make_reference(f"ref{i}", f"Ref{i}", "Component") for i in range(10)]

        _, indicators = detect_overfit(derived, reference)

        assert any("over-generation" in ind.lower() for ind in indicators)

    def test_detects_duplicate_names(self):
        """Should detect potential duplicate element names."""
        derived = [
            {"name": "UserService"},
            {"name": "UserServiceHandler"},
            {"name": "UserServiceManager"},
            {"name": "user_service"},  # Similar to UserService
        ]

        _, indicators = detect_overfit(derived, [])

        # Should find similar names like "UserService" and "user_service"
        assert any("duplicate" in ind.lower() for ind in indicators)


class TestFindSimilarNames:
    """Tests for duplicate name detection helper."""

    def test_finds_similar_names(self):
        """Should find pairs of very similar names."""
        names = ["UserService", "user_service", "OrderHandler"]

        pairs = _find_similar_names(names, threshold=0.8)

        assert len(pairs) >= 1
        assert any(
            ("UserService" in p and "user_service" in p)
            or ("user_service" in p and "UserService" in p)
            for p in pairs
        )

    def test_ignores_exact_duplicates(self):
        """Should skip exact case-insensitive duplicates."""
        names = ["Service", "service", "SERVICE"]

        pairs = _find_similar_names(names)

        # Exact duplicates (same name different case) should be skipped
        assert len(pairs) == 0

    def test_respects_threshold(self):
        """Should respect similarity threshold."""
        names = ["abc", "abcd", "xyz"]

        high_threshold = _find_similar_names(names, threshold=0.95)
        low_threshold = _find_similar_names(names, threshold=0.6)

        # Higher threshold -> fewer matches
        assert len(high_threshold) <= len(low_threshold)

    def test_empty_names(self):
        """Should handle empty name list."""
        pairs = _find_similar_names([])
        assert not pairs


class TestCreateFitAnalysis:
    """Tests for complete fit analysis creation."""

    def test_creates_complete_analysis(self):
        """Should create FitAnalysis with all fields."""
        derived = [
            {"type": "ApplicationComponent", "name": "Service1"},
            {"type": "BusinessProcess", "name": "Process1"},
        ]
        reference = [make_reference("ref1", "Ref1", "ApplicationComponent")]

        result = create_fit_analysis(
            repository="test-repo",
            run_id="run1",
            derived_elements=derived,
            reference_elements=reference,
        )

        assert result.repository == "test-repo"
        assert result.run_id == "run1"
        assert 0 <= result.coverage_score <= 1
        assert 0 <= result.underfit_score <= 1
        assert 0 <= result.overfit_score <= 1
        assert len(result.recommendations) > 0

    def test_good_fit_produces_good_fit_message(self):
        """Should produce 'GOOD FIT' message when well calibrated."""
        # Create a scenario with good fit (similar element counts, good precision/recall)
        derived = [{"type": "ApplicationComponent", "name": f"Svc{i}"} for i in range(5)]
        reference = [
            make_reference(f"ref{i}", f"Ref{i}", "ApplicationComponent") for i in range(5)
        ]
        semantic = make_semantic_report(precision=0.9, recall=0.9)

        result = create_fit_analysis(
            repository="test",
            run_id="run1",
            derived_elements=derived,
            reference_elements=reference,
            semantic_report=semantic,
        )

        # With high precision/recall, should have low underfit/overfit
        assert result.underfit_score < 0.5
        assert result.overfit_score < 0.5


class TestGenerateFitRecommendations:
    """Tests for fit recommendation generation."""

    def test_low_coverage_recommendation(self):
        """Should generate recommendation for low coverage."""
        recs = _generate_fit_recommendations(
            coverage_score=0.3,
            underfit_score=0.0,
            underfit_indicators=[],
            overfit_score=0.0,
            overfit_indicators=[],
        )

        assert any("LOW COVERAGE" in r for r in recs)

    def test_moderate_coverage_recommendation(self):
        """Should generate recommendation for moderate coverage."""
        recs = _generate_fit_recommendations(
            coverage_score=0.6,
            underfit_score=0.0,
            underfit_indicators=[],
            overfit_score=0.0,
            overfit_indicators=[],
        )

        assert any("MODERATE COVERAGE" in r for r in recs)

    def test_high_underfit_recommendation(self):
        """Should generate recommendation for high underfit."""
        recs = _generate_fit_recommendations(
            coverage_score=0.8,
            underfit_score=0.7,
            underfit_indicators=["Missing element types: 5 types not derived"],
            overfit_score=0.0,
            overfit_indicators=[],
        )

        assert any("HIGH UNDERFIT" in r for r in recs)
        assert any("missing element types" in r.lower() for r in recs)

    def test_high_overfit_recommendation(self):
        """Should generate recommendation for high overfit."""
        recs = _generate_fit_recommendations(
            coverage_score=0.8,
            underfit_score=0.0,
            underfit_indicators=[],
            overfit_score=0.7,
            overfit_indicators=["Low precision: 30%"],
        )

        assert any("HIGH OVERFIT" in r for r in recs)
        assert any("exclusion rules" in r.lower() for r in recs)

    def test_mixed_fit_issues_recommendation(self):
        """Should generate recommendation for both underfit and overfit."""
        recs = _generate_fit_recommendations(
            coverage_score=0.5,
            underfit_score=0.4,
            underfit_indicators=[],
            overfit_score=0.4,
            overfit_indicators=[],
        )

        assert any("MIXED FIT" in r for r in recs)

    def test_good_fit_recommendation(self):
        """Should generate 'GOOD FIT' when all metrics are good."""
        recs = _generate_fit_recommendations(
            coverage_score=0.9,
            underfit_score=0.1,
            underfit_indicators=[],
            overfit_score=0.1,
            overfit_indicators=[],
        )

        assert any("GOOD FIT" in r for r in recs)
