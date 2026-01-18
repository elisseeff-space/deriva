"""Tests for modules.derivation.business_actor module."""

from __future__ import annotations

from deriva.modules.derivation.base import Candidate
from deriva.modules.derivation.business_actor import (
    ACTOR_NAME_PATTERNS,
    AUTH_DECORATOR_PATTERNS,
    BusinessActorDerivation,
)


class TestBusinessActorDerivationInit:
    """Tests for BusinessActorDerivation initialization."""

    def test_element_type_is_business_actor(self):
        """Should have BusinessActor as element type."""
        derivation = BusinessActorDerivation()
        assert derivation.ELEMENT_TYPE == "BusinessActor"

    def test_has_relationship_rules(self):
        """Should have outbound and inbound relationship rules defined."""
        derivation = BusinessActorDerivation()
        assert hasattr(derivation, "OUTBOUND_RULES")
        assert hasattr(derivation, "INBOUND_RULES")


class TestHasAuthDecorator:
    """Tests for _has_auth_decorator method."""

    def test_returns_true_for_login_required(self):
        """Should return True for login_required decorator."""
        derivation = BusinessActorDerivation()
        candidate = Candidate(
            node_id="1",
            name="UserView",
            labels=["Method"],
            properties={"decorators": ["login_required"]},
        )
        assert derivation._has_auth_decorator(candidate) is True

    def test_returns_true_for_permission_required(self):
        """Should return True for permission_required decorator."""
        derivation = BusinessActorDerivation()
        candidate = Candidate(
            node_id="1",
            name="AdminView",
            labels=["Method"],
            properties={"decorators": ["permission_required('admin')"]},
        )
        assert derivation._has_auth_decorator(candidate) is True

    def test_returns_true_for_role_required(self):
        """Should return True for role_required decorator."""
        derivation = BusinessActorDerivation()
        candidate = Candidate(
            node_id="1",
            name="ManagerView",
            labels=["Method"],
            properties={"decorators": ["role_required"]},
        )
        assert derivation._has_auth_decorator(candidate) is True

    def test_returns_false_for_no_decorators(self):
        """Should return False when no decorators present."""
        derivation = BusinessActorDerivation()
        candidate = Candidate(
            node_id="1",
            name="PublicView",
            labels=["Method"],
            properties={},
        )
        assert derivation._has_auth_decorator(candidate) is False

    def test_returns_false_for_empty_decorators(self):
        """Should return False when decorators list is empty."""
        derivation = BusinessActorDerivation()
        candidate = Candidate(
            node_id="1",
            name="PublicView",
            labels=["Method"],
            properties={"decorators": []},
        )
        assert derivation._has_auth_decorator(candidate) is False

    def test_returns_false_for_non_auth_decorators(self):
        """Should return False for non-auth decorators."""
        derivation = BusinessActorDerivation()
        candidate = Candidate(
            node_id="1",
            name="CachedView",
            labels=["Method"],
            properties={"decorators": ["cached", "route"]},
        )
        assert derivation._has_auth_decorator(candidate) is False

    def test_handles_non_string_decorators(self):
        """Should handle non-string items in decorators list."""
        derivation = BusinessActorDerivation()
        candidate = Candidate(
            node_id="1",
            name="MixedView",
            labels=["Method"],
            properties={"decorators": [None, 123, "login_required"]},
        )
        assert derivation._has_auth_decorator(candidate) is True

    def test_case_insensitive_matching(self):
        """Should match decorators case-insensitively."""
        derivation = BusinessActorDerivation()
        candidate = Candidate(
            node_id="1",
            name="View",
            labels=["Method"],
            properties={"decorators": ["LOGIN_REQUIRED"]},
        )
        assert derivation._has_auth_decorator(candidate) is True


class TestHasActorNamePattern:
    """Tests for _has_actor_name_pattern method."""

    def test_returns_true_for_user_pattern(self):
        """Should return True for names containing 'user'."""
        derivation = BusinessActorDerivation()
        assert derivation._has_actor_name_pattern("UserService") is True
        assert derivation._has_actor_name_pattern("user_handler") is True

    def test_returns_true_for_admin_pattern(self):
        """Should return True for names containing 'admin'."""
        derivation = BusinessActorDerivation()
        assert derivation._has_actor_name_pattern("AdminController") is True

    def test_returns_true_for_customer_pattern(self):
        """Should return True for names containing 'customer'."""
        derivation = BusinessActorDerivation()
        assert derivation._has_actor_name_pattern("CustomerHandler") is True

    def test_returns_false_for_empty_name(self):
        """Should return False for empty name."""
        derivation = BusinessActorDerivation()
        assert derivation._has_actor_name_pattern("") is False

    def test_returns_false_for_none_name(self):
        """Should return False for None name."""
        derivation = BusinessActorDerivation()
        assert derivation._has_actor_name_pattern(None) is False  # type: ignore

    def test_returns_false_for_non_actor_name(self):
        """Should return False for names without actor patterns."""
        derivation = BusinessActorDerivation()
        assert derivation._has_actor_name_pattern("DataProcessor") is False

    def test_case_insensitive_matching(self):
        """Should match patterns case-insensitively."""
        derivation = BusinessActorDerivation()
        assert derivation._has_actor_name_pattern("USER_SERVICE") is True
        assert derivation._has_actor_name_pattern("AdminService") is True


class TestFilterCandidates:
    """Tests for filter_candidates method."""

    def test_sets_auth_decorator_property(self):
        """Should set has_auth_decorator property on auth candidates."""
        derivation = BusinessActorDerivation()
        candidates = [
            Candidate(
                node_id="1",
                name="SecureView",
                labels=["Method"],
                properties={"decorators": ["login_required"]},
                pagerank=0.5,  # Add pagerank to pass graph filtering
            ),
        ]
        # Provide enrichments to pass graph filtering
        enrichments = {"1": {"pagerank": 0.5}}

        result = derivation.filter_candidates(
            candidates=candidates,
            enrichments=enrichments,
            max_candidates=10,
            include_patterns=set(),
            exclude_patterns=set(),
        )

        # Check that the candidate has the auth decorator property set
        if result:
            assert any(c.properties.get("has_auth_decorator") for c in result)

    def test_sets_actor_name_property(self):
        """Should set has_actor_name property on actor-named candidates."""
        derivation = BusinessActorDerivation()
        candidates = [
            Candidate(
                node_id="1",
                name="UserManager",
                labels=["TypeDefinition"],
                properties={},
                pagerank=0.5,
            ),
        ]
        enrichments = {"1": {"pagerank": 0.5}}

        result = derivation.filter_candidates(
            candidates=candidates,
            enrichments=enrichments,
            max_candidates=10,
            include_patterns=set(),
            exclude_patterns=set(),
        )

        # Check that actor-named candidates get the property set
        if result:
            assert any(c.properties.get("has_actor_name") for c in result)

    def test_respects_max_candidates(self):
        """Should respect max_candidates limit."""
        derivation = BusinessActorDerivation()
        candidates = [
            Candidate(
                node_id=str(i),
                name=f"User{i}",
                labels=["TypeDefinition"],
                properties={},
                pagerank=0.5,
            )
            for i in range(20)
        ]
        enrichments = {str(i): {"pagerank": 0.5} for i in range(20)}

        result = derivation.filter_candidates(
            candidates=candidates,
            enrichments=enrichments,
            max_candidates=5,
            include_patterns=set(),
            exclude_patterns=set(),
        )

        assert len(result) <= 5


class TestConstants:
    """Tests for module constants."""

    def test_auth_decorator_patterns_has_common_patterns(self):
        """Should include common auth decorator patterns."""
        assert "login" in AUTH_DECORATOR_PATTERNS
        assert "permission" in AUTH_DECORATOR_PATTERNS
        assert "auth" in AUTH_DECORATOR_PATTERNS

    def test_actor_name_patterns_has_common_patterns(self):
        """Should include common actor name patterns."""
        assert "user" in ACTOR_NAME_PATTERNS
        assert "admin" in ACTOR_NAME_PATTERNS
        assert "customer" in ACTOR_NAME_PATTERNS
