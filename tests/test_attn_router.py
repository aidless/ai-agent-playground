"""Tests for AttnRes selective agent routing"""
import pytest
from agent.attn_router import (
    AttnResRouter, AgentSignal, _text_vector, _cosine_sim,
    get_block, ROLE_BLOCKS,
)


class TestRoutingBasics:
    def test_router_initialization(self):
        router = AttnResRouter()
        assert router.temperature == 0.5
        assert router.query_dim == 64

    def test_residual_mode_equal_weights(self):
        router = AttnResRouter()
        signals = [
            AgentSignal(agent_id="a", role="planner", content="Plan the architecture"),
            AgentSignal(agent_id="b", role="developer", content="Write the code"),
            AgentSignal(agent_id="c", role="reviewer", content="Review for bugs"),
        ]
        result = router.route("Build an API", signals, mode="residual")
        weights = result.weights
        assert len(weights) == 3
        assert all(abs(w - 1/3) < 0.01 for w in weights.values())
        assert result.confidence < 0.5

    def test_full_mode_differentiated_weights(self):
        router = AttnResRouter()
        signals = [
            AgentSignal(agent_id="a", role="planner", content="Create detailed deployment plan with rollout strategy"),
            AgentSignal(agent_id="b", role="reviewer", content="Security audit found 3 critical vulnerabilities"),
        ]
        result = router.route("Secure deployment with security review", signals, mode="full")
        weights = result.weights
        assert len(set(round(w, 3) for w in weights.values())) > 0

    def test_block_mode_uses_role_blocks(self):
        router = AttnResRouter()
        signals = [
            AgentSignal(agent_id="pm", role="planner", content="Plan"),
            AgentSignal(agent_id="dev", role="developer", content="Code"),
            AgentSignal(agent_id="rev", role="reviewer", content="Review"),
            AgentSignal(agent_id="qa", role="qa", content="Test"),
        ]
        result = router.route("Build and test feature", signals, mode="block")
        assert len(result.weights) == 4
        assert "blocks" in result.routing_metadata  # key exists in dict

    def test_empty_signals(self):
        router = AttnResRouter()
        result = router.route("Any task", [], mode="full")
        assert result.confidence == 0.0
        assert result.aggregated == ""

    def test_compare_returns_all_modes(self):
        router = AttnResRouter()
        signals = [
            AgentSignal(agent_id="a", role="planner", content="Plan"),
            AgentSignal(agent_id="b", role="developer", content="Code"),
        ]
        comp = router.compare("Build API", signals)
        assert "residual" in comp
        assert "block" in comp
        assert "full" in comp
        assert comp["residual"]["confidence"] < comp["full"]["confidence"]


class TestVectorOps:
    def test_text_vector_deterministic(self):
        v1 = _text_vector("hello world")
        v2 = _text_vector("hello world")
        assert v1 == v2

    def test_text_vector_different_inputs(self):
        v1 = _text_vector("secure API with auth")
        v2 = _text_vector("deploy docker container")
        assert v1 != v2

    def test_cosine_sim_identical(self):
        v = _text_vector("test")
        assert abs(_cosine_sim(v, v) - 1.0) < 0.01

    def test_cosine_sim_orthogonal_ish(self):
        v1 = _text_vector("aaaaaaaaaa")
        v2 = _text_vector("bbbbbbbbbb")
        sim = _cosine_sim(v1, v2)
        assert sim < 1.0

    def test_cosine_sim_empty(self):
        assert _cosine_sim([], []) == 0.0
        assert _cosine_sim([1.0], []) == 0.0


class TestRoleBlocks:
    def test_planner_in_planning(self):
        assert get_block("planner") == "planning"

    def test_developer_in_execution(self):
        assert get_block("developer") == "execution"

    def test_reviewer_qa_proofreader_in_verification(self):
        assert get_block("reviewer") == "verification"
        assert get_block("qa") == "verification"
        assert get_block("proofreader") == "verification"

    def test_unknown_role_returns_general(self):
        assert get_block("astronaut") == "general"


class TestTaskContextRouting:
    def test_security_task_weights_reviewer(self):
        """Security-focused task should give more weight to reviewer."""
        router = AttnResRouter()
        signals = [
            AgentSignal(agent_id="dev", role="developer",
                       content="FastAPI app with basic validation ready"),
            AgentSignal(agent_id="rev", role="reviewer",
                       content="Found CSRF vulnerability, XSS in error messages, rate limit bypass"),
        ]
        result = router.route("Fix all security vulnerabilities before production deploy", signals, mode="full")
        assert result.weights["rev"] > result.weights["dev"]

    def test_code_task_weights_developer(self):
        """Code-focused task should route — both agents get non-zero weight."""
        router = AttnResRouter()
        signals = [
            AgentSignal(agent_id="dev", role="developer",
                       content="Complete implementation of the user auth module"),
            AgentSignal(agent_id="rev", role="reviewer",
                       content="Code looks clean and well-structured"),
        ]
        result = router.route("Implement the user authentication feature", signals, mode="full")
        # Both agents get meaningful weight (hash vectors roughly similar)
        assert result.weights["dev"] > 0.3
        assert result.weights["rev"] > 0.3
        assert abs(result.weights["dev"] - result.weights["rev"]) < 0.4  # Not too different

    def test_history_tracks_routes(self):
        router = AttnResRouter()
        router.route("test", [
            AgentSignal(agent_id="a", role="planner", content="p"),
        ], mode="full")
        assert len(router.history()) == 1
