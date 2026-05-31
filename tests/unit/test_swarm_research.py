from __future__ import annotations

import math

import pytest

from ruview.swarm import (
    CoveragePhase,
    CoverageStrategy,
    CsiDetection,
    DroneState,
    FusedDetection,
    GossipState,
    LeaderFollower,
    MeshTopology,
    MultiViewFusion,
    NodeId,
    Position3D,
    ProbabilityGrid,
    RaftConfig,
    RaftNode,
    RaftRole,
    ReynoldsParams,
    RequestVote,
    Velocity3D,
    compute_mission_metrics,
    coverage_fraction,
    formation_vector_to_target,
    gdop,
    mean_uncertainty,
)


def test_mesh_nearest_k_ordering_and_remove_cluster_head() -> None:
    topology = MeshTopology(cluster_head=NodeId(3))
    topology.update_node(_state(0, 0.0, 0.0))
    topology.update_node(_state(1, 10.0, 0.0))
    topology.update_node(_state(2, 3.0, 4.0))
    topology.update_node(_state(3, 2.0, 0.0))

    assert topology.nearest_k(NodeId(0), 3) == [NodeId(3), NodeId(2), NodeId(1)]
    assert [state.id for state in topology.active_nodes()] == [NodeId(0), NodeId(1), NodeId(2), NodeId(3)]

    assert topology.remove(NodeId(3))
    assert topology.cluster_head is None
    assert topology.nearest_k(NodeId(0), 2) == [NodeId(2), NodeId(1)]


def test_gossip_merge_and_raft_eligibility_are_deterministic() -> None:
    older = GossipState(value="older", version=2, origin=NodeId(1), timestamp_ms=0)
    newer = GossipState(value="newer", version=5, origin=NodeId(2), timestamp_ms=0)
    tie_low = GossipState(value="low", version=5, origin=NodeId(1), timestamp_ms=0)

    assert GossipState.merge(older, newer).value == "newer"
    assert GossipState.merge(tie_low, newer).value == "newer"
    assert newer.spread(2, [NodeId(0), NodeId(1), NodeId(2), NodeId(3)], NodeId(0)) == [NodeId(1), NodeId(3)]

    eligible = DroneState(id=NodeId(9), battery_pct=50.0, link_quality=0.9)
    depleted = DroneState(id=NodeId(10), battery_pct=5.0, link_quality=0.9)
    config = RaftConfig(election_timeout_ms=100)
    node = RaftNode(NodeId(9), config=config)

    assert RaftNode.is_eligible_leader(eligible, config)
    assert not RaftNode.is_eligible_leader(depleted, config)
    message = node.tick(200, [])
    assert isinstance(message, RequestVote)
    assert node.role == RaftRole.LEADER


def test_formation_vectors_point_toward_target_and_away_when_too_close() -> None:
    current = Position3D(0.0, 0.0, -5.0)
    target = Position3D(4.0, -2.0, -5.0)

    vector = formation_vector_to_target(current, target, gain=0.5)

    assert vector.vx > 0.0
    assert vector.vy < 0.0
    assert vector.vz == pytest.approx(0.0)

    leader = LeaderFollower(NodeId(0))
    leader.add_follower(NodeId(1), (-5.0, 0.0, 0.0))
    follower_vector = leader.formation_vector(
        NodeId(1),
        [
            (NodeId(0), Position3D(10.0, 0.0, -5.0)),
            (NodeId(1), Position3D(0.0, 0.0, -5.0)),
        ],
    )
    assert follower_vector.vx > 0.0

    reynolds = ReynoldsParams(separation_dist_m=5.0, cohesion_weight=0.0)
    separation = reynolds.compute_velocity(
        NodeId(0),
        [
            (NodeId(0), Position3D(0.0, 0.0, 0.0)),
            (NodeId(1), Position3D(1.0, 0.0, 0.0)),
        ],
    )
    assert separation.vx < 0.0


def test_probability_grid_systematic_and_highest_priority_selection() -> None:
    grid = ProbabilityGrid(width=3, height=2, cell_size_m=2.0)

    assert grid.next_systematic_cell() == (0, 0)
    grid.mark_scanned((0, 0))
    assert grid.next_systematic_cell() == (1, 0)
    grid.mark_scanned((1, 0))
    grid.mark_scanned((2, 0))
    assert grid.next_systematic_cell() == (2, 1)

    grid.set_probability((1, 1), 0.95)
    assert grid.highest_priority_unscanned() == (1, 1)


def test_coverage_strategy_phase_transition_thresholds() -> None:
    grid = ProbabilityGrid(width=2, height=2, cell_size_m=5.0)
    strategy = CoverageStrategy(convergence_threshold=0.7)

    grid.set_probability((0, 0), 0.69)
    assert strategy.phase_transition(grid) == CoveragePhase.SYSTEMATIC

    grid.set_probability((0, 0), 0.75)
    assert strategy.phase_transition(grid) == CoveragePhase.PROBABILISTIC_PURSUIT

    grid.set_probability((1, 1), 0.91)
    assert strategy.phase_transition(grid, contributing_drones=[NodeId(1), NodeId(2)]) == CoveragePhase.CONVERGENCE
    assert strategy.convergence_drones == (NodeId(1), NodeId(2))


def test_multiview_fusion_rejects_single_view_and_improves_three_view_estimate() -> None:
    fusion = MultiViewFusion(min_viewpoints=2, min_confidence=0.5)
    victim = Position3D(50.0, 50.0, 0.0)
    detections = [
        CsiDetection(NodeId(0), 0.85, Position3D(52.0, 49.0, 0.0), 0),
        CsiDetection(NodeId(1), 0.80, Position3D(48.0, 52.0, 0.0), 0),
        CsiDetection(NodeId(2), 0.95, Position3D(50.0, 50.0, 0.0), 0),
    ]
    drone_positions = [
        (NodeId(0), Position3D(0.0, 0.0, -30.0)),
        (NodeId(1), Position3D(100.0, 0.0, -30.0)),
        (NodeId(2), Position3D(50.0, 86.6, -30.0)),
    ]

    assert fusion.fuse(detections[:1], drone_positions[:1]) is None

    result = fusion.fuse(detections, drone_positions)

    assert result is not None
    individual_errors = [detection.victim_position.distance_to(victim) for detection in detections if detection.victim_position]
    fused_error = result.estimated_position.distance_to(victim)
    assert fused_error < max(individual_errors)
    assert result.uncertainty_m < fusion.base_uncertainty_m
    assert result.contributing_drones == (NodeId(0), NodeId(1), NodeId(2))


def test_coverage_and_eval_metrics() -> None:
    grid = ProbabilityGrid(width=2, height=2, cell_size_m=5.0)
    grid.mark_scanned((0, 0))
    grid.mark_scanned((1, 1))

    fused = [
        FusedDetection(0.8, Position3D(0.0, 0.0, 0.0), (NodeId(0), NodeId(1)), 2.0, math.pi / 2.0),
        FusedDetection(0.9, Position3D(1.0, 1.0, 0.0), (NodeId(1), NodeId(2)), 4.0, math.pi / 2.0),
    ]
    target = Position3D(0.0, 0.0, 0.0)
    observers = [
        Position3D(10.0, 0.0, 0.0),
        Position3D(-5.0, 8.66, 0.0),
        Position3D(-5.0, -8.66, 0.0),
    ]

    assert coverage_fraction(grid) == pytest.approx(0.5)
    assert mean_uncertainty(fused) == pytest.approx(3.0)

    metrics = compute_mission_metrics(grid, uncertainties_m=[item.uncertainty_m for item in fused], opportunities=4)
    assert metrics.coverage_fraction == pytest.approx(0.5)
    assert metrics.mean_uncertainty_m == pytest.approx(3.0)
    assert metrics.detection_rate == pytest.approx(0.5)
    assert metrics.n_detections == 2
    assert gdop(observers, target) is not None


def _state(node_id: int, x: float, y: float) -> DroneState:
    return DroneState(
        id=NodeId(node_id),
        position=Position3D(x, y, -10.0),
        velocity=Velocity3D(),
    )
