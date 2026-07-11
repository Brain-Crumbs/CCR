from cognitive_runtime.cli import main
from cognitive_runtime.tools.episode_viewer import view_episode
from cognitive_runtime.tools.metrics_dashboard import dashboard
from cognitive_runtime.training.online_q_acceptance import run_simulated_online_acceptance


def test_simulated_online_q_beats_random_reproducibly():
    """Golden values recomputed for issue #42's expanded action space: a
    ~5x larger action space (14 -> ~130 actions) makes the fixed training
    budget below much harder for online Q (it now beats random only on
    survival ticks, not reward -- `RandomPolicy` sampling many more
    action-selection actions is a materially different baseline). The
    acceptance criterion (`accepted`, an OR over reward/ticks) still holds;
    only which specific metric wins changed."""
    result = run_simulated_online_acceptance()

    assert result.accepted
    assert result.acceptance_metric == "ticks"
    assert result.online_eval.total_reward == -18.62
    assert result.random_eval.total_reward == 12.26
    assert result.online_eval.total_ticks == 2307
    assert result.random_eval.total_ticks == 2287
    assert result.online_eval.total_ticks > result.random_eval.total_ticks
    assert result.training_ticks == 13318


def test_dashboard_and_view_work_with_online_sessions(tmp_path):
    record_dir = tmp_path / "sessions"
    checkpoint = tmp_path / "online-q.json"
    main(
        [
            "run",
            "--policy",
            "online",
            "--episodes",
            "1",
            "--episode-ticks",
            "30",
            "--world-size",
            "32",
            "--online-model",
            str(checkpoint),
            "--record-dir",
            str(record_dir),
            "--session-id",
            "online-tools",
        ]
    )
    session_dir = record_dir / "online-tools"

    viewed = view_episode(str(session_dir), "episode_00000", tail=3)
    dash = dashboard(str(record_dir))

    assert "policy_name: online" in viewed
    assert "action distribution:" in viewed
    assert "last 3 decisions:" in viewed
    assert "online" in dash
    assert "stream_events_per_sec" in dash

