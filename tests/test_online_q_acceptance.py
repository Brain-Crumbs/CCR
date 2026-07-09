from cognitive_runtime.cli import main
from cognitive_runtime.tools.episode_viewer import view_episode
from cognitive_runtime.tools.metrics_dashboard import dashboard
from cognitive_runtime.training.online_q_acceptance import run_simulated_online_acceptance


def test_simulated_online_q_beats_random_reproducibly():
    result = run_simulated_online_acceptance()

    assert result.accepted
    assert result.acceptance_metric == "reward"
    assert result.online_eval.total_reward == 15.75
    assert result.random_eval.total_reward == 11.76
    assert result.online_eval.total_ticks == 2400
    assert result.random_eval.total_ticks == 2337
    assert result.online_eval.total_reward > result.random_eval.total_reward
    assert result.online_eval.total_ticks > result.random_eval.total_ticks
    assert result.training_ticks == 13438


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

