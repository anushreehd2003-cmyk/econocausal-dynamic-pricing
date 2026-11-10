import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_pipeline as rp


def names(steps):
    return [script for script, _ in steps]


def test_every_step_script_exists_and_steps_are_unique():
    for steps in (rp.CORE_STEPS, rp.ALL_STEPS):
        assert len(names(steps)) == len(set(names(steps)))
        for script in names(steps):
            assert (rp.SRC / script).exists(), script
    assert set(names(rp.CORE_STEPS)) <= set(names(rp.ALL_STEPS))


def test_steps_run_in_dependency_order():
    for steps in (rp.CORE_STEPS, rp.ALL_STEPS):
        order = names(steps)

        def before(a, b):
            if a in order and b in order:
                assert order.index(a) < order.index(b), (a, b)

        before("generate_data.py", "dml_engine.py")
        before("dml_engine.py", "export_demo.py")
        before("export_demo.py", "efficiency.py")
        before("dml_engine.py", "optimizer.py")
        before("generate_data.py", "train_final_model.py")
        before("train_final_model.py", "drift_demo.py")
    assert names(rp.ALL_STEPS)[0] == "generate_data.py"


def test_build_steps_modes():
    assert rp.build_steps(False) is rp.CORE_STEPS
    assert rp.build_steps(True) is rp.ALL_STEPS


def test_missing_packages_reports_only_absent_ones():
    required = {"os": "os", "not-a-real-package": "definitely_not_a_module_xyz"}
    assert rp.missing_packages(required) == ["not-a-real-package"]
    assert rp.missing_packages({"os": "os"}) == []


def test_missing_files(tmp_path):
    expected = ["a.csv", "sub/b.json"]
    assert rp.missing_files(tmp_path, expected) == expected
    (tmp_path / "a.csv").write_text("x")
    assert rp.missing_files(tmp_path, expected) == ["sub/b.json"]


def test_run_steps_stops_at_the_first_failure(tmp_path, capsys):
    (tmp_path / "ok1.py").write_text("open('marker1', 'w').write('x')")
    (tmp_path / "bad.py").write_text("import sys; sys.exit(3)")
    (tmp_path / "ok2.py").write_text("open('marker2', 'w').write('x')")
    steps = [("ok1.py", "first"), ("bad.py", "second"), ("ok2.py", "third")]
    ok, results = rp.run_steps(steps, src_dir=tmp_path, python=sys.executable)
    assert ok is False
    assert [r[0] for r in results] == ["ok1.py", "bad.py"]
    assert results[1][1] == 3
    assert "--start-at bad.py" in capsys.readouterr().out


def test_run_steps_succeeds_when_all_scripts_pass(tmp_path):
    (tmp_path / "ok1.py").write_text("pass")
    (tmp_path / "ok2.py").write_text("pass")
    ok, results = rp.run_steps([("ok1.py", "a"), ("ok2.py", "b")], src_dir=tmp_path)
    assert ok is True and len(results) == 2


def test_start_at_resumes_from_the_named_step(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(rp, "missing_packages", lambda required=None: [])
    monkeypatch.setattr(rp, "ROOT", tmp_path)
    monkeypatch.setattr(rp, "run_steps", lambda steps, **kw: (seen.extend(names(steps)) or True, []))
    rp.main(["--start-at", "export_demo.py"])
    assert seen == ["export_demo.py", "efficiency.py", "train_final_model.py"]


def test_unknown_start_step_is_rejected(monkeypatch):
    monkeypatch.setattr(rp, "missing_packages", lambda required=None: [])
    assert rp.main(["--start-at", "nope.py"]) == 2


def test_missing_packages_stop_the_run_before_any_step(monkeypatch, capsys):
    monkeypatch.setattr(rp, "missing_packages", lambda required=None: ["dowhy"])
    monkeypatch.setattr(rp, "run_steps", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
    assert rp.main([]) == 1
    assert "pip install -r backend/requirements.txt" in capsys.readouterr().out


def test_check_mode_exit_code_follows_the_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "missing_packages", lambda required=None: [])
    monkeypatch.setattr(rp, "ROOT", tmp_path)
    assert rp.main(["--check"]) == 1
    for name in rp.EXPECTED_OUTPUTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    assert rp.main(["--check"]) == 0