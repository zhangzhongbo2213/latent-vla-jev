import tempfile
from pathlib import Path
import torch
import pytest

from jev.data import validate_cache, validate_pair, SubtaskDataset, collate_batch
from jev.checkpoint import save_checkpoint, load_policy
from jev.model import DecisionJEV, JEVConfig
from jev.codebook import WholeActionCodebook


def payload(ep="e", success=True):
    t, n, dv, dt, ds = 5, 2, 4, 3, 2
    return {"metadata": {"feature_contract": "terminal_conditioned_jev_v1", "vision_encoder": "v", "text_encoder": "t", "control_dt": .1, "action_frame": "body", "terminal_source": "predicted", "lwm_checkpoint": "l"}, "episodes": [{"episode_id": ep, "subtask_id": "s", "success": success, "current_features": torch.randn(t+1,n,dv), "terminal_features": torch.randn(n,dv), "task_features": torch.randn(2,dt), "task_mask": torch.ones(2,dtype=torch.bool), "proprio": torch.randn(t+1,ds), "actions": torch.randn(t,6), "gripper_targets": torch.zeros(t,dtype=torch.long)}]}


def test_boundary_labels_and_failed_mask():
    cache = validate_cache(payload())
    data = SubtaskDataset(cache, horizon=4)
    end = data[len(data)-1]
    assert end["completion"] == 1 and not end["action_mask"].any()
    failed = validate_cache(payload("f", False))
    item = SubtaskDataset(failed)[0]
    assert not item["action_mask"].any() and not item["progress_mask"].any()


def test_episode_split_leakage():
    with pytest.raises(ValueError):
        validate_pair(validate_cache(payload("same")), validate_cache(payload("same")))


def test_checkpoint_reload():
    acts = torch.randn(20, 6)
    cb = WholeActionCodebook.fit(acts, size=4, iterations=1)
    model = DecisionJEV(JEVConfig(4, 3, 2, codebook_size=4, hidden_dim=8, heads=2, context_layers=1, decoder_layers=1))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.pt"; save_checkpoint(p, model, cb, {"x": 1})
        loaded, codebook, meta = load_policy(p)
        assert loaded.config == model.config and codebook.fingerprint() == cb.fingerprint() and meta["x"] == 1
