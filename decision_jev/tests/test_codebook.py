import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from jev.codebook import WholeActionCodebook


def make_codebook() -> WholeActionCodebook:
    centers = torch.tensor(
        [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], dtype=torch.float32)
    return WholeActionCodebook(centers, torch.tensor([.01, .01, .01, .1, .1, .1]))


def test_encode_decode_nearest_and_physical_scale():
    book = make_codebook()
    actions = torch.tensor([[.009, 0, 0, 0, 0, 0], [0, .015, 0, 0, 0, 0]])
    ids = book.encode(actions)
    assert ids.tolist() == [1, 3]
    decoded = book.decode(ids)
    assert decoded.shape == actions.shape
    assert torch.allclose(decoded[0], torch.tensor([.01, 0, 0, 0, 0, 0]))


def test_zero_token_and_decode_validation():
    book = make_codebook()
    assert book.encode(torch.zeros(6)).item() == 0
    assert torch.equal(book.decode(torch.tensor(0)), torch.zeros(6))
    with pytest.raises(ValueError):
        book.decode(torch.tensor(book.size))
    with pytest.raises(ValueError):
        WholeActionCodebook(torch.ones(8, 6), torch.ones(6))


def test_soft_targets_are_local_normalized_and_ranked():
    book = make_codebook()
    targets = book.soft_targets(torch.tensor([[.009, 0, 0, 0, 0, 0]]), top_k=3,
                                temperature=.2, hard_weight=0)
    assert targets.shape == (1, book.size)
    assert torch.allclose(targets.sum(-1), torch.ones(1))
    support = torch.nonzero(targets[0]).flatten().tolist()
    assert len(support) == 3
    assert torch.argmax(targets[0]).item() == 1
    assert torch.all(targets[0, [i for i in range(book.size) if i not in support]] == 0)


def test_soft_target_hard_mixture_and_bad_arguments():
    book = make_codebook()
    pure = book.soft_targets(torch.tensor([.0, .0, .0, .0, .0, .0]), top_k=2,
                             hard_weight=1)
    assert pure[0] == 1 and pure.sum() == 1
    with pytest.raises(ValueError):
        book.soft_targets(torch.zeros(6), top_k=0)
    with pytest.raises(ValueError):
        book.soft_targets(torch.zeros(6), top_k=2, hard_weight=1.1)


def test_fit_is_seeded_and_reload_has_same_fingerprint(tmp_path):
    torch.manual_seed(3)
    actions = torch.randn(192, 6) * torch.tensor([.01, .01, .01, .1, .1, .1])
    actions[:8] = 0
    one = WholeActionCodebook.fit(actions, size=8, iterations=8, seed=11)
    two = WholeActionCodebook.fit(actions, size=8, iterations=8, seed=11)
    assert one.fingerprint() == two.fingerprint()
    assert torch.equal(one.centers[0], torch.zeros(6))
    path = tmp_path / "book.pt"
    one.save(path)
    loaded = WholeActionCodebook.load(path)
    assert loaded.fingerprint() == one.fingerprint()
    assert loaded.frame == one.frame and loaded.control_dt == one.control_dt


def test_fit_rejects_all_zero_or_insufficient_unique_data():
    with pytest.raises(ValueError, match="all-zero"):
        WholeActionCodebook.fit(torch.zeros(20, 6), size=4)
    with pytest.raises(ValueError, match="distinct"):
        WholeActionCodebook.fit(torch.tensor([[1., 0, 0, 0, 0, 0]]).repeat(20, 1), size=4)


def test_reconstruction_metrics_and_payload():
    book = make_codebook()
    actions = torch.stack((torch.zeros(6), torch.tensor([.01, 0, 0, 0, 0, 0])))
    metrics = book.reconstruction_metrics(actions)
    assert metrics["used_codes"] == 2 and metrics["dead_codes"] == book.size - 2
    assert metrics["translation_rmse_m"] == pytest.approx(0.0)
    assert WholeActionCodebook.from_payload(book.to_payload()).fingerprint() == book.fingerprint()
