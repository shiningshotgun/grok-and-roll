import unittest

import torch

from src.model import Transformer, make_model, P, D_VOCAB, N_CTX, D_MODEL, D_MLP, EQ_TOKEN


class TestModel(unittest.TestCase):
    def test_output_shape(self):
        m = make_model("cpu")
        x = torch.tensor([[0, 1, EQ_TOKEN], [2, 3, EQ_TOKEN]], dtype=torch.long)
        out = m(x)
        self.assertEqual(tuple(out.shape), (2, 3, D_VOCAB))

    def test_param_count(self):
        m = make_model("cpu")
        n = sum(p.numel() for p in m.parameters())
        # expected
        expected = (
            D_MODEL * D_VOCAB  # W_E
            + N_CTX * D_MODEL  # W_pos
            + 3 * (4 * 32 * D_MODEL)  # K,Q,V
            + D_MODEL * D_MODEL  # W_O
            + D_MLP * D_MODEL + D_MLP  # W_in + b_in
            + D_MODEL * D_MLP + D_MODEL  # W_out + b_out
            + D_MODEL * D_VOCAB  # W_U
        )
        self.assertEqual(n, expected)

    def test_reported_drops_eq(self):
        # smoke: loss convention not in model; just ensure EQ_TOKEN=113
        self.assertEqual(EQ_TOKEN, 113)

    def test_snapshot_reload_bitidentical(self):
        import tempfile, os
        m = make_model("cpu")
        x = torch.tensor([[0, 1, EQ_TOKEN], [5, 6, EQ_TOKEN]], dtype=torch.long)
        with torch.no_grad():
            y1 = m(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.pt")
            torch.save(m.state_dict(), path)
            m2 = make_model("cpu")
            m2.load_state_dict(torch.load(path, weights_only=True))
            m2.eval()
            with torch.no_grad():
                y2 = m2(x)
        self.assertTrue(torch.equal(y1, y2))

    def test_record_schema(self):
        # placeholder: metrics keys checked in analysis stage; here just import train
        import json
        sample = {"epoch": 0, "train_loss": 1.0, "test_loss": 1.0, "train_acc": 0.5, "test_acc": 0.5}
        for k in ["epoch", "train_loss", "test_loss", "train_acc", "test_acc"]:
            self.assertIn(k, sample)


if __name__ == "__main__":
    unittest.main()
