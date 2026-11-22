from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from model import eval_model, train_model
from model.main import _policy_flat, cf_6M_2024, compute_loss, PolicyHead


class SyntheticDataset(torch.utils.data.Dataset):
    """Small finite fixture for trainer/resume tests; no user data required."""
    history = 8
    repetition_features = "binary"
    return_legal_mask = True
    return_game_id = False
    plies_sha256 = "synthetic-training-data"

    def __len__(self):
        return 4

    def __getitem__(self, index):
        board = torch.zeros(64, 112)
        board[:, 111] = 1
        board[index, 0] = 1
        mask = torch.zeros(4352, dtype=torch.bool)
        mask[:4] = True
        return {"board": board, "policy_target": index, "value_target": index % 3,
                "legal_mask": mask}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = torch.nn.Parameter(torch.zeros(64, 64))
        self.promotion = torch.nn.Parameter(torch.zeros(8, 8, 4))
        self.value = torch.nn.Parameter(torch.zeros(3))

    def forward(self, inputs):
        n = len(inputs)
        return (self.policy.expand(n, -1, -1), self.promotion.expand(n, -1, -1, -1),
                self.value.expand(n, -1))


class PrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_half_loss_overflow_is_fixed_without_changing_objective(self):
        policy = torch.zeros(2, 64, 64, dtype=torch.float16)
        policy[:, 0, 0] = 60000
        policy[:, 0, 1] = -60000
        policy.requires_grad_()
        promotion = torch.zeros(2, 8, 8, 4, dtype=torch.float16, requires_grad=True)
        value = torch.zeros(2, 3, dtype=torch.float16, requires_grad=True)
        target = torch.ones(2, dtype=torch.long)
        mask = torch.zeros(2, 4352, dtype=torch.bool)
        mask[:, :2] = True
        old = F.cross_entropy(_policy_flat(policy, promotion).masked_fill(~mask, -torch.inf), target)
        self.assertFalse(torch.isfinite(old))
        loss = compute_loss(policy, promotion, value, target, target, legal_mask=mask)
        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, torch.tensor(120000.) + torch.log(torch.tensor(3.)))
        loss.backward()
        for tensor in (policy, promotion, value):
            self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertTrue(torch.all(policy.grad[:, 0, 2:] == 0))

    def test_bfloat16_and_float32_loss_match_float32_reference(self):
        generator = torch.Generator().manual_seed(11)
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                policy = torch.randn(2, 64, 64, generator=generator).to(dtype)
                promotion = torch.randn(2, 8, 8, 4, generator=generator).to(dtype)
                value = torch.randn(2, 3, generator=generator).to(dtype)
                targets = torch.tensor([1, 4096])
                outcomes = torch.tensor([0, 2])
                mask = torch.zeros(2, 4352, dtype=torch.bool)
                mask[:, [1, 3, 4096]] = True
                reference = (F.cross_entropy(_policy_flat(policy, promotion).float().masked_fill(
                    ~mask, -torch.inf), targets) + F.cross_entropy(value.float(), outcomes))
                loss = compute_loss(policy, promotion, value, targets, outcomes, mask)
                self.assertEqual(loss.dtype, torch.float32)
                torch.testing.assert_close(loss, reference, rtol=0, atol=0)

    def test_bfloat16_policy_head_handles_large_finite_activations(self):
        head = PolicyHead()
        with torch.no_grad():
            head.embed.weight.zero_()
            head.embed.bias.fill_(100)
            head.q_proj.weight.copy_(torch.eye(256))
            head.k_proj.weight.copy_(torch.eye(256))
            inputs = torch.zeros(1, 64, 256)
            reference, _ = head(inputs)
            with torch.autocast("cpu", dtype=torch.float16):
                fp16_logits, _ = head(inputs)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                bf16_logits, _ = head(inputs)
        self.assertFalse(torch.isfinite(fp16_logits).all())
        self.assertTrue(torch.isfinite(bf16_logits).all())
        torch.testing.assert_close(bf16_logits.float(), reference, rtol=0.01, atol=0)

    def test_old_amp_metadata_requires_explicit_precision_migration(self):
        checkpoint = {"amp": True, "seed": 0}
        settings = {"amp": True, "amp_dtype": "bfloat16", "loss_dtype": "float32", "seed": 0}
        self.assertEqual(train_model.checkpoint_precision(checkpoint),
                         {"amp": True, "amp_dtype": "float16", "loss_dtype": "float16"})
        with self.assertRaisesRegex(ValueError, "allow-precision-change"):
            train_model.validate_resume(checkpoint, settings)
        train_model.validate_resume(checkpoint, settings, allow_precision_change=True)
        with self.assertRaisesRegex(ValueError, "seed"):
            train_model.validate_resume(checkpoint, {**settings, "seed": 1}, allow_precision_change=True)

    def test_legacy_cpu_checkpoint_needs_no_precision_override(self):
        train_model.validate_resume({"amp": False},
            {"amp": False, "amp_dtype": "float32", "loss_dtype": "float32"})

    def test_corrupted_model_or_optimizer_state_is_rejected(self):
        train_model.ensure_finite_state({"step": torch.tensor(3.), "nested": [0.1]}, "state")
        for bad in (float("nan"), torch.tensor([float("inf")])):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "non-finite checkpoint"):
                train_model.ensure_finite_state({"state": {1: {"exp_avg": bad}}}, "optimizer")

    def test_failure_details_distinguish_invalid_masks_and_logits(self):
        inputs = torch.zeros(2, 64, 112)
        policy = torch.zeros(2, 64, 64)
        policy[0, 0, 0] = torch.inf
        mask = torch.zeros(2, 4352, dtype=torch.bool)
        mask[1, 0] = True
        details = train_model.loss_diagnostics(inputs, (policy, torch.zeros(2, 8, 8, 4),
            torch.zeros(2, 3)), torch.tensor([0, 1]), mask)
        self.assertEqual(details["nonfinite_inputs"], 0)
        self.assertEqual(details["nonfinite_policy_logits"], 1)
        self.assertEqual(details["empty_legal_masks"], 1)
        self.assertEqual(details["masked_out_targets"], 2)

    def test_resume_keeps_optimizer_state_and_records_precision_transition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                loader = torch.utils.data.DataLoader(SyntheticDataset(), batch_size=2)
                with (patch.object(train_model, "TCECPlies", return_value=SyntheticDataset()),
                      patch.object(train_model.np, "load", return_value=[0, 1, 2, 3]),
                      patch.object(train_model, "sha256_file", return_value="index-hash"),
                      patch.object(train_model, "build_eval_loader", return_value=loader),
                      patch.object(train_model, "cf_6M_2024", side_effect=TinyModel),
                      redirect_stdout(io.StringIO())):
                    args = ["--history", "8", "--repetition-features", "binary", "--workers", "0",
                            "--device", "cpu", "--batch-size", "2", "--lr-horizon", "100000"]
                    train_model.main(args + ["--run-name", "reference", "--steps", "4"])
                    train_model.main(args + ["--run-name", "recover", "--steps", "2"])
                    checkpoint_path = root / "checkpoints/recover/model_step_2.pt"
                    checkpoint = torch.load(checkpoint_path, weights_only=False)
                    # Simulate metadata from the old trainer. Tensor contents stay
                    # identical so we can also verify exact optimizer restoration.
                    checkpoint["amp"] = True
                    for key in ("amp_dtype", "loss_dtype", "precision_history"):
                        checkpoint.pop(key)
                    torch.save(checkpoint, checkpoint_path)
                    with self.assertRaisesRegex(ValueError, "allow-precision-change"):
                        train_model.main(args + ["--run-name", "recover", "--steps", "4", "--resume"])
                    train_model.main(args + ["--run-name", "recover", "--steps", "4", "--resume",
                                             "--allow-precision-change"])
                recovered = torch.load(root / "checkpoints/recover/model_step_4.pt", weights_only=False)
                reference = torch.load(root / "checkpoints/reference/model_step_4.pt", weights_only=False)
                for key, expected in reference["model_state_dict"].items():
                    torch.testing.assert_close(recovered["model_state_dict"][key], expected, rtol=0, atol=0)
                self.assertEqual(recovered["precision_history"][0]["amp_dtype"], "float16")
                self.assertEqual(recovered["precision_history"][1]["first_step"], 3)
                self.assertEqual(recovered["precision_history"][1]["amp_dtype"], "float32")
                config = json.loads((root / "checkpoints/recover/run_config.json").read_text())
                self.assertEqual(config["precision_history"], recovered["precision_history"])
            finally:
                os.chdir(old_cwd)

    def test_nonfinite_gradient_stops_before_unscaled_optimizer_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_cwd = Path.cwd()
            model = TinyModel()
            model.policy.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
            before = {key: value.clone() for key, value in model.state_dict().items()}
            try:
                os.chdir(root)
                loader = torch.utils.data.DataLoader(SyntheticDataset(), batch_size=2)
                with (patch.object(train_model, "TCECPlies", return_value=SyntheticDataset()),
                      patch.object(train_model.np, "load", return_value=[0, 1, 2, 3]),
                      patch.object(train_model, "sha256_file", return_value="index-hash"),
                      patch.object(train_model, "build_eval_loader", return_value=loader),
                      patch.object(train_model, "cf_6M_2024", return_value=model),
                      redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "non-finite")):
                    train_model.main(["--run-name", "bad-grad", "--steps", "2", "--workers", "0",
                                      "--batch-size", "2", "--device", "cpu"])
                for key, expected in before.items():
                    torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)
                self.assertFalse(list((root / "checkpoints/bad-grad").glob("model_step_*.pt")))
            finally:
                os.chdir(old_cwd)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                         "requires a CUDA GPU with bfloat16 support")
    def test_actual_model_cuda_bfloat16_updates_stay_finite(self):
        from model.encoder import encode_position, legal_move_mask, encode_move
        import chess
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(0)
            board = chess.Board()
            inputs = torch.from_numpy(encode_position([board.fen()])).repeat(2, 1, 1).cuda()
            mask = torch.from_numpy(legal_move_mask(board)).repeat(2, 1).cuda()
            target = torch.full((2,), encode_move(chess.Move.from_uci("e2e4"), board.turn),
                                dtype=torch.long, device="cuda")
            value_target = torch.ones(2, dtype=torch.long, device="cuda")
            model = cf_6M_2024().cuda().train()
            optimizer = torch.optim.NAdam(model.parameters(), lr=train_model.LR,
                betas=(train_model.BETA1, train_model.BETA2), eps=train_model.EPS)
            for _ in range(3):
                optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(inputs)
                loss = compute_loss(*logits, target, value_target, mask)
                self.assertEqual(loss.dtype, torch.float32)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
                optimizer.step()
            train_model.ensure_finite_state(model.state_dict(), "CUDA model")
            train_model.ensure_finite_state(optimizer.state_dict(), "CUDA optimizer")


if __name__ == "__main__":
    unittest.main()
