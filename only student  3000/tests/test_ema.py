import contextlib
import io
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ema import ModelEMA, rng_state, restore_rng, scheduler_state, restore_scheduler
from warmup_scheduler import GradualWarmupScheduler
import train_online_2d


class EMATests(unittest.TestCase):
    def test_delayed_update_buffers_and_skipped_step(self):
        model = torch.nn.BatchNorm1d(2)
        ema = ModelEMA()
        for epoch in range(1, 4):
            ema.begin_epoch(model, epoch)
            ema.update(model)
            self.assertIsNone(ema.model)
        ema.begin_epoch(model, 4)
        self.assertFalse(ema.model.training)
        self.assertTrue(all(not p.requires_grad for p in ema.model.parameters()))
        with torch.no_grad():
            model.weight.fill_(3)
            model.num_batches_tracked.fill_(7)
        ema.update(model, False)
        self.assertEqual(ema.num_updates, 0)
        ema.update(model)
        torch.testing.assert_close(ema.model.weight, torch.full((2,), 1.002))
        self.assertEqual(ema.model.num_batches_tracked.item(), 7)
        self.assertEqual(ema.num_updates, 1)

    def test_resume_before_and_after_activation(self):
        for stop in (2, 4, 6):
            torch.manual_seed(12)
            def build():
                model = torch.nn.Linear(2, 1)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
                scheduler = GradualWarmupScheduler(optimizer, 1, 3,
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 7))
                return model, optimizer, scheduler, ModelEMA()
            model, optimizer, scheduler, ema = build()
            def step(epoch, model, optimizer, scheduler, ema):
                ema.begin_epoch(model, epoch)
                optimizer.zero_grad()
                model(torch.randn(4, 2)).square().mean().backward()
                optimizer.step()
                ema.update(model)
                scheduler.step()
            for epoch in range(1, stop + 1):
                step(epoch, model, optimizer, scheduler, ema)
            with tempfile.TemporaryFile() as output:
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler_state(scheduler), ema=ema.state_dict(), rng=rng_state()), output)
                output.seek(0)
                saved = torch.load(output, weights_only=False)
            for epoch in range(stop + 1, 9):
                step(epoch, model, optimizer, scheduler, ema)
            model2, optimizer2, scheduler2, ema2 = build()
            model2.load_state_dict(saved['model'])
            optimizer2.load_state_dict(saved['optimizer'])
            restore_scheduler(scheduler2, saved['scheduler'])
            ema2.load_state_dict(saved['ema'], model2)
            restore_rng(saved['rng'])
            for epoch in range(stop + 1, 9):
                step(epoch, model2, optimizer2, scheduler2, ema2)
            for first, second in ((model, model2), (ema.model, ema2.model)):
                for key, value in first.state_dict().items():
                    torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(ema.num_updates, ema2.num_updates)
            self.assertEqual(optimizer.param_groups[0]['lr'], optimizer2.param_groups[0]['lr'])
            with self.assertRaises(ValueError):
                ModelEMA(decay=0.9).load_state_dict(saved['ema'], model2)

    def test_launcher_only_adds_ema_and_experiment_suffix(self):
        commands = []
        with patch.dict(os.environ, {'DRY_RUN': '1', 'RESUME': '0', 'RUN_TAG': ''}):
            for enabled in (False, True):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    train_online_2d.main(ema=enabled)
                commands.append(shlex.split(output.getvalue().strip().splitlines()[-1]))
        base, ema = commands
        self.assertEqual(ema.count('--ema'), 1)
        ema.remove('--ema')
        index = ema.index('--env') + 1
        self.assertEqual(ema[index], base[index] + '_ema_e4_d0p999')
        ema[index] = base[index]
        self.assertEqual(base, ema)


if __name__ == '__main__':
    unittest.main()
