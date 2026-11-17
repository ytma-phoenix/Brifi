"""Synthetic functional checks; no downloads, optimizer steps, or training."""
import tempfile
import random
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from brifi import BudgetedFilter, canonicalize_detector
from config import AttackConfig, DatasetConfig, ExperimentConfig, MethodConfig, TrainConfig


class PixelScore(torch.nn.Module):
    def forward(self, images):
        return images[:, 0, 0, 0] * 100


def image_batch(scores):
    return torch.tensor(scores, dtype=torch.float32)[:, None, None, None].expand(-1, 3, 2, 2) / 100


class CoreTests(unittest.TestCase):
    def test_classifier_initialization_is_independent_of_detector_rng(self):
        import fl_utils

        class Data:
            def __len__(self):
                return 4
            def __getitem__(self, index):
                return Image.new('RGB', (32, 32)), index % 2

        initial = []
        def make_model(config):
            model = torch.nn.Linear(1, 1)
            initial.append(model.weight.detach().clone())
            return model
        def detector_with_random_work(config):
            if config.method.name == 'brifi':
                torch.rand(1000)
            return PixelScore()
        train = TrainConfig(fl=True, fl_n_clients=1, fl_rounds=1,
                            fl_participation_rate=1., fl_malicious_frac=0.)
        for method in (MethodConfig.baseline(), MethodConfig.brifi()):
            config = ExperimentConfig(DatasetConfig.cifar10(), AttackConfig('badnets'), method, train)
            with patch.object(fl_utils, 'get_raw_dataset', return_value=Data()), \
                 patch.object(fl_utils, 'dirichlet_partition', return_value=[np.arange(4)]), \
                 patch.object(fl_utils, 'build_and_train_detector', side_effect=detector_with_random_work), \
                 patch.object(fl_utils, 'build_classifier', side_effect=make_model), \
                 patch.object(fl_utils, 'train_epoch', return_value=(0., 0., 0, 0)), \
                 patch.object(fl_utils, 'eval_acc', return_value=.5), \
                 patch.object(fl_utils, 'eval_asr_suite', return_value={}):
                fl_utils.run_fedavg_experiment(config)
        self.assertTrue(torch.equal(initial[0], initial[1]))

    def test_asr_family_draws_repeat_and_restore_random_state(self):
        import trainer

        class Data:
            def __len__(self):
                return 2
            def __getitem__(self, index):
                return Image.new('RGB', (32, 32), color=(70, 80, 90)), 1

        def inspect_trigger(model, loader, target, device):
            return sum(float(loader.dataset[i][0].sum()) for i in range(2))
        config = ExperimentConfig(DatasetConfig.cifar10(), AttackConfig('badnets'), MethodConfig.baseline())
        torch.manual_seed(5)
        random.seed(6)
        before_torch, before_python = torch.get_rng_state().clone(), random.getstate()
        with patch.object(trainer, 'get_raw_dataset', return_value=Data()), \
             patch.object(trainer, 'eval_asr', side_effect=inspect_trigger):
            first = trainer.eval_asr_suite(None, config)
            second = trainer.eval_asr_suite(None, config)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {'badnets', 'blended', 'dba', 'pattern', 'noise', 'wanet'})
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())

    def test_bootstrap_orders_raw_scores_and_respects_cap(self):
        filt = BudgetedFilter(PixelScore(), MethodConfig.brifi(), det_img_size=0)
        scores = torch.linspace(0, 99, 128)
        keep = filt.keep(image_batch(scores.tolist()))
        self.assertEqual(int(keep.sum()), 103)
        self.assertTrue(keep[:103].all())
        self.assertFalse(keep[103:].any())
        self.assertEqual(len(filt.buf), 103)

    def test_retained_only_update_and_no_clipped_ranking(self):
        cfg = MethodConfig.brifi(rho_max=0.25, mad_buffer_size=64, min_history_size=2)
        filt = BudgetedFilter(PixelScore(), cfg, det_img_size=0)
        filt.buf.extend([0., 1.])
        keep = filt.keep(image_batch([50, 99, 10, 0]))
        self.assertEqual(keep.tolist(), [True, False, True, True])
        self.assertEqual(len(filt.buf), 5)
        self.assertAlmostEqual(list(filt.buf)[-3], 50.)
        self.assertNotIn(99., list(filt.buf))

    def test_history_is_bounded_and_client_local(self):
        cfg = MethodConfig.brifi(rho_max=0.0, mad_buffer_size=4, min_history_size=2)
        first = BudgetedFilter(PixelScore(), cfg, det_img_size=0)
        other = BudgetedFilter(PixelScore(), cfg, det_img_size=0)
        first.keep(image_batch([1, 2, 3, 4]))
        first.keep(image_batch([5, 6]))
        self.assertEqual(len(other.buf), 0)
        self.assertEqual(len(first.buf), 4)
        self.assertTrue(np.allclose(list(first.buf), [3, 4, 5, 6]))

    def test_single_sided_threshold_retains_low_scores(self):
        cfg = MethodConfig.brifi(rho_max=1.0, min_history_size=2)
        filt = BudgetedFilter(PixelScore(), cfg, det_img_size=0)
        filt.buf.extend([50., 51.])
        keep = filt.keep(image_batch([0, 50, 51, 90]))
        self.assertEqual(keep.tolist(), [True, True, True, False])

    def test_detector_parameters_and_batchnorm_are_frozen(self):
        detector = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.AdaptiveAvgPool2d(1),
                                       torch.nn.Flatten(), torch.nn.Linear(3, 1))
        filt = BudgetedFilter(detector, MethodConfig.brifi(), det_img_size=32)
        mean = detector[0].running_mean.clone()
        detector.train()  # Caller cannot accidentally change scoring-time BN behavior.
        filt.keep(torch.rand(4, 1, 64, 64))
        self.assertTrue(torch.equal(mean, detector[0].running_mean))
        self.assertFalse(detector.training)
        self.assertTrue(all(not p.requires_grad for p in detector.parameters()))

    def test_shared_auxiliary_canonicalization(self):
        from datasets import AuxiliaryImageDataset, DetectorDataset
        from attacks import TriggerSampler
        from torchvision.transforms.functional import to_tensor
        from dataclasses import replace
        from datasets import _DETECTOR_REF_DCFG
        pixels = np.random.RandomState(4).randint(0, 256, (61, 47, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            Image.fromarray(pixels).save(Path(tmp) / 'sample.png')
            data = AuxiliaryImageDataset(tmp, 32)
            actual, _ = data[0]
            expected = canonicalize_detector(to_tensor(Image.fromarray(pixels))[None], 32)[0]
            self.assertTrue(torch.equal(actual, expected))
            cfg = replace(_DETECTOR_REF_DCFG, image_size=32)
            pairs = DetectorDataset(data, cfg, TriggerSampler(cfg, ['badnets']), 1)
            for i in range(2):
                image, label = pairs[i]
                self.assertEqual(tuple(image.shape), (3, 32, 32))
                self.assertEqual(float(label), float(i))

    def test_missing_checkpoint_never_uses_downstream_data(self):
        from trainer import build_and_train_detector
        config = ExperimentConfig(DatasetConfig.cifar10(), AttackConfig('badnets'),
                                  MethodConfig.brifi(), TrainConfig())
        with patch('trainer.get_raw_dataset', side_effect=AssertionError('must not download data')):
            with self.assertRaisesRegex(ValueError, 'requires an existing'):
                build_and_train_detector(config)

    def test_checkpoint_load_freezes_without_training(self):
        from models import DetectorCNN
        from trainer import build_and_train_detector
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'detector.pt'
            torch.save({'state_dict': DetectorCNN().state_dict(),
                        'metadata': {'architecture': 'DetectorCNN', 'image_size': 32}}, checkpoint)
            config = ExperimentConfig(DatasetConfig.mnist(), AttackConfig('badnets'),
                                      MethodConfig.brifi(), TrainConfig(detector_ckpt_path=str(checkpoint)))
            detector = build_and_train_detector(config)
            self.assertFalse(detector.training)
            self.assertTrue(all(not p.requires_grad for p in detector.parameters()))

    def test_zero_contamination_and_partial_participation(self):
        import fl_utils

        class FakeData:
            def __len__(self):
                return 32
            def __getitem__(self, index):
                return Image.new('RGB', (32, 32)), index % 2

        partitions = [np.arange(i * 8, (i + 1) * 8) for i in range(4)]
        observed_poison = []
        def no_training(model, loader, optimizer, device, pf, flags):
            observed_poison.append(int(loader.dataset.is_poisoned.sum()))
            return 0., 0., 0, 0
        train = TrainConfig(fl=True, fl_n_clients=4, fl_rounds=2, fl_local_epochs=2,
                            fl_participation_rate=0.5, fl_malicious_frac=0.)
        config = ExperimentConfig(DatasetConfig.cifar10(), AttackConfig('badnets'), MethodConfig.baseline(), train)
        with patch.object(fl_utils, 'get_raw_dataset', return_value=FakeData()), \
             patch.object(fl_utils, 'dirichlet_partition', return_value=partitions), \
             patch.object(fl_utils, 'build_and_train_detector', return_value=PixelScore()), \
             patch.object(fl_utils, 'build_classifier', return_value=torch.nn.Linear(1, 1)), \
             patch.object(fl_utils, 'train_epoch', side_effect=no_training), \
             patch.object(fl_utils, 'eval_acc', return_value=.5), \
             patch.object(fl_utils, 'eval_asr_suite', return_value={}):
            result = fl_utils.run_fedavg_experiment(config)
        self.assertEqual(result['contaminated_clients'], [])
        self.assertEqual(observed_poison, [0] * 8)
        self.assertTrue(all(len(r['selected_clients']) == 2 for r in result['round_history']))
        self.assertTrue(all(r['seen'] == 32 for r in result['round_history']))
        self.assertEqual(result['drop_rate'], 0.)


if __name__ == '__main__':
    unittest.main()
