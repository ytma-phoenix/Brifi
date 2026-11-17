"""Command-line entry points for the public BRIFi implementation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

from config import ALL_DATASETS, CORE_FAMILIES, AttackConfig, ExperimentConfig, MethodConfig, TrainConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='Run BRIFi or FedAvg on a downstream benchmark')
    run.add_argument('--dataset', choices=ALL_DATASETS, default='cifar10')
    run.add_argument('--data-root', default='./data')
    run.add_argument('--attack', choices=CORE_FAMILIES, default='badnets')
    run.add_argument('--method', choices=['brifi', 'baseline'], default='brifi')
    run.add_argument('--mode', choices=['federated', 'centralized'], default='federated')
    run.add_argument('--detector-checkpoint')
    run.add_argument('--detector-image-size', type=int, default=32)
    run.add_argument('--poison-rate', type=float, default=0.1, help='Rate within each contaminated client')
    run.add_argument('--clients', type=int, default=100)
    run.add_argument('--rounds', type=int, default=20)
    run.add_argument('--local-epochs', type=int, default=1)
    run.add_argument('--participation-rate', type=float, default=0.3)
    run.add_argument('--contaminated-client-ratio', type=float, default=0.2)
    run.add_argument('--dirichlet-alpha', type=float, default=0.5)
    run.add_argument('--epochs', type=int, default=20, help='Centralized mode only')
    run.add_argument('--batch-size', type=int, default=128)
    run.add_argument('--lr', type=float, default=0.01)
    run.add_argument('--drop-cap', type=float, default=0.2)
    run.add_argument('--threshold-alpha', type=float, default=0.05)
    run.add_argument('--history-size', type=int, default=2048)
    run.add_argument('--min-history', type=int, default=64)
    run.add_argument('--seed', type=int, default=1)
    run.add_argument('--workers', type=int, default=0)
    run.add_argument('--device', default='cpu')
    run.add_argument('--output', type=Path)
    run.add_argument('--overwrite', action='store_true')
    run.add_argument('--dry-run', action='store_true', help='Print configuration only; no imports of torch, data downloads, or training')
    detector = commands.add_parser('train-detector', help='Pretrain once on a user-provided, task-disjoint auxiliary training split')
    detector.add_argument('--aux-root', required=True, help='Directory containing only auxiliary training images')
    detector.add_argument('--output', type=Path, required=True)
    detector.add_argument('--samples', type=int, default=10000)
    detector.add_argument('--epochs', type=int, default=10)
    detector.add_argument('--image-size', type=int, default=32)
    detector.add_argument('--lr', type=float, default=1e-3)
    detector.add_argument('--seed', type=int, default=1)
    detector.add_argument('--workers', type=int, default=0)
    detector.add_argument('--device', default='cpu')
    commands.add_parser('smoke', help='Test the filter on synthetic tensors; no data downloads or training')
    return parser, parser.parse_args(argv)


def make_experiment(args):
    method = MethodConfig.baseline() if args.method == 'baseline' else MethodConfig.brifi(
        lrt_alpha=args.threshold_alpha, rho_max=args.drop_cap,
        mad_buffer_size=args.history_size, min_history_size=args.min_history)
    train = TrainConfig(
        seed=args.seed, device=args.device, num_workers=args.workers,
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
        detector_ckpt_path=args.detector_checkpoint, detector_image_size=args.detector_image_size,
        fl=args.mode == 'federated', fl_n_clients=args.clients, fl_rounds=args.rounds,
        fl_local_epochs=args.local_epochs, fl_alpha=args.dirichlet_alpha,
        fl_malicious_frac=args.contaminated_client_ratio,
        fl_participation_rate=args.participation_rate)
    return ExperimentConfig(ALL_DATASETS[args.dataset](args.data_root),
                            AttackConfig(args.attack, poison_rate=args.poison_rate), method, train)


def smoke():
    import torch
    from brifi import BudgetedFilter

    class PixelScore(torch.nn.Module):
        def forward(self, images):
            return images[:, 0, 0, 0]

    filt = BudgetedFilter(PixelScore(), MethodConfig.brifi(), det_img_size=0)
    images = torch.arange(128, dtype=torch.float32)[:, None, None, None].expand(-1, 3, 2, 2) / 128
    keep = filt.keep(images)
    assert int(keep.sum()) == 103
    assert not keep[103:].any()
    assert len(filt.buf) == 103
    keep2 = filt.keep(images)
    assert int((~keep2).sum()) <= 25
    print('PASS: raw-score bootstrap, retained-only history, and rejection budget')
    print('No benchmark data were downloaded and no model was trained.')


def main(argv=None):
    parser, args = parse_args(argv)
    if args.command == 'smoke':
        smoke()
        return
    if args.command == 'run':
        for name in ['clients', 'rounds', 'local_epochs', 'epochs', 'batch_size', 'history_size', 'min_history', 'detector_image_size']:
            if getattr(args, name) <= 0:
                parser.error(f'{name} must be positive')
        if not 0 <= args.poison_rate <= 1 or not 0 <= args.contaminated_client_ratio <= 1:
            parser.error('poison and contaminated-client rates must lie in [0, 1]')
        if not 0 < args.participation_rate <= 1 or args.dirichlet_alpha <= 0:
            parser.error('participation rate must lie in (0, 1] and Dirichlet alpha must be positive')
        if not 0 <= args.drop_cap <= 1 or not 0 < args.threshold_alpha < 1:
            parser.error('drop cap must lie in [0, 1] and threshold alpha in (0, 1)')
        if args.min_history > args.history_size or args.workers < 0 or args.lr <= 0:
            parser.error('Invalid history size, worker count, or learning rate')
        experiment = make_experiment(args)
        if args.dry_run:
            print(json.dumps(asdict(experiment), indent=2))
            return
        if args.method == 'brifi' and not (args.detector_checkpoint and Path(args.detector_checkpoint).is_file()):
            parser.error('BRIFi requires an existing --detector-checkpoint; use train-detector with independent auxiliary images first')
        if args.output is None:
            parser.error('--output is required for an experiment')
        if args.output.exists() and not args.overwrite:
            parser.error('Output exists; choose another path or explicitly use --overwrite')
        if experiment.train.fl:
            from fl_utils import run_fedavg_experiment
            result = run_fedavg_experiment(experiment)
        else:
            from trainer import run_one_experiment
            result = run_one_experiment(experiment)
        result['run_config'] = asdict(experiment)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        return
    if args.output.exists():
        parser.error('Detector output already exists; choose a new checkpoint path')
    if min(args.samples, args.epochs, args.image_size) <= 0 or args.lr <= 0 or args.workers < 0:
        parser.error('Invalid detector training parameters')
    if not Path(args.aux_root).is_dir():
        parser.error('--aux-root must be an existing directory of independent auxiliary training images')
    import numpy as np
    import torch
    from trainer import build_and_train_detector
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = TrainConfig(detector_aux_root=args.aux_root, detector_ckpt_path=str(args.output),
                        detector_image_size=args.image_size, detector_base_n=args.samples,
                        detector_epochs=args.epochs, detector_lr=args.lr,
                        seed=args.seed, device=args.device, num_workers=args.workers)
    experiment = ExperimentConfig(ALL_DATASETS['cifar10'](),
                                  AttackConfig('badnets'), MethodConfig.brifi(), train)
    build_and_train_detector(experiment)
    print(f'Saved frozen detector checkpoint to {args.output}')


if __name__ == '__main__':
    main()
