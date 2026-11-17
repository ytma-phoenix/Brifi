from __future__ import annotations
import copy, random, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from config import ExperimentConfig
from attacks import TriggerSampler, get_eval_triggers
from datasets import (CleanTestSet, PoisonedDataset, TriggeredTestSet,
                       get_raw_dataset)
from models import build_classifier, build_detector_model
from trainer import (Prefilter, build_and_train_detector, eval_acc, eval_asr,
                      eval_asr_suite, train_epoch)

def dirichlet_partition(labels: np.ndarray, n_clients: int,
                        alpha: float = 0.5, seed: int = 7) -> List[np.ndarray]:
    rng     = np.random.RandomState(seed)
    classes = np.unique(labels)
    clients: List[list] = [[] for _ in range(n_clients)]
    for c in classes:
        idx    = np.where(labels == c)[0]
        rng.shuffle(idx)
        props  = rng.dirichlet(np.full(n_clients, alpha))
        splits = (np.cumsum(props[:-1]) * len(idx)).astype(int)
        for k, chunk in enumerate(np.split(idx, splits)):
            clients[k].extend(chunk.tolist())
    return [np.array(c, dtype=np.int64) for c in clients]

def fedavg_aggregate(global_model: nn.Module, local_models: List[nn.Module],
                     client_sizes: List[int]) -> None:
    total = sum(client_sizes)
    g_sd  = global_model.state_dict()
    for key in g_sd:
        avg = sum(
            local_models[k].state_dict()[key].float() * (client_sizes[k] / total)
            for k in range(len(local_models))
        )
        g_sd[key].copy_(avg.to(g_sd[key].dtype))

class SubsetPoisoned(torch.utils.data.Dataset):

    def __init__(self, ds: PoisonedDataset, indices: np.ndarray):
        self.ds          = ds
        self.indices     = indices
        self.is_poisoned = ds.is_poisoned[indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, lbl, _ = self.ds[self.indices[i]]
        return img, lbl, i

def run_fedavg_experiment(ecfg: ExperimentConfig) -> dict:
    """FedAvg runner with persistent client-local BRIFi histories.

    Poison rate is local to contaminated clients; participation is sampled
    without replacement each round. This driver does not implement the paper's
    server-defense baselines or the complete advanced-attack experiments.
    """
    import math
    from brifi import BudgetedFilter
    dcfg, acfg, mcfg, tcfg = ecfg.dataset, ecfg.attack, ecfg.method, ecfg.train
    dev = tcfg.device
    if not 0 <= tcfg.fl_malicious_frac <= 1:
        raise ValueError('fl_malicious_frac must lie in [0, 1]')
    if not 0 < tcfg.fl_participation_rate <= 1:
        raise ValueError('fl_participation_rate must lie in (0, 1]')
    np.random.seed(tcfg.seed)
    random.seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tcfg.seed)
    detector = build_and_train_detector(ecfg)
    raw_tr = get_raw_dataset(dcfg, train=True)
    raw_te = get_raw_dataset(dcfg, train=False)
    clean_loader = DataLoader(CleanTestSet(raw_te, dcfg), 512, num_workers=tcfg.num_workers)
    labels = np.asarray([raw_tr[i][1] for i in range(len(raw_tr))])
    partitions = dirichlet_partition(labels, tcfg.fl_n_clients, tcfg.fl_alpha, tcfg.seed)
    nonempty = [k for k, indexes in enumerate(partitions) if len(indexes)]
    if not nonempty:
        raise ValueError('No training examples in the client partitions')
    rng = np.random.RandomState(tcfg.seed)
    contaminated = set(rng.choice(tcfg.fl_n_clients,
                                  int(tcfg.fl_n_clients * tcfg.fl_malicious_frac),
                                  replace=False).tolist())
    client_datasets, client_filters = [], []
    for k, indexes in enumerate(partitions):
        client_datasets.append(PoisonedDataset(
            Subset(raw_tr, indexes.tolist()), dcfg,
            acfg.poison_rate if k in contaminated else 0.,
            TriggerSampler(dcfg, families=[acfg.name]), aug=True, seed=tcfg.seed+k))
        if mcfg.name == 'baseline':
            client_filters.append(None)
        else:
            cls = BudgetedFilter if mcfg.name == 'brifi' else Prefilter
            client_filter = cls(detector, mcfg, dev, det_img_size=tcfg.detector_image_size)
            if mcfg.name == 'brifi' and tcfg.fl_warmup_rounds > 0:
                client_filter.set_active(False)
            client_filters.append(client_filter)
    # Scorer construction/loading must not change the classifier initialization
    # or downstream augmentation stream in BRIFi versus the undefended run.
    np.random.seed(tcfg.seed)
    random.seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tcfg.seed)
    global_model = build_classifier(dcfg).to(dev)
    selected_count = min(len(nonempty), max(1, int(tcfg.fl_n_clients * tcfg.fl_participation_rate)))
    history = []
    seen = poison_seen = dropped = poison_dropped = 0
    for rnd in range(tcfg.fl_rounds):
        chosen = rng.choice(nonempty, selected_count, replace=False).tolist()
        learning_rate = tcfg.lr * (1. + math.cos(math.pi * rnd / tcfg.fl_rounds)) / 2.
        local_models, local_sizes = [], []
        round_seen = round_poison = round_drop = round_drop_poison = 0
        for k in chosen:
            data, pf = client_datasets[k], client_filters[k]
            loader = DataLoader(data, tcfg.batch_size, shuffle=True, num_workers=tcfg.num_workers)
            if pf is not None:
                pf.set_epoch(rnd)
                if rnd >= tcfg.fl_warmup_rounds:
                    pf.set_active()
            local_model = copy.deepcopy(global_model)
            optimizer = optim.SGD(local_model.parameters(), lr=learning_rate,
                                  momentum=tcfg.momentum, weight_decay=tcfg.weight_decay)
            for _ in range(tcfg.fl_local_epochs):
                _, _, count_drop, count_poison_drop = train_epoch(
                    local_model, loader, optimizer, dev, pf, data.is_poisoned if pf else None)
                round_drop += count_drop
                round_drop_poison += count_poison_drop
                round_seen += len(data)
                round_poison += int(data.is_poisoned.sum())
            local_models.append(local_model)
            local_sizes.append(len(data))
        fedavg_aggregate(global_model, local_models, local_sizes)
        seen += round_seen
        poison_seen += round_poison
        dropped += round_drop
        poison_dropped += round_drop_poison
        accuracy = eval_acc(global_model, clean_loader, dev)
        item = {'round': rnd + 1, 'selected_clients': chosen,
                'learning_rate': learning_rate, 'clean_acc': accuracy,
                'seen': round_seen, 'poison_seen': round_poison,
                'dropped': round_drop, 'poison_dropped': round_drop_poison,
                'drop_rate': round_drop / max(round_seen, 1),
                'precision': round_drop_poison / max(round_drop, 1),
                'recall': round_drop_poison / max(round_poison, 1),
                'retained_poison_rate': (round_poison - round_drop_poison) / max(round_seen - round_drop, 1)}
        history.append(item)
        print(f"Round {rnd+1}/{tcfg.fl_rounds}: ACC={accuracy:.4f} DR={item['drop_rate']:.4f}")
    return {'exp_id': ecfg.exp_id, 'tag': ecfg.tag,
            'clean_acc': eval_acc(global_model, clean_loader, dev),
            'asr': eval_asr_suite(global_model, ecfg),
            'drop_rate': dropped / max(seen, 1),
            'precision': poison_dropped / max(dropped, 1),
            'recall': poison_dropped / max(poison_seen, 1),
            'round_history': history,
            'contaminated_clients': sorted(contaminated),
            'client_sizes': [len(d) for d in client_datasets],
            'config': {'dataset': dcfg.name, 'attack': acfg.name, 'method': mcfg.name,
                       'fl': True, 'rounds': tcfg.fl_rounds, 'poison_rate': acfg.poison_rate,
                       'fl_n_clients': tcfg.fl_n_clients, 'fl_alpha': tcfg.fl_alpha,
                       'fl_malicious_frac': tcfg.fl_malicious_frac,
                       'fl_participation_rate': tcfg.fl_participation_rate}}
