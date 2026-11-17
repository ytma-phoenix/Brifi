# BRIFi

**BRIFi: Client-Calibrated Online Poisoned-Sample Mining for Backdoor-Resilient Federated Learning**  
Yitong Ma and Gang Yan · WSDM 2027  
[Paper](https://doi.org/10.1145/3862322.3863472) · [Repository](https://github.com/ytma-phoenix/Brifi)

BRIFi filters suspicious samples inside each client's local training pipeline,
before they contribute to a model update. A separately pretrained, frozen CNN
scores each sample. A client-local rolling history of **retained samples only**
provides the median/MAD calibration; a one-sided score threshold selects eligible
samples and a hard per-batch rejection budget limits data removal. Before enough
history is available, the same budget is applied to the highest raw detector
scores. The Gaussian quantile parameterizes the cutoff; this is not a fitted
likelihood-ratio test and does not assert Gaussian score distributions.

## What is included

- `brifi.py`: the paper's core keep-mask algorithm, including shared detector
  preprocessing, raw-score initialization, retained-only calibration, and cap.
- `run.py`: command-line entry points for auxiliary detector training and
  downstream BRIFi/FedAvg experiments.
- `fl_utils.py`: Dirichlet partitions, sampled client participation, persistent
  per-client histories, local training, and sample-weighted FedAvg.
- `models.py`, `datasets.py`, `attacks.py`: detector/classifier models, data
  wrappers, and six attack-family instantiations.
- `trainer.py`: training/evaluation routines and retained exploratory variants.
- `fl_attacks.py`: legacy advanced-attack helper prototypes, **not integrated**
  into the public experiment runner.
- `tests/`: synthetic functional checks; they do not train a model or download data.

This release supplies the core method and a runnable experiment driver. It does
**not** include the original auxiliary corpus/splits, pretrained detector weights,
per-seed experiment outputs, server-defense implementations, or the complete
advanced/adaptive-attack experiment scripts. It therefore does not establish
reproduction of all paper tables. The `brifi_freq`, `brifi_warp`, and `brifi_u`
configurations retained in the source are exploratory variants, not the method
used by the public `brifi` command. See [RELEASE_NOTES.md](RELEASE_NOTES.md) for
implementation corrections and the precise validation scope.

## Installation

Use Python 3.10–3.12 and a matching PyTorch/torchvision pair. A CPU-compatible
validation environment used Python 3.12, PyTorch 2.6.0, and torchvision 0.21.0.
For CUDA, install the matching build following the
[official PyTorch instructions](https://pytorch.org/get-started/locally/).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --help
python run.py smoke
python -m unittest discover -s tests -v
```

`scikit-learn` is optional and enables AUROC diagnostics in the centralized
helper. The rejection filter does not depend on it.

## Prepare a task-disjoint scorer

**No pretrained checkpoint is bundled.** A BRIFi run requires an existing
checkpoint and never silently trains the detector on the downstream task.
Prepare independent auxiliary images and split the **base images** into
train/validation/test sets before generating clean/triggered variants. None of
MNIST, CIFAR-10, CIFAR-100, or Tiny-ImageNet may enter an auxiliary corpus intended
to reproduce the paper's cross-task protocol.

The paper uses GTSRB, EuroSAT-RGB, and procedural images. Their original images,
procedural generation recipes, exact splits, and tuned weights are not present in
this source package. Supply your own properly sourced auxiliary training split;
a newly trained checkpoint is a new run, not the paper's original checkpoint.
Do not use auxiliary validation/test images in `--aux-root`.

```text
auxiliary/
  train/        # images; nested directories are accepted, class labels ignored
  validation/   # reserve for independent tuning/evaluation
  test/         # reserve for independent evaluation
```

```bash
python run.py train-detector \
  --aux-root auxiliary/train \
  --output checkpoints/detector.pt \
  --samples 10000 --epochs 10 --image-size 32 --seed 1 --device cuda
```

The command pairs each selected base image with a clean and a triggered example.
It uniformly samples patch, blended, additive-noise, and spatial-warp surrogate
families. Images are converted to RGB float tensors in `[0,1]` and use the same
bilinear canonicalization helper as downstream BRIFi inputs. Validation and
hyperparameter selection are not automatically performed by this command.
The scorer parameters, BatchNorm statistics, and dropout state are frozen during
FL. Keep the same checkpoint for comparisons that require a fixed scorer.

## Downstream data and runs

CIFAR-10, CIFAR-100, and MNIST are obtained through torchvision on an actual run.
Tiny-ImageNet must already be extracted under
`data/tiny-imagenet-200/`, with `wnids.txt`, `train/`, and the original `val/`
annotation/image layout. Dataset licenses remain the responsibility of their
providers and users; no dataset is redistributed here.

Inspect configuration without importing torch, downloading data, or training:

```bash
python run.py run --dataset cifar10 --attack badnets --dry-run
```

BRIFi:

```bash
python run.py run \
  --dataset cifar10 --data-root data --attack badnets --method brifi \
  --detector-checkpoint checkpoints/detector.pt \
  --clients 100 --participation-rate 0.3 --contaminated-client-ratio 0.2 \
  --poison-rate 0.1 --rounds 20 --local-epochs 1 \
  --batch-size 128 --lr 0.01 --seed 1 --device cuda \
  --output results/cifar10_badnets_brifi_seed1.json
```

Undefended FedAvg, with the same partition/participation/poisoning parameters:

```bash
python run.py run \
  --dataset cifar10 --data-root data --attack badnets --method baseline \
  --seed 1 --device cuda --output results/cifar10_badnets_fedavg_seed1.json
```

Use `--device cpu` for CPU execution. Change `--seed` explicitly for each seed
(the manuscript uses 1, 12, and 123). The default FL driver uses 100 clients,
30% participation, Dirichlet alpha 0.5, 20 rounds, one local epoch, 20%
contaminated clients, and a **10% poison rate within each contaminated client**.
A contaminated-client ratio of zero means no contaminated clients. Changing this
ratio does not renormalize the local poison rate.

The `--attack` choices are `badnets`, `blended`, `dba`, `pattern`, `noise`, and
`wanet`. These are the supplied trigger-family instantiations; in particular,
the `dba` sampler creates multiple local patches and does not implement the full
original distributed attacker assignment. They must not be described as exact
reimplementations of every original attack paper. Evaluation reports each of the
six named families with deterministic evaluation random seeds, excludes samples
already in the target class, and leaves their original labels intact.

Each JSON result includes the run configuration, clean accuracy, ASR by family,
round-selected clients, and filtering statistics. Drop rate/recall denominators
count actual sample exposures, including repeated local epochs. The retained
poison fraction describes the current pass; rejection does not permanently
remove an example from the dataset. Existing result files require explicit
`--overwrite` before replacement.

`--mode centralized` is available for local diagnostics, but those results are
not federated results. The release does not implement the server-defense rows,
complete LOFO sweep, component-ablation sweep, or advanced/adaptive experiments
from the paper.

## Core API

```python
from brifi import BudgetedFilter
from config import MethodConfig

# detector is a separately pretrained torch.nn.Module returning one logit/image.
filter_ = BudgetedFilter(detector, MethodConfig.brifi(), device="cuda", det_img_size=32)
keep = filter_.keep(images)  # CPU Boolean mask; images are NCHW floats in [0,1]
retained_images = images[keep]
```

Create one `BudgetedFilter` per client and retain that object across rounds.
Do not share calibration histories across clients. With batch size 128 and the
default rejection cap 0.2, at most 25 samples can be rejected; cold-start keeps
103 samples. `lrt_alpha` is retained as an internal legacy field name for the
nominal threshold parameter, not a statistical LRT claim.

## Citation

```bibtex
@inproceedings{ma2027brifi,
  author = {Ma, Yitong and Yan, Gang},
  title = {{BRIFi}: Client-Calibrated Online Poisoned-Sample Mining for Backdoor-Resilient Federated Learning},
  booktitle = {Proceedings of the Twentieth ACM International Conference on Web Search and Data Mining},
  year = {2027},
  doi = {10.1145/3862322.3863472}
}
```

## Acknowledgments

This work was supported by the Excellent Young Scientists Fund (Overseas) of the
National Natural Science Foundation of China (awarded in 2024) and the Jilin
Provincial Department of Science and Technology (Grant No. 20260102309JC).

## License and contact

The repository code is distributed under the [MIT License](LICENSE). Third-party
dependencies and datasets retain their own licenses.

Corresponding author: Gang Yan, [gyan8@jlu.edu.cn](mailto:gyan8@jlu.edu.cn).
