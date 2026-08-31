Develop and submit a CIFAR-10 image classifier. Maximize hidden test-set top-1 accuracy while producing an ONNX submission that meets the configured STM32N6 runtime target, with preprocessing and postprocessing hooks. Use the available training service for substantial training and use evaluation results to guide improvements.

## Dataset available to you

The sandbox and training image contain exactly the 50,000 examples from the public CIFAR-10 training split. Use the benchmark-provided API rather than downloading the dataset:

```python
from odbench import dataset_manifest, load_dataset

print(dataset_manifest())
dataset = load_dataset()  # equivalent to split="train"
```

`load_dataset` returns a PyTorch `Dataset` named `CIFAR10Train`:

- `len(dataset) == 50_000`.
- Without transforms, `dataset[index]` is `(image, target)`, where `image` is a 32×32 RGB `PIL.Image` and `target` is an integer from 0 through 9.
- It accepts the familiar `transform=` and `target_transform=` keyword arguments.
- `dataset.data` is a NumPy array with shape `(50000, 32, 32, 3)` containing uint8 RGB pixels.
- `dataset.targets` is the list of integer targets.
- `dataset.classes` is `("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")` in label order.

For example:

```python
from odbench import load_dataset
from torchvision.transforms import ToTensor

training_data = load_dataset(transform=ToTensor())
```

There is no provided validation split and no test split in either the agent or training image. `load_dataset(split="val")` and `load_dataset(split="test")` fail, and the original CIFAR-10 test batch is deliberately absent. The held-out evaluation images and labels cannot be read through workspace or training commands; only aggregate top-1 accuracy is returned by epoch evaluation, `evaluate`, or final `submit`.

Create any development/validation split you need from the 50,000 training examples, preferably with a fixed seed and without training on the validation indices while selecting the model. Do not attempt to download CIFAR-10 through torchvision because the environment is offline.

## Artifact constraint

The self-contained ONNX file must be no larger than 16 MiB (16,777,216 bytes). This corresponds to at most roughly four million float32 parameter values before accounting for graph metadata and other constants; quantized weights have a different size relationship. The serialized file-size limit is authoritative. Set `external_data=False` when exporting.
