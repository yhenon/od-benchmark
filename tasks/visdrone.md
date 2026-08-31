Develop and submit a VisDrone2019-DET object detector. Maximize validation-set
AP while producing an ONNX submission that meets the configured STM32N6
runtime target, with preprocessing and postprocessing hooks. Use the training
service for substantial training and use
aggregate evaluation results to guide improvements.

## Dataset available to you

The sandbox and training image contain exactly the 6,471 images from the public
VisDrone2019-DET training split. Use the benchmark API rather than downloading
the dataset:

```python
from odbench import dataset_manifest, load_dataset

print(dataset_manifest())
dataset = load_dataset()  # equivalent to split="train"
```

`load_dataset` returns a PyTorch `Dataset` named `VisDroneDetectionTrain`:

- `dataset[index]` is `(image, target)`, where `image` is an RGB `PIL.Image`.
- `target["boxes"]` is a float32 tensor in original-image pixel `xyxy` format.
- `target["labels"]` contains VisDrone class IDs 1 through 10. Zero is reserved
  for background/ignored regions, which matches torchvision detector training.
- `target` also provides `image_id`, `area`, `iscrowd`, and `ignored_boxes`.
- `dataset.classes` lists the ten evaluated classes in class-ID order:
  pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle,
  bus, and motor.
- A joint `transforms(image, target)` callable may be supplied. Separate
  `transform=` and `target_transform=` callables are also accepted.

VisDrone source images have variable, often large resolutions. A worker that
converts each original image to a float32 tensor before resizing can place tens
of megabytes per sample into the multiprocessing queue. Keep the product of
DataLoader workers, prefetched batches, and batch size comfortably within the
`training_hardware.shared_memory` value in the effective run configuration.
Prefer a small worker count and `prefetch_factor=1`, or resize the image and
boxes to the training resolution before worker results are queued. Smoke-test a
few batches before launching a long job.

There is no validation split in the agent or training image. The held-out 548
validation images and annotations can only be accessed through aggregate epoch
evaluation, `evaluate`, or final `submit`.

## Offline pretrained initializers

The sandbox and training image contain the same verified, offline torchvision
initializers. Inspect the exact catalog with `list_pretrained()`; loading never
uses the network:

```python
from odbench import list_pretrained, load_backbone, load_detector

print(list_pretrained())
backbone = load_backbone("torchvision/mobilenet_v3_small_imagenet1k_v1")
features = backbone(images)  # tuple at reductions 4, 8, 16, and 32

# Alternatively retain compatible COCO SSDLite parameters while replacing its
# classification tensors for background plus the ten VisDrone classes.
detector = load_detector(
    "torchvision/ssdlite320_mobilenet_v3_large_coco_v1",
    num_classes=11,
)
```

The effective run configuration lists every available model, feature channel
count, feature reduction, normalization, and upstream reference compute. The
current catalog's reference costs are:

| Initializer | Reference input | Reference GFLOPS |
| --- | ---: | ---: |
| MobileNetV3 Small backbone | 224 x 224 | 0.057 |
| MobileNetV2 backbone | 224 x 224 | 0.301 |
| ShuffleNetV2 x0.5 backbone | 224 x 224 | 0.040 |
| SSDLite MobileNetV3 Large detector | 320 x 320 | 0.583 |

For an otherwise unchanged fully convolutional graph, a first-order estimate at
a different resolution is `new_cost ~= reference_cost * new_pixels /
reference_pixels`. This is only an architecture-sizing hint: operator support,
activation traffic, graph boundaries, and software fallbacks determine actual
STM32N6 latency. Before spending a long training job on a new graph or input
size, export and quantize an initializer or untrained instance and call
`analyze_for_hw`. Weights learned later do not normally change that graph's
runtime or memory footprint.

Also inspect image aspect ratios and the distribution of box widths and heights
after the proposed resize using the public training split. Optimize useful image
content, not merely tensor dimensions: padding consumes activation memory and
compute without preserving more object detail. A static rectangular input can
be preferable to a square letterbox when the dataset's aspect ratios support it,
provided its quantized graph passes target analysis and its geometry is undone
correctly in postprocessing.

Prefer the offline initializers over scratch training when they fit the artifact
and target-hardware constraints. Arbitrary timm weights remain unavailable
offline, so do not request `pretrained=True` from timm.

### Bundled SSDLite MobileNetV3 Large notes

`load_detector(..., num_classes=11)` returns torchvision's
`ssdlite320_mobilenet_v3_large` with its COCO-trained backbone, extra feature
blocks, and box-regression tensors retained. Classification tensors whose shape
depends on the class count are newly initialized. The full training API expects
a list of float `CHW` image tensors in `[0, 1]` and a matching list of target
dictionaries:

```python
detector.train()
losses = detector(
    [image_tensor],
    [{"boxes": boxes_xyxy, "labels": labels_1_through_10}],
)
loss = sum(losses.values())
```

The torchvision detector owns a fixed 320 x 320 transform, six feature maps,
the default-box generator, box coder, score filtering, and NMS. Its internal
normalization maps `[0, 1]` to `[-1, 1]` using mean and standard deviation 0.5
per channel. Inspect the installed object rather than guessing these details:

```python
print(detector.transform.fixed_size)
print(detector.transform.image_mean, detector.transform.image_std)
print(detector.anchor_generator.aspect_ratios)
batch = torch.stack(equal_sized_image_tensors)  # float values in [0, 1]
normalized = (batch - 0.5) / 0.5
features = list(detector.backbone(normalized).values())
raw = detector.head(features)
print([value.shape for value in features])
print({name: value.shape for name, value in raw.items()})
```

For deployment, exporting the full torchvision detector usually puts resizing,
box decoding, filtering, and NMS into the ONNX graph. A more target-friendly
split is to export a wrapper containing the normalization, `detector.backbone`,
and `detector.head`, returning raw `bbox_regression` and `cls_logits`; reproduce
the same default boxes and box-coder equations in `postprocess.py`, followed by
thresholding and NMS. Keep those operations synchronized if the input size or
anchor generator changes. Before training, compare the wrapper plus
postprocessing with `detector.eval()` on the same images, and verify decoded box
coordinates and class IDs. This catches anchor-order, normalization, and resize
geometry errors before they consume training or hidden-evaluation budget.

## Detection submission contract

The submission directory uses the same `submission.json`, self-contained ONNX,
`preprocess.py`, and `postprocess.py` layout as other tasks.

`preprocess(image)` receives one original RGB `PIL.Image`. It must return either
a NumPy array for a single-input ONNX model or a dictionary mapping ONNX input
names to NumPy arrays. Include the batch dimension expected by the model; a
typical detector input is contiguous float32 `NCHW` with shape `[1, 3, H, W]`.

The evaluator runs ONNX and constructs `outputs` as a dictionary mapping each
declared ONNX output name to its NumPy array. ONNX output order is irrelevant:
read `outputs["name"]` using the exact names supplied to `torch.onnx.export`.
For example, a graph exported with `output_names=["bbox_regression",
"cls_logits"]` is received under those two keys.

Declare postprocessing as `postprocess(outputs, image_size)`, where
`image_size` is the original `(width, height)` before preprocessing. Return
exactly:

```python
{
    "boxes": boxes,      # NumPy [N, 4], xyxy pixels in the original image
    "scores": scores,    # NumPy [N], finite values in [0, 1]
    "classes": classes,  # integer NumPy [N], VisDrone IDs 1..10
}
```

The three arrays describe the same rows and must have shapes `[N, 4]`, `[N]`,
and `[N]`. Return at most 500 rows. Scores are probabilities after sigmoid or
softmax, not logits. Class IDs are one-based VisDrone IDs: if a ten-channel head
uses channel indices 0 through 9, add one; a background-inclusive SSDLite head
already uses indices 1 through 10 for foreground. Detection row order is not
semantically significant because the evaluator sorts by score.

Boxes are continuous `xyxy` coordinates in pixels of the **original image**,
not normalized coordinates, feature-grid coordinates, or model-input pixels.
They must satisfy `0 <= x1 < x2 <= image.width` and
`0 <= y1 < y2 <= image.height`; `x2 == image.width` and
`y2 == image.height` are valid. Clip boxes and remove zero-area rows after
mapping them back. For no detections, return arrays with shapes `(0, 4)`, `(0,)`,
and `(0,)`, including an integer dtype for `classes`.

If preprocessing letterboxes an image, postprocessing must invert both the
padding and the *effective* x/y scales after integer rounding. For example:

```python
import numpy as np
from PIL import Image

MODEL_W, MODEL_H = 320, 192

def preprocess(image):
    original_w, original_h = image.size
    nominal = min(MODEL_W / original_w, MODEL_H / original_h)
    resized_w = max(1, round(original_w * nominal))
    resized_h = max(1, round(original_h * nominal))
    pad_x = (MODEL_W - resized_w) // 2
    pad_y = (MODEL_H - resized_h) // 2
    resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (MODEL_W, MODEL_H), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(array)  # apply model-specific normalization too

def boxes_to_original(boxes_in_model_pixels, image_size):
    original_w, original_h = image_size
    nominal = min(MODEL_W / original_w, MODEL_H / original_h)
    resized_w = max(1, round(original_w * nominal))
    resized_h = max(1, round(original_h * nominal))
    pad_x = (MODEL_W - resized_w) // 2
    pad_y = (MODEL_H - resized_h) // 2
    scale_x = resized_w / original_w
    scale_y = resized_h / original_h
    boxes = np.asarray(boxes_in_model_pixels, dtype=np.float32).copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale_x
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale_y
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h)
    return boxes
```

One-argument `postprocess(outputs)` hooks are accepted only when the ONNX output
is already expressed in original-image coordinates or original geometry is
otherwise unnecessary. Prefer the two-argument form for resized detectors.
Before training, make a synthetic box in an original image, apply the exact
training/preprocess geometry, invert it with the submitted postprocess code, and
assert the round trip within a small tolerance.

### Required local submission smoke test

Before calling `evaluate`, run the exact submitted `preprocess.py`, ONNX model,
and `postprocess.py` end to end on a representative sample of public training
images without augmentation. Compare the final original-image detections with
`target["boxes"]` and `target["labels"]`; at minimum report:

- detection counts and score ranges,
- class-aware recall at IoU 0.50 for several dozen images,
- decoded float-versus-INT8 agreement, and
- the synthetic resize/letterbox round-trip error.

Do not spend a hidden evaluation on a candidate that produces empty detections,
near-zero class-aware training recall, invalid class offsets, or materially
different decoded float and quantized predictions. A successful ONNX Runtime or
quantizer smoke test validates execution and shapes, not detection accuracy.

Evaluation follows the official VisDrone DET convention: AP is averaged over
IoU thresholds 0.50 through 0.95 for the ten classes with at most 500 detections
per image. AP50, AP75, and AR at 1, 10, 100, and 500 detections are also
reported. Detections substantially covered by annotated ignored regions do not
affect the result.

## Artifact constraint

The self-contained ONNX file must be no larger than 16 MiB (16,777,216 bytes).
The serialized file-size limit is authoritative. Set `external_data=False` when
exporting.
