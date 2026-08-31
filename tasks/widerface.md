Develop and submit a WIDER FACE detector. Maximize validation-set AP while
producing an ONNX submission that meets the configured STM32N6 runtime target,
with preprocessing and postprocessing hooks. Use the training service for
substantial training and aggregate evaluation results to guide improvements.

## Dataset available to you

The sandbox and training image contain exactly the 12,880 images from the
public WIDER FACE training split. Use the benchmark API rather than downloading
the dataset:

```python
from odbench import dataset_manifest, load_dataset

print(dataset_manifest())
dataset = load_dataset()  # equivalent to split="train"
```

`load_dataset` returns a PyTorch `Dataset` named `WiderFaceDetectionTrain`:

- `dataset[index]` is `(image, target)`, where `image` is an RGB `PIL.Image`.
- `target["boxes"]` is a float32 tensor in original-image pixel `xyxy` format.
- `target["labels"]` contains the single foreground class ID 1 (`face`). Zero
  is reserved for background, matching torchvision detector training.
- `target` also provides `image_id`, `area`, `iscrowd`, and `ignored_boxes`.
  Invalid source annotations are excluded from training boxes and exposed in
  `ignored_boxes`.
- `dataset.classes == ("face",)` and `dataset.class_ids == (1,)`.
- A joint `transforms(image, target)` callable may be supplied. Separate
  `transform=` and `target_transform=` callables are also accepted.

WIDER FACE images have variable resolutions and include very small faces and
highly crowded scenes. Inspect image aspect ratios and face sizes after the
proposed resize. A worker that converts each original image to a float32 tensor
before resizing can put tens of megabytes per sample into the multiprocessing
queue. Keep the product of DataLoader workers, prefetched batches, and batch
size within the configured shared memory. Prefer a small worker count and
`prefetch_factor=1`, or resize images and boxes before worker results are
queued. Smoke-test several crowded batches before a long job.

There is no validation split in the agent or training image. The held-out 3,226
validation images and annotations can only be accessed through aggregate epoch
evaluation, `evaluate`, or final `submit`.

## Offline pretrained initializers

The sandbox and training image contain verified offline torchvision
initializers. Inspect the exact catalog with `list_pretrained()`; loading never
uses the network:

```python
from odbench import list_pretrained, load_backbone, load_detector

print(list_pretrained())
backbone = load_backbone("torchvision/mobilenet_v3_small_imagenet1k_v1")
features = backbone(images)  # tuple at reductions 4, 8, 16, and 32

detector = load_detector(
    "torchvision/ssdlite320_mobilenet_v3_large_coco_v1",
    num_classes=2,  # background plus face
)
```

The effective run configuration lists available models, feature channels,
feature reductions, normalization, and reference compute. For an otherwise
unchanged fully convolutional graph, input-resolution compute scales roughly
with pixel count. This is only a sizing hint: operator support, activation
traffic, graph boundaries, and software fallbacks determine actual STM32N6
latency. Export, quantize, and call `analyze_for_hw` before spending a long
training job on a new graph or input size.

The bundled SSDLite detector owns a fixed 320 x 320 transform, six feature
maps, default boxes, box decoding, filtering, and NMS. Its full training API
expects a list of float `CHW` tensors in `[0, 1]` and matching target mappings.
For deployment, exporting the full detector usually places resizing, decoding,
filtering, and NMS inside ONNX. A more target-friendly split exports the
normalization, backbone, and raw head, then reproduces default-box decoding and
NMS in `postprocess.py`. Verify raw-head decoding against `detector.eval()`
before training so anchor order, normalization, and class offsets agree.

## Detection submission contract

The submission directory uses `submission.json`, a self-contained ONNX model,
`preprocess.py`, and `postprocess.py`.

`preprocess(image)` receives one original RGB `PIL.Image`. It must return a
NumPy array for a single-input model or a dictionary mapping ONNX input names
to arrays. Include the batch dimension; a typical detector input is contiguous
float32 `NCHW` with shape `[1, 3, H, W]`.

The evaluator runs ONNX and passes a dictionary of named NumPy outputs to
`postprocess`. Declare `postprocess(outputs, image_size)`, where `image_size`
is the original `(width, height)`, and return exactly:

```python
{
    "boxes": boxes,      # NumPy [N, 4], original-image xyxy pixels
    "scores": scores,    # NumPy [N], finite probabilities in [0, 1]
    "classes": classes,  # integer NumPy [N], all values equal to 1
}
```

The arrays must have shapes `[N, 4]`, `[N]`, and `[N]`, with at most 2,000
rows. A background-inclusive head uses class ID 1 for faces; if a one-channel
foreground head uses index 0 internally, add one. Scores are probabilities,
not logits. Return empty arrays with shapes `(0, 4)`, `(0,)`, and `(0,)` when
there are no detections, including an integer dtype for `classes`.

Boxes must be continuous `xyxy` coordinates in the original image. They must
satisfy `0 <= x1 < x2 <= image.width` and
`0 <= y1 < y2 <= image.height`. Clip boxes and remove zero-area rows after
mapping them back. If preprocessing resizes or letterboxes, undo both padding
and the effective x/y scales after integer rounding. Prefer the two-argument
postprocess form, and test a synthetic resize/letterbox box round trip before
training.

### Required local submission smoke test

Before calling `evaluate`, run the submitted preprocess, ONNX model, and
postprocess end to end on representative public training images, including
crowded scenes. At minimum check detection counts and score ranges, face recall
at IoU 0.50 over several dozen images, decoded float-versus-INT8 agreement, and
the resize/letterbox round-trip error. Do not spend a hidden evaluation on a
candidate with empty detections, near-zero training recall, invalid class IDs,
or materially different float and quantized decoding.

Evaluation reports one `AP` value at IoU 0.50 for the single face class across
all valid validation annotations. It does not calculate the original WIDER FACE
easy, medium, and hard subsets. Detections matching source annotations marked
invalid do not affect the result.

## Artifact constraint

The self-contained ONNX file must be no larger than 16 MiB (16,777,216 bytes).
The serialized file-size limit is authoritative. Set `external_data=False`
when exporting.
