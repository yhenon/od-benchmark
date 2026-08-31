"""Small two-boundary CUDA lifecycle used to verify the SSH trainer."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
from odbench import load_dataset
from odbench_train import epoch_end
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor


if not torch.cuda.is_available():
    raise RuntimeError("the remote GPU training smoke test requires CUDA")

torch.manual_seed(0)
device = torch.device("cuda")
dataset = load_dataset(transform=ToTensor())
loader = DataLoader(
    Subset(dataset, range(512)),
    batch_size=128,
    shuffle=True,
    pin_memory=True,
)
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((4, 4)),
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 10),
).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
loss_function = nn.CrossEntropyLoss()
output = Path("/job/output")
input_root = Path("/job/input")
script_directory = Path(__file__).resolve().parent.relative_to(input_root)
preprocess_path = (script_directory / "preprocess.py").as_posix()
postprocess_path = (script_directory / "postprocess.py").as_posix()
start_epoch = 0
resume_checkpoint = os.environ.get("ODBENCH_RESUME_CHECKPOINT")
if resume_checkpoint:
    saved = torch.load(resume_checkpoint, map_location=device)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    start_epoch = int(saved["epoch"]) + 1

for epoch in range(start_epoch, 2):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = loss_function(model(images), targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())

    checkpoint = output / f"checkpoint-{epoch}.pt"
    artifact = output / f"model-{epoch}.onnx"
    torch.save(
        {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
        checkpoint,
    )
    export_model = copy.deepcopy(model).cpu().eval()
    torch.onnx.export(
        export_model,
        (torch.zeros(1, 3, 32, 32),),
        artifact,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
        external_data=False,
    )
    decision = epoch_end(
        epoch=epoch,
        artifact=artifact.name,
        checkpoint=checkpoint.name,
        preprocess=preprocess_path,
        postprocess=postprocess_path,
        metrics={
            "device": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "train_loss": total_loss / len(loader),
        },
    )
    if decision.stop:
        break
