from pathlib import Path

import torch
from odbench import load_dataset
from odbench_train import epoch_end
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor


torch.manual_seed(0)
dataset = load_dataset(transform=ToTensor())
loader = DataLoader(Subset(dataset, range(256)), batch_size=64, shuffle=True)
model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()
output = Path("/job/output")

for epoch in range(2):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
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
    model.eval()
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, 32, 32),),
        artifact,
        input_names=["input"],
        output_names=["logits"],
        dynamo=True,
        external_data=False,
    )
    decision = epoch_end(
        epoch=epoch,
        artifact=artifact.name,
        checkpoint=checkpoint.name,
        preprocess="preprocess.py",
        postprocess="postprocess.py",
        metrics={"train_loss": total_loss / len(loader)},
    )
    if decision.stop:
        break
