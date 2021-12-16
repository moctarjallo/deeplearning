```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

from tqdm import tqdm
```

```python
print("torch.__version__:", torch.__version__)
```

```python
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device}")
```

```python
class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.flatten(input_data)
        logits = self.dense_layers(x)
        predictions = self.softmax(logits)
        return predictions
    
model = FeedForwardNet().to(device)
```

```python
# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)
```

```python
# Download train & validation data
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
```

```python
# Instantiate data loader
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
```

```python
for i in range(EPOCHS):
    print(f"Epoch {i+1}")
    # Train on single epoch
    for inp, target in tqdm(train_dataloader):
        inp, target = inp.to(device), target.to(device)

        # calculate loss
        prediction = model(inp)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")

```

```python
# save model
torch.save(model.state_dict(), "model.pth")
```

```python
# load model
model = FeedForwardNet()
state_dict = torch.load("model.pth")
model.load_state_dict(state_dict)
```

```python
# get validation data
validation_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```

```python
# retrieve one image from validation data
inp, target = validation_data[0][0], validation_data[0][1]
```

```python
plt.imshow(inp[0, :, :])
print(target)
```

```python
model.eval() # inference mode
with torch.no_grad():
    predictions = model(inp)
    predicted = predictions[0].argmax(0)
print(predicted.item())
```
