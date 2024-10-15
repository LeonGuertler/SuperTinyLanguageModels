"""from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Example string
test_string = " ".join(["monkey"]*4096)

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.half()

# Tokenize the input and move it to the same device
tokenized = tokenizer(test_string, return_tensors="pt").to(device)

# Get the model output
# with torch.no_grad():
output = model(**tokenized)

# Output logits or other relevant information
logits = output.logits
print(logits)"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.optim import AdamW

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Example string repeated to simulate a large input
test_string = " ".join(["monkey"] * 3500)  # Adjust the length of the string for testing

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device and enable FP16 precision
model.to(device)
model.half()

# Set batch size
batch_size = 1  # You can change this value to specify your batch size

# Tokenize the input and create a batch
tokenized = tokenizer([test_string] * batch_size, return_tensors="pt").to(device)

# Define an optimizer (AdamW in this case)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Get the model output (forward pass)
output = model(**tokenized, labels=tokenized['input_ids'])  # Adding labels for the causal LM loss

# Compute the loss (Causal LM usually includes the loss when you pass labels)
loss = output.loss

# Zero gradients
optimizer.zero_grad()

# Backward pass (compute gradients)
loss.backward()

# Update model parameters
optimizer.step()

# Output logits and loss
logits = output.logits
print(f"Logits: {logits}")
print(f"Loss: {loss.item()}")
