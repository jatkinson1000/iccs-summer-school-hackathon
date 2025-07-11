import torch
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(
    description="Convert PyTorch .pt model to TorchScript .ts model."
)
parser.add_argument(
    "--filename",
    type=str,
    default="L96_emulator",
    help="Base filename for the model (default: L96_emulator)"
)
args = parser.parse_args()
filename = args.filename

# Load the PyTorch model
model = torch.jit.load(f"{filename}.pt")

# Convert the model to TorchScript and save to file
scripted_model = torch.jit.script(model)
scripted_model.save(f"../pdaf-code/{filename}.ts")
