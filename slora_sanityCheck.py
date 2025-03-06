import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler
from peft import LoraConfig, get_peft_model, LoraModel, TaskType
from torch.optim import AdamW
from accelerate import Accelerator
from options import Options
from utils import get_model_and_tokenizer


# Custom SLoRAConfig for Continual Learning
class SLoRAConfig(LoraConfig):
    def __init__(self, *args, past_task_loras=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.past_task_loras = past_task_loras or {}  # Store LoRA {task_name: {layer_name: (A, B, scale)}}

    def update_past_loras(self, task_name, layer_name, lora_A, lora_B, scale):
        """ Updates past LoRA parameters and scaling factors for continual learning at a per-layer level. """
        if task_name not in self.past_task_loras:
            self.past_task_loras[task_name] = {}
        self.past_task_loras[task_name][layer_name] = (lora_A, lora_B, scale)


# SLoRAModel: Extends LoraModel for Continual Learning
class SLoRAModel(LoraModel):
    def __init__(self, model, peft_config):
        super().__init__(model, peft_config)

        # Attach LoRA layers for the current task
        self.model = get_peft_model(model, peft_config)

        # Dictionary to store past task LoRA parameters per layer
        self.past_loras = {}  
        for task_name, task_layers in peft_config.past_task_loras.items():
            for layer_name, (lora_A, lora_B, scale) in task_layers.items():
                if layer_name not in self.past_loras:
                    self.past_loras[layer_name] = {}
                self.past_loras[layer_name][task_name] = (
                    lora_A.clone().detach(),
                    lora_B.clone().detach(),
                    nn.Parameter(torch.tensor(scale, dtype=torch.float32, requires_grad=True))  # Trainable scale
                )

        # Trainable scaling factors for past tasks
        self.scaling_factors = nn.ModuleDict({
            layer_name: nn.ParameterDict({
                task_name: scale
                for task_name, (_, _, scale) in self.past_loras[layer_name].items()
            }) for layer_name in self.past_loras.keys()
        })

        # Trainable scaling factors for the current task per layer
        self.current_scales = nn.ParameterDict({
            layer_name: nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))
            for layer_name in self.model.state_dict().keys() if "lora_A" in layer_name
        })

        # Freeze past LoRA parameters
        for layer_name, task_loras in self.past_loras.items():
            for task_name, (lora_A, lora_B, _) in task_loras.items():
                lora_A.requires_grad = False
                lora_B.requires_grad = False

    def forward(self, x):
        for layer_name, layer in self.model.named_modules():
            if hasattr(layer, "forward"):
                # Skip base model computation if LoRA is present
                if hasattr(layer, "lora_A") or hasattr(layer, "lora_B"):
                    base_output = torch.zeros_like(x)  # No base computation

                    # Apply current task LoRA
                    delta_W_current = torch.matmul(layer.lora_B, layer.lora_A)
                    current_task_output = self.current_scales[layer_name] * torch.matmul(x, delta_W_current.T)
                else:
                    # Compute base model output normally
                    base_output = layer(x)
                    current_task_output = torch.zeros_like(base_output)

                # Apply past task LoRA layers
                past_task_output = torch.zeros_like(base_output)
                if layer_name in self.past_loras:
                    for task_name, (lora_A, lora_B, scale) in self.past_loras[layer_name].items():
                        delta_W = torch.matmul(lora_B, lora_A)  # Compute LoRA update
                        past_task_output += self.scaling_factors[layer_name][task_name] * torch.matmul(x, delta_W.T)

                # Sum all contributions
                x = base_output + past_task_output + current_task_output

        return x


# Sanity Check Function
def sanity_check():
    """ Runs a sanity check to verify SLoRAModel initialization, trainable layers, forward and backward pass. """
    
    # Auto-detect GPU and use if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    args = Options().parse()
    model, tokenizer = get_model_and_tokenizer(args)
    model.to(device)  # Move model to GPU if available
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Define SLoRA Configuration
    slora_config = SLoRAConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        bias="none",
        r=4,  # LoRA rank for testing
        lora_alpha=16,
        lora_dropout=0.1,
        fan_in_fan_out=False,
        past_task_loras={}  # No past tasks for the sanity check
    )

    # Apply SLoRA modifications to the model
    model = SLoRAModel(model, slora_config)

    # Print model layers and trainable parameters
    print("\nTrainable Layers in SLoRAModel:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")

    # Create a dummy input tensor
    batch_size, seq_length = 1, 10  # Simulating a single input sequence
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Forward pass with mixed precision if on GPU
    print("\nPerforming Forward Pass...")
    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    print(f"Logits Shape: {logits.shape}")

    # Compute a dummy loss
    labels = torch.randint(0, logits.shape[-1], (batch_size, seq_length)).to(device)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
    print("\nLoss Computed:", loss.item())

    # Backward pass
    print("\nPerforming Backward Pass...")
    loss.backward()
    print("Backward Pass Completed.")


if __name__ == "__main__":
    sanity_check()

