import torch
import os
import json

def normalize_rows(matrix):
    """Normalize each row of the matrix to unit norm."""
    print(f"Matrix shape before normalization: {matrix.shape}")
    normalized_matrix = matrix / (torch.norm(matrix, dim=1, keepdim=True) + 1e-8)
    print(f"Matrix shape after normalization: {normalized_matrix.shape}")
    return normalized_matrix

def load_lora_matrices(experiment_dir):
    """Load stored LoRA A and B matrices from the experiment directory."""
    a_matrices_path = os.path.join(experiment_dir, "olora_adapters.pt")
    b_matrices_path = os.path.join(experiment_dir, "olora_B_adapters.pt")

    if not os.path.exists(a_matrices_path) or not os.path.exists(b_matrices_path):
        raise FileNotFoundError("LoRA A and B matrices files not found!")

    A_matrices = torch.load(a_matrices_path, map_location="cpu")
    B_matrices = torch.load(b_matrices_path, map_location="cpu")

    return A_matrices, B_matrices

def compute_orthogonality_metrics(A_matrices, B_matrices):
    """
    Compute orthogonality metrics for LoRA matrices across different tasks.

    A_matrices: Dictionary where each key corresponds to a different task/layer
    B_matrices: Dictionary where each key corresponds to a different task/layer
    """
    task_names = list(A_matrices.keys())
    num_tasks = len(task_names)

    results = {
        "O-LoRA": {},
        "O-LoRA (Normalized)": {},
        "Cosine Similarity": {},
        "Raw Dot Product": {}
    }

    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):  # Avoid redundant comparisons
            task_i, task_j = task_names[i], task_names[j]
            A_i_layers, B_i_layers = A_matrices[task_i], B_matrices[task_i]
            A_j_layers, B_j_layers = A_matrices[task_j], B_matrices[task_j]

            AB_i_flat, AB_j_flat = [], []
            olora_sum = 0  # Accumulate O-LoRA metric
            olora_normalized_sum = 0  # Accumulate normalized O-LoRA metric

            for layer in A_i_layers.keys():  # Iterate through layers
                A_i = A_i_layers[layer]
                A_j = A_j_layers[layer]
                
                # Match B matrix key by replacing "lora_A" with "lora_B"
                layer_b = layer.replace("lora_A", "lora_B")

                if layer_b in B_i_layers and layer_b in B_j_layers:
                    B_i = B_i_layers[layer_b]
                    B_j = B_j_layers[layer_b]
                else:
                    print(f"Warning: Missing B matrix for layer {layer}, skipping...")
                    continue  # Skip this layer if B matrices are missing
                

                # Print the original shapes before transposing
                print(f"Layer: {layer}")
                print(f"A_i original shape: {A_i.shape}, B_i original shape: {B_i.shape}")
                print(f"A_j original shape: {A_j.shape}, B_j original shape: {B_j.shape}")

                # **Transpose A and B to match expected dimensions**
                A_i, B_i = A_i.T, B_i.T
                A_j, B_j = A_j.T, B_j.T

                # Print the new shapes after transposing
                print(f"A_i transposed shape: {A_i.shape}, B_i transposed shape: {B_i.shape}")
                print(f"A_j transposed shape: {A_j.shape}, B_j transposed shape: {B_j.shape}")

                # Print the shape of A and B matrices for debugging
                print(f"Layer: {layer}")
                print(f"A_i shape: {A_i.shape}, B_i shape: {B_i.shape}")
                print(f"A_j shape: {A_j.shape}, B_j shape: {B_j.shape}")
                # Compute AB for each layer
                AB_i = torch.matmul(A_i, B_i)
                AB_j = torch.matmul(A_j, B_j)

                # **Normalize AB matrices before flattening**
                AB_i_norm = normalize_rows(AB_i)
                AB_j_norm = normalize_rows(AB_j)

                # Flatten normalized AB matrices
                AB_i_flat.append(AB_i_norm.flatten())
                AB_j_flat.append(AB_j_norm.flatten())

                # Compute and accumulate O-LoRA metric (dot product of A matrices)
                olora_sum += torch.norm(A_i @ A_j.T, p='fro')**2

                # Compute and accumulate O-LoRA (Normalized) metric
                A_i_norm = normalize_rows(A_i)
                A_j_norm = normalize_rows(A_j)
                olora_normalized_sum += torch.norm(A_i_norm @ A_j_norm.T, p='fro')**2

            # Concatenate all layers into a single vector
            AB_i_flat = torch.cat(AB_i_flat)
            AB_j_flat = torch.cat(AB_j_flat)

            key = f"{task_i} | {task_j}"  # Convert tuple to string

            # 1. O-LoRA Metric (Summed over all layers)
            results["O-LoRA"][key] = olora_sum.item()

            # 2. O-LoRA Metric (Normalized, Summed over all layers)
            results["O-LoRA (Normalized)"][key] = olora_normalized_sum.item()

            # 3. Cosine Similarity (After Normalizing AB Matrices)
            cosine_similarity = torch.dot(AB_i_flat, AB_j_flat) / (torch.norm(AB_i_flat) * torch.norm(AB_j_flat) + 1e-8)
            results["Cosine Similarity"][key] = cosine_similarity.item()

            # 4. Raw Dot Product of Flattened AB
            raw_dot_product = torch.dot(AB_i_flat, AB_j_flat)
            results["Raw Dot Product"][key] = raw_dot_product.item()

    return results

if __name__ == "__main__":
    # Set experiment directory where LoRA matrices are stored
    experiment_dir = "/scratch/hsk8171/llmContinualLearning/CL4LLMs/OBT2.7B-ACC-O-LORA-dbpedia,amazon,yahoo,agnews-r=16-a=16-dp=0.1-2025-02-22"

    # Load LoRA A and B matrices
    A_matrices, B_matrices = load_lora_matrices(experiment_dir)

    # Compute orthogonality metrics
    results = compute_orthogonality_metrics(A_matrices, B_matrices)

    # Save results to a JSON file
    results_path = os.path.join(experiment_dir, "orthogonality_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")

