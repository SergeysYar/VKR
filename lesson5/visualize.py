import torch
import matplotlib.pyplot as plt
import fire
import numpy as np
from predict import get_torch_model, load_ckpt, get_device, load_ply_file, predict
from dataset import sample_point_cloud, normalize_point_cloud


def save_labeled_point_cloud(
        points: torch.Tensor,
        labels: torch.Tensor,
        save_path: str,
        format: str = "ply"
):
    """
    Saves a point cloud with labels to a file.

    Args:
        points (torch.Tensor): Tensor of 3D coordinates with shape (N, 3) or (3, N).
        labels (torch.Tensor): Predicted or true labels for each point with shape (N,).
        save_path (str): Path where to save the point cloud file.
        format (str): Output format - 'ply' or 'txt'. Defaults to 'ply'.

    Returns:
        None
    """
    # Ensure points are in shape (N, 3)
    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.transpose(0, 1)

    points = points.cpu().numpy()
    labels = labels.cpu().numpy().reshape(-1, 1)

    # Combine points and labels
    labeled_points = np.hstack([points, labels])

    if format.lower() == 'ply':
        # Save as PLY with header
        with open(save_path, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(labeled_points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property int label\n")
            f.write("end_header\n")

            # Write points with labels
            for point in labeled_points:
                f.write(f"{point[0]} {point[1]} {point[2]} {int(point[3])}\n")

        print(f"Saved labeled point cloud to {save_path} (PLY format with label field)")

    elif format.lower() == 'txt':
        # Save as simple text file (x y z label)
        np.savetxt(save_path, labeled_points, fmt='%.6f %.6f %.6f %d')
        print(f"Saved labeled point cloud to {save_path} (TXT format: x y z label)")

    elif format.lower() == 'npy':
        # Save as numpy array
        np.save(save_path, labeled_points)
        print(f"Saved labeled point cloud to {save_path}.npy (NumPy format)")

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'ply', 'txt', or 'npy'.")


def visualize_point_cloud(
        points: torch.Tensor,
        pred_labels: torch.Tensor,
        true_labels: torch.Tensor = None,
        save2path: str = None,
        save_labeled: str = None,
        save_format: str = "ply"
):
    """
    Visualizes predicted and true labels for a 3D point cloud and optionally saves labeled data.

    Args:
        points (torch.Tensor): Tensor of 3D coordinates with shape (N, 3) where N is the number of points.
        pred_labels (torch.Tensor): Predicted labels for each point with shape (N,).
        true_labels (torch.Tensor, optional): True labels for each point with shape (N,). If provided,
                                           a side-by-side comparison will be created showing both
                                           predicted and true labels. Defaults to None.
        save2path (str, optional): Path to save the visualization. If None, the plot will be displayed
                                 on screen. Defaults to None.
        save_labeled (str, optional): Path to save the labeled point cloud. If None, labeled data won't be saved.
                                    Defaults to None.
        save_format (str): Format for saving labeled point cloud ('ply', 'txt', or 'npy'). Defaults to 'ply'.

    Returns:
        None: Either displays the plot or saves it to the specified path.
    """
    # Save labeled point cloud if requested
    if save_labeled:
        # Save predictions
        pred_save_path = save_labeled.replace('.', '_pred.')
        save_labeled_point_cloud(points, pred_labels, pred_save_path, format=save_format)

        # Save true labels if available
        if true_labels is not None and len(true_labels) > 0:
            true_save_path = save_labeled.replace('.', '_true.')
            save_labeled_point_cloud(points, true_labels, true_save_path, format=save_format)

    # Continue with visualization
    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.transpose(0, 1)

    points = points.cpu().numpy()
    fig = plt.figure(figsize=(12, 5) if true_labels is not None else (6, 5))

    ax = fig.add_subplot(121 if true_labels is not None else 111, projection="3d")
    ax.set_title("Predicted Labels")

    labels = pred_labels.cpu().numpy()
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=labels, cmap="jet", s=1
    )
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if true_labels is not None and len(true_labels) > 0:
        ax = fig.add_subplot(122, projection="3d")
        ax.set_title("True Labels")

        labels = true_labels.cpu().numpy()
        scatter = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2], c=labels, cmap="jet", s=1
        )
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()

    if save2path:
        plt.savefig(save2path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save2path}")
    else:
        plt.show()


def main(
        path2model: str,
        input: str,
        save2path: str = None,
        save_labeled: str = None,
        save_format: str = "ply",
        num_points: int = None
):
    """
    Main function to load a trained model, perform prediction on a point cloud, and visualize/save the results.

    Args:
        path2model (str): Path to the model checkpoint file.
        input (str): Path to the input point cloud file (PLY format).
        save2path (str, optional): Path to save the visualization. If None, the plot will be displayed.
        save_labeled (str, optional): Path to save the labeled point cloud. If None, labeled data won't be saved.
        save_format (str): Format for saving labeled point cloud ('ply', 'txt', or 'npy').
        num_points (int, optional): Number of points to sample. If None, uses all points.

    Returns:
        None
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {path2model}...")
    model = get_torch_model(load_ckpt(path2model)).to(device)
    model.eval()

    # Load and process point cloud
    print(f"Loading point cloud from {input}...")
    verts, labels = load_ply_file(input)

    # Sample points if needed
    if num_points is not None:
        point_cloud, labels = sample_point_cloud(verts, labels, num_points=num_points)
        print(f"Sampled {num_points} points")
    else:
        point_cloud = verts
        print(f"Using all {len(verts)} points")

    # Normalize
    point_cloud = normalize_point_cloud(point_cloud)

    # Convert to tensors
    point_cloud = torch.FloatTensor(point_cloud).to(device)
    labels = torch.LongTensor(labels)

    # Prepare for model (batch dimension)
    point_cloud_batch = point_cloud.transpose(0, 1).unsqueeze(0)

    # Predict
    print("Running prediction...")
    with torch.no_grad():
        pred_labels = predict(model, point_cloud_batch)

    # Count classes
    unique_pred, counts_pred = torch.unique(pred_labels, return_counts=True)
    print(f"Predicted classes: {dict(zip(unique_pred.cpu().numpy(), counts_pred.cpu().numpy()))}")

    # Visualize and save
    visualize_point_cloud(
        point_cloud,
        pred_labels,
        labels,
        save2path=save2path,
        save_labeled=save_labeled,
        save_format=save_format
    )


if __name__ == "__main__":
    fire.Fire(main)