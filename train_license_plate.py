import os
import yaml
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime
import time
from tqdm import tqdm


class LicensePlateTrainer:
    def __init__(
        self, data_yaml_path, model_size="n", epochs=100, imgsz=640, batch_size=16
    ):
        """
        Initialize the License Plate Trainer

        Args:
            data_yaml_path (str): Path to the data.yaml file
            model_size (str): YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            epochs (int): Number of training epochs
            imgsz (int): Image size for training
            batch_size (int): Batch size for training
        """
        self.data_yaml_path = data_yaml_path
        self.model_size = model_size
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.project_name = f"license_plate_detection_{model_size}"

        # Create runs directory
        self.runs_dir = Path("runs/detect")
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = YOLO(f"yolov8{model_size}.pt")

    def setup_training_config(self):
        """Setup training configuration"""
        # Check if CUDA is available
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {device}")

        # Training configuration
        self.train_config = {
            "data": self.data_yaml_path,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch_size,
            "device": device,
            "project": "runs/detect",
            "name": self.project_name,
            "save": True,
            "save_period": 10,  # Save checkpoint every 10 epochs
            "cache": True,  # Cache images for faster training
            "workers": 8,
            "patience": 50,  # Early stopping patience
            "optimizer": "SGD",
            "lr0": 0.01,  # Initial learning rate
            "lrf": 0.01,  # Final learning rate factor
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 7.5,  # Box loss gain
            "cls": 0.5,  # Class loss gain
            "dfl": 1.5,  # DFL loss gain
            "pose": 12.0,
            "kobj": 2.0,
            "label_smoothing": 0.0,
            "nbs": 64,  # Nominal batch size
            "hsv_h": 0.015,  # Image HSV-Hue augmentation (fraction)
            "hsv_s": 0.7,  # Image HSV-Saturation augmentation (fraction)
            "hsv_v": 0.4,  # Image HSV-Value augmentation (fraction)
            "degrees": 0.0,  # Image rotation (+/- deg)
            "translate": 0.1,  # Image translation (+/- fraction)
            "scale": 0.5,  # Image scale (+/- gain)
            "shear": 0.0,  # Image shear (+/- deg)
            "perspective": 0.0,  # Image perspective (+/- fraction), range 0-0.001
            "flipud": 0.0,  # Image flip up-down (probability)
            "fliplr": 0.5,  # Image flip left-right (probability)
            "mosaic": 1.0,  # Image mosaic (probability)
            "mixup": 0.0,  # Image mixup (probability)
            "copy_paste": 0.0,  # Segment copy-paste (probability)
        }

    def train(self):
        """Train the YOLOv8 model"""
        print(
            f"Starting training with YOLOv8{self.model_size} for license plate detection..."
        )
        print(f"Data: {self.data_yaml_path}")
        print(f"Epochs: {self.epochs}")
        print(f"Image size: {self.imgsz}")
        print(f"Batch size: {self.batch_size}")

        # Setup training configuration
        self.setup_training_config()

        # Start training with tqdm progress bar
        with tqdm(total=self.epochs, desc="Training Progress", unit="epoch") as pbar:
            for epoch in range(self.epochs):
                results = self.model.train(**self.train_config)
                pbar.update(1)

        return results

    def validate(self):
        """Validate the trained model"""
        print("Running validation...")
        val_results = self.model.val()
        return val_results

    def export_model(self, export_format="onnx"):
        """Export the trained model"""
        print(f"Exporting model to {export_format} format...")
        self.model.export(format=export_format)

    def plot_results(self):
        """Plot training results"""
        results_dir = self.runs_dir / self.project_name

        if (results_dir / "results.png").exists():
            print(f"Training results plot saved at: {results_dir}/results.png")

        if (results_dir / "confusion_matrix.png").exists():
            print(f"Confusion matrix saved at: {results_dir}/confusion_matrix.png")

    def test_inference(self, test_image_path=None):
        """Test inference on a sample image"""
        if test_image_path is None:
            # Use a test image from the dataset
            test_dir = Path(self.data_yaml_path).parent / "test" / "images"
            if test_dir.exists():
                test_images = list(test_dir.glob("*.jpg")) + list(
                    test_dir.glob("*.png")
                )
                if test_images:
                    test_image_path = test_images[0]

        if test_image_path and Path(test_image_path).exists():
            print(f"Running inference on: {test_image_path}")
            results = self.model(test_image_path)
            results[0].show()
            return results
        else:
            print("No test image found for inference")


class M1LicensePlateTrainer:
    def __init__(self, data_yaml_path, model_size="n", epochs=50):
        """Optimized for M1 MacBook Air"""
        self.data_yaml_path = data_yaml_path
        self.model_size = model_size
        self.epochs = epochs

        # M1 optimized settings
        self.batch_size = self.get_optimal_batch_size()
        self.imgsz = 640  # Standard size works well

        # Check MPS availability
        if torch.backends.mps.is_available():
            print("✅ MPS (Metal Performance Shaders) is available!")
            self.device = "mps"
        else:
            print("⚠️ MPS not available, using CPU")
            self.device = "cpu"

        self.model = YOLO(f"yolov8{model_size}.pt")

    def get_optimal_batch_size(self):
        """Get optimal batch size for M1 Air"""
        if self.model_size == "n":
            return 16  # YOLOv8n can handle larger batches
        elif self.model_size == "s":
            return 8  # YOLOv8s - moderate batch size
        else:
            return 4  # Larger models - smaller batches

    def train_optimized(self):
        """M1 optimized training configuration"""
        print(f"🚀 Starting M1 optimized training...")
        print(f"📊 Model: YOLOv8{self.model_size}")
        print(f"🔧 Device: {self.device}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"🔁 Epochs: {self.epochs}")

        start_time = time.time()

        # M1 optimized training parameters - TRAIN ONCE, NOT IN A LOOP!
        results = self.model.train(
            data=self.data_yaml_path,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch_size,
            device=self.device,
            # Performance optimizations for M1
            cache=True,  # Cache images in memory
            workers=4,  # Optimal for M1
            patience=20,  # Early stopping
            # Prevent thermal throttling
            save_period=10,  # Save checkpoints frequently
            # Learning rate optimized for small dataset
            lr0=0.01,
            lrf=0.1,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            # Data augmentation (moderate for small dataset)
            hsv_h=0.01,
            hsv_s=0.5,
            hsv_v=0.3,
            degrees=5.0,
            translate=0.1,
            scale=0.2,
            fliplr=0.5,
            mosaic=0.8,
            project="runs/detect",
            name=f"license_plate_m1_{self.model_size}",
        )

        training_time = time.time() - start_time
        print(f"⏱️ Training completed in {training_time/60:.1f} minutes")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 License Plate Detection Training"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="neet/train-data/data.yaml",
        help="Path to data.yaml file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size for training"
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--validate", action="store_true", help="Run validation after training"
    )
    parser.add_argument(
        "--export",
        type=str,
        choices=["onnx", "tensorrt", "tflite"],
        help="Export format after training",
    )
    parser.add_argument("--test", action="store_true", help="Run test inference")

    args = parser.parse_args()

    # Initialize trainer
    trainer = LicensePlateTrainer(
        data_yaml_path=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
    )

    try:
        # Train the model
        print("=" * 50)
        print("STARTING LICENSE PLATE DETECTION TRAINING")
        print("=" * 50)

        results = trainer.train()

        # Plot results
        trainer.plot_results()

        # Validate if requested
        if args.validate:
            trainer.validate()

        # Export if requested
        if args.export:
            trainer.export_model(args.export)

        # Test inference if requested
        if args.test:
            trainer.test_inference()

        print("=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)

        # Print best model path
        best_model_path = (
            trainer.runs_dir / trainer.project_name / "weights" / "best.pt"
        )
        print(f"Best model saved at: {best_model_path}")

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


# Additional utility functions
def load_trained_model(model_path):
    """Load a trained YOLOv8 model"""
    model = YOLO(model_path)
    return model


def predict_on_image(model_path, image_path, conf_threshold=0.25):
    """Make predictions on a single image"""
    model = load_trained_model(model_path)
    results = model(image_path, conf=conf_threshold)
    return results


def predict_on_video(model_path, video_path, output_path=None, conf_threshold=0.25):
    """Make predictions on a video"""
    model = load_trained_model(model_path)

    if output_path is None:
        output_path = f"output_{Path(video_path).stem}.mp4"

    results = model(
        video_path,
        conf=conf_threshold,
        save=True,
        project="runs/detect",
        name="video_inference",
    )
    return results


def quick_m1_test():
    """Quick test function for M1 MacBook Air"""
    # Test with YOLOv8n first (fastest)
    trainer = M1LicensePlateTrainer(
        data_yaml_path="train-data/data.yaml",
        model_size="n",
        epochs=25,  # Start with fewer epochs for testing
    )

    print("🧪 Running quick test with YOLOv8n...")
    results = trainer.train_optimized()

    # Quick validation
    val_results = trainer.model.val()
    print(f"📈 Quick test mAP50: {val_results.box.map50:.3f}")

    return trainer


if __name__ == "__main__":
    main()
