import torch
from rfdetr import RFDETRPoseMedium


if __name__ == "__main__":
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA devices available")

    model = RFDETRPoseMedium(num_keypoints=2, keypoint_names=["one", "two"], skeleton=[[0, 1]])

    model.train(
        dataset_dir="./data/detect_bed_rails_coco",
        epochs=10,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="./results/train_bed_rails",
        device="cuda",
        num_keypoints=2
    )
