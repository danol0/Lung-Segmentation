from src.model import UNet, SegmentationDataset, HUWindowAndScale
from src.options import parse_args
from src.utils import load_data_splits, binary_accuracy, DSC
import torch
import torchvision.transforms as transforms
from accelerate import Accelerator
from tqdm import tqdm
import pandas as pd
import numpy as np


@torch.no_grad()
def test():
    opt = parse_args()

    # Define Model
    model = UNet()
    print(f"Loading model from {opt.model_path}")
    model.load_state_dict(torch.load(opt.model_path)["model_state_dict"])

    # Load datasets
    splits = load_data_splits("splits.pkl", opt.data_dir)
    tf = transforms.Compose(
        [
            HUWindowAndScale(lower=-1000, upper=0),
        ]
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=SegmentationDataset(splits, "train", img_transform=tf),
        batch_size=1,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=SegmentationDataset(splits, "test", img_transform=tf),
        batch_size=1,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    # Accelerator
    accelerator = Accelerator()
    model, train_loader, test_loader = accelerator.prepare(
        model, train_loader, test_loader
    )

    # Test loop
    model.eval()
    scores = []
    for loader, split in zip([train_loader, test_loader], ["train", "test"]):
        print(f"Evaluating model on {split} set...")
        for x, y, id in tqdm(loader):
            y_pred = model(x)
            acc = binary_accuracy(y_pred, y).item()
            sds = DSC(y_pred, y).item() if y.sum() > 0 else np.nan
            scores.append([id[0], split, acc, sds])

    # Save scores
    df = pd.DataFrame(
        scores, columns=["id", "split", "accuracy", "dice_score"]
    ).set_index("id")
    df.sort_values("accuracy", ascending=True, inplace=True)
    df.to_csv("scores.csv")
    print("Saved scores to scores.csv")
    print(
        f'Average accuracy on test set: {df[df["split"] == "test"]["accuracy"].mean():.2f}\n'
        f'Average dice score on test set: {df[df["split"] == "test"]["dice_score"].mean():.2f}'
    )


if __name__ == "__main__":
    test()
