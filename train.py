from src.model import UNet, CombinedLoss, SegmentationDataset, HUWindowAndScale
from src.options import parse_args
from src.utils import load_data_splits, binary_accuracy, DSC
import torch
import torchvision.transforms as transforms
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np

# import wandb


def train():
    opt = parse_args()

    # Define Model
    model = UNet()

    # Load dataset
    splits = load_data_splits("splits.pkl", opt.data_dir)
    tf = transforms.Compose(
        [
            HUWindowAndScale(lower=-1000, upper=0),
        ]
    )
    dataset = SegmentationDataset(splits, "train", img_transform=tf)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=3, shuffle=True, drop_last=True
    )

    # wandb.init(project="Segmentation", config=opt)

    # Optimizer and Loss
    optim = torch.optim.Adam(model.parameters(), lr=opt.lr)
    loss_fn = CombinedLoss(opt.lambda_DSC, opt.lambda_BCE)

    # Accelerator
    accelerator = Accelerator()
    model, optim, train_loader = accelerator.prepare(model, optim, train_loader)
    print(f"Training on {accelerator.device}")
    # wandb.config.update({"device": accelerator.device})

    # Training Loop
    print(f"Total parameter updates: {len(train_loader) * opt.n_epoch}")
    logs = []
    pbar = tqdm(total=len(train_loader), position=0, leave=True)
    for epoch in range(opt.n_epoch):
        pbar.reset()
        model.train()

        for x, y, _ in train_loader:
            optim.zero_grad()

            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            accelerator.backward(loss)

            with torch.no_grad():
                mask = y.sum(dim=(2, 3)) == 0
                accuracy = binary_accuracy(y_pred, y).mean().item()
                dice = DSC(y_pred, y)[~mask].mean().item()

            logs.append([loss.item(), accuracy, dice])
            avg_loss = np.mean([log[0] for log in logs[-100:]])
            avg_accuracy = np.mean([log[1] for log in logs[-100:]])
            avg_dice = np.mean([log[2] for log in logs[-100:]])

            pbar.set_description(
                f"Epoch: {epoch}, Loss: {avg_loss:.3g}, Accuracy: {avg_accuracy:.3g}, DSC: {avg_dice:.3g}"
            )
            pbar.update(1)

            optim.step()

        epoch_loss = np.mean([log[0] for log in logs[-len(train_loader) :]])
        epoch_accuracy = np.mean([log[1] for log in logs[-len(train_loader) :]])
        epoch_dice = np.mean([log[2] for log in logs[-len(train_loader) :]])
        pbar.set_description(
            f"Epoch Avg Loss: {epoch_loss:.3g}, Avg Accuracy: {epoch_accuracy:.3g}, Avg DSC: {epoch_dice:.3g}"
        )
        # wandb.log(
        #     {"loss": epoch_loss, "accuracy": epoch_accuracy, "dice_score": epoch_dice}
        # )

    pbar.close()
    print("Training complete")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "logs": logs,
            "config": opt,
        },
        opt.model_path,
    )

    print(f"Model saved to {opt.model_path}")


if __name__ == "__main__":
    train()
