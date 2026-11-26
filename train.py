
import math
import torch
import tiktoken

from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from data_preparation.dataset import GPTDataset
from gpt import GPTModel

from generate import generate_and_print_sample

from config import (
    GPT_CONFIG_124M,
    TRAIN_CONFIG,
    DATA_PATH,
    MODEL_PATH,
    get_device
)


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDataset(
        txt,
        tokenizer,
        max_length,
        stride
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )


def calc_loss_batch(
    input_batch,
    target_batch,
    model,
    device
):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    loss = cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )

    return loss


def calc_loss_loader(
    data_loader,
    model,
    device,
    num_batches=None
):
    total_loss = 0.0

    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(
            num_batches,
            len(data_loader)
        )

    if num_batches <= 0:
        return float("nan")

    for i, (input_batch, target_batch) in enumerate(data_loader):

        if i >= num_batches:
            break

        loss = calc_loss_batch(
            input_batch,
            target_batch,
            model,
            device
        )

        total_loss += loss.item()

    return total_loss / num_batches


def evaluate_model(
    model,
    train_loader,
    val_loader,
    device,
    eval_iter
):
    model.eval()

    with torch.no_grad():

        train_loss = calc_loss_loader(
            train_loader,
            model,
            device,
            num_batches=eval_iter
        )

        val_loss = calc_loss_loader(
            val_loader,
            model,
            device,
            num_batches=eval_iter
        )

    model.train()

    return train_loss, val_loss


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    n_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    warmup_steps,
    context_size,
    initial_lr=3e-5,
    min_lr=1e-6
):
    train_losses = []
    val_losses = []
    track_tokens_seen = []
    track_lrs = []

    tokens_seen = 0
    global_step = -1

    peak_lr = optimizer.param_groups[0]["lr"]

    total_training_steps = len(train_loader) * n_epochs

    if total_training_steps == 0:
        raise ValueError("Training loader is empty.")

    warmup_steps = min(
        max(0, warmup_steps),
        total_training_steps - 1
    )

    lr_increment = (
        (peak_lr - initial_lr) / max(1, warmup_steps)
    )

    for epoch in range(n_epochs):

        model.train()

        for input_batch, target_batch in train_loader:

            optimizer.zero_grad()

            global_step += 1

            # Linear learning-rate warmup
            if global_step < warmup_steps:

                lr = initial_lr + global_step * lr_increment

            # Cosine learning-rate decay
            else:

                progress = (
                    (global_step - warmup_steps)
                    / max(1, total_training_steps - warmup_steps - 1)
                )

                lr = min_lr + (
                    (peak_lr - min_lr)
                    * 0.5
                    * (1 + math.cos(math.pi * progress))
                )

            # Update the optimizer's learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            track_lrs.append(lr)

            # Forward pass and calculate loss
            loss = calc_loss_batch(
                input_batch,
                target_batch,
                model,
                device
            )

            # Backpropagation
            loss.backward()

            # Gradient clipping after warmup
            if global_step >= warmup_steps:

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

            # Update model parameters
            optimizer.step()

            tokens_seen += input_batch.numel()

            # Evaluate the model
            if global_step % eval_freq == 0:

                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Ep {epoch + 1} "
                    f"(Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}, "
                    f"LR {lr:.8f}"
                )

        # Generate a sample after each epoch
        generate_and_print_sample(
            model=model,
            tokenizer=tokenizer,
            device=device,
            start_context=start_context,
            context_size=context_size
        )

    return (
        train_losses,
        val_losses,
        track_tokens_seen,
        track_lrs
    )


def main():

    config = TRAIN_CONFIG
    model_config = GPT_CONFIG_124M

    torch.manual_seed(config["seed"])

    device = get_device()

    print(f"Using device: {device}")

    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(model_config)
    model.to(device)

    if config["max_length"] > model_config["context_length"]:
        raise ValueError(
            "Dataset max_length exceeds model context length."
        )

    # Read training data
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        text_data = file.read()

    # Split training and validation data
    split_idx = int(
        config["train_ratio"] * len(text_data)
    )

    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=config["batch_size"],
        max_length=config["max_length"],
        stride=config["stride"],
        shuffle=config["shuffle"],
        drop_last=True
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=config["batch_size"],
        max_length=config["max_length"],
        stride=config["stride"],
        shuffle=False,
        drop_last=False
    )

    if len(train_loader) == 0:
        raise ValueError(
            "Not enough training data for the batch size "
            "and context length."
        )

    if len(val_loader) == 0:
        raise ValueError(
            "Not enough validation data for the context length."
        )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["peak_lr"],
        weight_decay=config["weight_decay"]
    )

    total_steps = (
        len(train_loader) * config["n_epochs"]
    )

    warmup_steps = int(
        config["warmup_ratio"] * total_steps
    )

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    (
        train_losses,
        val_losses,
        tokens_seen,
        lrs
    ) = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        n_epochs=config["n_epochs"],
        eval_freq=config["eval_freq"],
        eval_iter=config["eval_iter"],
        start_context=config["start_context"],
        tokenizer=tokenizer,
        warmup_steps=warmup_steps,
        context_size=model_config["context_length"],
        initial_lr=config["initial_lr"],
        min_lr=config["min_lr"]
    )

    # Create the checkpoint directory
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save model and optimizer
    torch.save(
        {
            "config": model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "tokens_seen": tokens_seen,
            "learning_rates": lrs
        },
        MODEL_PATH
    )

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
