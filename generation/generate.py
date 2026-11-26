
import argparse
import torch
import tiktoken

from config import MODEL_PATH, get_device
from pretraining.transformer.gpt import GPTModel


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(
        text,
        allowed_special={"<|endoftext|>"}
    )

    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None
):
    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]

        logits = model(idx_cond)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(
                logits,
                top_k
            )

            min_val = top_logits[:, -1:]

            logits = logits.masked_fill(
                logits < min_val,
                float("-inf")
            )

        if temperature > 0.0:

            logits = logits / temperature

            probs = torch.softmax(
                logits,
                dim=-1
            )

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

        else:
            idx_next = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

        if eos_id is not None:
            if (idx_next == eos_id).all().item():
                break

        idx = torch.cat(
            (idx, idx_next),
            dim=1
        )

    return idx


def generate_text_simple(
    model,
    idx,
    max_new_tokens,
    context_size
):
    return generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size
    )


def generate_and_print_sample(
    model,
    tokenizer,
    device,
    start_context,
    context_size
):
    was_training = model.training
    model.eval()

    try:
        encoded = text_to_token_ids(
            start_context,
            tokenizer
        ).to(device)

        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )

        decoded_text = token_ids_to_text(
            token_ids,
            tokenizer
        )

        print(decoded_text.replace("\n", " "))

    finally:
        model.train(was_training)


def load_model(device):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run train.py first."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model_config = checkpoint["config"]

    model = GPTModel(model_config)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, model_config


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("prompt", type=str)

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None
    )

    args = parser.parse_args()

    device = get_device()

    tokenizer = tiktoken.get_encoding("gpt2")

    model, model_config = load_model(device)

    encoded = text_to_token_ids(
        args.prompt,
        tokenizer
    ).to(device)

    token_ids = generate(
        model=model,
        idx=encoded,
        max_new_tokens=args.max_new_tokens,
        context_size=model_config["context_length"],
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.eot_token
    )

    print(token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
