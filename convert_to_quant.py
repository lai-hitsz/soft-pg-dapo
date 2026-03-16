import torch
import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from .quantization.step_aware_linear import convert_to_fake_quant, load_stepaware_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--use_pg", action="store_true")
    parser.add_argument("--tbits", type=int, default=0)

    parser.add_argument("--dtype", type=str, default="float16")

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # ---------- load step-aware model ----------
    model, tokenizer = load_stepaware_model(
        args.model_path,
        wbits=args.wbits,
        tbits=args.tbits,
        group_size=args.group_size
    )
    convert_to_fake_quant(model)

    print("Saving model...")
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()