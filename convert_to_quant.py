import torch
import argparse
import os
from typing import Dict
import transformers
from transformers import LlamaTokenizer
# from .quantization.step_aware_linear import convert_to_fake_quant, load_stepaware_model
from .quantization.fourier_linear import convert_to_fake_quant, load_stepaware_model

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


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

    # print(tokenizer)

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )    

    if isinstance(tokenizer, LlamaTokenizer):
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id != None else tokenizer.pad_token_id
                )
        })

    convert_to_fake_quant(model, args)

    print("Saving model...")
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()