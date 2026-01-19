import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from .quantization.fake_linear import convert_to_fake_quant
from accelerate import infer_auto_device_map, dispatch_model


def setup_logger():
    logger = logging.getLogger("eval_logger")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def test_model_quant_levels(model: torch.nn.Module, verbose=True):
    from collections import Counter
    total_counter = Counter()
    total_groups = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight.detach().clone()
            _, dim2 = w.shape
            group_size = 128
            assert dim2 % group_size == 0
            w_reshaped = w.reshape(-1, group_size)

            group_level_counts = []
            for g in w_reshaped:
                g = g.unsqueeze(0)
                levels = torch.unique(g).numel()
                group_level_counts.append(levels)

            counter = Counter(group_level_counts)
            total_counter.update(counter)
            total_groups += len(group_level_counts)

            if verbose:
                print(f"[Module: {name}] group-level counts: {dict(counter)}")

    print("\n[Global Group-Level Distribution]")
    for levels, count in sorted(total_counter.items()):
        print(f"  Groups with {levels} levels: {count} / {total_groups} ({count/total_groups*100:.2f}%)")

    return total_counter, total_groups

@torch.no_grad()
def evaluate_tasks(model, tokenizer, args, logger):
    results = {}
    if args.eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table

        task_list = args.eval_tasks.split(',')
        task_manager = lm_eval.tasks.TaskManager()
        model = HFLM(pretrained=model, batch_size=args.eval_batch_size, add_bos_token=True)      
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=task_list,
            num_fewshot=args.shots,
            task_manager=task_manager,
            confirm_run_unsafe_code=True,
        )

        table_str = make_table(results)
        logger.info(table_str)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_memory", type=str, default="70GiB",help="The maximum memory of each GPU")

    # args with quant
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)

    parser.add_argument("--use_pg", action="store_true")
    parser.add_argument("--tbits", type=int, default=0)

    # args with tasks
    parser.add_argument("--eval_tasks", type=str, default="")
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=16)

    # args with ppl
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--ppl_seqlen", type=int, default=2048)

    args = parser.parse_args()
    logger = setup_logger()

    logger.info(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    logger.info(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    model = dispatch_model(model, device_map=device_map)

    convert_to_fake_quant(model, wbits=args.wbits, group_size=args.group_size)
    # test_model_quant_levels(model)

    evaluate_tasks(model, tokenizer, args, logger)


if __name__ == "__main__":
    main()
