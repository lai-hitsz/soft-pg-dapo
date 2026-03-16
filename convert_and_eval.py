import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table


import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

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
def evaluate_tasks(args, logger):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table

    results = {}

    if args.eval_tasks == "":
        return results

    task_list = args.eval_tasks.split(',')
    task_manager = lm_eval.tasks.TaskManager()

    hflm = HFLM(
        pretrained=args.model_path,
        batch_size=args.eval_batch_size,
        parallelize=True,           # 对齐 CLI
        trust_remote_code=True,
    )

    # test_model_quant_levels(hflm.model)

    results = lm_eval.simple_evaluate(
        model=hflm,
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

    # args with tasks
    parser.add_argument("--eval_tasks", type=str, default="")
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=16)

    # args with ppl
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--ppl_seqlen", type=int, default=2048)

    args = parser.parse_args()
    logger = setup_logger()

    evaluate_tasks(args, logger)


if __name__ == "__main__":
    main()
