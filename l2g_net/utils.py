import os
import yaml
import numpy as np
import torch

class Logger:
    def __init__(self, args, metric, num_data_splits):
        self.save_dir = self.get_save_dir(base_dir=args.save_dir, dataset=args.dataset, name=args.name)
        self.verbose = args.verbose
        self.metric = metric
        self.val_metrics = []
        self.test_metrics = []
        self.best_steps = []
        self.num_runs = args.num_runs
        self.num_data_splits = num_data_splits
        self.cur_run = None
        self.cur_data_split = None

        print(f'Results will be saved to {self.save_dir}.')
        with open(os.path.join(self.save_dir, 'args.yaml'), 'w') as file:
            yaml.safe_dump(vars(args), file, sort_keys=False)

    def start_run(self, run, data_split):
        self.cur_run = run
        self.cur_data_split = data_split
        self.val_metrics.append(0)
        self.test_metrics.append(0)
        self.best_steps.append(None)

        if self.num_data_splits == 1:
            print(f'Starting run {run}/{self.num_runs}...')
        else:
            print(f'Starting run {run}/{self.num_runs} (using data split {data_split}/{self.num_data_splits})...')

    def update_metrics(self, metrics, step):
        if metrics[f'val {self.metric}'] > self.val_metrics[-1]:
            self.val_metrics[-1] = metrics[f'val {self.metric}']
            self.test_metrics[-1] = metrics[f'test {self.metric}']
            self.best_steps[-1] = step

        if self.verbose:
            print(f'run: {self.cur_run:02d}, step: {step:03d}, '
                  f'train {self.metric}: {metrics[f"train {self.metric}"]:.4f}, '
                  f'val {self.metric}: {metrics[f"val {self.metric}"]:.4f}, '
                  f'test {self.metric}: {metrics[f"test {self.metric}"]:.4f}')

    def finish_run(self):
        self.save_metrics()
        print(f'Finished run {self.cur_run}. '
              f'Best val {self.metric}: {self.val_metrics[-1]:.4f}, '
              f'corresponding test {self.metric}: {self.test_metrics[-1]:.4f} '
              f'(step {self.best_steps[-1]}).\n')

    def save_metrics(self):
        num_runs = len(self.val_metrics)
        val_metric_mean = np.mean(self.val_metrics).item()
        val_metric_std = np.std(self.val_metrics, ddof=1).item() if len(self.val_metrics) > 1 else np.nan
        test_metric_mean = np.mean(self.test_metrics).item()
        test_metric_std = np.std(self.test_metrics, ddof=1).item() if len(self.test_metrics) > 1 else np.nan

        metrics = {
            'num runs': num_runs,
            f'val {self.metric} mean': val_metric_mean,
            f'val {self.metric} std': val_metric_std,
            f'test {self.metric} mean': test_metric_mean,
            f'test {self.metric} std': test_metric_std,
            f'val {self.metric} values': self.val_metrics,
            f'test {self.metric} values': self.test_metrics,
            'best steps': self.best_steps
        }

        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'w') as file:
            yaml.safe_dump(metrics, file, sort_keys=False)

    def print_metrics_summary(self):
        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'r') as file:
            metrics = yaml.safe_load(file)

        print(f'Finished {metrics["num runs"]} runs.')
        print(f'Val {self.metric} mean: {metrics[f"val {self.metric} mean"]:.4f}')
        print(f'Val {self.metric} std: {metrics[f"val {self.metric} std"]:.4f}')
        print(f'Test {self.metric} mean: {metrics[f"test {self.metric} mean"]:.4f}')
        print(f'Test {self.metric} std: {metrics[f"test {self.metric} std"]:.4f}')

    @staticmethod
    def get_save_dir(base_dir, dataset, name):
        idx = 1
        save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')
        while os.path.exists(save_dir):
            idx += 1
            save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')

        os.makedirs(save_dir)

        return save_dir

def get_parameter_groups(
    model,
    base_lr=None,
    filter_lr_multiplier=200.0,
    step_lr_multiplier=0.01,
    knot_lr_multiplier=0.01,
    knot_lr=None,
):
    """
    Separates parameters into 5 groups:
    1. Standard weights (decay, normal LR)
    2. Biases/Norms (no decay, normal LR)
    3. Spline filters (no decay, high LR)
    4. Step parameters (no decay, reduced LR)
    5. Knot logits (no decay, custom LR)
    """
    no_weight_decay_names = ['bias', 'normalization', 'label_embeddings']
    filter_names = ['coeffs', 'means', 'stds', 'combinator']
    step_names = ['step']
    knot_names = ['knots_', '_logits']  # matches e.g. knots_S._logits

    groups = {
        "decay": {"params": []},
        "no_decay": {"params": [], "weight_decay": 0.0},
        "filter": {"params": [], "weight_decay": 0.0},
        "step": {"params": [], "weight_decay": 0.0},
        "knot": {"params": [], "weight_decay": 0.0},
    }

    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))

        n = name.lower()

        # Priority order avoids overlap
        if ('knots_' in n) and ('_logits' in n):
            groups["knot"]["params"].append(param)
        elif any(s in n for s in step_names):
            groups["step"]["params"].append(param)
        elif any(f in n for f in filter_names):
            groups["filter"]["params"].append(param)
        elif any(nd in n for nd in no_weight_decay_names):
            groups["no_decay"]["params"].append(param)
        else:
            groups["decay"]["params"].append(param)

    parameter_groups = [
        groups["decay"],
        groups["no_decay"],
        groups["filter"],
        groups["step"],
        groups["knot"],
    ]

    if base_lr is not None:
        parameter_groups[2]['lr'] = base_lr * filter_lr_multiplier
        parameter_groups[3]['lr'] = base_lr * step_lr_multiplier
        effective_knot_lr = knot_lr if knot_lr is not None else base_lr * knot_lr_multiplier
        parameter_groups[4]['lr'] = effective_knot_lr

        print(f"Filter parameters boosted to LR: {parameter_groups[2]['lr']:.5f}")
        print(f"Step parameters reduced to LR: {parameter_groups[3]['lr']:.5f}")
        print(f"Knot parameters set to LR: {parameter_groups[4]['lr']:.5f}")

    # Drop empty groups
    return [g for g in parameter_groups if len(g["params"]) > 0]

def get_lr_scheduler_with_warmup(optimizer, num_warmup_steps=None, num_steps=None,
                                 warmup_proportion=None, last_step=-1):
    if num_warmup_steps is None and (num_steps is None or warmup_proportion is None):
        raise ValueError('Either num_warmup_steps or num_steps and warmup_proportion should be provided.')
    if num_warmup_steps is None:
        num_warmup_steps = int(num_steps * warmup_proportion)

    def get_lr_multiplier(step):
        if step < num_warmup_steps:
            return (step + 1) / (num_warmup_steps + 1)
        return 1.0  # flat after warmup

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=last_step)

"""
def get_lr_scheduler_with_warmup(optimizer, num_warmup_steps=None, num_steps=None, warmup_proportion=None, last_step=-1):
    if num_warmup_steps is None and (num_steps is None or warmup_proportion is None):
        raise ValueError('Either num_warmup_steps or num_steps and warmup_proportion should be provided.')

    if num_warmup_steps is None:
        num_warmup_steps = int(num_steps * warmup_proportion)

        def get_lr_multiplier(step):
            if step < num_warmup_steps:
                return (step + 1) / (num_warmup_steps + 1)
            else:
                return 1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=last_step)

    return lr_scheduler
"""
""" 
def get_lr_scheduler_with_warmup(optimizer, num_warmup_steps=None, num_steps=None, warmup_proportion=None, last_step=-1):
    if num_warmup_steps is None and (num_steps is None or warmup_proportion is None):
        raise ValueError('Either num_warmup_steps or num_steps and warmup_proportion should be provided.')

    if num_warmup_steps is None:
        num_warmup_steps = int(num_steps * warmup_proportion)

    # --- Schedule 1: Standard (Warmup -> Constant 1.0) ---
    def standard_schedule(step):
        if step < num_warmup_steps:
            return (step + 1) / (num_warmup_steps + 1)
        return 1.0

    # --- Schedule 2: Decay (Warmup -> Linear Decay to 0.0) ---
    def decay_schedule(step):
        if step < num_warmup_steps:
            return (step + 1) / (num_warmup_steps + 1)
        
        steps_completed_decay = step - num_warmup_steps
        total_decay_steps = num_steps - num_warmup_steps
        
        if total_decay_steps <= 0: return 0.0
            
        return max(0.0, (total_decay_steps - steps_completed_decay) / total_decay_steps)

    # Auto-detect the 3-group structure from get_parameter_groups
    if len(optimizer.param_groups) == 3:
        schedule_list = [standard_schedule, standard_schedule, decay_schedule]
    else:
        schedule_list = standard_schedule

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule_list, last_epoch=last_step)

    return lr_scheduler
"""