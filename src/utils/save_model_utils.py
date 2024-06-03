import os.path

import torch

from src.models.factory import get_model


CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_solver_model(t_idx, solver_model, exp_name=None):
    ckpt_path = get_solver_store_path(t_idx, exp_name)
    torch.save(solver_model.state_dict(), ckpt_path)


def get_trained_solver(t_idx, model_args, exp_dict, exp_name=None):
    solver = get_model(exp_dict, model_args).cuda()
    ckpt_path = get_solver_store_path(t_idx, exp_name)
    solver.load_state_dict(torch.load(ckpt_path))
    return solver


def get_solver_store_path(t_idx, exp_name=None):
    if exp_name:
        ckpt_path = os.path.join(CHECKPOINT_DIR, exp_name, f"solver_{t_idx}.pth")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        return ckpt_path
    return os.path.join(CHECKPOINT_DIR, f"solver_{t_idx}.pth")
