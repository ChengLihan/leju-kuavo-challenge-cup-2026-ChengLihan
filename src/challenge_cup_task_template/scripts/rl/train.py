#!/usr/bin/env python3
"""
PPO training script — deployment-realistic grasping RL.

Single policy: 23-dim obs → 7-dim residual arm correction.
No velocity, torque, or force in observations (unavailable at ROS deployment).
"""

import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from rl.grasp_env import GraspEnv
from rl.policy_nets import ResidualActor, Critic

# ── Status file for monitoring ──
_STATUS_FILE = os.environ.get("RL_STATUS_FILE", "/tmp/rl_status.txt")

def _status(msg):
    print(msg, flush=True)
    try:
        with open(_STATUS_FILE, "a") as f:
            f.write(msg + "\n"); f.flush()
    except Exception:
        pass


# ── PPO Buffer ──────────────────────────────────────────────────────

class PPOBuffer:
    def __init__(self, n_envs, n_steps, obs_dim, device):
        self.n_envs, self.n_steps, self.device = n_envs, n_steps, device
        self.idx = 0
        self.obs      = torch.zeros(n_steps, n_envs, obs_dim, device=device)
        self.actions  = torch.zeros(n_steps, n_envs, 7, device=device)
        self.logps    = torch.zeros(n_steps, n_envs, device=device)
        self.rewards  = torch.zeros(n_steps, n_envs, device=device)
        self.values   = torch.zeros(n_steps, n_envs, device=device)
        self.dones    = torch.zeros(n_steps, n_envs, device=device)
        self.advantages = torch.zeros(n_steps, n_envs, device=device)
        self.returns  = torch.zeros(n_steps, n_envs, device=device)

    def store(self, obs, action, logp, value, reward, done):
        self.obs[self.idx]     = torch.from_numpy(obs)
        self.actions[self.idx] = torch.from_numpy(action)
        self.logps[self.idx]   = torch.from_numpy(logp)
        self.values[self.idx]  = torch.from_numpy(value)
        self.rewards[self.idx] = torch.from_numpy(reward)
        self.dones[self.idx]   = torch.from_numpy(done)
        self.idx += 1

    def compute_gae(self, last_value, gamma, gae_lambda):
        last_val = torch.from_numpy(last_value).to(self.device)
        gae = torch.zeros(self.n_envs, device=self.device)
        for step in reversed(range(self.n_steps)):
            nv = last_val if step == self.n_steps-1 else self.values[step+1]
            nd = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * nv * nd - self.values[step]
            gae = delta + gamma * gae_lambda * nd * gae
            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]
            last_val = self.values[step]
        # Normalize both
        for tensor in [self.advantages, self.returns]:
            flat = tensor.view(-1)
            s = flat.std()
            if s > 1e-6:
                tensor.copy_((tensor - flat.mean()) / s)

    def get_batches(self, batch_size):
        total = self.n_envs * self.n_steps
        indices = torch.randperm(total, device=self.device)
        flat = {"obs": self.obs.view(total, -1), "actions": self.actions.view(total, 7),
                "logps": self.logps.view(total), "advantages": self.advantages.view(total),
                "returns": self.returns.view(total)}
        for start in range(0, total, batch_size):
            end = min(start+batch_size, total)
            idx = indices[start:end]
            yield {k: v[idx] for k, v in flat.items()}


# ── PPO Trainer ─────────────────────────────────────────────────────

class PPOTrainer:
    def __init__(self, actor, critic, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, device="cuda"):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.device = device
        self.optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.clip_range, self.ent_coef = clip_range, ent_coef
        self.vf_coef, self.max_grad_norm = vf_coef, max_grad_norm

    def update(self, buffer, batch_size, n_epochs):
        stats = {"p_loss":0,"v_loss":0,"ent":0,"kl":0}
        n_up = 0
        for _ in range(n_epochs):
            for batch in buffer.get_batches(batch_size):
                new_logp, entropy = self.actor.evaluate(batch["obs"], batch["actions"])
                new_values = self.critic(batch["obs"])
                old_logp = batch["logps"]
                ratio = torch.exp(new_logp - old_logp)
                adv = batch["advantages"]

                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range)
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = nn.functional.mse_loss(new_values, batch["returns"])
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters())+list(self.critic.parameters()), self.max_grad_norm)
                self.optimizer.step()

                stats["p_loss"] += policy_loss.item()
                stats["v_loss"] += value_loss.item()
                stats["ent"] += entropy.mean().item()
                with torch.no_grad():
                    stats["kl"] += (new_logp - old_logp).mean().item()
                n_up += 1
        return {k: v/max(n_up,1) for k,v in stats.items()}


# ── Vectorized Env ──────────────────────────────────────────────────

from concurrent.futures import ThreadPoolExecutor, as_completed

class VecGraspEnv:
    """Thread-pool vectorized env.

    Each env has its OWN MjModel/MjData, so mj_step is thread-safe.
    v2 training ran 1500+ FPS with this for 500k+ steps without issues.
    """
    def __init__(self, n_envs, difficulty=0.0, arms=None):
        if arms is None:
            arms = ["right"]*n_envs
        self.envs = [GraspEnv(arm=arms[i%len(arms)], difficulty=difficulty) for i in range(n_envs)]
        self.n_envs = n_envs
        self.obs_dim = self.envs[0].obs_dim
        # Pre-warm: run 1 step sequentially to avoid MuJoCo global init race in ThreadPool
        for env in self.envs:
            env.reset()
            env.step(np.zeros(7, dtype=np.float32))
        self._pool = ThreadPoolExecutor(max_workers=n_envs)
        self._actions = None

    def reset(self):
        obs_list = [env.reset()[0] for env in self.envs]
        return np.stack(obs_list)

    def _step_one(self, i):
        env = self.envs[i]
        o, r, d, t, _ = env.step(self._actions[i])
        if d or t:
            o, _ = env.reset()
        return i, o, r, d, t

    def step(self, actions):
        self._actions = actions
        n = self.n_envs
        obs_arr = np.empty((n, self.obs_dim), dtype=np.float32)
        rew_arr = np.empty(n, dtype=np.float32)
        done_arr = np.empty(n, dtype=np.float32)
        trunc_arr = np.empty(n, dtype=np.float32)
        futures = [self._pool.submit(self._step_one, i) for i in range(n)]
        for f in as_completed(futures, timeout=10):  # 10s timeout prevents indefinite hang
            i, o, r, d, t = f.result()
            obs_arr[i] = o; rew_arr[i] = r; done_arr[i] = d; trunc_arr[i] = t
        return obs_arr, rew_arr, done_arr, trunc_arr

    def set_difficulty(self, d):
        for env in self.envs:
            env.difficulty = d


# ── Training Loop ───────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _status(f"Using device: {device}")

    arms = ["right"]*(args.n_envs//2) + ["left"]*(args.n_envs - args.n_envs//2)
    envs = VecGraspEnv(n_envs=args.n_envs, difficulty=0.0, arms=arms)
    _status(f"Created {args.n_envs} envs ({args.n_envs//2}R, {args.n_envs - args.n_envs//2}L)")

    actor = ResidualActor(obs_dim=envs.obs_dim)
    critic = Critic(obs_dim=envs.obs_dim)
    n_params = sum(p.numel() for p in list(actor.parameters())+list(critic.parameters()))
    _status(f"Policy: {n_params:,} params")

    trainer = PPOTrainer(actor, critic, lr=args.lr, gamma=args.gamma,
                         gae_lambda=args.gae_lambda, clip_range=args.clip_range,
                         ent_coef=args.ent_coef, device=device)
    buffer = PPOBuffer(args.n_envs, args.n_steps, envs.obs_dim, device)

    writer = SummaryWriter(log_dir=os.path.join(args.logdir, f"run_{int(time.time())}"))
    os.makedirs(args.output, exist_ok=True)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        _status(f"Resumed from {args.resume} at {start_step}")

    open(_STATUS_FILE, "w").close()
    _status(f"Starting training for {args.total_timesteps:,} steps...")
    t_start = time.time()

    obs = envs.reset()
    total_steps = start_step
    next_ckpt = start_step + args.checkpoint_interval

    while total_steps < args.total_timesteps:
        # Curriculum
        progress = total_steps / args.total_timesteps
        if args.curriculum:
            difficulty = 0.0 if progress<0.2 else (0.33 if progress<0.5 else (0.66 if progress<0.8 else 1.0))
        else:
            difficulty = 0.5
        envs.set_difficulty(difficulty)

        # Rollout
        for step in range(args.n_steps):
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().to(device)
                action, logp = actor.get_action(obs_t)
                value = critic(obs_t)
            next_obs, rewards, dones, truncs = envs.step(action.cpu().numpy())
            done = np.logical_or(dones, truncs).astype(np.float32)
            buffer.store(obs, action.cpu().numpy(), logp.cpu().numpy(),
                         value.cpu().numpy(), rewards, done)
            obs = next_obs
            total_steps += args.n_envs

        # Last value
        with torch.no_grad():
            last_value = critic(torch.from_numpy(obs).float().to(device)).cpu().numpy()
        buffer.compute_gae(last_value, trainer.gamma, trainer.gae_lambda)

        # Update
        stats = trainer.update(buffer, args.batch_size, args.n_epochs)
        buffer.idx = 0

        fps = total_steps / max(time.time()-t_start, 0.1)
        _status(f"Step {total_steps:>8,} | p_loss={stats['p_loss']:.4f} "
                f"v_loss={stats['v_loss']:.4f} ent={stats['ent']:.3f} "
                f"kl={stats['kl']:.4f} diff={difficulty:.2f} fps={fps:.0f}")

        writer.add_scalar("train/policy_loss", stats["p_loss"], total_steps)
        writer.add_scalar("train/value_loss", stats["v_loss"], total_steps)
        writer.add_scalar("train/entropy", stats["ent"], total_steps)
        writer.add_scalar("train/fps", fps, total_steps)
        writer.add_scalar("train/difficulty", difficulty, total_steps)

        # Checkpoint
        if total_steps >= next_ckpt:
            ckpt_path = os.path.join(args.output, f"checkpoint_{total_steps}.pt")
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                        "optimizer": trainer.optimizer.state_dict(), "step": total_steps}, ckpt_path)
            _status(f"  Saved checkpoint: {ckpt_path}")
            next_ckpt = total_steps + args.checkpoint_interval

    final_path = os.path.join(args.output, "final_policy.pt")
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "optimizer": trainer.optimizer.state_dict(), "step": total_steps}, final_path)
    _status(f"Training complete! Final policy saved to {final_path}")
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="PPO Training — Kuavo Grasping RL v2")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=500_000)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(_SCRIPT_DIR, "models")
    if args.logdir is None:
        args.logdir = os.path.join(_SCRIPT_DIR, "runs")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
