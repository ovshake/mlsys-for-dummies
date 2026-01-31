#!/usr/bin/env python3
"""
Minimal PPO Implementation on CartPole

This script demonstrates PPO's core concepts:
- Actor-Critic architecture
- GAE (Generalized Advantage Estimation)
- PPO-Clip objective

This is a simplified implementation for educational purposes.
For production RLHF, see verl, trl, or OpenRLHF.

Usage:
    pip install gymnasium  # if not installed
    python ppo_cartpole.py
"""

import argparse
from dataclasses import dataclass
from typing import List, Tuple
import random
import math

# Try to import gymnasium, fall back to simulation if not available
try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    print("Note: gymnasium not installed. Using simulated environment.")


@dataclass
class Experience:
    """Single step of experience."""
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: bool
    log_prob: float
    value: float


class SimpleNetwork:
    """
    Simple neural network simulation for demonstration.

    In real implementations, use PyTorch or JAX.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Random initialization (simplified)
        self.w1 = [[random.gauss(0, 0.1) for _ in range(hidden_size)]
                   for _ in range(input_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [[random.gauss(0, 0.1) for _ in range(output_size)]
                   for _ in range(hidden_size)]
        self.b2 = [0.0] * output_size

    def forward(self, x: List[float]) -> List[float]:
        """Forward pass."""
        # Hidden layer
        hidden = []
        for j in range(self.hidden_size):
            h = self.b1[j]
            for i in range(self.input_size):
                h += x[i] * self.w1[i][j]
            hidden.append(max(0, h))  # ReLU

        # Output layer
        output = []
        for j in range(self.output_size):
            o = self.b2[j]
            for i in range(self.hidden_size):
                o += hidden[i] * self.w2[i][j]
            output.append(o)

        return output

    def update(self, grads: List[float], lr: float):
        """Simplified gradient update (demonstration only)."""
        # In real implementations, use proper backpropagation
        pass


class ActorCritic:
    """
    Actor-Critic network for PPO.

    Actor: Outputs action probabilities
    Critic: Outputs state value
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        self.actor = SimpleNetwork(state_size, hidden_size, action_size)
        self.critic = SimpleNetwork(state_size, hidden_size, 1)
        self.action_size = action_size

    def get_action(self, state: List[float]) -> Tuple[int, float]:
        """
        Sample action from policy.

        Returns: (action, log_probability)
        """
        logits = self.actor.forward(state)

        # Softmax
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]

        # Sample action
        r = random.random()
        cumsum = 0
        action = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                action = i
                break

        log_prob = math.log(probs[action] + 1e-8)
        return action, log_prob

    def get_value(self, state: List[float]) -> float:
        """Get state value from critic."""
        return self.critic.forward(state)[0]

    def get_action_prob(self, state: List[float], action: int) -> float:
        """Get probability of specific action."""
        logits = self.actor.forward(state)
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        return exp_logits[action] / sum_exp


def compute_gae(experiences: List[Experience], gamma: float = 0.99,
                lam: float = 0.95) -> List[float]:
    """
    Compute Generalized Advantage Estimation.

    GAE balances bias and variance in advantage estimation:
    - λ=0: Just TD error (high bias, low variance)
    - λ=1: Full returns (low bias, high variance)
    """
    advantages = []
    gae = 0

    # Iterate backwards through experiences
    for i in reversed(range(len(experiences))):
        exp = experiences[i]

        if exp.done:
            next_value = 0
        else:
            next_value = experiences[i + 1].value if i + 1 < len(experiences) else 0

        # TD error
        delta = exp.reward + gamma * next_value - exp.value

        # GAE
        gae = delta + gamma * lam * (0 if exp.done else gae)
        advantages.insert(0, gae)

    return advantages


def ppo_update(actor_critic: ActorCritic, experiences: List[Experience],
               advantages: List[float], clip_epsilon: float = 0.2,
               lr: float = 3e-4) -> dict:
    """
    PPO update step.

    Key components:
    1. Compute probability ratio (new policy / old policy)
    2. Clip the ratio to prevent large updates
    3. Take minimum of clipped and unclipped objectives
    """
    # Compute returns for value function update
    returns = []
    for i, exp in enumerate(experiences):
        returns.append(exp.value + advantages[i])

    # PPO objectives (computed but not applied in this demo)
    policy_losses = []
    value_losses = []
    clip_fractions = []

    for i, exp in enumerate(experiences):
        # New probability
        new_prob = actor_critic.get_action_prob(exp.state, exp.action)
        new_log_prob = math.log(new_prob + 1e-8)

        # Probability ratio
        ratio = math.exp(new_log_prob - exp.log_prob)

        # Advantage
        adv = advantages[i]

        # Clipped objective
        unclipped = ratio * adv
        clipped = max(min(ratio, 1 + clip_epsilon), 1 - clip_epsilon) * adv

        # PPO loss (take minimum)
        policy_loss = -min(unclipped, clipped)
        policy_losses.append(policy_loss)

        # Track clipping
        clip_fractions.append(1 if abs(ratio - 1) > clip_epsilon else 0)

        # Value loss
        new_value = actor_critic.get_value(exp.state)
        value_loss = (new_value - returns[i]) ** 2
        value_losses.append(value_loss)

    return {
        'policy_loss': sum(policy_losses) / len(policy_losses),
        'value_loss': sum(value_losses) / len(value_losses),
        'clip_fraction': sum(clip_fractions) / len(clip_fractions),
    }


class SimulatedCartPole:
    """Simple CartPole simulation for when gymnasium isn't available."""

    def __init__(self):
        self.reset()

    def reset(self) -> List[float]:
        self.x = random.uniform(-0.05, 0.05)
        self.x_dot = random.uniform(-0.05, 0.05)
        self.theta = random.uniform(-0.05, 0.05)
        self.theta_dot = random.uniform(-0.05, 0.05)
        self.steps = 0
        return [self.x, self.x_dot, self.theta, self.theta_dot]

    def step(self, action: int) -> Tuple[List[float], float, bool]:
        # Simplified physics
        force = 10.0 if action == 1 else -10.0

        self.x_dot += 0.02 * force + random.gauss(0, 0.01)
        self.x += 0.02 * self.x_dot
        self.theta_dot += 0.05 * force * (1 if self.theta > 0 else -1)
        self.theta_dot += random.gauss(0, 0.01)
        self.theta += 0.02 * self.theta_dot

        self.steps += 1

        done = (abs(self.x) > 2.4 or abs(self.theta) > 0.21 or self.steps > 200)
        reward = 1.0 if not done else 0.0

        return [self.x, self.x_dot, self.theta, self.theta_dot], reward, done


def run_episode(env, actor_critic: ActorCritic) -> List[Experience]:
    """Run one episode and collect experiences."""
    if HAS_GYM:
        state, _ = env.reset()
        state = list(state)
    else:
        state = env.reset()

    experiences = []
    done = False

    while not done:
        action, log_prob = actor_critic.get_action(state)
        value = actor_critic.get_value(state)

        if HAS_GYM:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = list(next_state)
        else:
            next_state, reward, done = env.step(action)

        experiences.append(Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value,
        ))

        state = next_state

    return experiences


def main():
    parser = argparse.ArgumentParser(description="Minimal PPO on CartPole")
    parser.add_argument("--episodes", "-e", type=int, default=100,
                        help="Number of episodes to train")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float, default=0.2,
                        help="PPO clip parameter")
    args = parser.parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║" + " MINIMAL PPO ON CARTPOLE".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Create environment
    if HAS_GYM:
        env = gym.make("CartPole-v1")
        print("\nUsing gymnasium CartPole-v1")
    else:
        env = SimulatedCartPole()
        print("\nUsing simulated CartPole")

    # Create actor-critic
    actor_critic = ActorCritic(state_size=4, action_size=2)

    print(f"\nConfiguration:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Gamma: {args.gamma}")
    print(f"  GAE Lambda: {args.lam}")
    print(f"  Clip Epsilon: {args.clip_epsilon}")

    # Training loop
    print("\n" + "=" * 60)
    print(" TRAINING")
    print("=" * 60)

    episode_rewards = []

    for episode in range(args.episodes):
        # Collect episode
        experiences = run_episode(env, actor_critic)

        # Compute advantages using GAE
        advantages = compute_gae(experiences, args.gamma, args.lam)

        # PPO update
        update_stats = ppo_update(actor_critic, experiences, advantages,
                                   args.clip_epsilon)

        # Track rewards
        total_reward = sum(exp.reward for exp in experiences)
        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
            print(f"Episode {episode + 1:4d} | Reward: {total_reward:6.1f} | "
                  f"Avg(10): {avg_reward:6.1f} | "
                  f"Clip: {update_stats['clip_fraction']:.2f}")

    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)

    avg_first_10 = sum(episode_rewards[:10]) / 10
    avg_last_10 = sum(episode_rewards[-10:]) / 10

    print(f"\nAverage reward (first 10 episodes): {avg_first_10:.1f}")
    print(f"Average reward (last 10 episodes): {avg_last_10:.1f}")
    print(f"Improvement: {avg_last_10 - avg_first_10:.1f}")

    # Explain PPO
    print("\n" + "=" * 60)
    print(" PPO EXPLAINED")
    print("=" * 60)
    print(f"""
What just happened:

1. EPISODE COLLECTION
   Agent interacted with environment
   Stored: states, actions, rewards, log probs, values

2. ADVANTAGE COMPUTATION (GAE)
   For each step, computed "how much better than expected"
   λ={args.lam} balances bias/variance

3. PPO UPDATE
   Computed policy gradient with clipped objective
   Clip ε={args.clip_epsilon} prevents too large updates

Key PPO Components:

   ratio = π_new(a|s) / π_old(a|s)

   L^CLIP = min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)

   - If A > 0 (good action): ratio clipped at 1+ε (prevent overconfidence)
   - If A < 0 (bad action): ratio clipped at 1-ε (prevent overcorrection)

Why PPO for RLHF:
   - Stable training (no huge policy shifts)
   - Sample efficient (reuse trajectories)
   - Simple to implement and tune
   - Proven at scale (ChatGPT, Claude, etc.)
""")

    if HAS_GYM:
        env.close()


if __name__ == "__main__":
    main()
