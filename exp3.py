import math
import matplotlib.pyplot as plt
import numpy as np


def get_reward(rew_avg) -> np.ndarray:
    # Add epsilon (sub-gaussian noise) to reward.
    mean = np.zeros(rew_avg.size)
    cov = np.eye(rew_avg.size)
    epsilon = np.random.multivariate_normal(mean, cov)
    reward = rew_avg + epsilon

    return reward


def run_algo(rew_avg, eta, num_iter, num_trial) -> np.ndarray:
    regret = np.zeros((num_trial, num_iter))
    max_arm = np.argmax(rew_avg)

    for i in range(num_trial):
        means = np.zeros(rew_avg.size)
        S = np.zeros(rew_avg.size)
        cum = [0]
        probs_arr = np.ones(rew_avg.size) / rew_avg.size

        if (i + 1) % 10 == 0:
            print('Trial number = ', i + 1)

        for t in range(num_iter - 1):
            rew = get_reward(rew_avg)
            # Choose arm
            chosen_arm = np.random.choice(len(means), 1, p=probs_arr)
            # Add regret to cumulation
            reg = rew[max_arm] - rew[chosen_arm]
            reg += cum[-1]
            cum.append(reg)
            # Update prob distribution
            probs_arr = np.exp(eta * S) / np.sum(np.exp(eta * S))
            # Update total estimated reward
            for j in range(rew_avg.size):
                if chosen_arm == j:
                    S[j] += 1 - ((1 - rew[chosen_arm]) /
                                 probs_arr[chosen_arm])
                else:
                    S[j] += 1
        regret[i, :] = np.asarray(cum, dtype=object)

    return regret


if __name__ == '__main__':
    # Initialize experiment parameters.
    rew_avg = np.asarray([0.6, 0.9, 0.95, 0.8, 0.7, 0.3])
    num_iter, num_trial = int(1e4), 30
    eta = [1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 1e-2]

    # Run experiments.
    regrets = []
    for eta_val in eta:
        reg = run_algo(rew_avg, eta_val, num_iter, num_trial)
        avg_reg = np.mean(reg, axis=0)
        regrets.append(avg_reg)

    # Plot results.
    for eta_val, reg in zip(eta, regrets):
        plt.plot(reg, label="eta=" + str(eta_val))
    plt.xlabel('iterations')
    plt.ylabel('cumulative regret')
    plt.title('Cumulative Regret with EXP3 Bandit')
    plt.legend()
    plt.show()
