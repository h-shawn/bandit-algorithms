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


def run_algo(rew_avg, delta, num_iter, num_trial) -> np.ndarray:
    regret = np.zeros((num_trial, num_iter))
    k = rew_avg.size
    max_arm = np.argmax(rew_avg)

    # TODO: return a (num_trial x num_iter) matrix of cumulative regrets for each trial.
    for i in range(num_trial):
        means = np.zeros(rew_avg.size)
        num = np.zeros(rew_avg.size)
        cum = [0]
        time = 0.0
        ucb_arr = 1e5 * np.ones(k)

        if (i + 1) % 10 == 0:
            print('Trial number = ', i + 1)

        for t in range(num_iter - 1):
            rew = get_reward(rew_avg)
            chosen_arm = np.argmax(ucb_arr)

            reg = rew[max_arm] - rew[chosen_arm]
            reg += cum[-1]
            cum.append(reg)

            num[chosen_arm] += 1
            means[chosen_arm] = (means[chosen_arm] * (num[chosen_arm] - 1) + rew[chosen_arm]) / num[chosen_arm]
            time += 1

            func = 2 * np.log(1 / delta)
            for j in range(k):
                if num[j] == 0:
                    continue
                else:
                    ucb_arr[j] = means[j] + np.sqrt(func / num[j])
        regret[i, :] = np.asarray(cum)

    return regret


if __name__ == '__main__':
    # Initialize experiment parameters.
    rew_avg = np.asarray([0.6, 0.9, 0.95, 0.8, 0.7, 0.3])
    num_iter, num_trial = int(1e4), 30
    delta = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    # Run experiments.

    regrets = []
    for delta_val in delta:
        reg = run_algo(rew_avg, delta_val, num_iter, num_trial)
        avg_reg = np.mean(reg, axis=0)
        regrets.append(avg_reg)

    # Plot results.
    # TODO: plot cumulative regret vs. iteration number

    # Normal scale
    for delta_val, reg in zip(delta, regrets):
        plt.plot(reg, label="delta=" + str(delta_val))
    plt.xlabel('iterations')
    plt.ylabel('cumulative regret')
    plt.title('Cumulative Regret with UCB Bandit')
    plt.legend()
    plt.show()

    # Log scale x-axis
    for delta_val, reg in zip(delta, regrets):
        plt.semilogx(reg, label="delta=" + str(delta_val))
    plt.xlabel('iterations')
    plt.ylabel('cumulative regret')
    plt.title('Cumulative Regret with UCB Bandit (Semilogx)')
    plt.legend()
    plt.show()
