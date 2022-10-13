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


def run_algo(rew_avg, m, num_iter, num_trial) -> np.ndarray:
    regret = np.zeros((num_trial, num_iter))
    k = rew_avg.size
    max_arm = np.argmax(rew_avg)

    for i in range(num_trial):
        means = np.zeros(rew_avg.size)
        num = np.zeros(rew_avg.size)
        cum = [0]

        if (i + 1) % 10 == 0:
            print('Trial number = ', i + 1)

        for t in range(num_iter - 1):
            rew = get_reward(rew_avg)
            # Choose arm
            if t <= m * k:
                chosen_arm = t % k
                # Calculate mean of each arm
                num[chosen_arm] += 1
                means[chosen_arm] += (rew[chosen_arm] -
                                      means[chosen_arm]) / num[chosen_arm]
            else:
                chosen_arm = np.argmax(means)
            # Add regret to cumulation
            reg = rew[max_arm] - rew[chosen_arm]
            reg += cum[-1]
            cum.append(reg)
        regret[i, :] = np.asarray(cum)

    return regret


if __name__ == '__main__':
    # Initialize experiment parameters.
    rew_avg = np.asarray([0.6, 0.9, 0.95, 0.8, 0.7, 0.3])
    num_iter, num_trial = int(1e4), 30
    m = [10, 100, 200, 500, 1000]

    # Choose optimal m
    sorted_rew = np.sort(rew_avg)
    delta = sorted_rew[-1] - sorted_rew[0]
    m_optim = max(1, math.ceil((4 / delta ** 2) *
                  (math.log(num_iter * delta ** 2 / 4))))
    m.append(m_optim)
    m.sort()

    # Run experiments.
    regrets = []
    for m_val in m:
        reg = run_algo(rew_avg, m_val, num_iter, num_trial)
        avg_reg = np.mean(reg, axis=0)
        regrets.append(avg_reg)

    # Plot results.
    for m_val, reg in zip(m, regrets):
        plt.plot(reg, label="m=" + str(m_val))

    plt.xlabel('iterations')
    plt.ylabel('cumulative regret')
    plt.title('Cumulative Regret with ETC Bandit')
    plt.legend()
    plt.show()
