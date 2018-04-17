import numpy as np
import gym
import tensorflow as tf

env = gym.make('ReinforcementLearnig-v0')
env.reset()
H = 50
batch_size = 25
gamma = 0.99
D = 4
learning_rate = 1e-1
tf.reset_default_graph()


def train():

    observations = tf.placeholder("float", [None, D], name='input_x')
    W1 = tf.get_variable("W1", shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
    layer1 = tf.nn.relu(tf.matmul(observations, W1))
    W2 = tf.get_variable("W2", shape=[H, 1], initializer=tf.contrib.layers.xavier_initializer())
    probability = tf.nn.sigmoid(tf.matmul(layer1, W2))

    tvars = tf.trainable_variables()
    input_y = tf.placeholder("float", [None, 1], name="input_y")
    advantages = tf.placeholder("float", name="reward_signal")

    loglik = tf.log(input_y*(input_y - probability)) + (1 - input_y)*(input_y + probability)
    loss = -1 * tf.reduce_mean(loglik * advantages)
    newGrads = tf.gradients(loss, tvars)

    adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
    W1Grad = tf.placeholder("float", name="batch_grad1")
    W2Grad = tf.placeholder("float", name="batch_grad2")
    batchGrad = [W1Grad, W2Grad]
    updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

    xs, ys, drs = [], [], []
    reward_sum = 0
    episode_num = 1
    total_episode = 10000
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        rendering = False
        sess.run(init)
        observation = env.reset()
        gradBuffer = sess.run(tvars)
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        while episode_num <= total_episode:
            if reward_sum/batch_size > 100 or rendering is True:
                env.render()
                rendering = True
            x = np.reshape(observation, [1, D])
            tfprob = sess.run(probability, feed_dict={observations: x})
            action = 1 if np.random.uniform() < tfprob else 0
            xs.append(x)
            y = 1 - action
            ys.append(y)

            observation, reward, done, info = env.step(action)
            reward_sum += reward
            drs.append(reward)
            if done:
                episode_num += 1
                epx = np.vstack(xs)
                epy = np.vstack(ys)
                epr = np.vstack(drs)
                xs, ys, drs = [], [], []
                discounted_epr = discount_reward(epr)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)

                tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
                for ix, grad in enumerate(tGrad):
                    gradBuffer[ix] += grad

                if episode_num & batch_size == 0:
                    sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                    print('Average reward for episode %d : %f. ' % (episode_num, reward_sum/batch_size))

                    if reward_sum / batch_size > 200:
                        print("Task solved in ", episode_num, "episode!")
                        break

                observation = env.reset()


def discount_reward(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


if __name__ == '__main__':
    train()