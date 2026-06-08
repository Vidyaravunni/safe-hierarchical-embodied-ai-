import keras

import random
import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
import os
from keras.regularizers import l2
np.random.seed(97)


class DDQN:
    """ Implementation of deep q learning algorithm """

    def __init__(self, state_space, action_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .95
        self.batch_size = 64
        self.epsilon_min = 0.05  # Keep some exploration
        self.epsilon_decay = 0.99  # Slower decay (per episode now, not per step)
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.target_net = self.build_model()
        #print(self.model.summary())
        self.initializeModel()
        self.lossPlot = []

    def initializeModel(self):
        if os.path.exists('ddqn_Reward/ddqn_weights'):
            self.target_net.load_weights('ddqn_Reward/ddqn_weights')
            self.model.load_weights('ddqn_Reward/ddqn_weights')
            self.epsilon = 0.01


    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=(self.state_space,), activation='relu'))
        # model.add(BatchNormalization())  # Add batch normalization layer here
        model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
        # model.add(BatchNormalization())  # Add batch normalization layer here
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def loadModel(self):
        if (os.path.exists('5_step/ddqn_weights')):
            self.model.load_weights('5_step/ddqn_weights')
            self.model.epsilon = 0.01

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            val =  random.randrange(self.action_space)
            return val
        # state = np.squeeze(state)
        # state = np.reshape(state, ( 1, self.state_space))
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, minibatch, ae_weight=0.0):
        """
        Experience replay for DDQN training.

        Args:
            minibatch: List of (state, action, reward, next_state, done, predReward) tuples
            ae_weight: Weight for Anticipation Engine prediction (0.0 = disabled)
        """
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        predReward = np.array([i[5] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        q = self.model.predict(states, verbose=0)
        next_q = self.model.predict(next_states, verbose=0)
        target_q = self.target_net.predict(next_states, verbose=0)

        for i in range(states.shape[0]):
            if dones[i]:
                # Terminal state: Q = immediate reward only
                q[i, actions[i]] = rewards[i]
            else:
                # Non-terminal: Q = reward + gamma * max(Q_target)
                next_best_action = np.argmax(next_q[i, :])
                Q_tar = rewards[i] + self.gamma * target_q[i, next_best_action]

                # Optionally add AE prediction (weighted)
                if ae_weight > 0:
                    Q_ae = predReward[i]
                    q[i, actions[i]] = Q_tar + ae_weight * Q_ae
                else:
                    q[i, actions[i]] = Q_tar

        self.model.fit(states, q, epochs=1, verbose=0)

        # NOTE: Epsilon decay moved to decay_epsilon() - called per episode, not per step

    def decay_epsilon(self):
        """Decay exploration rate. Call once per episode, not per step."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update(self):
        self.target_net.set_weights(self.model.get_weights())
        # temp = self.model
        # self.model = self.target_net
        # self.target_net = temp
