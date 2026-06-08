"""
Task DDQN — Double Deep Q-Network with LSTM for Task Execution RL.

Extends the navigation DDQN pattern with:
- LSTM layer for sequential memory (captures action history)
- LLM-guided action selection (bonus for following LLM suggestion)
- 10-action task space (pick, place, open, close, toggle, slice, cook, break, navigate, noop)

The LLM plan acts as an advisor — the RL agent gets a bonus for following
the suggestion but can deviate when the environment demands it.
"""

import os
import random
import numpy as np
import keras
from keras import Model, Input
from keras.layers import Dense, LSTM
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

np.random.seed(97)


class TaskDDQN:
    """DDQN with LSTM for task execution action selection."""

    def __init__(self, state_space: int, action_space: int, sequence_length: int = 5):
        """
        Args:
            state_space: Dimension of task state vector (90)
            action_space: Number of task actions (10)
            sequence_length: LSTM window size (number of past states)
        """
        self.state_space = state_space
        self.action_space = action_space
        self.sequence_length = sequence_length

        # Hyperparameters
        self.epsilon = 1.0
        self.gamma = 0.95
        self.batch_size = 32
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.98  # Faster decay for ~200 episodes
        self.learning_rate = 0.0005

        # LLM guidance weight (decays as agent learns)
        self.llm_bonus = 0.3

        # Build networks
        self.model = self.build_model()
        self.target_net = self.build_model()
        self.update_target()

        # Try loading saved weights
        self._try_load_weights()

        self.loss_history = []

    def build_model(self) -> Model:
        """
        Build LSTM-based Q-network.

        Architecture: Input(seq_len, state_dim) → LSTM(128) → Dense(256) →
                     Dense(128) → Dense(64) → Dense(action_space)
        """
        input_layer = Input(shape=(self.sequence_length, self.state_space))

        # LSTM for sequential memory
        lstm_out = LSTM(128)(input_layer)

        # Dense layers (matching navigation DDQN pattern)
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(lstm_out)
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
        output = Dense(self.action_space, activation='linear')(x)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state_sequence: np.ndarray, llm_suggested_action: int = None) -> int:
        """
        Select action using epsilon-greedy with LLM guidance.

        During exploration: 50% chance to follow LLM suggestion, 50% random.
        During exploitation: Q-values + bonus for LLM-suggested action.

        Args:
            state_sequence: Shape (1, sequence_length, state_space)
            llm_suggested_action: Action index the LLM plan suggests (or None)

        Returns:
            Selected action index
        """
        if np.random.rand() <= self.epsilon:
            # Exploration: strongly prefer LLM suggestion
            if llm_suggested_action is not None and np.random.rand() < 0.8:
                return llm_suggested_action
            return random.randrange(self.action_space)

        # Exploitation: Q-values with LLM bonus
        q_values = self.model.predict(state_sequence, verbose=0)[0]

        if llm_suggested_action is not None:
            # Scale bonus by how much training we've done
            # High bonus early (agent untrained), decays as Q-values improve
            effective_bonus = self.llm_bonus + max(0, self.epsilon * 5.0)
            q_values[llm_suggested_action] += effective_bonus

        return int(np.argmax(q_values))

    def replay(self, minibatch):
        """
        Experience replay with Double DQN targets.

        Args:
            minibatch: List of (state_seq, action, reward, next_state_seq, done) tuples
        """
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        # Squeeze extra dims if present
        if states.ndim == 4:
            states = np.squeeze(states, axis=1)
        if next_states.ndim == 4:
            next_states = np.squeeze(next_states, axis=1)

        # Double DQN: use model to select action, target_net to evaluate
        q = self.model.predict(states, verbose=0)
        next_q = self.model.predict(next_states, verbose=0)
        target_q = self.target_net.predict(next_states, verbose=0)

        for i in range(len(minibatch)):
            if dones[i]:
                q[i, actions[i]] = rewards[i]
            else:
                best_action = np.argmax(next_q[i])
                q[i, actions[i]] = rewards[i] + self.gamma * target_q[i, best_action]

        history = self.model.fit(states, q, epochs=1, verbose=0)
        self.loss_history.append(history.history['loss'][0])

    def decay_epsilon(self):
        """Decay exploration rate. Call once per episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        """Sync target network weights from model."""
        self.target_net.set_weights(self.model.get_weights())

    def save_weights(self, path: str = "outputs/weights/task_ddqn_weights"):
        """Save model weights."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save_weights(path)

    def load_weights(self, path: str = "outputs/weights/task_ddqn_weights"):
        """Load model weights."""
        status_main = self.model.load_weights(path)
        status_target = self.target_net.load_weights(path)
        # Silence expected partial-restoration warnings (optimizer slots are not used here).
        if hasattr(status_main, "expect_partial"):
            status_main.expect_partial()
        if hasattr(status_target, "expect_partial"):
            status_target.expect_partial()
        self.epsilon = self.epsilon_min

    def _try_load_weights(self):
        """Try to load previously saved weights."""
        paths = [
            "outputs/weights/task_ddqn_weights",
            "outputs/weights/task_ddqn_weights.index",
        ]
        for path in paths:
            base = path.replace(".index", "")
            if os.path.exists(path):
                try:
                    self.load_weights(base)
                    print(f"  Task DDQN weights loaded from: {base}")
                    return
                except Exception as e:
                    print(f"  Failed to load task DDQN weights: {e}")
