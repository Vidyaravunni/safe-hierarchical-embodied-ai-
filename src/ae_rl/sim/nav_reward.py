import numpy as np
from keras import Model, Input
from keras.layers import Dense, concatenate
from keras.regularizers import l2


class RewardModel:
    """
    Reward model for kitchen navigation.

    Design principles:
    1. Progress-based: Reward moving toward target
    2. Sparse success: Large reward for reaching target
    3. Gentle penalties: Small collision penalty, no episode termination
    4. Exploration bonus: Reward visiting new areas
    """

    def __init__(self):
        self.state_space = 75  # 72 (6x12 grid) + 3 (target info)
        self.validation_split = 0.2
        self.num_epochs = 10
        self.reward = 0
        self.epsilon = 1
        self.gamma = 0.95
        self.batch_size = 64
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.buildModel()

        # ========== REWARD CONFIGURATION ==========
        # These values are tuned for kitchen navigation

        # Success reward (reaching target)
        self.reward_success = 10.0

        # Progress rewards
        self.reward_progress = 0.5        # Moving closer to target
        self.reward_regression = -0.3     # Moving away from target

        # Visibility rewards
        self.reward_target_visible = 0.2  # Target in view
        self.reward_target_found = 2.0    # First time seeing target

        # Collision penalty (NOT terminal)
        self.penalty_collision = -0.3

        # Step cost (encourages efficiency)
        self.penalty_step = -0.01

        # Exploration bonus
        self.reward_new_position = 0.1

        # Oscillation penalty (repeated back-forth)
        self.penalty_oscillation = -0.2

        # Tracking for exploration bonus
        self.visited_positions = set()
        self.recent_actions = []
        self.target_seen_this_episode = False

    def reset_episode(self):
        """Call at the start of each episode."""
        self.visited_positions = set()
        self.recent_actions = []
        self.target_seen_this_episode = False

    def buildModel(self):
        """Neural network for reward prediction (used by Anticipation Engine)."""
        input1 = Input(shape=(1,))  # action
        input2 = Input(shape=(self.state_space,))  # state
        input3 = Input(shape=(self.state_space,))  # next_state

        concat1 = concatenate([input1, input2, input3])

        dense1 = Dense(512, activation='relu')(concat1)
        dense2 = Dense(256, activation='relu')(dense1)
        output = Dense(1, activation='linear', kernel_regularizer=l2(0.001))(dense2)

        model = Model(inputs=[input1, input2, input3], outputs=output)
        model.compile(optimizer='adam', loss='mse')
        return model

    # ==========================================================================
    # NEW REWARD COMPUTATION FOR KITCHEN NAVIGATION
    # ==========================================================================

    def compute_navigation_reward(self, prev_distance, curr_distance,
                                   target_visible, action, collision,
                                   agent_position, reached_target):
        """
        Compute reward for navigation step.

        Args:
            prev_distance: Distance to target before action
            curr_distance: Distance to target after action
            target_visible: Whether target is currently visible
            action: Action taken (0-4)
            collision: Whether action resulted in collision
            agent_position: (x, z) tuple of agent position
            reached_target: Whether agent reached target (visible + close)

        Returns:
            float: Total reward for this step
        """
        reward = 0.0
        reward_breakdown = {}

        # 1. SUCCESS REWARD (highest priority)
        if reached_target:
            reward += self.reward_success
            reward_breakdown['success'] = self.reward_success
            return reward, reward_breakdown

        # 2. COLLISION PENALTY (but don't kill the episode)
        if collision:
            reward += self.penalty_collision
            reward_breakdown['collision'] = self.penalty_collision
            # Still continue with other rewards

        # 3. PROGRESS REWARD (most important learning signal)
        if prev_distance is not None and curr_distance is not None:
            distance_delta = prev_distance - curr_distance
            if distance_delta > 0.05:  # Moved closer (threshold to ignore noise)
                reward += self.reward_progress
                reward_breakdown['progress'] = self.reward_progress
            elif distance_delta < -0.05:  # Moved away
                reward += self.reward_regression
                reward_breakdown['regression'] = self.reward_regression

        # 4. TARGET VISIBILITY REWARDS
        if target_visible:
            reward += self.reward_target_visible
            reward_breakdown['target_visible'] = self.reward_target_visible

            # Bonus for first time seeing target
            if not self.target_seen_this_episode:
                reward += self.reward_target_found
                reward_breakdown['target_found'] = self.reward_target_found
                self.target_seen_this_episode = True

        # 5. EXPLORATION BONUS
        if agent_position is not None:
            pos_key = (round(agent_position[0], 1), round(agent_position[1], 1))
            if pos_key not in self.visited_positions:
                reward += self.reward_new_position
                reward_breakdown['exploration'] = self.reward_new_position
                self.visited_positions.add(pos_key)

        # 6. OSCILLATION PENALTY
        self.recent_actions.append(action)
        if len(self.recent_actions) > 4:
            self.recent_actions.pop(0)

        if len(self.recent_actions) >= 4:
            # Check for left-right rotation oscillation (RotateLeft=3, RotateRight=4)
            if self.recent_actions == [3, 4, 3, 4] or self.recent_actions == [4, 3, 4, 3]:
                reward += self.penalty_oscillation
                reward_breakdown['oscillation'] = self.penalty_oscillation
            # Check for left-right strafe oscillation (MoveLeft=0, MoveRight=2)
            elif self.recent_actions == [0, 2, 0, 2] or self.recent_actions == [2, 0, 2, 0]:
                reward += self.penalty_oscillation
                reward_breakdown['oscillation'] = self.penalty_oscillation

        # 7. STEP COST (small, encourages efficiency)
        reward += self.penalty_step
        reward_breakdown['step_cost'] = self.penalty_step

        return reward, reward_breakdown

    # ==========================================================================
    # LEGACY METHODS (kept for backward compatibility with existing code)
    # ==========================================================================

    def getReward(self, state, action):
        """
        Legacy method - returns minimal rewards.
        New code should use compute_navigation_reward() instead.
        """
        # Return near-zero values to not interfere if old code calls this
        return 0.0, 0.0, 0.0

    def Rg(self, state):
        """
        DEPRECATED: Grid penalty removed.
        This was the main source of negative reward accumulation.
        """
        return 0.0

    def Ra(self, action):
        """
        DEPRECATED: Action-based rewards removed.
        Progress-based rewards are now used instead.
        """
        return 0.0

    # ==========================================================================
    # Reward model prediction (for Anticipation Engine)
    # ==========================================================================

    def act(self, state, next_state, action):
        state = np.squeeze(state)
        action = np.squeeze(action)
        state = np.expand_dims(state, axis=0)
        action = np.expand_dims(action, axis=0)
        predicted_reward = self.model.predict([action, state, next_state], verbose=0)
        return predicted_reward

    def replay(self, states, actions, next_states, rewards):
        self.model.fit(
            [np.array(actions), np.array(states), np.array(next_states)],
            rewards,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            validation_split=self.validation_split,
            verbose=0
        )
