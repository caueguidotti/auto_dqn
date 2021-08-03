import os
import random
import numpy as np
from collections import deque
from tensorflow.keras import models, layers, optimizers, activations, losses
from tensorflow.keras.callbacks import TensorBoard


class DQNAgent:
    """
    Represents a Deep Q-Networks (DQN) agent.
    """

    def __init__(self, state_size, action_size, model_layers_dicts, gamma=0.95, epsilon=0.5, epsilon_min=0.01,
                 epsilon_decay=0.98, learning_rate=0.001, buffer_size=4098, model_name=None, verbose=False):
        """
        Creates a Deep Q-Networks (DQN) agent.

        :param state_size: number of dimensions of the feature vector of the state.
        :type state_size: int.
        :param action_size: number of actions.
        :type action_size: int.
        :param gamma: discount factor.
        :type gamma: float.
        :param epsilon: epsilon used in epsilon-greedy policy.
        :type epsilon: float.
        :param epsilon_min: minimum epsilon used in epsilon-greedy policy.
        :type epsilon_min: float.
        :param epsilon_decay: decay of epsilon per episode.
        :type epsilon_decay: float.
        :param learning_rate: learning rate of the action-value neural network.
        :type learning_rate: float.
        :param buffer_size: size of the experience replay buffer.
        :type buffer_size: int.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=buffer_size)  # giving a maximum length makes this buffer forget old memories
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.tensor_board = [TensorBoard(log_dir=os.path.join("logs", "{}".format(model_name)),
                                         histogram_freq=1, update_freq=1)] \
            if model_name is not None else None
        self.model = self.make_model(model_layers_dicts)

    def make_model(self, layers_dicts):
        """
        Makes the action-value neural network model using Keras.

        :param layers_dicts: list of dict containing the following keys: num_neurons, activation, activation_par_dict
        :return: action-value neural network.
        :rtype: Keras' model.
        """
        model = models.Sequential()

        for layer_n, layer_dict in enumerate(layers_dicts):
            num_neurons = layer_dict['num_neurons']
            activation = layer_dict['activation']
            activation_par_dict = layer_dict['activation_par_dict'] if 'activation_par_dict' in layer_dict else None

            if layer_n == 0:
                # first layer - inform input shape
                model.add(layers.Dense(num_neurons, input_shape=(self.state_size,)))
            else:
                model.add(layers.Dense(num_neurons))

            if activation in ['softplus', 'softsign', 'swish', 'relu', 'tanh', 'sigmoid', 'exponential', 'hard_sigmoid',
                              'linear', 'serialize', 'deserialize']:
                # regular activation functions
                if activation_par_dict is None:
                    activation_fn = lambda x: getattr(activations, activation)(x)
                else:
                    activation_fn = lambda x: getattr(activations, activation)(x, **activation_par_dict)
                model.add(layers.Activation(activation_fn))
            elif activation in ['LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU', 'Softmax', 'ReLU']:
                # advanced activation functions - used after linear activation
                model.add(layers.Activation(activations.linear))
                if activation_par_dict is None:
                    model.add(getattr(layers, activation)())
                else:
                    model.add(getattr(layers, activation)(**activation_par_dict))
            else:
                raise ValueError('Activation function not identified.')

        # layer last output shape is always equal to action size and activation is linear
        model.add(layers.Dense(self.action_size, activation=activations.linear))

        model.compile(loss=losses.mse,
                      optimizer=optimizers.Adam(lr=self.learning_rate))
        if self.verbose:
            model.summary()
        return model

    def act(self, state):
        """
        Chooses an action using an epsilon-greedy policy.

        :param state: current state.
        :type state: NumPy array with dimension (1, 2).
        :return: chosen action.
        :rtype: int.
        """
        # choose greedy with probability 1-epsilon
        if 1 - self.epsilon > np.random.uniform():
            # choose action with maximum return
            return self.model.predict(state)[0].argmax()
        else:
            # choose random action
            return np.random.randint(self.action_size)

    def append_experience(self, state, action, reward, next_state, done):
        """
        Appends a new experience to the replay buffer (and forget an old one if the buffer is full).

        :param state: state.
        :type state: NumPy array with dimension (1, 2).
        :param action: action.
        :type action: int.
        :param reward: reward.
        :type reward: float.
        :param next_state: next state.
        :type next_state: NumPy array with dimension (1, 2).
        :param done: if the simulation is over after this experience.
        :type done: bool.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        Learns from memorized experience.

        :param batch_size: size of the minibatch taken from the replay buffer.
        :type batch_size: int.
        :return: loss computed during the neural network training.
        :rtype: float.
        """
        minibatch = random.sample(self.replay_buffer, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if not done:
                target[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            else:
                target[0][action] = reward
            # Filtering out states and targets for training
            states.append(state[0])
            targets.append(target[0])
        history = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        return loss

    def load(self, folder, name):
        """
        Loads the neural network's weights from disk.

        param folder: folder where files will be saved
        :param name: model's name.
        :type folder: str.
        :type name: str.
        """
        self.model.load_weights(os.path.join(folder, name))

    def save(self, folder, model_file, weights_file):
        """
        Saves the neural network's model and weights to disk.

        :param folder: folder where files will be saved
        :param model_file: model file name
        :param weights_file: model's weights file name
        :type folder: str.
        :type model_file: str.
        :type model_file: str.
        """
        if not os.path.isdir(folder):
            os.mkdir(folder)
        try:
            self.model.save_weights(os.path.join(folder, weights_file))
        except:
            pass
        try:
            self.model.save(os.path.join(folder, model_file))
        except:
            pass

    def update_epsilon(self):
        """
        Updates the epsilon used for epsilon-greedy action selection.
        """
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
