# Import files and libs
from env import TetrisEnv
from tqdm import tqdm
import numpy as np
import random

class QLearningModel:
    def __init__(self):
        # Define env
        self.env = TetrisEnv()

        # Empty tables used for rewriting over the existing files (To reset)
        # Define the main tables
        self.states_index = []
        self.q_table = []

        self.write_model_to_file()
        # Load the model from files
        self.load_model_from_file()

        # Hyperparameters
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.1

        # Set training size
        self.training_size = 250
        self.total_trained = 0
        self.eval_episodes = 10

        # We train the model in 5 sections to save a gif of the best game at the end of that training
        for i in range(5):
            # Train the model
            print("[!] Training Q Learning Model")
            self.train()
            self.total_trained += self.training_size

            # Write to file
            self.write_model_to_file()

            # Evaluate and then render the best episode as gif
            print(f"[!] Evaluating Q Learning Model after {self.total_trained} training games.")
            self.gif_file_name = str(self.total_trained)

            print(len(self.q_table))
            self.evaluate(render_best=True)


    # Main function that trains the model
    def train(self):
        # Used TQDM for a progressbar when training
        for i in tqdm(range(self.training_size), leave=False):
            # Reset the env and define starter variables
            self.env.reset()
            reward, done = 0, False
            state = self.env.state()

            while not done:
                # Get state index
                state_index = self.get_state_index(state)

                # Explores action space
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()

                # Explots q learning
                else:
                    # Get normal state actions
                    actions = self.q_table[state_index]

                    # Get the action with the highest value
                    action = np.argmax(actions)

                # Take the next step in the env
                next_state, reward, done = self.env.step(action)
                next_state_index = self.get_state_index(next_state)

                # Old Q table values
                old_value = self.q_table[state_index][action]

                # Next Q table values
                next_max = max(self.q_table[next_state_index])

                # New values
                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

                # Update new values
                self.q_table[state_index][action] = new_value

                # Ready for next iteration
                state = next_state
    

    # Main function that evaluates the model
    def evaluate(self, render_best = False):
        # Variables to keep track of evaluation
        best_episode = None
        average = 0
        for episode in range(self.eval_episodes):
            # Reset the env and define starter variables
            self.env = TetrisEnv()
            state = self.env.reset()
            score, done = 0, False

            while not done:
                # Get table indexes for the given states
                state_index = self.get_state_index(state)

                # Get the actions for the given states
                actions = self.q_table[state_index]

                # Get the highest valued action
                action = np.argmax(actions)

                # Take the step in the env
                state, reward, done = self.env.step(action)

                # Apply reward
                score += reward

            # Add to the average
            average += score

            # Check if it's the best episode
            if best_episode == None:
                best_episode = [self.env, score]

            else:
                if score > best_episode[1]:
                    best_episode = [self.env, score]

        
        # Calculate and print average score
        average = int(average / self.eval_episodes)
        print(f"[!] Episodes average: {average}")

        # If render is True it will save the gif to the file name "TrainingSessionAmount_Score.gif"
        if render_best:
            print("[!] Best episode score: {}".format(best_episode[1]))
            best_episode[0].render(self.gif_file_name + "_" + str(best_episode[1]) + ".gif")

    
    # Get the index location of where the given state is located in the q_table
    def get_state_index(self, state):
        state_index = [i for i in range(len(self.states_index)) if state == self.states_index[i]]

        if state_index == []:
            self.states_index.append(state)
            state_index = len(self.states_index) - 1
            self.q_table.append([0 for i in range(self.env.action_space.n)])
        
        else:
            state_index = state_index[0]

        return state_index

    # Write the model to files
    def write_model_to_file(self):
        np.save("q_learning_files/states_index", self.states_index)
        np.save("q_learning_files/q_table", self.q_table)
        
    # Load the model from files
    def load_model_from_file(self):
        self.q_table = np.load("q_learning_files/q_table.npy", allow_pickle=True).tolist()
        self.states_index = np.load("q_learning_files/states_index.npy", allow_pickle=True).tolist()


QLearningModel()