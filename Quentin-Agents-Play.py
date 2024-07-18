##This class allows you to test the agents among them, and provides a interactive visualization during the game.

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from Quentin_DQN import load_model, convert_state, rank_actions
from Quentin import QuentinGame


class Play_Quentin(QuentinGame):
    def __init__(self, size, agent_black=None, agent_white=None):
        super().__init__(size)
        self.agent_black = load_model(model_path=agent_black, state_size=size*size+1, action_size=size*size) if agent_black is not None else None
        self.agent_white = load_model(model_path=agent_white, state_size=size*size+1, action_size=size*size) if agent_white is not None else None
        self.fig, self.ax = plt.subplots()
        plt.ion()
    
    ##visualize
    def plot_board(self):
        self.ax.clear()
        self.ax.set_xticks(np.arange(0, self.size, 1))
        self.ax.set_yticks(np.arange(0, self.size, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])  # Reverse y labels
        self.ax.grid(which='both', color='black', linestyle='-', linewidth=1)
        self.ax.set_xlim(-0.5, self.size-0.5)
        self.ax.set_ylim(-0.5, self.size-0.5)
        self.ax.set_aspect('equal')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # Add letters (A-M) to the left and right of the board
        for i in range(self.size):
            self.ax.text(-1, i, chr(ord('A') + self.size - i - 1), va='center', ha='center')  # Left side
            self.ax.text(self.size, i, chr(ord('A') + self.size - i - 1), va='center', ha='center')  # Right side
        # Add numbers (0-12) to the top and bottom of the board
        for i in range(self.size):
            self.ax.text(i, self.size, str(i), va='center', ha='center')  # Bottom side
            self.ax.text(i, -1, str(i), va='center', ha='center')  # Top side

        for idx, value in enumerate(self.board):
            row, col = divmod(idx, self.size)
            row = self.size-row-1
            if value != -1:
                color = 'black' if value == 0 else 'white'
                circle = Circle((col, row), 0.4, facecolor=color, edgecolor='black', linewidth=1)
                self.ax.add_patch(circle)      

        plt.draw()
        plt.pause(0.1)


    def next_move(self, is_black, illegal):        
        filled_locations = [i for i, x in enumerate(self.board) if x != -1]
        valid_move = False
        tmp = None

        if (self.agent_black is not None) == is_black:
            tmp = self.agent_black
        if (self.agent_white is not None) == (not is_black):
            tmp = self.agent_white
            
        if tmp is not None:
            state = convert_state(self.board, self.size, not is_black)
            ranked_actions, q_values = rank_actions(tmp, state)
            i = -1
            while not valid_move:
                i = i+1
                next_best = ranked_actions[i]
                valid_move = next_best not in list(set(filled_locations) | set(illegal)) and 0 <= next_best < len(self.board)
                if valid_move:
                    self.board[next_best] = 0 if is_black else 1
            return next_best


    def update_board(self, is_black):
        illegal_moves =  []
        while True:
            last_move = self.next_move(is_black, illegal_moves)
            empty_locations = [i for i, x in enumerate(self.board) if x == -1]
            region = []
            territories = []
            while empty_locations:
                for loc in empty_locations:
                    if not region or self.adiacent_location(region, loc):
                        region.append(loc)
                if self.candidate_for_territory(region):
                    self.fill_territory(region, last_move)
                    territories.extend(region)
                empty_locations = [x for x in empty_locations if x not in region]
                region.clear()
            if not self.legal_move(last_move):
                self.board[last_move] = -1
                for idx in territories:
                    self.board[idx] = -1
                territories.clear()
                # print("Illegal move.")
                illegal_moves.append(last_move)
            else:
                break
    

    def play(self):
        while True:
            # print(black_string)
            self.update_board(True)
            self.plot_board()
            if self.gameover() != -1:
                break
            # print(white_string)
            self.update_board(False)
            self.plot_board()
            if self.gameover() != -1:
                break


def main():
    size = 7

    game = Play_Quentin(size, agent_black="../quentin_sz7_ep25__black", agent_white="../quentin_sz7_ep25__white")
    game.plot_board()
    game.play()

if __name__ == "__main__":
    main()
