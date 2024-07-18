##This function allows users to play against another person or against an agent. 
## Actions must be typed in command line. Interactive visualization included.

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import re
from Quentin_DQN import load_model, convert_state, rank_actions



class Play_Quentin:
    def __init__(self, size, agent=None, agent_black=True):
        self.size = size
        self.board = [-1] * (size * size)
        self.line_size = size
        self.agent = load_model(model_path=agent, state_size=size*size+1, action_size=size*size) if agent is not None else None
        self.agent_is_black = None if self.agent is None else agent_black
        self.fig, self.ax = plt.subplots()
        plt.ion()

    def __str__(self):
        rows = "abcdefghijklm"
        printed = "   " + " ".join(f"{i:2}" for i in range(self.size)) + "\n"
        for row in range(self.size):
            line = f"{rows[row]} "
            for col in range(self.size):
                value = self.board[row * self.size + col]
                if value == 0:
                    line += " B"
                elif value == 1:
                    line += " W"
                else:
                    line += " ."
            printed += line + "\n"
        return printed
    
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
        
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        plt.draw()
        plt.pause(0.01)

    def convert(self, values):
        converted_values = []
        for v in values:
            col = v % self.line_size
            row = v // self.line_size + 1
            converted_values.append(chr(ord('a') + row - 1) + str(col))
        return converted_values

    def gameover(self):
        winning_path = set()
        black_won = any(self.board[i] == 0 and self.winning_path_lookup(i, [], winning_path) for i in range(self.line_size))
        
        if black_won:
            print("BLACK WON!")
            print(self.convert(winning_path))
            # print(winning_path)
            return 0

        white_won = any(self.board[i * self.line_size] == 1 and self.winning_path_lookup(i * self.line_size, [], winning_path) for i in range(self.line_size))

        if white_won:
            print("WHITE WON!")
            print(self.convert(winning_path))
            # print(winning_path)
            return 1

        return -1

    def winning_path_lookup(self, idx, exclude, winning_path):
        if idx in winning_path:
            return False
        winning_path.add(idx)
        neighbors = self.neighbours(idx, exclude)
        arrival = list(range(self.size * (self.size - 1), self.size * self.size)) if self.board[idx] == 0 else list(range(self.size - 1, self.size * self.size, self.size))
        if idx in arrival:
            return True
        for neighbor in neighbors:
            if self.board[neighbor] == self.board[idx] and self.winning_path_lookup(neighbor, exclude, winning_path):
                return True
        winning_path.remove(idx)
        return False

    def neighbours(self, idx, exclude):
        mylist = []
        if idx + self.line_size < len(self.board) and idx + self.line_size not in exclude:
            mylist.append(idx + self.line_size)
        if idx + 1 < len(self.board) and idx // self.line_size == (idx + 1) // self.line_size and idx + 1 not in exclude:
            mylist.append(idx + 1)
        if idx - self.line_size >= 0 and idx - self.line_size not in exclude:
            mylist.append(idx - self.line_size)
        if idx - 1 >= 0 and idx // self.line_size == (idx - 1) // self.line_size and idx - 1 not in exclude:
            mylist.append(idx - 1)
        return mylist

    def diagonals(self, idx):
        mylist = []
        if idx + self.line_size + 1 < len(self.board) and (idx + 1) // self.line_size == idx // self.line_size:
            mylist.append(idx + self.line_size + 1)
        if idx + self.line_size - 1 < len(self.board) and (idx - 1) // self.line_size == idx // self.line_size and idx != 0:
            mylist.append(idx + self.line_size - 1)
        if idx - self.line_size - 1 >= 0 and (idx - 1) // self.line_size == idx // self.line_size:
            mylist.append(idx - self.line_size - 1)
        if idx - self.line_size + 1 >= 0 and (idx + 1) // self.line_size == idx // self.line_size:
            mylist.append(idx - self.line_size + 1)
        return mylist

    def next_move(self, is_black, illegal):        
        filled_locations = [i for i, x in enumerate(self.board) if x != -1]
        valid_move = False
        
        if self.agent_is_black == is_black:
            state = convert_state(self.board, self.size, not is_black)
            ranked_actions, q_values = rank_actions(self.agent, state)
            i = -1
            while not valid_move:
                i = i+1
                next_best = ranked_actions[i]
                valid_move = next_best not in list(set(filled_locations) | set(illegal)) and 0 <= next_best < len(self.board)
                if valid_move:
                    self.board[next_best] = 0 if is_black else 1
            return next_best
        
        else:
            pattern = r"[a-zA-Z]\d+"
            while not valid_move:
                user_input = input("Enter a position in the board (e.g.: a0): ")
                if not re.match(pattern, user_input):
                    print("Incorrect format.")
                    continue
                row = ord(user_input.lower()[0]) - ord('a')
                col = int(user_input[1:])
                location = row * self.line_size + col
                valid_move = location not in list(set(filled_locations) | set(illegal)) and 0 <= location < len(self.board) and col < self.line_size
                if valid_move:
                    self.board[location] = 0 if is_black else 1
                else:
                    print("Invalid location.")
            return location

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
                print("Illegal move.")
                illegal_moves.append(last_move)
            else:
                break

    def legal_move(self, last_move):
        for i in self.neighbours(last_move, []):
            if self.board[i] == self.board[last_move]:
                return True
        for i in self.diagonals(last_move):
            if self.board[i] == self.board[last_move]:
                return False
        return True

    def adiacent_location(self, region, idx):
        region_neighbours = set()
        for curr in region:
            region_neighbours.update(self.neighbours(curr, []))
            region_neighbours.difference_update([i for i in region_neighbours if self.board[i] != -1])
        extended_idx_neighbours = set(self.neighbours(idx, []))
        extended_idx_neighbours.difference_update([i for i in extended_idx_neighbours if self.board[i] != -1])
        extended_idx_neighbours.add(idx)
        temp_set = extended_idx_neighbours.intersection(region_neighbours)
        if temp_set:
            return True
        while True:
            for x in extended_idx_neighbours:
                temp_set.update(self.neighbours(x, []))
                temp_set.difference_update([i for i in temp_set if self.board[i] != -1])
            if extended_idx_neighbours == temp_set:
                return False
            extended_idx_neighbours = temp_set
            temp_set = extended_idx_neighbours.intersection(region_neighbours)
            if temp_set:
                return True

    def candidate_for_territory(self, region):
        for curr in region:
            cnt = sum(1 for i in self.neighbours(curr, []) if self.board[i] != -1)
            if cnt < 2:
                return False
        return True

    def fill_territory(self, region, last_move):
        neighbours_union = set()
        for i in region:
            neighbours_union.update(self.neighbours(i, []))
        neighbours_union.difference_update(region)
        distinct_neighbours = list(neighbours_union)
        cnt_black = sum(1 for i in distinct_neighbours if self.board[i] == 0)
        cnt_white = sum(1 for i in distinct_neighbours if self.board[i] == 1)
        if cnt_black == cnt_white:
            if self.board[last_move] == 1:
                cnt_black += 1
            else:
                cnt_white += 1
        replacement = 0 if cnt_black > cnt_white else 1
        for index in region:
            self.board[index] = replacement


    def play(self):
        black_string = "Player 1:" if self.agent_is_black is not True else ""
        white_string = "Player 2:" if self.agent_is_black is not False else ""
        while True:
            print(black_string)
            self.update_board(True)
            self.plot_board()
            if self.gameover() != -1:
                break

            print(white_string)
            self.update_board(False)
            self.plot_board()
            if self.gameover() != -1:
                break




def main():
    size = 5

    num_players = input("Number of players: ")
    if int(num_players) == 1:
        player_color = input("Choose your color (0=black, 1=white): ")
        if int(player_color) == 1:
            game = Play_Quentin(size, "quentin_sz7_ep25__black", agent_black=True)
        else:
            game = Play_Quentin(size, "quentin_sz7_ep25__white", agent_black=False)
    else:
        game = Play_Quentin(size)
    
    game.plot_board()
    game.play()

if __name__ == "__main__":
    main()
