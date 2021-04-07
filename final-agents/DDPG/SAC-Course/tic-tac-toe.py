import random
import numpy as np


class Agent():
    def __init__(self, player1=True, epsilon=1, eps_dec=0.000001, eps_min=0.02):
        if player1:
            self.marker = 1
        else:
            self.marker = -1
        self.epsilon = epsilon # takes random move (epsilon%) of the time
        self.epsilon_min = eps_min
        self.epsilon_dec = eps_dec

        self.Q_table = {} # tracks (state, action) tuples to track the value of an agent taking x in action in a state
        self.visited = [] # tracks visited states in an episode

    # Updates the agent epsilon (called at the end of each episode)
    def update_epsilon(self):
        self.epsilon -= self.epsilon_dec
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def select_action(self, state):
        t_state = tuple(state)
        action = 0
        legal_moves = [idx for idx, i in enumerate(t_state) if i == 0]
        if (t_state, 0) not in self.Q_table: # adding states to the table as agents come to them
            for move in range(9):
                if (t_state, move) not in self.Q_table:
                    self.Q_table[(t_state, move)] = 0.05 # adding possible moves

        if random.random() > self.epsilon: # if choosing 'optimal' move
            action_scores = [self.Q_table[t_state, move] for move in legal_moves] # get each legal action
            action_pos = np.argmax(action_scores) # get the best 'legal' action
            action = legal_moves[action_pos] # find which action it relates to
        else: # if random move
            action = random.choice(legal_moves)
        self.visited.append((t_state, action)) # store a list of 'visited' nodes for the episode
        return action


    def learn(self, winner):
        # self.marker is -1 for O player and 1 for X player
        # .. if O player gets reward of -1, can be inverted based on marker
        # .. if X player gets reward of -1, it will be negative

        # Episode reward
        reward = 0
        if self.marker == winner:
            reward = 1 # reward for a win
        elif self.marker == -winner: # episode was lost by current agent
            reward = -1 # reward for a loss
        else:
            reward = 0 # reward for a draw

        # Update states before the end
        for idx, (state, action) in enumerate(self.visited[:-1]):
            next_state, _ = self.visited[idx+1]
            max_Q = max([self.Q_table[(next_state, a)] for a in range(9)])
            self.Q_table[(state, action)] = 0.9 * self.Q_table[(state, action)] + 0.1 * (reward + 0.97 * max_Q - self.Q_table[(state, action)])

        # Update the last state
        (last_state, last_action) = self.visited[-1]  # check update
        self.Q_table[(last_state, last_action)] = 0.9 * self.Q_table[(last_state, last_action)] + 0.1 * (reward - self.Q_table[(last_state, last_action)])

        # clearing states visited for the episode
        self.visited = []


class Board():
    def __init__(self):
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def reset(self):
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def placePiece(self, position, player1=True):
        if self.board[position] != 0:
            print("Invalid move made")

        if player1:
            self.board[position] = 1
        else:
            self.board[position] = -1

    def __str__(self):
        def lamba(x):
            if x == 1: return 'x'
            elif x == -1: return 'o'
            else: return '_'

        rep = list(map(lamba, self.board))
        rep.insert(3, '\n')
        rep.insert(7, '\n')

        return ''.join(rep)

    def gameDone(self):
        if self.board.count(0) == 0:
            return True
        elif self.gameWinner() != 0:
            return True
        else:
            return False

    def gameWinner(self): # since these columns need to go through pos 4, can remove != 0 check for each
        if self.board[4] != 0:
            if self.board[0] == self.board[4] == self.board[8]: # Diagonal
                return self.board[4]
            elif self.board[2] == self.board[4] == self.board[6]: # Diagonal
                return self.board[4]
            elif self.board[3] == self.board[4] == self.board[5]: # Row
                return self.board[4]
            elif self.board[1] == self.board[4] == self.board[7]: # Column
                return self.board[4]

        if self.board[0] != 0:
            if self.board[0] == self.board[1] == self.board[2]: # Row
                return self.board[0]
            elif self.board[0] == self.board[3] == self.board[6]: # Column
                return self.board[0]

        if self.board[8] != 0:
            if self.board[6] == self.board[7] == self.board[8]: # Row
                return self.board[8]
            elif self.board[2] == self.board[5] == self.board[8]: # Column
                return self.board[8]
        return 0


if __name__ == '__main__':
    player1 = Agent() # player 1 plays X's
    player2 = Agent(player1=False) # player 2 plays O's
    players = [player1, player2]
    wins = [] # [player1 wins, player2 wins, draws]

    board = Board()

    n_games = 1000000
    for game in range(n_games):
        board.reset() # reset board
        while not board.gameDone():
            board.placePiece(player1.select_action(board.board))
            if board.gameDone():
                break
            board.placePiece(player2.select_action(board.board), player1=False)
        winner = board.gameWinner() # player1 win: 1, draw: 0, player2 win: -1
        [player.learn(winner) for player in players] # learn from experiences for both players
        [player.update_epsilon() for player in players] # update epsilon for both agents

        wins.append(winner)

        if game % 10000 == 0 and game != 0:
            print("Played " + str(game) + " games")
            print("X wins for last 10000 games are {}".format((wins[-10000:].count(1) / 10000) * 100))
            print("O wins for last 10000 games are {}".format((wins[-10000:].count(-1) / 10000) * 100))
            print("Draws for last 10000 games are {}".format((wins[-10000:].count(0) / 10000) * 100))
            print("Player epsilon is {}".format(player1.epsilon))
            print("")
"""
        if game > 990000:
            print(board)
            print("winner is " + str(wins[-1]))
            print("")
"""
