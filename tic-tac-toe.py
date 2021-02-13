import random
import numpy as np
import time
from itertools import permutations

class Agent():
    def __init__(self, player1=True, epsilon=1, eps_dec=0.0000007, eps_min=0.04):
        if player1:
            self.marker = 1
        else:
            self.marker = -1
        self.epsilon = epsilon # takes random move (epsilon%) of the time
        self.epsilon_min = eps_min
        self.epsilon_dec = eps_dec

        self.Q_table = {}
        self.returns = {}
        self.visited = []

    # Updates the agent epsilon (called once per episode)
    def update_epsilon(self):
        self.epsilon -= self.epsilon_dec
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def select_action(self, state):
        t_state = tuple(state)
        action = 0
        if (t_state, 0) not in self.Q_table:
            for move in range(9):
                if (t_state, move) not in self.Q_table:
                    self.Q_table[(t_state, move)] = 0 # adding all possible moves to dict
                    self.returns[(t_state, move)] = []

        if random.random() > self.epsilon: # if choosing best move
            # get each action from the dictionary
            legal_moves = [idx for idx, i in enumerate(t_state) if i == 0]
            action_scores = [self.Q_table[t_state, move] for move in legal_moves]
            action = np.argmax(action_scores)
            self.visited.append((t_state, action))
        else: # if random move and in q_table already
            legal_moves = [idx for idx, i in enumerate(t_state) if i == 0]
            action = random.choice(legal_moves)
            self.visited.append((t_state, action))
        return action

    def learn(self, winner):
        # get the episode reward
        reward = 0
        if self.marker == winner: # episode was won by current agent
            reward = 1
        elif self.marker == -winner: # episode was lost by current agent
            reward = -1

        # discount for future steps
        discount = 0.92

        for idx, (state, action) in enumerate(self.visited[::-1]): # go from end to start
            state_return = reward * round(discount ** idx, 3) # discount reduces for each step (round to 3dp)
            self.returns[(state, action)].append(state_return)

        for state, action in self.visited:
            self.Q_table[(state, action)] = np.mean(self.returns[(state, action)])
            if len(self.returns[(state, action)]) >= 12000: # trimming returns for performance
                self.returns[(state, action)] = self.returns[(state, action)][1000:]
                print(len(self.returns[(state, action)]))

        self.visited = []


class Board():
    def __init__(self):
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def reset(self):
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def placePiece(self, position, player1=True):
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

    def gameWinner(self):
        # since these columns need to go through pos 4, can remove != 0 check for each
        if self.board[4] != 0:
            if self.board[0] == self.board[4] == self.board[8]: # Diagonal
                return self.board[0]
            elif self.board[2] == self.board[4] == self.board[6]: # Diagonal
                return self.board[2]
            elif self.board[3] == self.board[4] == self.board[5]: # Row
                return self.board[6]
            elif self.board[1] == self.board[4] == self.board[7]: # Column
                return self.board[1]

        if self.board[0] != 0:
            if self.board[0] == self.board[1] == self.board[2]: # Row
                return self.board[0]
            elif self.board[0] == self.board[3] == self.board[6]: # Column
                return self.board[0]

        if self.board[8] != 0:
            if self.board[6] == self.board[7] == self.board[8]: # Row
                return self.board[6]
            elif self.board[2] == self.board[5] == self.board[8]: # Column
                return self.board[2]
        return 0 # otherwise 0


if __name__ == '__main__':
    player1 = Agent()
    player2 = Agent(player1=False)
    players = [player1, player2]
    x_wins = 0
    o_wins = 0
    draws = 0
    wins = [] # [player1 wins, player2 wins, draws]

    board = Board()

    n_games = 5000000
    for game in range(n_games):
        board.reset()
        while not board.gameDone():
            board.placePiece(player1.select_action(board.board))
            if board.gameDone():
                break
            board.placePiece(player2.select_action(board.board), player1=False)
        winner = board.gameWinner() # 1 if player1 won, 0 if noone won and -1 if player2 won
        [player.learn(winner) for player in players] # learn from experiences for both players
        [player.update_epsilon() for player in players] # update epsilon for both agents

        wins.append(winner)
        if winner == 0:
            draws+=1
        elif winner == -1:
            o_wins += 1
        else:
            x_wins += 1

        if game % 10000 == 0 and game != 0:
            print("Played " + str(game) + " games")
            print("X wins for last 10000 games are {}".format((wins[-10000:].count(1) / 10000) * 100))
            print("O wins for last 10000 games are {}".format((wins[-10000:].count(-1) / 10000) * 100))
            print("Draws for last 10000 games are {}".format((wins[-10000:].count(0) / 10000) * 100))
            print("")
            print("Total X wins are {}".format((x_wins/game)*100))
            print("Total O wins are {}".format((o_wins/game)*100))
            print("Total draws are {}".format((draws/game)*100))
            print("Player epsilon is {}".format(player1.epsilon))
            print("\n")

        if game > 499000:
            print(board)
            print("winner is " + str(wins[-1]))
            print("")

    print("total games are {}, total X wins are {}, total O wins are {}, total draws are {}".format(n_games, wins[0], wins[1], wins[2]))
