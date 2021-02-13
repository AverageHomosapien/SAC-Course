import random


# Player 1 is X, player 2 is O
INIT_WEIGHT = 10000

class boardState:
    def __init__(self, placements, children, weights, print_weights=False):
        self.placements = placements
        self.children = children
        self.weights = weights

        if print_weights:
            for weight in self.weights:
                print(weight)

    def __str__(self):
        def lamba(x):
            if x == 1: return 'x'
            if x == -1: return 'o'
            if x == 0: return '_'

        rep = list(map(lamba, self.placements))
        rep.insert(3, '\n')
        rep.insert(7, '\n')

        return ''.join(rep)

def checkWin(board):
    # Check rows
    if board[0] == board[1] == board[2] and board[0] != 0:
        return board[0]
    if board[3] == board[4] == board[5] and board[3] != 0:
        return board[6]
    if board[6] == board[7] == board[8] and board[6] != 0:
        return board[6]

    # Check columns
    if board[0] == board[3] == board[6] and board[0] != 0:
        return board[0]
    if board[1] == board[4] == board[7] and board[1] != 0:
        return board[1]
    if board[2] == board[5] == board[8] and board[2] != 0:
        return board[2]

    # Check diagonals
    if board[0] == board[4] == board[8] and board[0] != 0:
        return board[0]
    if board[2] == board[4] == board[6] and board[2] != 0:
        return board[2]

    # Otherwise return 0
    return 0

def populateMovelist(board, player1=True):
    # Check that we are not at a won game
    if checkWin(board.placements) != 0: return

    # Loop over positions in the board for empty ones
    for i in range(9):
        # If we find an empty one we place there
        if board.placements[i] == 0:
            child_placements = board.placements.copy()
            if player1:
                child_placements[i] = 1
            else:
                child_placements[i] = -1

            # We create the child object, and append it and its initial weight to our list
            child = boardState(child_placements, [], [])
            board.children.append(child)
            board.weights.append(INIT_WEIGHT)

            # Finally call this method for our child
            populateMovelist(child, not player1)

def makeMove(boardstate):
    """Asks the AI to make a move in a current board state"""
    move = random.choices(
        population=boardstate.children,
        weights=boardstate.weights,
        k=1
    )

    return move[0]

def playGame():
    moves_made = []

    current_player = ai_player

    # Go through the rest of the game
    while checkWin(current_player.placements) == 0:
        # Make a move and save it
        current_player = makeMove(current_player)
        moves_made.append(current_player)

        # Print board position
        #print(current_player)
        #print()

        # Check if we need to break at a draw
        if all(current_player.placements):
            break

    # Print that the game has ended, and update weights
    #print("End of game!")
    foo = ai_player
    for i in range(0,len(moves_made)):
        # Find the index of the move made
        weight_index = foo.children.index(moves_made[i])

        # If X won
        if checkWin(current_player.placements) == 1:
            if i % 2 == 0:
                foo.weights[weight_index] += 10
            else:
                foo.weights[weight_index] -= 10
                if foo.weights[weight_index] <= 0: foo.weights[weight_index] = 0

        # If O won
        if checkWin(current_player.placements) == -1:
            if i % 2 == 0:
                foo.weights[weight_index] -= 10
                if foo.weights[weight_index] <= 0: foo.weights[weight_index] = 0
            else:
                foo.weights[weight_index] += 10

        # If it was a draw
        if checkWin(current_player.placements) == 0:
            if i % 2 == 0:
                foo.weights[weight_index] += 5
            else:
                foo.weights[weight_index] += 5

        foo = foo.children[weight_index]

    return checkWin(current_player.placements)


# Create initial player
board = [0,0,0,0,0,0,0,0,0]
ai_player = boardState(board, [], [])

# Recursively fill out possible game states
populateMovelist(ai_player)

x_wins = 0
o_wins = 0
draws = 0
games = 0

while True:
    winner = playGame()
    if winner == 1: x_wins += 1
    if winner == -1: o_wins += 1
    if winner == 0: draws += 1
    games += 1

    if games % 10000 == 0:
        print(x_wins / games * 100)
        print(o_wins / games * 100)
        print(draws / games * 100)
        print(games)
        print()

    if games % 50000 == 0:
        break

    
