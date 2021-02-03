import random

## TODO: ace wrap (for dealer only)

class PlayerDealer():
    def __init__(self, dealer=False, stop_hitting=20):
        self.dealer = dealer
        self.stop_hitting = stop_hitting
        if not dealer:
            self.experience = [[], 0, False] # holds [[0, 2], 0, False]
        else:
            self.cards = [[], False]


    def deal(self, cards=1):
        """
        appends cards to the player and dealer's experience
        param cards (int): deals this many cards to the agent
        """
        for i in range(cards):
            new_card = card_dealt()
            if new_card == 11:
                if not self.dealer:
                    self.experience[2] = True
                else:
                    self.cards[1] = True
            if not self.dealer:
                self.experience[0].append(new_card)
            else:
                self.cards[0].append(new_card)

    def dealer_card(self, dealer_card):
        """
        Sets the dealer card for the player agent
        """
        self.experience[1] = dealer_card

    def reset(self):
        """
        resets dealer and player experience
        """
        if not self.dealer:
            self.experience = [[], 0, False]
        else:
            self.cards = [[], False]

    def policy(self):
        """
        policy for dealer - hits if < 16
        policy for player - hits if < 20
        returns True if hitting, or False if sticking
        """
        if not self.dealer:
            if sum(self.experience[0]) < self.stop_hitting:
                return True
            if self.experience[2] == True:
                idxs = [idx for idx, i in enumerate(self.experience[0]) if i == 11]
                if idxs == []:
                    return False
                else:
                    for pos in idxs:
                        self.experience[0][pos] = 1
                    self.experience[2] = False
                    return True
            return False
        if sum(self.cards[0]) < self.stop_hitting:
            return True
        return False


def card_dealt():
    """
    returns a card value (2-11) (all face cards are 10 and ace returned as 11)
    """
    card = random.randint(1,13)
    if card > 9:
        return 10
    elif card == 1:
        return 11
    return card

def state_vals(): # Stored in format STATE, TIMES-REACHED, AVG
    """
    create the state-value states
    """
    k = [True, False]
    for i in range(4, 22):
        for j in range(2, 12):
            for k in [True, False]:
                state_values.append([[i, j, k], 0, 0])

def update_experience(experience, score): #18 states for each number (4-21)
    """
    adds experiences to the state-value calculator
    """
    if sum(experience[0]) > 21:
        for i in range(1, len(experience)):
            score *= discount
            start_idx = (sum(experience[0][:-i]) - 4) * 20
            for j in range(start_idx, len(state_values)):
                if state_values[j][0] == [sum(experience[0][:-i]), experience[1], experience[2]]:
                    state_values[j][2] *= state_values[j][1]
                    state_values[j][2] += score
                    state_values[j][1] += 1
                    state_values[j][2] /= state_values[j][1]
                    break
    else:
        for i in range(0, len(experience[0])-1):
            val = sum(experience[0]) # [:-sum(experience[0])]
            start_idx = (val - 4) * 18
            for j in range(start_idx, len(state_values)):
                if state_values[j][0] == [val, experience[1], experience[2]]:
                    state_values[j][2] *= state_values[j][1]
                    state_values[j][2] += score
                    state_values[j][1] += 1
                    state_values[j][2] /= state_values[j][1]
                    break
            score *= discount


def calc_score(player_cards, dealer_cards):
    if sum(player_cards) > 21: # bust
        return -1
    elif sum(dealer_cards) > 21: # dealer bust
        return 1
    elif len(player_cards) >= 5: # full house
        return 1
    elif len(dealer_cards) >= 5: # full house
        return -1
    elif sum(player_cards) > sum(dealer_cards): # player beaten dealer
        return 1
    elif sum(player_cards) < sum(dealer_cards): # player lost to dealer (player 20, dealer 21)
        return -1
    else: return 0 # player equal to dealer

discount = 0.75
EPISODES = 500000
state_values = []
PLAYER_STOP_HITTING = 20
DEALER_STOP_HITTING = 16


if __name__ == '__main__':
    state_vals() # 306 states

    player = PlayerDealer(dealer=False, stop_hitting=PLAYER_STOP_HITTING)
    dealer = PlayerDealer(dealer=True, stop_hitting=DEALER_STOP_HITTING)
    players = [player, dealer]

    for i in range(EPISODES):
        [p.reset() for p in players]
        player.deal(2)
        dealer.deal()
        player.dealer_card(dealer.cards[0][0])
        # stupid blackjack
        while(player.policy()): # player deal all cards
            player.deal()
        while(dealer.policy()): # dealer deal all cards
            dealer.deal()
        score = calc_score(player.experience[0], dealer.cards[0])
        update_experience(player.experience, score)

    for state in state_values:
        print(state)
