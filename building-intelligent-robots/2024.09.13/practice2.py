import random
import matplotlib.pyplot as plt

def play_once():
    if random.random() <=0.5:
        res = 1
    else:
        res = -1
    return res

def do_games(n):
    num_win = 0
    for i in range(n):
        num_win += play_once()
    return num_win

# n: the number of total games
# k: the number of minigames per game

def game_simulate(n,k):
    list_game_result = []
    for i in range(n):
        exp_winning = (1/k)*do_games(k)
        list_game_result.append(exp_winning)
    return list_game_result

list_game = game_simulate(1000,1000)
print(list_game)

plt.hist(list_game, bins=10)