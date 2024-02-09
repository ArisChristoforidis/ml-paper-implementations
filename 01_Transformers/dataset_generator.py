import random as rnd
from tqdm import tqdm
import csv
def main():
    data = []
    data_file = open('data.txt', 'w')
    coins = [(2, '2E'), (1, '1E'), (0.5, '50c'), (0.2, '20c') , (0.1, '10c'), (0.05, '5c'), (0.02, '2c'),  (0.01, '1c')]
    for _ in tqdm(range(100000)):
        n_selections = rnd.randint(2, 5)
        selections = rnd.sample(coins, n_selections)
        coin_strings = []
        coin_sum = 0
        for selection in selections:
            amount = rnd.randint(1, 20)
            coin_sum += amount * selection[0]
            coin_strings.append(f'{amount} {selection[1]}')
        coin_str = ', '.join(coin_strings)
        if coin_sum >= 1:
            sum_str = f"{coin_sum:.2f}E"
        else:
            sum_str = f"{coin_sum*100:.2f}c"
        data.append(f"{coin_str}: {sum_str}\n")
    
    data_file.writelines(data)
    data_file.close()


if __name__ == "__main__":
    main()