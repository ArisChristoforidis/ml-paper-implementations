# Transformers

This is my personal implementation of the transformer architecture based on the original [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper. Some implementation details are taken from [Andrej Karpathy's GPT tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=950s).

The purpose of this project is __purely educational__ and is not meant to provide a performant implementation of the architecture. If you want to use transformers, you should use [PyTorch's native transformer class](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html), or a library like [Hugging Face's Transformers](https://huggingface.co/docs/transformers/en/index), which offers optimized implementations and a wide range of pre-trained models for various natural language processing tasks.

# Testing

I made a small synthetic dataset that consists of entries representing combinations of euro coins, along with their respective quantities and total sum. The general format for each sample is the following:

`quantity_1 x coin_1, quantity_2 x coin_2, ... , quantity_N * coin_N: sum of coins `

For example the line `2 2E, 3 50c: 5.50E` describes the following:
    
    - 2 * 2€
    - 3 * 0.5€ (50 cents)
    
    The sum is 5.50€

Note that the letters __E__ and __c__ are used to denote Euros(€) and cents respectively. 

__The proposed task is to guess the sum of the coins given the sequence of quantities for each coin.__

I also included a small train script (see `train.py`) that can train the Transformer on `data.txt`. I trained the transformer on a colab notebook and it behaves as expected.

```
Epoch 1: 100%|██████████| 2812/2812 [04:48<00:00,  9.74it/s]
Epoch 1 train loss: 0.816
Epoch 2: 100%|██████████| 2812/2812 [04:47<00:00,  9.79it/s]
Epoch 2 train loss: 0.727
Epoch 3: 100%|██████████| 2812/2812 [04:45<00:00,  9.83it/s]
Epoch 3 train loss: 0.645
Epoch 4: 100%|██████████| 2812/2812 [04:46<00:00,  9.81it/s]
Epoch 4 train loss: 0.584
Epoch 5: 100%|██████████| 2812/2812 [04:47<00:00,  9.77it/s]
Epoch 5 train loss: 0.541
Epoch 6: 100%|██████████| 2812/2812 [04:47<00:00,  9.78it/s]
Epoch 6 train loss: 0.511
Epoch 7: 100%|██████████| 2812/2812 [04:49<00:00,  9.72it/s]
Epoch 7 train loss: 0.492
Epoch 8: 100%|██████████| 2812/2812 [04:48<00:00,  9.74it/s]
Epoch 8 train loss: 0.469
Epoch 9: 100%|██████████| 2812/2812 [04:49<00:00,  9.71it/s]
Epoch 9 train loss: 0.453
Epoch 10: 100%|██████████| 2812/2812 [04:46<00:00,  9.80it/s]
Epoch 10 train loss: 0.434
```

# LICENCE

MIT