from model import Transformer
from random import shuffle
from tqdm import tqdm 
import torch

DATASET_FILE = "data.txt"
TRAIN_PCT = 0.9
START_TOKEN = 'S'
END_TOKEN = 'F'
PAD_TOKEN = 'P'
MAX_ENCODE_SEQ_LENGTH = 40
MAX_DECODE_SEQ_LENGTH = 9

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3

N_HEADS = 4
D_MODEL = 128
N_LAYERS = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_data():
    """
    Reads the data file.

    Returns:
        list[tuple]: A list of training samples.
        list[str]: A list containing the unique characters in the data set.
    """

    with open(DATASET_FILE, 'r') as f:
        text = f.read()
        lines = text.split('\n')
        unique_chars = sorted(list(set(text)))
    data = []
    for line in lines:
        data.append(tuple(line.split(':')))
    return data, unique_chars

def split_data(data: list[tuple], train_pct: float = TRAIN_PCT):
    """
    Split the data into train and test splits.

    Args:
        data (list[tuple]): The full dataset.
        train_pct (float): The percentage of data to contain in the train dataset.
    
    Returns:
        list[tuple]: The train dataset.
        list[tuple]: The test dataset.
    """
    assert type(train_pct) is float and train_pct >= 0 and train_pct <= 1, "The train percentage must be between 0 and 1"
    shuffle(data)
    split_idx = int(len(data) * train_pct)
    train = data[:split_idx]
    test = data[split_idx:]
    return train, test

def create_batches(data: list[tuple], pad_token: int):
    """
    Create batches from the data.

    Args:
        data (list[tuple]): The data to create batches from.
        pad_token (int): The encoded value of the pad token.
    
    Returns:
        torch.Tensor: The encoder inputs.
        torch.Tensor: The decoder inputs.
        torch.Tensor: The targets.
    """
    assert type(pad_token) is int, "The encoded value of the pad token must be an integer"
    encode_inputs = []
    decode_inputs = []
    targets = []
    for e_tokens, d_tokens in tqdm(data, desc="Creating batches"):
        e_pad = (1, MAX_ENCODE_SEQ_LENGTH - len(e_tokens))
        e_tokens = torch.nn.functional.pad(e_tokens, e_pad, value=pad_token)

        d_pad = (1, MAX_DECODE_SEQ_LENGTH - len(d_tokens))
        d_tokens = torch.nn.functional.pad(d_tokens, d_pad, value=pad_token)
        e_input = e_tokens
        d_input = d_tokens[:-1]
        target = d_tokens[1:]

        encode_inputs.append(e_input)
        decode_inputs.append(d_input)
        targets.append(target)
    
    encode_inputs = torch.stack(encode_inputs)
    decode_inputs = torch.stack(decode_inputs)
    targets = torch.stack(targets)
    
    return encode_inputs, decode_inputs, targets

def get_batch(data, index):
    """
    Get a slice of data for training.

    Args:
        data (list(torch.Tensor)): A list of samples.
        index (int): The batch index.
    
    Returns:
        torch.Tensor: The encoder inputs.
        torch.Tensor: The decoder inputs.
        torch.Tensor: The targets.
    """
    start = index * BATCH_SIZE % len(data[0])
    stop = (index+1) * BATCH_SIZE % len(data[0])
    e_inputs = data[0][start: stop]
    d_inputs = data[1][start: stop]
    targets = data[2][start: stop]
    return e_inputs, d_inputs, targets

def main():
    data, vocabulary = get_data()
    data = data[:100]
    # Remove the special characters and add the start/end/pad tokens.
    vocabulary.remove('\n')
    vocabulary.remove(':')
    vocabulary.append(START_TOKEN)
    vocabulary.append(END_TOKEN)
    vocabulary.append(PAD_TOKEN)
    
    vocab_size = len(vocabulary)
    # Tokenize characters: create a mapping between characters and integers and back.
    stoi = {vocabulary[i]: i for i in range(vocab_size)}
    itos = {i: vocabulary[i] for i in range(vocab_size)}
    encode = lambda string: [stoi[c] for c in string]
    decode = lambda int_list: ''.join([itos[i] for i in int_list])

    # Encode data.
    for i in tqdm(range(len(data))):
        inputs, targets = data[i]
        encode_input = torch.tensor(encode(inputs), device=device)
        decode_input = torch.tensor(encode(START_TOKEN + targets + END_TOKEN), device=device)
        data[i] = (encode_input, decode_input)

    train, _ = split_data(data)
    train_data = create_batches(train,encode(PAD_TOKEN)[0])

    model = Transformer(
        vocab_size=vocab_size,
        start_token=encode(START_TOKEN)[0],
        end_token=encode(END_TOKEN)[0],
        max_tokens=MAX_ENCODE_SEQ_LENGTH + 1, # Hacky, compensates for the extra START/FINISH/PAD tokens.
        n_heads=N_HEADS,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=LR)
    batch_count = len(train_data[0]) // BATCH_SIZE
    for epoch in range(EPOCHS):
        mean_loss = 0
        for i in tqdm(range(batch_count), desc=f'Epoch {epoch}'):
            encoder_inputs, decoder_inputs, targets = get_batch(train_data, i)
            _, loss = model(encoder_inputs, decoder_inputs, targets)
            mean_loss += loss.item()
            # Backprop.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss /= batch_count
        print(f"Epoch {epoch} train loss: {mean_loss:.2f}")         

if __name__ == "__main__":
    main()