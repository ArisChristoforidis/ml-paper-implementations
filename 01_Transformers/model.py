import torch
import torch.nn as nn
import torch.nn.functional as F

TEMP_TRIL_SIZE = 50
DROPOUT = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FeedForward(nn.Module):

    def __init__(self, d_model: int):
        """
        A simple feed forward module.
        
        Args:
            d_model (int): The number of hidden dimensions.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, head_size: int, max_tokens: int, mask: bool):
        """
        Self Attention calculates how relevant each token of a sequence is to all other
        tokens in the same sequence.

        Args:
            d_model (int): The number of hidden dimensions.
            head_size (int): The output dimension for each head.
            max_tokens (int): The max length of the input sequence.
            mask (bool): Whether to mask future tokens or not (True for Decoder, false for Encoder).
        """

        super().__init__()
        self.mask = mask
        # Intuition for the three layers:
        # Query: What does each token request?
        # Key: What does each token contain?
        # Value: What will each token communicate with other tokens?
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        if self.mask:
            # Tril is not a trainable model parameter so we initialize it like a buffer. (From Karpathy's video)
            self.register_buffer('tril', torch.tril(torch.ones(max_tokens, max_tokens)))
        pass

    def forward(self, x):
        _, L, d_k = x.shape

        q = self.query(x)
        k = self.query(x)
        v = self.query(x)
        # We perform a dot product between the token queries (requests) and the
        # token keys (information contained). Relevant information gets higher scores.
        # We divide by the square root of d_k in order to prevent the softmax from
        # becoming peaky and converging to a 1-hot, which would mean that each token
        # only considers a single other token. (From Karpathy's video)
        attention_weights = q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(d_k))
        # Mask using a lower triangular (decoder attention) so we don't consider information from future tokens.
        if self.mask:
            attention_weights = attention_weights.masked_fill(self.tril[:L,:L] == 0, float('-inf'))
        # Normalize our attention weights
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        result = attention_weights @ v
        return result

class CrossAttention(nn.Module):

    def __init__(self, d_model: int, head_size: int, max_tokens: int, mask: bool = True):
        """
        Cross Attention is nearly identical to Self Attention, but uses the encoder output
        as the key and value inputs and the decoder outputs for the query generation.

        Args:
            d_model (int): The number of hidden dimensions.
            head_size (int): The output dimension for each head.
            max_tokens (int): The max length of the input sequence.
            mask (bool): Whether to mask future tokens or not (True for Decoder, false for Encoder).
        """
        super().__init__()
        self.mask = mask
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        if self.mask:
            # Tril is not a trainable model parameter so we initialize it like a buffer. (From Karpathy's video)
            self.register_buffer('tril', torch.tril(torch.ones(max_tokens, max_tokens)))

    def forward(self, x):
        enc_outputs, x = x
        _, L, d_k = x.shape
        q = self.query(x)
        k = self.query(enc_outputs)
        v = self.query(enc_outputs)
        # We perform a dot product between the token queries (requests) and the
        # token keys (information contained). Relevant information gets higher scores.
        # We divide by the square root of d_k in order to prevent the softmax from
        # becoming peaky and converging to a 1-hot, which would mean that each token
        # only considers a single other token. (From Karpathy's video)
        attention_weights = q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(d_k))
        # Mask using a lower triangular (decoder attention) so we don't consider information from future tokens.
        if self.mask:
            attention_weights = attention_weights.masked_fill(self.tril[:attention_weights.shape[1],:attention_weights.shape[2]] == 0, float('-inf'))
        # Normalize our attention weights
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        result = attention_weights @ v
        return result


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, max_tokens: int, mask: bool, attention_type = 'self'):
        """
        A multihead attention module. Contains a number of attention heads that 
        focus on different parts of the input and concatenates their results.

        Args:
            n_heads (int): The number of attention heads per attention module.
            d_model (int): The number of hidden dimensions.
            max_tokens (int): The max length of the input sequence.
            mask (bool): Whether to mask future tokens or not (True for Decoder, false for Encoder).
            attention_type (str): Either 'self' or 'cross'. Defines whether to use Self Attention or Cross Attention.
        """
        super().__init__()
        assert d_model % n_heads == 0, "The number of heads (n_heads) must be divisible by the number of hidden dimensions (d_model)"
        assert attention_type in ['self', 'cross'], "The attention type must be either 'self' or 'cross'"
        head_size = d_model // n_heads
        if attention_type == 'self':
            self.attention_heads = nn.ModuleList([SelfAttention(d_model, head_size, max_tokens, mask) for _ in range(n_heads)])
        elif attention_type == 'cross':
            self.attention_heads = nn.ModuleList([CrossAttention(d_model, head_size, max_tokens, mask) for _ in range(n_heads)])
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # Calculate each head result, concatenate and pass through a linear layer.
        result = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        result = self.linear(result)
        result = self.dropout(result)
        return result

class EncoderBlock(nn.Module):
    def __init__(self, n_heads: int, d_model: int, max_tokens: int):
        """
        An encoder block as defined in the Attention is All You Need paper.

        Args:
            n_heads (int): The number of attention heads per attention module.
            d_model (int): The number of hidden dimensions.
            max_tokens (int): The max length of the decoder input sequence.
        """

        super().__init__()
        self.attention = MultiHeadAttention(n_heads, d_model, max_tokens, mask=False)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        pass

    def forward(self, x):
        attention = self.attention(x)
        # Add & LayerNorm 1
        x = self.layer_norm_1(x + attention)
        # Feed Forward.
        attention = self.feed_forward(x)
        # Add & LayerNorm 2
        result = self.layer_norm_2(x + attention)
        return result

class Encoder(nn.Module):

    def __init__(self, vocab_size: int, n_heads: int, d_model: int, n_layers: int, max_tokens: int):
        """
        The encoder stack.

        Args:
            vocab_size (int): The number of unique tokens in the dataset.
            n_heads (int): The number of attention heads per attention module.
            d_model (int): The number of hidden dimensions.
            n_layers (int): The number of layers in the encoder and decoder modules.
            max_tokens (int): The max length of the encoder input sequence.
        """
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(41, d_model)
        blocks = [EncoderBlock(n_heads, d_model, max_tokens) for _ in range(n_layers)]
        self.net = nn.Sequential(*blocks)

    def forward(self, encoder_inputs):
        result = self.input_embedding(encoder_inputs) + self.pos_embedding(torch.arange(encoder_inputs.shape[1],device=device))
        result = self.net(result)
        return result

class DecoderBlock(nn.Module):

    def __init__(self, n_heads: int, d_model: int, max_tokens: int):
        """
        A decoder block as defined in the Attention is All You Need paper.

        Args:
            n_heads (int): The number of attention heads per attention module.
            d_model (int): The number of hidden dimensions.
            max_tokens (int): The max length of the decoder input sequence.
        """
        super().__init__()
        self.masked_attention = MultiHeadAttention(n_heads, d_model, max_tokens, mask=True)
        self.layer_norm_1 = nn.LayerNorm(d_model)

        self.cross_attention = MultiHeadAttention(n_heads, d_model, max_tokens, mask=True, attention_type='cross')
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.feed_forward = FeedForward(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)

    def forward(self, x):
        encoder_output, x = x
        # Masked attention sub block.
        attention = self.masked_attention(x)
        x = self.layer_norm_1(x + attention)

        # Attention + encoder outputs sub block.
        attention = self.cross_attention([encoder_output, x])
        x = self.layer_norm_2(x + attention)

        # Feed forward
        feed_forward = self.feed_forward(x)
        result = self.layer_norm_3(x + feed_forward)

        return result

class Decoder(nn.Module):

    def __init__(self, vocab_size: int, n_heads: int, d_model: int, n_layers: int, max_tokens: int):
        """
        The decoder stack.

        Args:
            vocab_size (int): The number of unique tokens in the dataset.
            n_heads (int): The number of attention heads per attention module.
            d_model (int): The number of hidden dimensions.
            n_layers (int): The number of layers in the encoder and decoder modules.
            max_tokens (int): The max length of the decoder input sequence.
        """
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(10, d_model)
        self.net = nn.ModuleList([DecoderBlock(n_heads, d_model, max_tokens) for _ in range(n_layers)])


    def forward(self, encoder_output, decoder_input):
        x = self.input_embedding(decoder_input) + self.pos_embedding(torch.arange(decoder_input.shape[1], device=device))
        for decoder_block in self.net:
            x = decoder_block([encoder_output, x])
        return x

class Transformer(nn.Module):

    def __init__(self, vocab_size: int, start_token: int, end_token: int, max_tokens: int, n_heads: int = 4, d_model: int = 128, n_layers: int = 6):
        """
        A transformer module.

        Args:
            vocab_size (int): The number of unique tokens in the dataset.
            start_token (int): The start token.
            end_token (int): The end token.
            max_tokens (int): The max length of the input sequence.
            n_heads (int): The number of attention heads per attention module.
            d_model (int): The number of hidden dimensions.
            n_layers (int): The number of layers in the encoder and decoder modules.
        """
        super().__init__()
        assert type(vocab_size) is int and vocab_size > 0, "The vocabulary size must be a positive integer"
        assert type(start_token) is int, "The start token must be an integer"
        assert type(end_token) is int, "The end token must be an integer"
        assert type(max_tokens) is int and max_tokens > 0, "max_tokens must be a positive integer"
        assert type(vocab_size) is int and vocab_size > 0, "The vocabulary size must be a positive integer"
        assert type(n_heads) is int and n_heads > 0, "The number of heads must be a positive integer"
        assert type(d_model) is int and d_model > 0, "The number of hidden dimensions must be a positive integer"
        assert type(n_layers) is int and n_layers > 0, "The number of layers must be a positive integer"

        self.start_token = start_token
        self.end_token = end_token

        self.encoder = Encoder(vocab_size, n_heads, d_model, n_layers, max_tokens)
        self.decoder = Decoder(vocab_size, n_heads, d_model, n_layers, max_tokens)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, encoder_inputs, decoder_inputs, targets):
        encoder_out = self.encoder(encoder_inputs)
        decoder_out = self.decoder(encoder_out, decoder_inputs)
        logits = self.linear(decoder_out)

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def predict(self, input_tokens: torch.Tensor, max_length: int = 10):
        """
        Generate a prediction.

        Args:
            input_tokens (torch.Tensor): The encoder inputs (tokenized text).
            max_length (int): The max length of the predicted sequence.

        Returns:
            list: A list of predicted tokens that can be decoded.
        """
        # Run the encoder input through the encoder once, since it stays constant.
        input_tokens = torch.tensor(input_tokens).unsqueeze(0)
        enc_out = self.encoder(input_tokens)
        decode_tokens = torch.tensor([self.start_token], dtype=torch.int, device=device).unsqueeze(0)
        while decode_tokens.shape[1] < max_length:
            # Run the decoder.
            dec_out = self.decoder(enc_out, decode_tokens)
            # Softmax on the last position.
            logits = self.linear(dec_out)
            logits = logits[:, -1, :]
            distribution = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(distribution, num_samples=1)
            decode_tokens = torch.cat((decode_tokens, next_token), dim=1)
            if next_token.item() == self.end_token: break

        return decode_tokens.squeeze().tolist()