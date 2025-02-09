import torch

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, max_len=5000, empty=False):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.empty = empty
        self.__compute_pe__(d_model, max_len)


    def __compute_pe__(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)

        if not self.empty:
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            # pe = pe.unsqueeze(0).transpose(0, 1)

        pe = pe.unsqueeze(1) 
        self.register_buffer('pe', pe)
    
    def __str__(self):
        if self.empty:
            return f"PositionalEncoding(EMPTY)"
        else:
            return f"PositionalEncoding(Dim={self.d_model}, MaxLen={self.max_len})"

    def __repr__(self):
        return str(self)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x.dim0 = sequence length
            output: [sequence length, batch_size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        if x.size(0) > self.pe.shape[0]: 
            self.__compute_pe__(self.d_model, x.size(0)+10)
            self.pe = self.pe.to(x.device)

        return self.pe[:x.size(0), :]