# https://github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.py

import torch
import torch.nn as nn

class SoftEmbedding(nn.Module):
    def __init__(self,
                wte: nn.Embedding,
                n_tokens: int = 10,
                random_range: float = 0.5,
                initialize_from_vocab: bool = True,
                prefix_token_id: int = 50256
                ):
        """appends learned embedding to

        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.prefix_token_id = prefix_token_id
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               n_tokens,
                                                                               random_range,
                                                                               initialize_from_vocab))

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             n_tokens: int = 10,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = True
                             ):
        """initializes learned embedding

        Args:
            same as __init__

        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    def forward(self, tokens):
        """run forward pass

        Args:
            tokens (torch.long): input tokens before encoding

        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        if tokens.size(1) < self.n_tokens:
            return self.wte(tokens)
        input_embedding = []
        if not (tokens[:, 0] == self.prefix_token_id).all(dim=0):
            for i in range(tokens.size(0)):
                prefix_start = torch.nonzero(tokens[i] == self.prefix_token_id)[0][0]
                input_embedding.append(torch.cat([self.wte(tokens[i][:prefix_start]), self.learned_embedding, self.wte(tokens[i][prefix_start+self.n_tokens:])], 0))
            return torch.stack(input_embedding, 0)
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)