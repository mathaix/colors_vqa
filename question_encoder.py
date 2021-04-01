import torch
import torch.nn as nn
import torch.nn.functional as F 


class QuestionEncoder(nn.Module):
    def __init__(self,
            vocab_size,
            embed_dim,
            hidden_dim,
            embedding=None,
            freeze_embedding=False):
        """
        Simple Question Encoder based on a GRU cell

        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.freeze_embedding = freeze_embedding
        self.embedding = self._define_embedding(
            embedding, self.vocab_size, embed_dim, self.freeze_embedding)
        self.embed_dim = self.embedding.embedding_dim
        self.rnn = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, question_seqs, question_length,hidden=None):
        """
        Parameters
        ----------
        question_seqs: 

        Returns
        -------
        hidden : torch.FloatTensor
            These are the final hidden state of the RNN for this batch,
            shape `(m, p) where `m` is the batch_size and `p` is
             the color dimensionality.
        output, hidden = self.rnn(color_seqs)
        return hidden

        """

        embs = self.get_embeddings(question_seqs)
        output, hidden = self.rnn(embs, hidden)
        output = self.output_layer(output)
        return output, hidden

    def get_embeddings(self, word_seqs):
        """
        Gets the input token representations. At present, these are
        just taken directly from `self.embedding`, but `target_colors`
        can be made available in case the user wants to subclass this
        function to append these representations to each input token.

        Parameters
        ----------
        word_seqs : torch.LongTensor
            This is a padded sequence, dimension (m, k), where k is
            the length of the longest sequence in the batch.

        target_colors : torch.FloatTensor
            Dimension (m, c), where m is the number of examples and
            c is the dimensionality of the color representations.

        """
        return self.embedding(word_seqs)

    @staticmethod
    def _define_embedding(embedding, vocab_size, embed_dim, freeze_embedding):
        if embedding is None:
            emb = nn.Embedding(vocab_size, embed_dim)
            emb.weight.requires_grad = not freeze_embedding
            return emb
        else:
            embedding = torch.FloatTensor(embedding)
            return nn.Embedding.from_pretrained(
                embedding, freeze=freeze_embedding)
        

    