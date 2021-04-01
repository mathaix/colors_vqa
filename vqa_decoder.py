import torch
import torch.nn as nn
import torch.nn.functional as F 

class VQADecoder(nn.Module):
    def __init__(self,
            vocab_size,
            embed_dim,
            hidden_dim,
            embedding=None,
            freeze_embedding=False):
        """
        Simple Decoder model based on a GRU cell. The hidden
        representations of the GRU are passed through a dense linear
        layer, and those logits are used to train the language model
        according to a softmax objective in `ContextualColorDescriber`.

        Parameters
        ----------
        vocab_size : int

        embed_dim : int

        hidden_dim : int

        embedding : np.array or None
            If `None`, a random embedding is created. If `np.array`, this
            value becomes the embedding.

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

    def forward(self, word_seqs, seq_lengths=None, hidden=None, target_colors=None):
        """
        Core computation for the model.

        Parameters
        ----------
        word_seqs : torch.LongTensor
            This is a padded sequence, dimension (m, k), where k is
            the length of the longest sequence in the batch. The `forward`
            method uses `self.get_embeddings` to mape these indices to their
            embeddings.

        seq_lengths : torch.LongTensor
            Shape (m, ) where `m` is the number of examples in the batch.

        hidden : torch.FloatTensor
            Shape `(m, self.hidden_dim)`. When training, this is always the
            final state of the `Encoder`. During prediction, this might be
            recursively computed as the sequence is processed.

        target_colors : torch.FloatTensor
            Dimension (m, c), where m is the number of examples and
            c is the dimensionality of the color representations.

        Returns
        -------
        output : torch.FloatTensor
            The full sequence of outputs states. When we are training, the
            shape is `(m, hidden_dim, k)` to accommodate the expectations
            of the loss function. During prediction, the shape is
            `(m, k, hidden_dim)`. In both cases, m is the number of examples in
            the batch and `k` is the maximum length of sequences in this batch.

        hidden : torch.FloatTensor
            The final output state of the network. Shape `(m, hidden_dim)`
            where m is the number of examples in the batch.

        """
        embs = self.get_embeddings(word_seqs, target_colors=target_colors)

        if self.training:
            # Packed sequence for performance:
            embs = torch.nn.utils.rnn.pack_padded_sequence(
                embs,
                batch_first=True,
                lengths=seq_lengths,
                enforce_sorted=False)
            # RNN forward:
            output, hidden = self.rnn(embs, hidden)
            # Unpack:
            output, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True)
            # Output dense layer to get logits:
            output = self.output_layer(output)
            # Drop the final element:
            output = output[:, : -1, :]
            # Reshape for the sake of the loss function:
            output = output.transpose(1, 2)
            return output, hidden
        else:
            output, hidden = self.rnn(embs, hidden)
            output = self.output_layer(output)
            return output, hidden

    def get_embeddings(self, word_seqs, target_colors=None):
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