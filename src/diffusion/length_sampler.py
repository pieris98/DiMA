import numpy as np


class LengthSampler:
    def __init__(self, path, max_sequence_len):
        """
        path: path to numpy array that contains probabilities of sequence lengths
        max_sequence_len: maximum sequence length without special tokens to sample from
        """
        self.distrib = np.load(path)
        min_len = 128  # DiMA uses min length of 128
        if len(self.distrib) != max_sequence_len - min_len + 1:
            self.distrib = self.distrib[:max_sequence_len + 1 - min_len]
        self.distrib = self.distrib / np.sum(self.distrib)
        self.lengths = np.arange(min_len, max_sequence_len + 1)
            
    def sample(self, num_samples):
        # Sample sequence lengths according to the probability distribution
        sampled_lengths = np.random.choice(self.lengths, size=num_samples, p=self.distrib)
        return sampled_lengths
        