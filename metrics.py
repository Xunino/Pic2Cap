import collections
import numpy as np
import tensorflow as tf


def MaskedSoftmaxCELoss(label, pred):
    """
    :param label: shape (batch_size, max_length, vocab_size)
    :param pred: shape (batch_size, max_length)
    :return: weighted_loss: shape (batch_size, max_length)
    """
    weights_mask = 1 - np.equal(label, 0)
    unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label, pred)
    weighted_loss = tf.reduce_mean(unweighted_loss * weights_mask)
    return weighted_loss


def accuracy_function(real, pred):
    # Check equal same -> True
    real = tf.cast(real, dtype=tf.int64)
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    # Ìf both mask and accuracies is True -> True
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    # Return True value -> 1
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


class Bleu_score:
    """
        We can evaluate a predicted sequence by comparing it with the label sequence.
        BLEU (Bilingual Evaluation Understudy) "https://aclanthology.org/P02-1040.pdf",
        though originally proposed for evaluating machine translation results,
        has been extensively used in measuring the quality of output sequences for different applications.
        In principle, for any n-grams in the predicted sequence, BLEU evaluates whether this n-grams appears
        in the label sequence.
    """

    def __init__(self):
        super().__init__()

    def remove_oov(self, sentence):
        return [i for i in sentence.split(" ") if i not in ["<sos>", "<eos>"]]

    def __call__(self, pred, target, n_grams=3):
        pred = self.remove_oov(pred)
        target = self.remove_oov(target)
        pred_length = len(pred)
        target_length = len(target)

        if pred_length < n_grams:
            return 0
        else:
            score = np.exp(np.minimum(0, 1 - target_length / pred_length))
            for k in range(1, n_grams + 1):
                label_subs = collections.defaultdict(int)
                for i in range(target_length - k + 1):
                    label_subs[" ".join(target[i:i + k])] += 1

                num_matches = 0
                for i in range(pred_length - k + 1):
                    if label_subs[" ".join(pred[i:i + k])] > 0:
                        label_subs[" ".join(pred[i:i + k])] -= 1
                        num_matches += 1
                score *= np.power(num_matches / (pred_length - k + 1), np.power(0.5, k))
            return score
