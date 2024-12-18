import tensorflow as tf
import numpy as np
import tqdm

__all__ = ('pad_ragged_2d', 'shuffle_ragged_2d',
           'inputs_to_labels', 'get_pos_encoding',
           'get_quant_time', 'softmax_with_temp',
           'generate_midis')


def pad_ragged_2d(ragged_tensor, pad_idx):

    # ragged_tensor -> RAGGED(batch_size, None)
    lens = ragged_tensor.row_lengths(axis=-1)
    maxlen = tf.math.reduce_max(lens)
    mask = tf.sequence_mask(lens, maxlen, tf.bool)

    zero_padded = ragged_tensor.to_tensor()
    # zero_padded -> (batch_size, maxlen)

    padding = tf.constant(pad_idx, dtype=zero_padded.dtype)

    padded_tensor = tf.where(mask, zero_padded, padding)
    # padded_tensor -> (batch_size, maxlen)

    return padded_tensor


def shuffle_ragged_2d(ragged_tensors, pad_idx, lowest_idx=5):

    if not isinstance(ragged_tensors, (list, tuple)):
        ragged_tensors = [ragged_tensors]

    # ragged_tensor -> RAGGED(batch_size, None)
    lens = ragged_tensors[0].row_lengths(axis=-1)
    # kth_lowest defines which size you want to consider. if kth_lowerst=3 it will take the size of the 3th smallest song
    kth_lowest = -tf.nn.top_k(-lens, lowest_idx).values[-1]
    shuffled_tensors = [[] for _ in ragged_tensors]

    for len_, *rows in zip(lens, *ragged_tensors):

        assert all(row.shape[0] == len_ for row in rows)
        if len_ <= kth_lowest:
            new_rows = [tf.pad(row, paddings=[[0, kth_lowest - len_]],
                               constant_values=pad_idx) for row in rows]
        else:
            start_idx = tf.random.uniform(
                (), minval=0, maxval=len_ - kth_lowest + 1, dtype=tf.int64)
            new_rows = [row[start_idx: start_idx + kth_lowest]
                        for row in rows]

        for tensor, row in zip(shuffled_tensors, new_rows):
            tensor.append(row[tf.newaxis, :])

    shuffled_tensors = [tf.concat(shuffled_tensor, axis=0)
                        for shuffled_tensor in shuffled_tensors]

    return shuffled_tensors


def inputs_to_labels(inputs, pad_idx):

    # inputs -> (batch_size, seq_len)
    inputs_padded = tf.pad(inputs[:, 1:], paddings=[
                           [0, 0], [0, 1]], constant_values=pad_idx)

    return inputs_padded


def get_pos_encoding(seq_len, d_model):

    numerator = np.arange(seq_len, dtype=np.float32)
    numerator = numerator[:, np.newaxis]

    denominator = np.arange(0, d_model, 2, dtype=np.float32)
    denominator = denominator / d_model

    denominator = np.power(np.array(10000, dtype=np.float32), denominator)

    denominator = 1 / denominator
    denominator = np.repeat(denominator, 2)
    denominator = denominator[np.newaxis, :]

    encoding = np.matmul(numerator, denominator)
    encoding[:, ::2] = np.sin(encoding[:, ::2])
    encoding[:, 1::2] = np.cos(encoding[:, 1::2])
    #encoding = encoding[np.newaxis, ...]
    encoding = tf.cast(encoding, dtype=tf.float32)

    return encoding


def get_quant_time():

    step = 0.001
    coef = 1.16
    delta = 0
    total_reps = 64
    local_reps = 2
    quant_time = []
    for _ in range(total_reps // local_reps):
        for _ in range(local_reps):
            delta += step
            quant_time.append(delta)

        step *= coef

    quant_time = np.sort(quant_time + [5.0, 0.0])
    return quant_time


def softmax_with_temp(x, temp=1.0):

    assert isinstance(temp, float)
    assert temp > 0
    assert all(map(lambda a: a > 0, x))

    x = x / np.sum(x) / temp
    x = tf.nn.softmax(x).numpy()
    return x


def generate_midis(model, seq_len, mem_len, max_len, parser, filenames, pad_idx, top_k=1, temp=1.0):

    assert isinstance(seq_len, int)
    assert seq_len > 0

    assert isinstance(mem_len, int)
    assert mem_len >= 0

    assert isinstance(max_len, int)
    assert max_len > 1

    batch_size = len(filenames)

    sounds, deltas = zip(*[parser.load_features(filename)
                           for filename in filenames])

    min_len = min([len(s) for s in sounds])

    orig_len = np.random.randint(1, min(2 * mem_len, min_len))
    assert orig_len >= 1

    sounds = np.array([sound[:orig_len] for sound in sounds])
    deltas = np.array([delta[:orig_len] for delta in deltas])
    # sounds -> (batch_size, orig_len)

    full_len = mem_len + seq_len - 1

    inputs_sound = tf.constant(sounds[:, -seq_len:])
    inputs_delta = tf.constant(deltas[:, -seq_len:])

    outputs_sound, outputs_delta, next_mem_list, attention_weight_list, attention_loss_list = model(
        inputs=(inputs_sound, inputs_delta),
        mem_list=None,
        next_mem_len=mem_len,
        training=False
    )

    for _ in tqdm.tqdm(range(max_len)):

        outputs_sound = outputs_sound[:, -1, :]
        probs_sound = tf.nn.softmax(outputs_sound, axis=-1).numpy()
        probs_sound[:, pad_idx] = 0
        # probs_sound -> (batch_size, n_sounds)

        outputs_delta = outputs_delta[:, -1, :]
        probs_delta = tf.nn.softmax(outputs_delta, axis=-1).numpy()
        probs_delta[:, pad_idx] = 0
        # probs_delta -> (batch_size, n_deltas)

        new_sounds = []

        for batch_probs in probs_sound:

            best_idxs = batch_probs.argsort()[-top_k:][::-1]
            best_probs = softmax_with_temp(batch_probs[best_idxs], temp)
            new_sound = np.random.choice(best_idxs, p=best_probs)
            new_sounds.append(new_sound)

        new_sounds = np.array(new_sounds)[:, np.newaxis]
        # new_sounds -> (batch_size, 1)
        sounds = np.concatenate((sounds, new_sounds), axis=-1)

        new_deltas = []

        for batch_probs in probs_delta:

            best_idxs = batch_probs.argsort()[-top_k:][::-1]
            best_probs = softmax_with_temp(batch_probs[best_idxs], temp)
            new_delta = np.random.choice(best_idxs, p=best_probs)
            new_deltas.append(new_delta)

        new_deltas = np.array(new_deltas)[:, np.newaxis]
        # new_deltas -> (batch_size, 1)
        deltas = np.concatenate((deltas, new_deltas), axis=-1)

        inputs_sound = tf.constant(new_sounds)
        inputs_delta = tf.constant(new_deltas)

        outputs_sound, outputs_delta, next_mem_list, attention_weight_list, attention_loss_list = model(
            inputs=(inputs_sound, inputs_delta),
            mem_list=next_mem_list,
            next_mem_len=mem_len,
            training=False
        )

    sounds = sounds[:, orig_len:]
    deltas = deltas[:, orig_len:]

    midi_list = [parser.features_to_midi(
        sound, delta) for sound, delta in zip(sounds, deltas)]

    return midi_list, next_mem_list, attention_weight_list, attention_loss_list

