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
    # reshape
    numerator = numerator[:, np.newaxis]

    # stepsize=2 to alternate beweent the sine and cosine (used later)
    denominator = np.arange(0, d_model, 2, dtype=np.float32)
    # normalize denominator
    denominator = denominator / d_model

    denominator = np.power(np.array(10000, dtype=np.float32), denominator)

    # invert values -> decreasing values for encoding positions
    denominator = 1 / denominator
    # duplicate each value to compute sin and cos for each value
    denominator = np.repeat(denominator, 2)
    # reshape
    denominator = denominator[np.newaxis, :]

    # matrix multiplication of numerator and denominator
    encoding = np.matmul(numerator, denominator)
    # apply sin for all even indiceses
    encoding[:, ::2] = np.sin(encoding[:, ::2])
    # apply cos for all odd indiceses
    encoding[:, 1::2] = np.cos(encoding[:, 1::2])
    #encoding = encoding[np.newaxis, ...]
    # cast to tenforflow variable
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


def saveToBuffer(inputs, parser, path):
    sound, delta = inputs
    midi_file = parser.features_to_midi(sound, delta)
    midi_file.save(path)


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
    #test: set orig_len manually
    orig_len = seq_len * 2
    assert orig_len >= 1

    sounds = np.array([sound[:orig_len] for sound in sounds])
    deltas = np.array([delta[:orig_len] for delta in deltas])
    # sounds -> (batch_size, orig_len)

    full_len = mem_len + seq_len - 1

    inputs_sound = tf.constant(sounds[:, -seq_len:])
    inputs_delta = tf.constant(deltas[:, -seq_len:])

    result_sound = sounds[:, :-seq_len]
    result_delta = deltas[:, :-seq_len]
    saveToBuffer((sounds, deltas), parser, 'generated_midis/buffer/full_song.midi')
    saveToBuffer((result_sound, result_delta), parser, 'generated_midis/buffer/input_notes.midi')
    cut = 200
    saveToBuffer((result_sound[:, -cut], result_delta[:, -cut]), parser, 'generated_midis/buffer/input_notes_cut'+str(cut)+'.midi')

    outputs_sound, outputs_delta, next_mem_list, attention_weight_list, attention_loss_list = model(
        inputs=(inputs_sound, inputs_delta),
        mem_list=None,
        next_mem_len=mem_len,
        training=False
    )
    cnt = 1
    # tqdm used to output a process bar
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

        result_sound = np.concatenate((result_sound, new_sounds), axis=-1)
        result_delta = np.concatenate((result_delta, new_deltas), axis=-1)
        if cnt % 100 == 0:
            print('size sound:', result_sound.size)
            print('size delta:', result_delta.size)
            saveToBuffer((result_sound, result_delta), parser, 'generated_midis/buffer/' + str(cnt) + '_note.midi')

        inputs_sound = tf.constant(new_sounds)
        inputs_delta = tf.constant(new_deltas)

        outputs_sound, outputs_delta, next_mem_list, attention_weight_list, attention_loss_list = model(
            inputs=(inputs_sound, inputs_delta),
            mem_list=next_mem_list,
            next_mem_len=mem_len,
            training=False
        )
        cnt += 1

    sounds = sounds[:, orig_len:]
    deltas = deltas[:, orig_len:]

    midi_list = [parser.features_to_midi(
        sound, delta) for sound, delta in zip(sounds, deltas)]

    return midi_list, next_mem_list, attention_weight_list, attention_loss_list


def generate_text(model, seq_len, mem_len, max_len, tokenizer, start_idx, end_idx, blocked_idxs,
                  batch_size, beginning=None, top_k=3, temp=0.4):

    if isinstance(beginning, str):
        words = tokenizer.texts_to_sequences([beginning])
        words = np.repeat(words, batch_size, axis=0)
        start_idxs = np.full((batch_size, 1), start_idx,
                             dtype=words.dtype)
        words = np.concatenate((start_idxs, words), axis=-1)

    elif isinstance(beginning, list):
        assert len(beginning) == batch_size
        for string in beginning:
            assert isinstance(string, str)
        words = tokenizer.texts_to_sequences(beginning)
        min_len = min([len(x) for x in words])
        words = np.array([x[:min_len] for x in words])
        start_idxs = np.full((batch_size, 1), start_idx,
                             dtype=words.dtype)
        words = np.concatenate((start_idxs, words), axis=-1)

    else:
        words = np.full((batch_size, 1), start_idx)

    end_flags = [False] * batch_size
    end_cnt = 0

    orig_len = words.shape[1]
    assert orig_len >= 1
    # words -> (batch_size, orig_len)

    # ================================

    inputs = tf.constant(words[:, -seq_len:])

    outputs, next_mem_list, attention_weight_list, attention_loss_list = model(
        inputs=inputs,
        mem_list=None,
        next_mem_len=mem_len,
        training=False
    )

    for _ in tqdm.tqdm(range(max_len)):

        outputs = outputs[:, -1, :]
        probs = tf.nn.softmax(outputs, axis=-1).numpy()
        probs[:, blocked_idxs] = 0
        # probs -> (batch_size, n_words)

        new_words = []

        for batch_idx, batch_probs in enumerate(probs):

            best_idxs = batch_probs.argsort()[-top_k:][::-1]
            best_probs = softmax_with_temp(batch_probs[best_idxs], temp)
            new_word = np.random.choice(best_idxs, p=best_probs)
            new_words.append(new_word)

            if new_word == end_idx and not end_flags[batch_idx]:
                end_flags[batch_idx] = True
                end_cnt += 1

        new_words = np.array(new_words)[:, np.newaxis]
        # new_words -> (batch_size, 1)
        words = np.concatenate((words, new_words), axis=-1)

        if end_cnt >= batch_size:
            break

        inputs = tf.constant(new_words)

        outputs, next_mem_list, attention_weight_list, attention_loss_list = model(
            inputs=inputs,
            mem_list=next_mem_list,
            next_mem_len=mem_len,
            training=False
        )

    return words, end_flags
