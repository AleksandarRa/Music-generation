import csv
from midi_parser import MIDI_parser
from model import Music_transformer
import config_music as config
from utils import shuffle_ragged_2d, inputs_to_labels, get_quant_time
import tensorflow as tf
import argparse
import pathlib
import numpy as np

CHECKPOINT_EPOCH = 30
N_FILES = 1
BATCHSIZE = 1
FILENAME = '0.npz'


@tf.function
def evaluate_model(inputs_sound, inputs_delta, labels_sound, labels_delta, alpha):
    with tf.GradientTape() as tape:

        logits_sound, logits_delta, next_mem_list, attention_weight_list, attention_loss_list = model(
            inputs=(inputs_sound, inputs_delta),
            mem_list=None,
            next_mem_len=mem_len,
            training=False,
            alpha=alpha
        )

        if use_attn_reg:
            attention_loss = 4 * tf.math.reduce_mean(attention_loss_list)
        else:
            attention_loss = None

        loss, pad_mask = model.get_loss(
            logits_sound=logits_sound,
            logits_delta=logits_delta,
            labels_sound=labels_sound,
            labels_delta=labels_delta,
            attention_loss=attention_loss
        )

    outputs_sound = tf.nn.softmax(logits_sound, axis=-1)
    # outputs_sound -> (batch_size, seq_len, n_sounds)
    outputs_delta = tf.nn.softmax(logits_delta, axis=-1)
    # outputs_delta -> (batch_size, seq_len, n_deltas)

    non_padded_labels_sound = tf.boolean_mask(labels_sound, pad_mask)
    non_padded_outputs_sound = tf.boolean_mask(outputs_sound, pad_mask)

    non_padded_labels_delta = tf.boolean_mask(labels_delta, pad_mask)
    non_padded_outputs_delta = tf.boolean_mask(outputs_delta, pad_mask)

    loss_metric(loss)
    acc_metric_sound(non_padded_labels_sound, non_padded_outputs_sound)
    acc_metric_delta(non_padded_labels_delta, non_padded_outputs_delta)

    return next_mem_list


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-np', '--npz_dir', type=str, default='npz_music',
                            help='Directory where the npz files are stored')

    arg_parser.add_argument('-w', '--weights', type=str,
                            default='checkpoints_music/transformerXL/transformerXL_checkpoint' + str(CHECKPOINT_EPOCH) +'.weights.h5', help='Path to saved model weights')

    args = arg_parser.parse_args()

    assert pathlib.Path(args.npz_dir).is_dir()
    if not args.weights is None:
        assert pathlib.Path(args.weights).is_file()

    # False = Tensorflow buils use computational graph
    # better for performance but harder to debug
    # True = use eager execution mode. evalues operations immediately without building graphs
    tf.config.run_functions_eagerly(True)

    idx_to_time = get_quant_time()

    midi_parser = MIDI_parser.build_from_config(config, idx_to_time)

    dataset = midi_parser.get_tf_dataset(
        file_directory=args.npz_dir, batch_size=BATCHSIZE,
        n_samples=N_FILES, filename=FILENAME)

    model, optimizer = Music_transformer.build_from_config(config=config, checkpoint_path=args.weights)
    #model, optimizer = Music_transformer.build_from_config(config=config, checkpoint_path=None)

    loss_metric = tf.keras.metrics.Mean(name='loss')
    acc_metric_sound = tf.keras.metrics.SparseCategoricalAccuracy(
        name='acc_sound')
    acc_metric_delta = tf.keras.metrics.SparseCategoricalAccuracy(
        name='acc_delta')

    use_attn_reg = config.use_attn_reg
    values = []


    # evaluate model

    pad_idx = config.pad_idx
    mem_len = config.mem_len
    max_segs_per_batch = config.max_segs_per_batch
    loss_metric.reset_state()
    acc_metric_sound.reset_state()
    acc_metric_delta.reset_state()


    full_song = next(iter(dataset.take(1)))
    sound, delta = shuffle_ragged_2d(full_song, pad_idx, lowest_idx=1)
    # sound -> (batch_size, maxlen)
    # delta -> (batch_size, maxlen)

    labels_sound = inputs_to_labels(sound, pad_idx)
    # batch_labels_sound -> (batch_size, maxlen)
    labels_delta = inputs_to_labels(delta, pad_idx)
    # batch_labels_delta -> (batch_size, maxlen)

    maxlen = sound.shape[1]
    # ======================================================================================
    # train on random slices of the batch
    # ======================================================================================

    # seperate the song into 4 parts. first and third part are given, second and forth should be predicted
    seq_len = int(maxlen / 4)
    segs_per_batch = int(min(max_segs_per_batch, maxlen // seq_len))
    mem_list = None
    alphas = np.linspace(0,1,11)
    #alphas = tf.range(0.0, 1.1, delta=0.1)
    for alpha in alphas:
        # -100 for rounding error
        print('alpha: ', alpha)
        for start in range(0, maxlen - 100, seq_len*2):
            seg_sound = sound[:, start: start + seq_len]
            # seg_sound -> (batch_size, seq_len)
            seg_delta = delta[:, start: start + seq_len]
            # seg_delta -> (batch_size, seq_len)

            seg_labels_sound = labels_sound[:,
                               start: start + seq_len]
            # seg_labels_sound -> (batch_size, seq_len)
            seg_labels_delta = labels_delta[:,
                               start: start + seq_len]
            # seg_labels_delta -> (batch_size, seq_len)

            mem_list = evaluate_model(inputs_sound=seg_sound,
                                  inputs_delta=seg_delta,
                                  labels_sound=seg_labels_sound,
                                  labels_delta=seg_labels_delta,
                                  alpha = alpha)

            value = [('epochs trained', CHECKPOINT_EPOCH),
                     ('filename', FILENAME),
                     ('alpha', alpha),
                     ('range', str(start) + " - " + str(start + seq_len)),
                     ('acc_sound', acc_metric_sound.result()),
                     ('acc_delta', acc_metric_delta.result()),
                     ('loss', loss_metric.result())]
            with open('logs/evaluateModel.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                #writer.writerow([name for name, result in value])  # Headers
                writer.writerow([result.numpy() if hasattr(result, 'numpy') else result for name, result in value])

