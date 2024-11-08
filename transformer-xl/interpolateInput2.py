import csv

from midi_parser import MIDI_parser
from model import Music_transformer
import config_music as config
from utils import get_quant_time, softmax_with_temp
import numpy as np
import argparse
import os
import pathlib
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import tqdm

CHECKPOINT_EPOCH = 80
N_GEN_SEQ = 1

def computeLoss(model, logits_sound, logits_delta, labels_sound, labels_delta):

    loss_metric_mse = tf.keras.metrics.Mean(name='loss')
    loss_metric_mae = tf.keras.metrics.Mean(name='loss')
    acc_metric_sound = tf.keras.metrics.Accuracy(name='acc sound')
    acc_metric_delta = tf.keras.metrics.Accuracy(name='acc delta')

    # Compute MSE loss
    mse_loss = tf.keras.losses.MeanSquaredError()
    loss_mse_sound = mse_loss(labels_sound, logits_sound)
    loss_mse_delta = mse_loss(labels_delta, logits_delta)

    # compute MAE loss
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    loss_mae_sound = mae_loss(labels_sound, logits_sound)
    loss_mae_delta = mae_loss(labels_delta, logits_delta)

    # Adjust with class weights if available
    if model.weights_sound is not None:
        weights = tf.gather_nd(params=model.weights_sound, indices=labels_sound[..., tf.newaxis])
        loss_mse_sound = loss_mse_sound * weights
        loss_mae_sound = loss_mse_sound * weights

    # Adjust with class weights if available
    if model.weights_delta is not None:
        weights = tf.gather_nd(params=model.weights_delta, indices=labels_delta[..., tf.newaxis])
        loss_mse_delta = loss_mse_delta * weights
        loss_mae_delta = loss_mae_delta * weights

    # Combine the losses
    loss_mse = loss_mse_sound + loss_mse_delta
    loss_mae = loss_mae_sound + loss_mae_delta

    # Average the loss over all elements
    loss_mse = tf.math.reduce_mean(loss_mse)
    loss_mae = tf.math.reduce_mean(loss_mae)

    loss_metric_mse(loss_mse)
    loss_metric_mae(loss_mae)
    acc_metric_sound.update_state(labels_sound, logits_sound)
    acc_metric_delta.update_state(labels_delta, logits_delta)

    return loss_metric_mse, loss_metric_mae, acc_metric_sound, acc_metric_delta


def generate(model, sounds, deltas, pad_idx, top_k=1, temp=1.0, alpha=0.0, interpol_len=0, sounds2=None, deltas2=None):

    max_len = sounds.size * N_GEN_SEQ
    seq_len = sounds.size
    mem_len = seq_len
    i=0
    full_len = mem_len + seq_len - 1

    inputs_sound = tf.constant(sounds[:, -seq_len:])
    inputs_delta = tf.constant(deltas[:, -seq_len:])

    outputs_sound, outputs_delta, next_mem_list, attention_weight_list, attention_loss_list = model(
        inputs=(inputs_sound, inputs_delta),
        mem_list=None,
        next_mem_len=mem_len,
        training=False,
        alpha=alpha,
        inputs2=(None, None)
    )

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

        inputs_sound = tf.constant(new_sounds)
        inputs_delta = tf.constant(new_deltas)

        inputs_sound2 = tf.constant(sounds2[:, i:(seq_len+i+1)])
        inputs_delta2 = tf.constant(deltas2[:, i:(seq_len+i)+1])
        i += 1

        outputs_sound, outputs_delta, next_mem_list, attention_weight_list, attention_loss_list = model(
            inputs=(inputs_sound, inputs_delta),
            mem_list=next_mem_list,
            next_mem_len=mem_len,
            training=False,
            alpha=alpha,
            inputs2=(inputs_sound2, inputs_delta2)
        )

    sounds = sounds[:, max_len:]
    deltas = deltas[:, max_len:]

    return sounds, deltas, next_mem_list, attention_weight_list, attention_loss_list

def saveValues(npz_filenames, npz_filenames2, song_len, cutted_song_len, interpol_len, acc_metric_sound, acc_metric_delta, loss_mse, loss_mae, alpha=0):

    values = [('filename', os.path.basename(npz_filenames[0])),
              ('mixing filename', os.path.basename(npz_filenames2[0])),
              ('aplha', alpha),
              ('song length', song_len),
              ('input length', cutted_song_len),
              ('interpolation length', interpol_len),
              ('output length', cutted_song_len * N_GEN_SEQ),
              ('acc_sound', acc_metric_sound),
              ('acc_delta', acc_metric_delta),
              ('loss mse', loss_mse),
              ('loss mae', loss_mae)]

    # Open the file in append mode and write the values
    with open('logs/interpolate_fullMemList_logs.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the values as a row
        #writer.writerow([name for name, result in values])  # Headers (Optional)
        # Write the actual numeric values
        writer.writerow([result.numpy() if hasattr(result, 'numpy') else result for name, result in values])

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-n', '--n_songs', type=int,
                            help='Number of files to generate', default=1)

    arg_parser.add_argument('-c', '--checkpoint_path', type=str,
                            help = 'Path to the saved weights',
                            default = "data/checkpoints_music/checkpoint" + str(CHECKPOINT_EPOCH) + ".weights.h5")

    arg_parser.add_argument('-np', '--npz_dir', type=str, default='data/npz_temp',
                            help='Directory with the npz files')

    arg_parser.add_argument('-o', '--dst_dir', type=str, default='data/generated_midis',
                            help='Directory where the generated midi files will be stored')

    arg_parser.add_argument('-k', '--top_k', type=int, default=1)

    arg_parser.add_argument('-t', '--temp', type=float, default=0.35,
                            help='Temperature of softmax')

    arg_parser.add_argument('-f', '--filenames', nargs='+', type=str, default=None,
                            help='Names of the generated midis. Length must be equal to n_songs')

    arg_parser.add_argument('-l', '--input_length', nargs='+', type=int, default=None,
                            help='Names of the generated midis. Length must be equal to n_songs')

    arg_parser.add_argument('-v', '--visualize_attention', action='store_true',
                            help='If activated, the attention weights will be saved as images')


    args = arg_parser.parse_args()

    assert isinstance(args.n_songs, int)
    assert args.n_songs > 0
    assert pathlib.Path(args.checkpoint_path).is_file()
    assert pathlib.Path(args.npz_dir).is_dir()
    if pathlib.Path(args.dst_dir).exists():
        assert pathlib.Path(args.dst_dir).is_dir()
    else:
        pathlib.Path(args.dst_dir).mkdir(parents=True, exist_ok=True)
    assert isinstance(args.top_k, int)
    assert args.top_k > 0
    assert isinstance(args.temp, float)
    assert args.temp > 0.0
    if args.filenames is None:
        midi_filenames = [str(i)+"_interpolateInput" for i in range(1, args.n_songs + 1)]
    else:
        midi_filenames = args.filenames
    midi_filenames = [f + '.midi' for f in midi_filenames]
    midi_filenames = [os.path.join(args.dst_dir, f) for f in midi_filenames]
    assert len(midi_filenames) == args.n_songs
    assert len(set(midi_filenames)) == len(midi_filenames)

    # ============================================================
    # ============================================================

    npz_filenames = list(pathlib.Path(args.npz_dir).rglob('0.npz'))
    assert len(npz_filenames) > 0
    filenames_sample = np.random.choice(
        npz_filenames, args.n_songs, replace=False)

    idx_to_time = get_quant_time()

    tf.config.run_functions_eagerly(False)
    midi_parser = MIDI_parser.build_from_config(config, idx_to_time)
    model, _ = Music_transformer.build_from_config(
        config=config, checkpoint_path=args.checkpoint_path)


    batch_size = len(filenames_sample)
    soundsAll, deltasAll = zip(*[midi_parser.load_features(filename)
                                 for filename in filenames_sample])


    song_len = soundsAll[0].shape[0]
    cutted_song_len = int(song_len / 4)
    interpol_len = cutted_song_len

    if args.input_length is not None:
        cutted_song_len = args.input_length[0]

    sounds = np.array([sound[cutted_song_len:cutted_song_len*(N_GEN_SEQ+1)] for sound in soundsAll])
    deltas = np.array([delta[cutted_song_len:cutted_song_len*(N_GEN_SEQ+1)] for delta in deltasAll])

    labels_sounds = np.array([sound[cutted_song_len*2:cutted_song_len*(N_GEN_SEQ+2)] for sound in soundsAll])
    labels_deltas = np.array([delta[cutted_song_len*2:cutted_song_len*(N_GEN_SEQ+2)] for delta in deltasAll])

    sounds2 = np.array([sound[:cutted_song_len] for sound in soundsAll])
    deltas2 = np.array([delta[:cutted_song_len] for delta in deltasAll])

    alphas = np.linspace(0,1,11)
    for alpha in alphas:
        print("alpha:", alpha)

        # compute the output of the whole song with the first 25% of the song
        sounds, deltas, attention_loss_list, attention_weight_list, _ = generate(model=model,
                                                                                 sounds=sounds,
                                                                                 deltas=deltas,
                                                                                 pad_idx=config.pad_idx,
                                                                                 top_k=args.top_k,
                                                                                 temp=args.temp,
                                                                                 alpha=alpha,
                                                                                 interpol_len=interpol_len,
                                                                                 sounds2=sounds2,
                                                                                 deltas2=deltas2)

        loss_mse, loss_mae, acc_metric_sound, acc_metric_delta = computeLoss(model, sounds,
                                                                             deltas, labels_sounds,
                                                                             labels_deltas)
        saveValues(npz_filenames, npz_filenames, song_len, cutted_song_len, interpol_len, acc_metric_sound.result(), acc_metric_delta.result(),
                  loss_mse.result(), loss_mae.result(), alpha)

    if args.visualize_attention:

        viz_dir = 'vizualized_attention'
        pathlib.Path(viz_dir).mkdir(parents=True, exist_ok=True)

        for layer_idx, layer_weights in enumerate(attention_weight_list, 1):
            for head_idx, head_weights in enumerate(layer_weights[0, ...].numpy(), 1):

                img_path = os.path.join(
                    viz_dir, f'layer{layer_idx}_head{head_idx}.png')
                plt.figure(figsize=(17, 14))
                plt.step(np.arange(head_weights.shape[1]), head_weights[0])
                #plt.imsave(img_path, head_weights, cmap='Reds')
                plt.savefig(img_path)
