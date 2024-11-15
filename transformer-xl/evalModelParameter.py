import csv
from midi_parser import MIDI_parser
from model import Music_transformer
import config_music as config
from utils import get_quant_time, softmax_with_temp
import numpy as np
import argparse
import os
import pathlib
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


def generate(model, sounds, deltas, pad_idx, seq_len, mem_len, gen_len,temp, top_k=1):

    max_len = gen_len
    i=0
    inputs_sound = tf.constant(sounds[:, -seq_len:])
    inputs_delta = tf.constant(deltas[:, -seq_len:])

    outputs_sound, outputs_delta, next_mem_list, attention_weight_list, attention_loss_list = model(
        inputs=(inputs_sound, inputs_delta),
        mem_list=None,
        next_mem_len=mem_len,
        training=False,
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
        i += 1
        outputs_sound, outputs_delta, next_mem_list, attention_weight_list, attention_loss_list = model(
            inputs=(inputs_sound, inputs_delta),
            mem_list=next_mem_list,
            next_mem_len=mem_len,
            training=False,
            inputs2=(None, None)
        )

    sounds = sounds[:, seq_len:]
    deltas = deltas[:, seq_len:]

    return sounds, deltas, next_mem_list, attention_weight_list, attention_loss_list

def saveValues(npz_filenames, song_len, seq_len, gen_len, mem_len, temp, acc_metric_sound, acc_metric_delta, loss_mse, loss_mae, alpha=0):

    values = [('filename', os.path.basename(npz_filenames[0])),
              ('song length', song_len),
              ('seq_len', seq_len ),
              ('gen_len', gen_len),
              ('mem_len', mem_len),
              ('temp', temp),
              ('acc_sound', acc_metric_sound),
              ('acc_delta', acc_metric_delta),
              ('loss mse', loss_mse),
              ('loss mae', loss_mae)]

    # Open the file in append mode and write the values
    with open('logs/analyseParameters.csv', mode='a', newline='') as file:
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

    arg_parser.add_argument('-f', '--filenames', nargs='+', type=str, default=None,
                            help='Names of the generated midis. Length must be equal to n_songs')


    args = arg_parser.parse_args()

    # ============================================================
    # ============================================================

    filenames_npz =['1787.npz', '1280.npz']
    for filename_npz in filenames_npz:
        print("filename:", filename_npz)
        npz_filenames = list(pathlib.Path("data/npz_temp").rglob(filename_npz))
        assert len(npz_filenames) > 0
        filenames_sample = np.random.choice(
            npz_filenames, args.n_songs, replace=False)

        idx_to_time = get_quant_time()

        tf.config.run_functions_eagerly(False)
        midi_parser = MIDI_parser.build_from_config(config, idx_to_time)


        batch_size = len(filenames_sample)
        soundsAll, deltasAll = zip(*[midi_parser.load_features(filename)
                                     for filename in filenames_sample])
        song_len = soundsAll[0].shape[0]

        model, _ = Music_transformer.build_from_config(
            config=config, checkpoint_path=args.checkpoint_path, max_seq_len=song_len)

        seq_len_list = [500, 1500, 2500]
        gen_len_list = [500, 1500, 2500]
        mem_len_list = [0, 500, 1500, 2500]
        temp = 0.5
        for seq_len in seq_len_list:
            print("-seq_len:", seq_len)

            sounds = np.array([sound[:seq_len] for sound in soundsAll])
            deltas = np.array([delta[:seq_len] for delta in deltasAll])

            for gen_len in gen_len_list:
                print("--gen_len:", gen_len)
                labels_sounds = np.array(
                    [sound[seq_len:seq_len + gen_len] for sound in soundsAll])
                labels_deltas = np.array(
                    [delta[seq_len:seq_len + gen_len] for delta in deltasAll])

                for mem_len in mem_len_list:
                    print("---mem_len:", mem_len)
                    # compute the output of the whole song with the first 25% of the song
                    out_sounds, out_deltas, attention_loss_list, attention_weight_list, _ = generate(model=model,
                                                                                             sounds=sounds,
                                                                                             deltas=deltas,
                                                                                             pad_idx=config.pad_idx,
                                                                                             seq_len=seq_len,
                                                                                             mem_len=mem_len,
                                                                                             gen_len=gen_len,
                                                                                             temp=temp)
                    loss_mse, loss_mae, acc_metric_sound, acc_metric_delta = computeLoss(model, out_sounds,
                                                                                         out_deltas, labels_sounds,
                                                                                         labels_deltas)
                    saveValues(npz_filenames, song_len, seq_len, gen_len, mem_len, temp, acc_metric_sound.result(), acc_metric_delta.result(),
                               loss_mse.result(), loss_mae.result())