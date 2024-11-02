import joblib

tempo = 500000
ppq = 480
numerator = 4
denominator = 4
clocks_per_click = 24
notated_32nd_notes_per_beat = 8

cc_kept = [64, 67]
cc_threshold = 64
cc_lower = 0
cc_upper = 127

vel_value = 64

n_notes = 128

n_cc = 2 * len(cc_kept)

n_sounds = 2 * n_notes + n_cc + 1

n_deltas = 66 + 1


pad_idx = 0
n_jobs = joblib.cpu_count()

# dimensions of embeddings for sound events, delta time and their combined representation
d_sound = 384
d_delta = 256
d_combined = d_sound + d_delta

n_heads_sound = 6
n_heads_delta = 4
n_heads_combined = n_heads_sound + n_heads_delta

# For lstm layer
n_units_sound = 6
n_units_delta = 4
n_heads_combined = n_heads_sound + n_heads_delta

n_layers_sound = 3
n_layers_delta = 3
n_layers_combined = 6

# max for seq_len = 2048, else the max_seq_len must be changed
seq_len = 256
mem_len = 384

batch_size = 1
dropout_rate = 0.1
n_epochs = 60 
max_segs_per_batch = 20
lr = 0.00002

use_attn_reg = True


dataset_url = 'https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip'
