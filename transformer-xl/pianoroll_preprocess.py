import pickle
from pypianoroll import Multitrack, Track, StandardTrack
from scipy.sparse import coo_matrix

with open("music.pk", "rb") as f:
    music = pickle.load(f)

def write_midi(mus):
    tr = StandardTrack(pianoroll=(mus["piece"] * 127).toarray(), program=0, is_drum=False)
    mt = Multitrack(name=mus["name"], tracks=[tr])
    fn = "midis/" + mus["name"] + ".mid"
    #fn = mus["name"] + ".mid"  # Add a file extension
    mt.write(fn)  # Save as MIDI file

for m in music:
    write_midi(m)
