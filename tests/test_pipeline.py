from __future__ import annotations
import os
import sys


import soundfile as sf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "tests", "test_data")
sys.path.insert(0, SRC_DIR)
from mix_engine.engine import mix_Engine
from mix_engine.songOrderEngine import song_Order_Engine
from mix_engine.dj_bridge import get_dj_mix_curves

order_Eng = song_Order_Engine()
mix_Eng = mix_Engine()

test_data = [
    os.path.join(TEST_DATA_DIR, "New Patek.wav"),
    os.path.join(TEST_DATA_DIR, "Gucci Mane_ Bruno Mars_ Kodak Black - Wake Up In The Sky.wav"),
    os.path.join(TEST_DATA_DIR, "KOODA.wav"),
    os.path.join(TEST_DATA_DIR, "Spin The Block _Feat_ Future_.wav"),
    os.path.join(TEST_DATA_DIR, "Startender feat_ Offset _ Tyga.wav"),
    os.path.join(TEST_DATA_DIR, "Swervin feat_ 6ix9ine.wav"),
    os.path.join(TEST_DATA_DIR, "whoa _mind in awe_.wav"),
    os.path.join(TEST_DATA_DIR, "Young Thug _ Gunna - Chanel _Go Get It_ ft_ Lil Baby.wav"),
    os.path.join(TEST_DATA_DIR, "ZEZE _feat_ Travis Scott _ Offset_.wav"),
    os.path.join(TEST_DATA_DIR, "eterna-cancao-wav-12569.wav"),
    os.path.join(TEST_DATA_DIR, "Cry Alone.wav"),
    os.path.join(TEST_DATA_DIR, "1942 -  G Eazy ft_ Yo Gotti YBN Nahmir.wav"),
    os.path.join(TEST_DATA_DIR, "BAD_.wav"),
    os.path.join(TEST_DATA_DIR, "Billy.wav"),
    os.path.join(TEST_DATA_DIR, "BlocBoy JB _LOOK ALIVE_ ft_ Drake.wav"),
    os.path.join(TEST_DATA_DIR, "Calling My Spirit.wav"),
    os.path.join(TEST_DATA_DIR, "Change Lanes.wav"),

]

song_Order, cost = order_Eng.solve_tsp(test_data)
print("Song Order: \n")

for song in song_Order["path"]:
    print(song)

print(f"total cost: {cost}")

test_param = [
    {
        # LOW band (kick/bass): let A hang on longer so low end doesn't instantly jump
        "f_lo_bin": 0,
        "f_hi_bin": 200,
        # A fades out slowly, starts basically at t=0, gentle slope
        "a_time": {"s": 0.0, "delta": 1.0},
        # B fades in a bit later so two kicks don't fully slam at once
        "b_time": {"s": 0.2, "delta": 1.0},
    },
    {
        # MID band (vocals / body): cross them more in the middle
        "f_lo_bin": 200,
        "f_hi_bin": 800,
        # A fades out starting kinda early-mid
        "a_time": {"s": 0.1, "delta": 1.5},
        # B fades in not too late
        "b_time": {"s": 0.1, "delta": 1.5},
    },
    {
        # HIGH band (hats / air): let B's highs in early so it "feels" like new song
        "f_lo_bin": 800,
        "f_hi_bin": 2047,
        # A highs drop fairly early
        "a_time": {"s": 0.05, "delta": 2.0},
        # B highs come in basically right away
        "b_time": {"s": 0.0, "delta": 2.0},
    },
]

for i in range(len(song_Order["path"]) - 1):
    prev_path = song_Order["path"][i]
    next_path = song_Order["path"][i+1]

    seam = get_dj_mix_curves(prev_path, next_path)
    print(f'Seam between:\n  {prev_path}\n  -> {next_path}:\n  {seam}')
seams = mix_Eng.mix_playlist(song_Order["d_Song"], test_param)
# Prepare output dir under tests/output
out_dir = os.path.join(PROJECT_ROOT, "tests", "output")
os.makedirs(out_dir, exist_ok=True)

for i in range(len(song_Order["path"]) - 1):
    seam_info = seams[i]

    # write file
    out_path = os.path.join(out_dir, f"seam_{i:02d}.wav")
    sf.write(out_path, seam_info["y_overlap"], seam_info["sr"])
    print(f"  wrote: {out_path}")
