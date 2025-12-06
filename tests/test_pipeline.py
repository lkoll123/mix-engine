from __future__ import annotations
import os
import sys

import soundfile as sf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "tests", "test_data")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mix_engine.engine import mix_Engine
from mix_engine.songOrderEngine import song_Order_Engine
from mix_engine.dj_bridge import get_dj_mix_curves

# -------------------------------
# Build test playlist
# -------------------------------

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
    os.path.join(TEST_DATA_DIR, "1942 -  1 G Eazy ft_ Yo Gotti YBN Nahmir.wav".replace("  1 ", " ")),
    os.path.join(TEST_DATA_DIR, "BAD_.wav"),
    os.path.join(TEST_DATA_DIR, "Billy.wav"),
    os.path.join(TEST_DATA_DIR, "BlocBoy JB _LOOK ALIVE_ ft_ Drake.wav"),
    os.path.join(TEST_DATA_DIR, "Calling My Spirit.wav"),
    os.path.join(TEST_DATA_DIR, "Change Lanes.wav"),
]

order_Eng = song_Order_Engine()
mix_Eng = mix_Engine()

def main():
    # 1) Order playlist
    song_Order, cost = order_Eng.solve_tsp(test_data)

    print("Song Order:\n")
    for song in song_Order["path"]:
        print(song)
    print(f"total cost: {cost}")

    paths = song_Order["path"]
    d_songs = song_Order["d_Song"]

    seams = []

    # 2) For each adjacent pair, get DJtransGAN curves and mix once
    for i in range(len(paths) - 1):
        prev_path = paths[i]
        next_path = paths[i + 1]
        prev_song = d_songs[i]
        next_song = d_songs[i + 1]

        print(f"\n=== Seam {i} ===")
        print(f"Prev: {prev_path}")
        print(f"Next: {next_path}")

        dj_curves = get_dj_mix_curves(prev_path, next_path)
        print(f"Got {len(dj_curves)} curve bands")

        seam = mix_Eng.mix_songs(prev_song, next_song, dj_curves)
        seams.append(seam)

    # 3) Write seams to tests/output
    out_dir = os.path.join(PROJECT_ROOT, "tests", "output")
    os.makedirs(out_dir, exist_ok=True)

    for i, seam_info in enumerate(seams):
        out_path = os.path.join(out_dir, f"seam_{i:02d}.wav")
        sf.write(out_path, seam_info["y_overlap"], seam_info["sr"])
        print(f"  wrote: {out_path}")


if __name__ == "__main__":
    main()
