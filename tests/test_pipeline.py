from __future__ import annotations

import os
import sys

import soundfile as sf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "tests", "test_data")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mix_engine.dj_bridge import get_dj_mix_curves
from mix_engine.engine import mix_Engine
from mix_engine.songOrderEngine import song_Order_Engine
from mix_engine.inference import run_mixer_for_playlist

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
    os.path.join(TEST_DATA_DIR, "I Kill People_ ft Tadoe _ Chief Keef _Produced by_ Ozmusiqe_ RR.wav"),
    os.path.join(TEST_DATA_DIR, "Demons and Angels feat_ Juice WRLD.wav"),
    os.path.join(TEST_DATA_DIR, "Meek Mill Ft_ Rick Ross - Ima Boss.wav"),
    os.path.join(TEST_DATA_DIR, "Creeping _feat_ Rich The Kid_prod_ by Menoh Beats_.wav"),
    os.path.join(TEST_DATA_DIR, "FEFE _Feat_ Nicki Minaj _ Murda Beatz_.wav"),
    os.path.join(TEST_DATA_DIR, "KIKA ft Tory Lanez.wav"),
    os.path.join(TEST_DATA_DIR, "MURDER ON MY MIND _Explicit_.wav"),
    os.path.join(TEST_DATA_DIR, "KEKE _Ft_ Fetty Wap _ A Boogie wit da Hoodie_.wav"),
    
]



order_Eng = song_Order_Engine()
mix_Eng = mix_Engine()

def main():
   run_mixer_for_playlist(test_data, os.path.join(PROJECT_ROOT, "tests", "output"))


if __name__ == "__main__":
    main()
