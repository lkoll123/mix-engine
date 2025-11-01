import os
import sys

import soundfile as sf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
from mix_engine.engine import mix_Engine
from mix_engine.songOrderEngine import song_Order_Engine

order_Eng = song_Order_Engine()
mix_Eng = mix_Engine()

test_data = [
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/New Patek.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Gucci Mane_ Bruno Mars_ Kodak Black - Wake Up In The Sky.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/KOODA.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Spin The Block _Feat_ Future_.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Startender feat_ Offset _ Tyga.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Swervin feat_ 6ix9ine.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/whoa _mind in awe_.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Young Thug _ Gunna - Chanel _Go Get It_ ft_ Lil Baby.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/ZEZE _feat_ Travis Scott _ Offset_.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/eterna-cancao-wav-12569.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Cry Alone.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/1942 -  G Eazy ft_ Yo Gotti YBN Nahmir.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/BAD_.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Billy.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/BlocBoy JB _LOOK ALIVE_ ft_ Drake.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Calling My Spirit.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Change Lanes.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Climax _feat_ 6lack_.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Demons and Angels feat_ Juice WRLD.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Creeping _feat_ Rich The Kid_prod_ by Menoh Beats_.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Dip.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/ESSKEETIT.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/FEFE _Feat_ Nicki Minaj _ Murda Beatz_.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Gave It All I Got - Prod_ By C - Clip Beatz.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Genie.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Gunna - Oh Okay _Ft Young Thug _ Lil Baby_.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/I Kill People_ ft Tadoe _ Chief Keef _Produced by_ Ozmusiqe_ RR.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Japan _Prod_ _JGramm_.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Kanye West _ Lil Pump - I Love It.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/KEKE _Ft_ Fetty Wap _ A Boogie wit da Hoodie_.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/KIKA ft Tory Lanez.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Leave Me Alone _Prod_ by Young Forever x Cast Beats_.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Look Back At It.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Meek Mill Ft_ Rick Ross - Ima Boss.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/Meek Mill Ft_ Rick Ross - Ima Boss.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/MURDER ON MY MIND _Explicit_.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/REEL IT IN.wav",
    "/Users/lukakoll/Documents/mix-engine/tests/test_data/sheck wes - mo bamba _prod_ 16yrold _ take a daytrip_.wav",
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

seams = mix_Eng.mix_playlist(song_Order["d_Song"], test_param)
# Prepare output dir under tests/output
out_dir = os.path.join(PROJECT_ROOT, "tests", "output")
os.makedirs(out_dir, exist_ok=True)

for i in range(len(song_Order["path"]) - 1):
    seam_info = seams[i]

    print(f'\nSeam between:\n  {song_Order["path"][i]}\n  -> {song_Order["path"][i+1]}')

    # write file
    out_path = os.path.join(out_dir, f"seam_{i:02d}.wav")
    sf.write(out_path, seam_info["y_overlap"], seam_info["sr"])
    print(f"  wrote: {out_path}")
