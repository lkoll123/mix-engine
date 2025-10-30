import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
from mix_engine.songOrderEngine import song_Order_Engine

testObj = song_Order_Engine()

test_data = [
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/1942 -  G Eazy ft_ Yo Gotti YBN Nahmir.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/BAD_.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Billy.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/BlocBoy JB _LOOK ALIVE_ ft_ Drake.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Calling My Spirit.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Change Lanes.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Climax _feat_ 6lack_.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Demons and Angels feat_ Juice WRLD.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Creeping _feat_ Rich The Kid_prod_ by Menoh Beats_.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Cry Alone.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Dip.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/ESSKEETIT.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/FEFE _Feat_ Nicki Minaj _ Murda Beatz_.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/eterna-cancao-wav-12569.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Gave It All I Got - Prod_ By C - Clip Beatz.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Genie.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Gucci Mane_ Bruno Mars_ Kodak Black - Wake Up In The Sky.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Gunna - Oh Okay _Ft Young Thug _ Lil Baby_.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/I Kill People_ ft Tadoe _ Chief Keef _Produced by_ Ozmusiqe_ RR.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Japan _Prod_ _JGramm_.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Kanye West _ Lil Pump - I Love It.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/KEKE _Ft_ Fetty Wap _ A Boogie wit da Hoodie_.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/KIKA ft Tory Lanez.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/KOODA.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Leave Me Alone _Prod_ by Young Forever x Cast Beats_.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Look Back At It.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Meek Mill Ft_ Rick Ross - Ima Boss.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Meek Mill Ft_ Rick Ross - Ima Boss.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/MURDER ON MY MIND _Explicit_.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/New Patek.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/REEL IT IN.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/sheck wes - mo bamba _prod_ 16yrold _ take a daytrip_.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Spin The Block _Feat_ Future_.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Startender feat_ Offset _ Tyga.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Swervin feat_ 6ix9ine.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/whoa _mind in awe_.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/Young Thug _ Gunna - Chanel _Go Get It_ ft_ Lil Baby.wav',
    '/Users/lukakoll/Documents/mix-engine/tests/test_data/ZEZE _feat_ Travis Scott _ Offset_.wav'
]

song_Order, cost = testObj.solve_tsp(test_data)
print("Song Order: \n")

for song in song_Order:
    print(song)

print(f'total cost: {cost}')