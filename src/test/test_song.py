from engine import MixEngine

testObj = MixEngine()

test_path_1 = "/Users/lukakoll/Documents/mix-engine/src/test/test_data/eterna-cancao-wav-12569.wav"
test_path_2 = "/Users/lukakoll/Documents/mix-engine/src/test/test_data/memphis-trap-wav-349366.wav"

print(f"Similarity Score {testObj.get_similarity_det(test_path_1, test_path_2)}")
