import pickle
from pprint import pprint as pprint 

with open("generated_prob_dist_data/prob_distributions_generated_NUM_SAMPLES_PER_BIN_2_and_NUM_BINS_20.pkl", "rb") as f:
    data = pickle.load(f)

pprint(data)
for entropy_bin, prob_distributions in data.items():
    print(f"Number of samples in entropy bin {entropy_bin} is: {len(prob_distributions)}")