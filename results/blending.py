import pandas as pd
import numpy as np

models = [
pd.read_csv("sub-0.csv"), #0.87603
pd.read_csv("sub-2-feature_net.csv"), #0.87929
pd.read_csv("sub-9-gru_res_conv.csv"), #0.87995
pd.read_csv("sub-10-gru_res_conv-whole_dataset.csv"), #0.88965
]

blend = models[0].copy()

for label in ["time_slot_0","time_slot_1","time_slot_2","time_slot_3","time_slot_4","time_slot_5","time_slot_6","time_slot_7","time_slot_8","time_slot_9","time_slot_10","time_slot_11","time_slot_12","time_slot_13","time_slot_14","time_slot_15","time_slot_16","time_slot_17","time_slot_18","time_slot_19","time_slot_20","time_slot_21","time_slot_22","time_slot_23","time_slot_24","time_slot_25","time_slot_26","time_slot_27"]:
	ttlProb = 0
	for m in range(len(models)):
		ttlProb += models[m][label]
	blend[label] = ttlProb / len(models)

blend.to_csv('sub-blend.csv', index = False)


