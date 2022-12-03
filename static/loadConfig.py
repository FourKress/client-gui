import sys
import numpy as np

dir_file = (sys.argv[1])
setting_info_loaded = np.load(dir_file,allow_pickle=True)
setting_info = setting_info_loaded.item()

print(setting_info)
sys.stdout.flush()
