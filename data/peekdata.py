import numpy as np

data = np.load("sketchrnn/angel.npz", encoding="latin1", allow_pickle=True)

print("Keys in NPZ:", data.files)

for key in data.files:
    print(f"{key} :", len(data[key]))

for key in data.files:
    arr = data[key]
    size_mb = arr.nbytes / (1024 * 1024)
    print(
        f"Key: {key} | Shape: {arr.shape} | Data Type: {arr.dtype} | Size: {size_mb:.2f} MB"
    )

##example: keyword"angel"
# Keys in NPZ: ['test', 'train', 'valid']
# test : 2500
# train : 70000
# valid : 2500
# Key: test | Shape: (2500,) | Data Type: object | Size: 0.02 MB
# Key: train | Shape: (70000,) | Data Type: object | Size: 0.53 MB
# Key: valid | Shape: (2500,) | Data Type: object | Size: 0.02 MB

