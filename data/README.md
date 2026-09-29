# Data

This directory will hold **locally downloaded and processed** QuickDraw data.

- Raw QuickDraw data should **not** be committed to GitHub. Everything in this
  directory except this README is ignored by `.gitignore`.
- Dataset source: [The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)

- Among the given [Quickdraw_dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22)), we are using the sketchrnn dataset only. 

## Download and preprocessing

- If you have the Google Cloud SDK installed, you can download the entire `sketchrnn` folder using:

gsutil -m cp -r gs://quickdraw_dataset/sketchrnn .


### About Sketchrnn & key features

- Vector Stroke Representation (Stroke-5 Format)
- Quick, Draw! dataset contains other raw, high-resolution stroke coordinates including ndjson, we chose the Sketchrnn dataset (.npz) specifically provides simplified, pre-processed stroke vectors. 
- Sketchrnn applies the Ramer-Douglas-Peucker (RDP) algorithm to prune redundant control points, resulting in much cleaner and more compact sequence data. Unlike the Full dataset's absolute coordinate values, Sketch-RNN normalizes movements into relative offsets $(\Delta x, \Delta y)$ along with explicit pen-state flags ($p_1, p_2, p_3$). This filtering drastically reduces sequence length and memory footprint, making .npz files optimized for recurrent architectures rather than dense image processing. 
- Sketchrnn format is tailored specifically for generative stroke modeling and vector sequence prediction.