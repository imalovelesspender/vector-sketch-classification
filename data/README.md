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
- 345 keywords, 70,000 shapes per keyword
- Quick, Draw! dataset contains other raw, high-resolution stroke coordinates including ndjson, we chose the Sketchrnn dataset (.npz) specifically provides simplified, pre-processed stroke vectors. 
- Sketchrnn applies the Ramer-Douglas-Peucker (RDP) algorithm to prune redundant control points, resulting in much cleaner and more compact sequence data. Unlike the Full dataset's absolute coordinate values, Sketch-RNN normalizes movements into relative offsets $(\Delta x, \Delta y)$ along with explicit pen-state flags ($p_1, p_2, p_3$). This filtering drastically reduces sequence length and memory footprint, making .npz files optimized for recurrent architectures rather than dense image processing. 
- Sketchrnn format is tailored specifically for generative stroke modeling and vector sequence prediction.

## Using the data for classification (LSTM pipeline)

The classification code expects one Sketch-RNN file per category in
`data/sketchrnn/<category>.npz`; the file name is the class label
(`angel.npz` -> `angel`). The `gsutil` command above downloads all 345
categories; for a small run you can fetch individual files instead:

```bash
mkdir -p data/sketchrnn
for c in angel cat apple; do
  curl -o data/sketchrnn/$c.npz https://storage.googleapis.com/quickdraw_dataset/sketchrnn/$c.npz
done
```

As stored in the files, each sketch is an `int16` array of shape `(seq_len, 3)`
in **stroke-3** format `(dx, dy, pen_lift)`, where `pen_lift = 1` marks the
last point of a stroke (this is what `drawNpz.py` uses as end-of-stroke).
Stroke-5 `(dx, dy, p1, p2, p3)` is the Sketch-RNN model's internal format; the
loader in `src/data/quickdraw.py` reads stroke-3 and can convert to stroke-5
with `"stroke_format": "stroke5"` in the config.
