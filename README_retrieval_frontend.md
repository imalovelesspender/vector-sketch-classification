## Sketch frontend

`frontend/` is a small drawing app (pygame) for trying the classifier by hand.
It preprocesses the drawing the same way as the training data (QuickDraw
simplification, stroke-3, the normalization in
`data_preprocessing/preprocess.ipynb`) and sends the points to a backend.

```bash
pip install -r requirements.txt
python frontend/app.py                                              # mock backend
python frontend/app.py --backend-url http://127.0.0.1:8000/predict  # real backend
```

The request/response JSON is documented in `frontend/backend.py`; settings are
in `configs/frontend.json`. The mock backend returns placeholder labels only.
