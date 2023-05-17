# Neural-Guided Room Layout Generation with Bubble Diagram Constraints

![teaser](./bin/teaser.png)

---
## Installation
- pytorch
- numpy
- tqdm
- shutil
- json
- opencv-python
- networkx
- ortools
- cvxopt
 
---
## Data

- download the RPLAN dataset from https://docs.google.com/forms/d/e/1FAIpQLSfwteilXzURRKDI5QopWCyOGkeb_CFFbRwtQ0SOPhEg0KGSfw/viewform?usp=sf_link
- extract png data into './data/rplan/'
- preprocess png to json, run 
```
python3 ./data_utils/prepropcessing.py --png_path [path to png] --json_path [path to json]
```
- split training/valid/test set, run 
```
python3 ./data_utils/data_split.py --out_path [path to split.txt]
```
---

## Training

- train topology, run 
```
python3 train.py --data_path [path to json] --split [path to train_split.txt] --log_dir [path for saving model]
```
- train geometry, run 
```
python3 train_geo.py --data_path [path to json] --split [path to split.txt] --log_dir [path for saving model]
```

---

## Reasoning

- predict topology, run 
```
python3 test.py --data_path [path to json] --split [path to test_split.txt] --log_dir [path for result_1]
```
- predict geometry, run 
```
python3 test_geo.py --step_1_path [path to result_1] --split [path to test_split.txt] --log_dir [path for result_2]
```

- optimization, run 
```
python3 ./post/orthogonal_drawing.py --json_path [path for result_2] --png_path [path for images]
```

