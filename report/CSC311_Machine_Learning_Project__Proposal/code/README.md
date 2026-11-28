# CSC311 Project: Data Exploration

`code.py` explores the survey data in `csc311_project_training_202609.csv`
(each student describes baseball, basketball and soccer). It splits the
students into training and test sets, prints summary statistics about the
training set, and saves a figure.

## Files

| File | Description |
| --- | --- |
| `code.py` | The data exploration script |
| `csc311_project_training_202609.csv` | Survey data (870 rows: 290 students × 3 sports) |
| `requirements.txt` | Python libraries the script needs |
| `figure.png` | Figure produced by the script |

## Setup

You need Python 3.10 or newer. Run every command below from inside this
`code/` folder.

### 1. Create a virtual environment

```bash
python3 -m venv .venv
```

### 2. Activate it

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Windows (Command Prompt):

```bat
.venv\Scripts\activate.bat
```

Your prompt should now start with `(.venv)`.

### 3. Install the libraries

```bash
pip install -r requirements.txt
```

This installs `numpy`, `pandas` and `matplotlib`.

## Running the script

With the virtual environment active, from inside `code/`:

```bash
python code.py
```

The script opens the CSV and saves the figure by relative path, so run it
from this folder. Running it from anywhere else gives a `FileNotFoundError`.

When you are done, leave the virtual environment with:

```bash
deactivate
```

## Outputs

### Train/test split

The script holds out 44 students (132 rows) as a test set, using a fixed
random seed (`311`), so you get the same split every time. All three rows from
one student stay in the same split. Every statistic below is computed on the
**training set only** (246 students, 738 rows).

### Printed to the terminal

The script prints five sections:

1. **Basic overview**: number of rows and students, class sizes, the
   train/test sizes, the test student ids, and a check that no student is in
   both splits.
2. **Missing values per feature**: how many training rows left each question
   blank (the song question is missing most often, at about 8%).
3. **Distributions and patterns**:
   - (a) mean and median ease-of-starting rating (0–10) for each sport
   - (b) for each Likert statement, the share of answers that are Agree or
     Strongly agree, by sport
4. **Price data**: how the free-text ticket prices were parsed (single number,
   range midpoint, refusal = $0, or unparseable), the overall median,
   95th percentile and maximum, the median price per sport, and the skewness
   before and after a `log(1 + p)` transform.
5. **Text fields** (food & drink, colour, song, feeling):
   - how many distinct colour answers remain after normalizing case and
     punctuation
   - vocabulary size per field, how many tokens appear in at least 3 rows
     (these become bag-of-words features), and the share of tokens that
     appear only once
   - the most frequent tokens per sport for each field
   - the most frequent full song answers per sport
   - the share of rows per sport whose feeling answer mentions "excited" or
     "bored"

Example (start of the output):

```
Basic overview:
rows: 870, students: 290
rows per student: [3]
class sizes (entire dataset): {'Soccer': 290, 'Baseball': 290, 'Basketball': 290}
id range: 1 to 290
test: 44 students, 132 rows
train: 246 students, 738 rows
...
Saved figure.png
```

### Saved figure: `figure.png`

The script writes `figure.png` to this folder (overwriting any existing copy).
It has three panels:

- **(a) Ease of starting**: box plot of the 0–10 ease rating per sport; the
  diamond marks the mean.
- **(b) Likert statements**: grouped bars showing the share who agree with
  each statement ("Strategy matters", "High injury risk",
  "Individual > teamwork"), coloured by sport.
- **(c) Parsed ticket price**: box plot of willingness to pay per sport, on a
  log scale ($0 to $10k).
