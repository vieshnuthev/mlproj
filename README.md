# MLproj

End-to-end dessert classification project for the GitHub submission.

## Team Members

- ALAN SYAHMI BIN SANIH @ SANI – BI23113044
- AZAM ABDULLAH – BI23110393
- Khuhan A/L S Magendran – BI23110085
- HARIESS KUMARAN – BI23110313
- MUHAMMAD IRFAN SHAH BIN MUHAMMAD AIMAN MARAN – BI23110214
- VIESHNU A/L VAGANANTHAN – BI23110185

## Project Structure

The repo is arranged so it is easier to navigate:

```text
MLproj/
├── end_to_end_pipeline.py
├── requirements.txt
├── dessert_dataset/
├── scripts/
│   └── milestones/
├── assets/
│   ├── figures/
│   └── videos/
└── outputs/
```

Each milestone script is stored under `scripts/milestones/` so it is easier to inspect the stages separately.

## Main Pipeline

The main entry point is `end_to_end_pipeline.py`. It runs the full pipeline locally:

1. Load the dataset from `dessert_dataset/train` and `dessert_dataset/validation`
2. Show a sampled image grid for the data pipeline milestone
3. Build and summarize the CNN architecture
4. Train the model and save learning curves
5. Run final multi-seed evaluation and export test results

## Full Dataset Source

This project was sampled from the full Food-101 dataset available on Kaggle:

https://www.kaggle.com/datasets/hari31416/food-101

Food-101 is a large benchmark image dataset with many food categories. For this project, we used the dessert portion of that dataset and organized it into training and validation folders for the milestone workflow. That keeps the project focused while still being based on a widely used real-world image classification dataset. This is also to address GitHub's limitation on uploading huge datasets (in this case, the original dataset is ~10GB after unzipped).

## Setup

Install the Python dependencies:

```powershell
pip install -r requirements.txt
```

## How To Run

Run the full pipeline:

```powershell
python end_to_end_pipeline.py
```

Optional faster demo run:

```powershell
python end_to_end_pipeline.py --epochs 1 --sample-per-class 1
```

Optional arguments:

```powershell
python end_to_end_pipeline.py --epochs 10 --seeds 42 123 456 --sample-per-class 0 --show-plots
```

## Dataset Layout

The scripts expect this structure:

```text
dessert_dataset/
  train/
    class_name_1/
    class_name_2/
    ...
  validation/
    class_name_1/
    class_name_2/
    ...
```

## Outputs

All generated artifacts are saved in the `outputs/` folder.

The reference images and videos used for the report are stored in `assets/figures/` and `assets/videos/`.
