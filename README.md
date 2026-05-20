# MLproj

End-to-end dessert classification project for the GitHub submission.

## What This Repo Contains

The main entrypoint is `end_to_end_pipeline.py`. It runs the full pipeline locally:

1. Load the dataset from `dessert_dataset/train` and `dessert_dataset/validation`
2. Show a sampled image grid for the data pipeline milestone
3. Build and summarize the CNN architecture
4. Train the model and save learning curves
5. Run final multi-seed evaluation and export test results

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

The script expects this structure:

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

## Output Files

All generated artifacts are saved in the `outputs/` folder:

- `milestone_1_sample_preview.png`
- `milestone_2_model_summary.txt`
- `milestone_3_training_curves.png`
- `milestone_5_multi_seed_accuracy.png`
- `milestone_5_best_run_curves.png`
- `milestone_5_confusion_matrix.png`
- `milestone_5_per_class_accuracy.png`
- `milestone_5_classification_report.txt`
