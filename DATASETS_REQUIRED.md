# Required real-world dataset files

The archive includes the GASD files but does not redistribute the other large or third-party datasets. Place the original files at the exact paths below before running the complete real-world matrix.

| Dataset | Required repository path | Source named by the project |
|---|---|---|
| CCPP | `Datasets/Real/Datasets_Generators_CSV/08_CCPP/008_Folds5x2_pp.csv` | UCI Combined Cycle Power Plant |
| MCPD | `Datasets/Real/Datasets_Generators_CSV/05_MCPD/005_insurance.csv` | Kaggle Medical Cost Personal Dataset |
| KCHSD | `Datasets/Real/Datasets_Generators_CSV/07_KCHSD/007_kc_house_data.csv` | Kaggle King County House Sales |
| 1KC | `Datasets/Real/Datasets_Generators_CSV/06_1KC/006_1000_Companies.csv` | Kaggle 1000 Companies Profit |
| UCIAQD | `Datasets/Real/Datasets_Generators_CSV/UCIAQD/AirQualityUCI.csv` | UCI Air Quality |
| GASD | `Datasets/Real/Datasets_Generators_CSV/GASD/batch1.dat` through `batch10.dat` | UCI Gas Sensor Array Drift at Different Concentrations |
| CalCOFI | `Datasets/Real/Datasets_Generators_CSV/CalCOFI/bottle.csv` and `cast.csv` | CalCOFI bottle and cast data |
| WSSF | `Datasets/Real/Datasets_Generators_CSV/WSSF/train.csv`, `features.csv`, `stores.csv`, and `test.csv` | Walmart Store Sales Forecasting dataset |

Run the validator after copying the files:

```bash
python RealWorldDatasetsEvaluation/000_validate_setup.py
```

The validator deliberately fails when required files are absent. Use `--allow-missing-datasets` only for code-level validation or synthetic-only runs.
