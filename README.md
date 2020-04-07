| ID  | creator     | code               | location                   | decription                      | start date | end date |
|-----|-------------|--------------------|----------------------------|---------------------------------|------------|----------|
| 001 | Pascal Roth | Cantera Python API | local, (STFS, GSC Cluster) | homogenous reaction simulations | 07.04.20   | ...      |
| 002 | Pascal Roth | pyTorch            | local                      | machine learning with pyTorch   | 07.04.20   | ...      |
| 000 | Pascal Roth |                    | local                      |                                 | 07.04.20   | ...      |

The used conda environment was to run all the codes was exported into "file/path/to/yml/file.yml".
To replicate this exact environment run 
```bash
conda env create -f /path/to/file.yml
```

| Software | Version | Source              | commit |
|----------|---------|---------------------|--------|
| pyTorch  | 1.4     | installed via conda |        |
| Cantera  | 2.2     | installed via conda |        |
