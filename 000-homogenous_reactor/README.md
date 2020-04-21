| ID    | creator     | code               | location                   | decription                                   | start date | end date |
|-------|-------------|--------------------|----------------------------|----------------------------------------------|------------|----------|
| 00000 | Pascal Roth | Cantera Python API | local, (STFS, GSC Cluster) | homogeneous reaction - constant pressure CH4 | 07.04.20   | ...      |
| 00001 | Pascal Roth | Cantera Python API | local                      | homogeneous reaction - constant volume CH4   | 07.04.20   | ...      |
| 00002 | Pascal Roth | Cantera Python API | local                      | homogeneous reaction - OME                   | 10.04.20   | ...      |
| 00003 | Pascal Roth |                    | local                      | conda environment                            | 07.04.20   | ...      |
| 00004 | Pascal Roth |                    | local                      | mechanism comparison                         | 17.04.20   | ...      |
| 00005 | Pascal Roth |                    | local                      | mechanism literature comparison              | 18.04.20   | ...      |

The used conda environment was to run all the codes was exported into "file/path/to/yml/file.yml".
To replicate this exact environment run 
```bash
conda env create -f /path/to/file.yml
```

| Software | Version | Source              | commit |
|----------|---------|---------------------|--------|
| pyTorch  | 1.4     | installed via conda |        |
| Cantera  | 2.2     | installed via conda |        |
