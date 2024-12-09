# ADRL

Codes for Artificial Intelligence In Medicine paper: *Enhancing Diagnosis Prediction with Adaptive Disease Representation Learning*

## Download the MIMIC-III and MIMIC-IV datasets
Go to [https://mimic.physionet.org/](https://mimic.physionet.org/gettingstarted/access/) for access. 
Once you have the authority for the dataset, download the dataset and extract the csv files to `data/mimic3/raw/` and `data/mimic4/raw/` in this project. For specific data processing, please refer to [Chet](https://github.com/LuChang-CS/Chet/).


# Motivation
Motivation:Diagnosis prediction predicts which diseases a patient is most likely to suffer from in the future based on their historical electronic health records. The time series model can better capture the temporal progression relationship of patient diseases, but ignores the semantic correlation between all diseases; in fact, multiple diseases that are often diagnosed at the same time reflect hidden patterns that are conducive to diagnosis, so predefined global disease co-occurrence graph can help the model understand disease relationships. But it may contain a lot of noise and ignore the semantic adaptation of the disease under the diagnosis target. 

## Requirements Packages

- python 3.7
- numpy
- sklearn
- pandas
- tensorflow

# Usage
## Prepare the environment

1. Install all required softwares and packages.

```bash
pip install -r requirement.txt
```


## Preprocess
```bash
python run_preprocess.py
```

## Train model
```bash
python train_codes.py
```

## Configuration
Please see `train_codes.py` for detailed configurations.