# DyMHGRec
The data and code for the DyMHGRec framework.
## Requirements
* PyTorch 3.8.0 and CUDA 11.1.
* The learning rate of 5e-4 for 50 epochs.
## Results
* Results of the paper:

  ``` python
  DDI: 0.0493 (0.0005) Ja: 0.6835 (0.0038) PRAUC: 0.8535 (0.0029) F1: 0.7866 (0.0029) 
  ```
  ## Folder Specification
  # Dataset 
  For the MIMIC-III dataset, we do not share the MIMIC-III dataset due to reasons of personal privacy, maintaining research standards, and legal considerations. Go to https://physionet.org/content/mimiciii/1.4/ to download the MIMIC-III dataset. We use the same data set and processing methods as ExpDrug (https://github.com/hyh0606/ExpDrug).
  * PRESCRIPTIONS.csv
  * DIAGNOSES_ICD.csv
  * PROCEDURES_ICD.csv
  * D_ICD_DIAGNOSES.csv
  * D_ICD_PROCEDURES.csv
  # Processing file
  data/data_new/
  - **ndc2atc_level4.csv**: this is a NDC-RXCUI-ATC4 mapping file, and we only need the RXCUI to ATC4 mapping. This file is obtained from https://github.com/hyh0606/ExpDrug.
  - **drug-atc.csv**: this is a CID-ATC file, which gives the mapping from CID code to detailed ATC code . This file is obtained from https://github.com/hyh0606/ExpDrug.
  - **ndc2rxnorm_mapping.txt**: NDC to RXCUI mapping file. This file is obtained from https://github.com/hyh0606/ExpDrug.
  - **drug-DDI.csv**: this a large file, containing the drug DDI information, coded by CID. The file could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing.
  data/data_divided/: This contains a classification of the drugs in the training set, the evaluation set, and the test set (run processing.py).
  * eval_divided.pkl
  * test_divided.pkl
  * train_divided.pkl
 Processing the data:

  ``` python
  python data/data_new/processing.py
  ```
  ## Run
  * Run DyMHGRec_main.py to reproduce the results of the paper.
  ```python
cd src
python DyMHGRec_main.py
```
  
  
