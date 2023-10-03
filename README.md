# KTGF
A Knowledge-enhanced Two-stage Generative Framework for Medical Dialogue Information Extraction, Machine Intelligence Research (MIR). 

## Requirement

* pytorch 1.10.0
* transformers 4.18.0

## Usage

### Preprocessing

Download the [Chinese T5 base](https://huggingface.co/uer/t5-base-chinese-cluecorpussmall) and save the files to pretrained_models/t5-base-chinese-cluecorpussmall Folder. 
```
sh preprocess_data.sh 
```

### Training:
```
sh train.sh
```

### Testing
```
sh generate_stage1.sh
sh generate_stage2.sh
```

