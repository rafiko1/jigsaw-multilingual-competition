### Introduction
Tensorflow code, used with Kaggle TPUv3 instances by the 1st place winner for the [2020 Jigsaw Multilingual Kaggle competition](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification). 
Our solution is detailed in this [Kaggle forum post](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160862). The code is part of the overall solution with another part leveraging similar [Pytorch models](https://github.com/leecming/jigsaw-multilingual) trained locally.

### Recipe for training: 
1. Bootstrap test-set predictions (pseudo-labels) for all languages using the 
XLM-Roberta-Large multilingual model (XLM-R)
2. Train a monolingual/multilingual 
model, and predict test-set samples in the relevant language(s)
3. Blend predictions from the current model with the previous ensemble 
predictions - update pseudo-labels 
4. Repeat steps 2 and 3 with various pretrained monolingual and multilingual 
models.

Relevant upgrades to training:
1. Implementing pseudolabels togehter with train labels in step 2
2. As final result, use a post-processing technique

We provide 3 sets of sample pseudo-labels (scoring on public LB: 9372, 9500, 9537) for you to test training monolingual and multilingual models against. <br>
For post-processing, we also provide 11 pseudolabels where subsequent Russian labels were updated after training. <br>
You can refer to the **Data and model files** section below for more info. 

### Code
| Training | Comment |
| ----- | ------  |
|[Template for es/it/tr](template-es-it-tr.ipynb) | monolingual approach for languages with validation set es/it/tr. Default model: XLM-R |
|[Template for pt/ru/fr](template-pt-ru-fr.ipynb) | monolingual approach for languages without validation set pt/ru/fr. Default model: XLM-R |  
|[Template train-bias](xlm-r-train-bias.ipynb) | monolingual approach using train-bias dataset. Default model: XLM-R |  

| Post-processing | Comment | 
| -------------- | ------- |
| [Post-processing example](post-processing-example.ipynb) | Post-processing technique using Russian pseudolabels as example |

### Data and model files
1. HuggingFace models are downloaded directly via API so there is no need to manually download them.
2. Translations of the Toxic 2018 dataset and pseudo-labels for public LB 9372, public LB 9500, public LB 9537 (used as sample inputs to training) can be found [here](https://www.kaggle.com/leecming/multilingual-toxic-comments-training-data).
3. Translation of the Toxic 2019 Unintended bias dataset is found [here](https://www.kaggle.com/rafiko1/translated-train-bias-all-langs)
4. For post-processing: 11 Russian-specific updated pseudolabels [here](https://www.kaggle.com/rafiko1/ru-changed-subs)

### Setup


### Example 1: Running a spanish monolingual Transformer model using public LB 9500 pseudo-labels 
1. We use the pretrained spanish model: mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es
2. Update [SETTINGS.json](SETTINGS.json) so that PSEUDO_LABELS_PATH and other paths are updated for your setup.
3. Run prepare_data.py
4. Run classifier_baseline.py
5. Run prepare_predictions.py

A submission file should be generated at $TRAIN_DATA_DIR/curr_run_submission.csv. 
For this, we were able to generate a 9502 public LB submission. 
Due to training variability, you may get different results.
To use the predictions as pseudo-labels for another model run, note that you'll have to merge the predicted labels with test.csv. 


### Example 2: Running a spanish monolingual FastText model using public LB9537 pseudo-labels
1. We use the pretrained official FastText embeddings for spanish (cc.es.300.bin) 
2. Update [SETTINGS.json](SETTINGS.json) so that PSEUDO_LABELS_PATH and other paths are updated for your setup.
3. Run prepare_data.py
4. Run classifier_bigru_fasttext_tf.py
5. Update prepare_predictions.py so that ENSEMBLE_WEIGHT=0.8 (give less weight to predictions from this model)
6. Run prepare_predictions.py

A submission file should be generated at $TRAIN_DATA_DIR/curr_run_submission.csv. 
For this, we were able to generate a 9540 public LB submission. 
Due to training variability, you may get different results.
To use the predictions as pseudo-labels for another model run, note that you'll have to merge the predicted labels with test.csv 


### Example 3: Running a multilingual Transformer model (XLM-Roberta-Large) using public LB9372 pseudo-labels
1. We use the pretrained multilingual model: xlm-roberta-large
2. Update [SETTINGS.json](SETTINGS.json) so that PSEUDO_LABELS_PATH and other paths are updated for your setup.
3. Update [prepare_data.py](prepare_data.py) so that the LANG_LIST = ['es', 'fr', 'it', 'pt', 'ru', 'tr'] and SAMPLE_FRAC=0.1
4. Update [classifier_base.py](classifier_baseline.py) so that PRETRAINED_MODEL = 'xlm-roberta-large' and BASE_MODEL_OUTPUT_DIM=1024. You'll likely need to adjust the ACCUM_FLAG and BATCH_SIZE given the model size (ACCUM_FLAG=2 and BATCH_SIZE=24 on a single RTX Titan)
5. Run classifier_base.py
6. Update prepare_predictions.py so that ENSEMBLE_WEIGHT=0.2 (give more weight to predictions from this model)
7. Run prepare_predictions.py

A submission file should be generated at $TRAIN_DATA_DIR/curr_run_submission.csv. 
For this, we were able to generate a 9409 public LB submission. 
Due to training variability, you may get different results.
To use the predictions as pseudo-labels for another model run, note that you'll have to merge the predicted labels with test.csv
 



### Notes
1. Below lists the various pretrained HuggingFace transformer models we used -

| Language | Models |
| -------- | ------ | 
| All | xlm-roberta-large |
| French | camembert-base, camembert/camembert-large, flaubert/flaubert_large_cased |
| Italian | dbmdz/bert-base-italian-xxl-cased, dbmdz/bert-base-italian-xxl-uncased, m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0 |
| Portuguese | neuralmind/bert-large-portuguese-cased | 
| Russian | DeepPavlov/rubert-base-cased-conversational |
| Spanish | dccuchile/bert-base-spanish-wwm-cased, dccuchile/bert-base-spanish-wwm-uncased, mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es |
| Turkish | dbmdz/bert-base-turkish-128k-cased, dbmdz/bert-base-turkish-128k-uncased, dbmdz/electra-base-turkish-cased-v0-discriminator |
