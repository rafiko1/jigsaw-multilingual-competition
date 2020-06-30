### Introduction

The Tensorflow code released here, was implemented using Kaggle TPUv3 by the 1st place winner of the [2020 Jigsaw Multilingual Kaggle competition](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification). It was the 3rd annual competition organized by the Jigsaw team. It followed [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), the original 2018 competition, and [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). <br> 
The goal was to run multilingual toxicity predictions on six different languages (i.e. es, tr, it, ru, pt and fr), and promote the usage of TPU.

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
1. Implement pseudolabels togehter with train labels in step 2
2. Use a post-processing technique

We provide 3 sets of sample pseudo-labels (scoring on public LB: 9372, 9500, 9537) for you to test training monolingual and multilingual models against. <br>
For post-processing, we also provide 11 pseudolabels where subsequent Russian labels were updated after training. <br>
You can refer to the **Data and model files** section below for more info. 

### Code
| Training | Comment |
| ----- | ------  |
|[Template for es/it/tr](template-es-it-tr.ipynb) | monolingual approach for languages with validation set (es/it/tr). Default model: XLM-R |
|[Template for pt/ru/fr](template-pt-ru-fr.ipynb) | monolingual approach for languages without validation set (pt/ru/fr). Default model: XLM-R |  
|[Template train-bias](template-train-bias.ipynb) | monolingual approach using train-bias dataset. Default model: XLM-R |  

| Post-processing | Comment | 
| -------------- | ------- |
| [Post-processing example](post-processing-example.ipynb) | Post-processing technique using Russian-specific pseudolabels as example |

### Data and model files
1. HuggingFace models are downloaded directly via API so there is no need to manually download them.
2. Translations of the Toxic 2018 dataset and pseudo-labels for public LB 9372, public LB 9500, public LB 9537 (used as sample inputs to training) can be found [here](https://www.kaggle.com/leecming/multilingual-toxic-comments-training-data).
3. Translation of the Toxic 2019 Unintended bias dataset is found [here](https://www.kaggle.com/rafiko1/translated-train-bias-all-langs)
4. For post-processing: 11 Russian-specific updated pseudolabels [here](https://www.kaggle.com/rafiko1/ru-changed-subs)

### Notes
1. Below lists the various pretrained HuggingFace transformer models we used -

| Language | Models |
| -------- | ------ | 
| All | jplu/tf-xlm-roberta-large |
| French | camembert-base, camembert/camembert-large, flaubert/flaubert_large_cased |
| Italian | dbmdz/bert-base-italian-xxl-cased, dbmdz/bert-base-italian-xxl-uncased, m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0 |
| Portuguese | neuralmind/bert-large-portuguese-cased | 
| Russian | DeepPavlov/rubert-base-cased-conversational |
| Spanish | dccuchile/bert-base-spanish-wwm-cased, dccuchile/bert-base-spanish-wwm-uncased, mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es |
| Turkish | dbmdz/bert-base-turkish-128k-cased, dbmdz/bert-base-turkish-128k-uncased, dbmdz/electra-base-turkish-cased-v0-discriminator |
