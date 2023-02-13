# Learning to Predict Visual Attributes Through Human Disagreement

Project for Grounded Language Processing Course at University of Trento

## Overview

Attribute prediction has made exciting recent progress with the emergence of attribute-centred datasets. However, the reliability of their ground truth labels does not reflect human disagreement. 
We present a comparative study of the use of soft and hard labels for this supervised multi-label classification problem. Specifically, we compare three types of training paradigms: hard labels training, soft labels training, and self-knowledge distillation.

Detailed information are available in `report/report.pdf`

## Dataset

We chose the [VAW dataset](https://vawdataset.com/) to perform attribute prediction. VAW is constructed with a large vocabulary of 620 attributes, each belonging to one of 8 parent classes: `action`, `color`, `material`, `other`, `shape`, `size`, `state`, `texture`. Each instance is annotated with positive, negative, and missing attributes.

We used a filtered version of the dataset maintaining the same proportions between train, val and test splits. In addition, we defined the test set taking approximately 35% of the total number of objects to be novel.

To create our filtered splits, run the dataset creation script as follows from the `data` folder:

```
python dataset_creation.py
```

The script creates four JSON inside `data`: 

`train_data.json`, `val_data.json`, `test_data.json`, `generalization_data.json`. 

`generalization_data.json` cotaines only the subset of the test split with approximately 35% of novel objects.



## Run evaluation

In the `eval` folder, we provide the prediction and ground truth for each of the three models.

Predictions and ground truth are named as following:

```
{default, kd, kd_xent, soft_labels}_test_{gt, preds}.pt
```
where `default` stands for the model trained with hard labels; `kd` stands for the model trained with knowledge distillation with KL divergence; `kd_xent` stands for the model trained with knowledge distillation with CE loss; `soft_labels` stands for the model trained with class-level soft labels obtained from human disgreement; `gt` stands for ground truth; `preds` stands for model predictions.

`eval/eval.py` is used to run the evaluation script. 

In particular, `eval.py` expects as input the followings:

1. `fpath_pred`: path to the model prediction tensor `{default, kd, kd_xent, soft_labels}_test_preds.pt`  
   (shape `(n_instances, n_class)`). `preds[i,j]` is the predicted probability 
   for attribute class `j` of instance `i`.
2. `fpath_label`: path to the groundtruth label tensor `{default, kd, kd_xent, soft_labels}_test_gt.pt` (shape `(n_instances, n_class)`).
   The groundtruth labels are different based on the model type: *hard labels* for `{default, kd, kd_xent}`, and *soft labels* for `soft_labels`. `gt_label[i,j]` is the ground truth label for attribute class `j` of instance `i`.

From the `eval` folder, run the evaluation script as follows (default model evaluation):
```
python eval.py --fpath_pred default_test_preds.pt --fpath_label default_test_gt.pt
```

Note: `{default, kd, kd_xent}` all use `default_test_gt.pt` as ground truth labels for the evaluation.

