# MXNMT: MXNet based Neural Machine Translation

This is an implementation of seq2seq with attention for neural machine translation with MXNet.

## Warning:
This repo is no longer maintained.
I recommend https://github.com/magic282/PyTorch_seq2seq

## Data

The current code uses IWSLT 2009 Chinese-English corpus as training, development and test data. Please request this data set or **use other available parallel corpus**. Data statistics,

| training | dev | test |
|----------|-----|------|
| 81819    | 446 | 504  |

## Attention
* This code does work with the latest mxnet. I made a new version with improved performance in the [next](https://github.com/magic282/MXNMT/tree/next) branch and it can run with the 0.9.5 mxnet. However, this branch is not complete since it lacks the decode part. **I will really appreciate it if you can contribute to this branch.** Also, I ***strongly*** recommend to use this commit (138344683e65c87af20250e3f4cdcc5a72ac3cc5) of mxnet because of [this issue](https://github.com/dmlc/mxnet/issues/5816).
* The author cannot distribute this dataset. **Any email requesting this dataset to the code author will not be replied.**

### Dev/Test Data Format
The reference number of IWSLT 2009 Ch-En is 7, for example:
```
在 找 给 家里 人 的 礼物 .

i 'm searching for some gifts for my family .
i want to find something for my family as presents .
i 'm about to buy some presents for my family .
i 'd like to buy my family something as a gift .
i 'm looking for a gift for my family .
i 'm looking for a present for my family .
i need a gift for my family .
有 $number 块 钱 以下 的 茶 吗 ? |||| {1 ||| 1 ||| one thousand ||| $number ||| 一千}

do you have any tea under one thousand yen ?
i 'd like to take a look at some tea cheaper than one thousand yen .
is there any tea less than one thousand yen here ?
i 'm looking for some tea under one thousand yen .
do you have any tea lower than one thousand yen ?
do you have any tea less than one thousand yen ?
i would like to buy some tea cheaper than one thousand yen .
```

## Result

According to my test, this code can achieve 44.18 BLEU score (with beam search) on IWSLT dev set without post-processing after 53 iteration. Specifically,
`1gram=72.65%  2gram=49.63%  3gram=37.62%  4gram=28.08%   BP = 1.0000 BLEU = 0.4418`


## Know Issues
*  Compatibility issue. The current version will ask to use Python 3 since it is annoying to handle Chinese encoding problems for Python 2.
*  In the attention part, `h.dot(U)` should be pre-computed. However it seems that it won't work properly if I do so.
*  The BLEU evaluator, which is an exe file and not included, should be replaced by nltk evaluator in the future.
*  The model can be modified to make it achieve about 50 BLEU score on this data set.
