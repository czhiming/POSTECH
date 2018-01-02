#!/bin/sh


# suffix of source language files
SRC=en
SRCTAG=en
# suffix of target language files
TRG=de
TRGTAG=de

tools=tools
# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=$tools/moses
# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=$tools/subword-nmt 

# tokenize
for prefix in train dev
 do
   cat tuning/$prefix.$SRCTAG | \
   $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC | \
   $tools/normalise-romanian.py | \
   $tools/remove-diacritics.py | \
   $tools/tokenizer.perl -a -l $SRC > tuning/$prefix.tok.$SRCTAG

   cat tuning/$prefix.$TRGTAG | \
   $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG | \
   $tools/tokenizer.perl -a -l $TRG > tuning/$prefix.tok.$TRGTAG

 done

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
$mosesdecoder/scripts/training/clean-corpus-n.perl tuning/train.tok $SRCTAG $TRGTAG tuning/train.tok.clean 1 80

# train truecaser
#$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus data/train.tok.clean.$SRCTAG -model model/truecase-model.$SRCTAG
#$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus data/train.tok.clean.$TRGTAG -model model/truecase-model.$TRGTAG

# apply truecaser (cleaned training corpus)
for prefix in train
 do
  $tools/scripts/truecase.perl -model model/truecase-model.$SRCTAG < tuning/$prefix.tok.clean.$SRCTAG > tuning/$prefix.tc.$SRCTAG
  $tools/scripts/truecase.perl -model model/truecase-model.$TRGTAG < tuning/$prefix.tok.clean.$TRGTAG > tuning/$prefix.tc.$TRGTAG
 done

# apply truecaser (dev/test files)
for prefix in dev
 do
  $tools/scripts/truecase.perl -model model/truecase-model.$SRCTAG < tuning/$prefix.tok.$SRCTAG > tuning/$prefix.tc.$SRCTAG
  $tools/scripts/truecase.perl -model model/truecase-model.$TRGTAG < tuning/$prefix.tok.$TRGTAG > tuning/$prefix.tc.$TRGTAG
 done


# train BPE   源语言和目标语言分别单独训练 bpe
#cat data/train.tc.$SRCTAG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$SRCTAG.bpe
#cat data/train.tc.$TRGTAG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$TRGTAG.bpe

# apply BPE

for prefix in train dev
 do
  $tools/scripts/apply_bpe.py -c model/$SRCTAG.bpe < tuning/$prefix.tc.$SRCTAG > tuning/$prefix.bpe.$SRCTAG
  $tools/scripts/apply_bpe.py -c model/$TRGTAG.bpe < tuning/$prefix.tc.$TRGTAG > tuning/$prefix.bpe.$TRGTAG
 done
###############################################
# build network dictionary
#$nematus/data/build_dictionary.py data/train.bpe.$SRCTAG data/train.bpe.$TRGTAG
