#!/bin/sh

# this sample script preprocesses a sample corpus, including tokenization,
# truecasing, and subword segmentation. 
# for application to a different language pair,
# change source and target prefix, optionally the number of BPE operations,
# and the file names (currently, data/corpus and data/newsdev2016 are being processed)

# in the tokenization step, you will want to remove Romanian-specific normalization / diacritic removal,
# and you may want to add your own.
# also, you may want to learn BPE segmentations separately for each language,
# especially if they differ in their alphabet

# suffix of source language files
SRC=en
SRCTAG=en
# suffix of target language files
TRG=de
TRGTAG=de
# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
#bpe_operations=89500
bpe_operations=45000

tools=tools
# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=$tools/moses
# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=$tools/subword-nmt 


# tokenize
for prefix in train dev
 do
   cat data/$prefix.$SRCTAG | \
   $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC | \
   $tools/normalise-romanian.py | \
   $tools/remove-diacritics.py | \
   $tools/tokenizer.perl -a -l $SRC > data/$prefix.tok.$SRCTAG

   cat data/$prefix.$TRGTAG | \
   $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG | \
   $tools/tokenizer.perl -a -l $TRG > data/$prefix.tok.$TRGTAG

 done
#############################################
# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
$mosesdecoder/scripts/training/clean-corpus-n.perl data/train.tok $SRCTAG $TRGTAG data/train.tok.clean 1 80

# train truecaser
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus data/train.tok.clean.$SRCTAG -model model/truecase-model.$SRCTAG
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus data/train.tok.clean.$TRGTAG -model model/truecase-model.$TRGTAG

# apply truecaser (cleaned training corpus)
for prefix in train
 do
  $tools/scripts/truecase.perl -model model/truecase-model.$SRCTAG < data/$prefix.tok.clean.$SRCTAG > data/$prefix.tc.$SRCTAG
  $tools/scripts/truecase.perl -model model/truecase-model.$TRGTAG < data/$prefix.tok.clean.$TRGTAG > data/$prefix.tc.$TRGTAG
 done

# apply truecaser (dev/test files)
for prefix in dev
 do
  $tools/scripts/truecase.perl -model model/truecase-model.$SRCTAG < data/$prefix.tok.$SRCTAG > data/$prefix.tc.$SRCTAG
  $tools/scripts/truecase.perl -model model/truecase-model.$TRGTAG < data/$prefix.tok.$TRGTAG > data/$prefix.tc.$TRGTAG
 done

############################################

# train BPE   源语言和目标语言分别单独训练 bpe
cat data/train.tc.$SRCTAG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$SRCTAG.bpe
cat data/train.tc.$TRGTAG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$TRGTAG.bpe

# apply BPE

for prefix in train dev
 do
  $tools/scripts/apply_bpe.py -c model/$SRCTAG.bpe < data/$prefix.tc.$SRCTAG > data/$prefix.bpe.$SRCTAG
  $tools/scripts/apply_bpe.py -c model/$TRGTAG.bpe < data/$prefix.tc.$TRGTAG > data/$prefix.bpe.$TRGTAG
 done

###############################################

# build network dictionary
$tools/build_dictionary.py data/train.bpe.$SRCTAG data/train.bpe.$TRGTAG


