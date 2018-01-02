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
for prefix in test16 test17 dev train
 do
   cat test/$prefix.$SRCTAG | \
   $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC | \
   $tools/tokenizer.perl -a -l $SRC > test/$prefix.tok.$SRCTAG

   cat test/$prefix.$TRGTAG | \
   $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG | \
   $tools/tokenizer.perl -a -l $TRG > test/$prefix.tok.$TRGTAG
   
   #pe
   cat test/$prefix.pe | \
   $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG | \
   $tools/tokenizer.perl -a -l $TRG > test/$prefix.tok.pe

 done

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
#$mosesdecoder/scripts/training/clean-corpus-n.perl tuning/train.tok $SRCTAG $TRGTAG tuning/train.tok.clean 1 80

# train truecaser
#$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus data/train.tok.clean.$SRCTAG -model model/truecase-model.$SRCTAG
#$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus data/train.tok.clean.$TRGTAG -model model/truecase-model.$TRGTAG

# apply truecaser (cleaned training corpus)
#for prefix in train
# do
#  $tools/scripts/truecase.perl -model model/truecase-model.$SRCTAG < tuning/$prefix.tok.clean.$SRCTAG > tuning/$prefix.tc.$SRCTAG
#  $tools/scripts/truecase.perl -model model/truecase-model.$TRGTAG < tuning/$prefix.tok.clean.$TRGTAG > tuning/$prefix.tc.$TRGTAG
# done

# apply truecaser (dev/test files)
for prefix in test16 test17 dev train
 do
  $tools/scripts/truecase.perl -model model/truecase-model.$SRCTAG < test/$prefix.tok.$SRCTAG > test/$prefix.tc.$SRCTAG
  $tools/scripts/truecase.perl -model model/truecase-model.$TRGTAG < test/$prefix.tok.$TRGTAG > test/$prefix.tc.$TRGTAG
  
  #pe
  $tools/scripts/truecase.perl -model model/truecase-model.$TRGTAG < test/$prefix.tok.pe > test/$prefix.tc.pe
  
 done

# train BPE   源语言和目标语言分别单独训练 bpe
#cat data/train.tc.$SRCTAG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$SRCTAG.bpe
#cat data/train.tc.$TRGTAG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$TRGTAG.bpe

# apply BPE
for prefix in test16 test17 dev train
 do
  $tools/scripts/apply_bpe.py -c model/$SRCTAG.bpe < test/$prefix.tc.$SRCTAG > test/$prefix.bpe.$SRCTAG
  $tools/scripts/apply_bpe.py -c model/$TRGTAG.bpe < test/$prefix.tc.$TRGTAG > test/$prefix.bpe.$TRGTAG
  
  #pe
  $tools/scripts/apply_bpe.py -c model/$TRGTAG.bpe < test/$prefix.tc.pe > test/$prefix.bpe.pe
  
 done
###############################################
# build network dictionary
#$nematus/data/build_dictionary.py data/train.bpe.$SRCTAG data/train.bpe.$TRGTAG
