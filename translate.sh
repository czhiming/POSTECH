#!/bin/sh

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=tools/moses

# theano device, in case you do not want to compute on gpu, change it to cpu
device=gpu

#model prefix
prefix=model/model.npz

#输入需要翻译的文件，和输出文件
dev=test/test17.bpe.en
ref=test/test17.tok.pe

# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python nematus/translate.py \
     -m $prefix.best_bleu \ #best bleu model
     -i $dev \
     -o $dev.output.dev \
     -k 12 -n -p 1

./postprocess-dev.sh < $dev.output.dev > $dev.output.postprocessed.dev

## get BLEU
BEST=`cat ${prefix}_best_bleu || echo 0`
$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev.output.postprocessed.dev >> ${prefix}_bleu_scores
BLEU=`$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev.output.postprocessed.dev | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`

echo "BLEU = $BLEU"

# save model with highest BLEU
if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $BLEU > ${prefix}_best_bleu
  cp ${prefix}.dev.npz ${prefix}.best_bleu
  cp ${prefix}.dev.npz.json ${prefix}.best_bleu.json
fi
