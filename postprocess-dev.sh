#/bin/sh

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=tools/moses

# suffix of target language files
#lng=pe

sed 's/\@\@ //g' | \
$mosesdecoder/scripts/recaser/detruecase.perl
