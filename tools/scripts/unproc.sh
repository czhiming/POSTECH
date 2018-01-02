#!/bin/bash

/work/wmt16/tools/scripts/cleanBPE | /data/smt/mosesMaster/scripts/recaser/detruecase.perl | /data/smt/mosesMaster/scripts/tokenizer/deescape-special-chars.perl 