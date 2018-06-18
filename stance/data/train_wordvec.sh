#!/usr/bin/env bash

# This script outputs word vectors with different settings via fastText.
# - per company or whole data
# - starting from pre-trained vectors or not
#     - word2vec Twitter: https://www.fredericgodin.com/software/
#     - fastText English

# Assume you are in the docker container of `fastText`
# ```shell
# (sudo) docker run --rm -v (slo/stance root)/data:/data -it xebxeb/fasttext-docker /bin/bash
# ```

# PREREQUISITES:
#   - text files of preprocessed tweets per company produced by ../tweet_preprocessor.py
#   - pre-trained word vectors in wordvec/pretrained
#     - recomend to use word2vec or fastText compatible files
#         - this script has been run using fastText's pretrained English vectors (.vec)
#     - if you use GloVe vectors, you should convert its format into the word2vec compatible one
#     - but GloVe uses the different algorithm than word2vec/fastText, so it's not recommended
#     - GloVe vectors can be easily converted via gensim.scripts.glove2word2vec

# =====
# just follow one of the best practices
# https://kvz.io/blog/2013/11/21/bash-best-practices/
# Bash3 Boilerplate. Copyright (c) 2014, kvz.io

set -o errexit
set -o pipefail
set -o nounset
# set -o xtrace

# Set magic variables for current file & dir
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
__file="${__dir}/$(basename "${BASH_SOURCE[0]}")"
__base="$(basename ${__file} .sh)"
__root="$(cd "$(dirname "${__dir}")" && pwd)" # <-- change this as it depends on your app

# arg1="${1:-}"
# =====


echo "train without pretrained vecs per dimension"
echo ""

dims=(100 200 300)
for file in "${__dir}"/preprocessed_tweets/slo/*; do
  if [[ "${file}" == *.txt ]]; then
    result="/data/wordvec/$(basename ${file} .txt)"
    # if [[ "$(basename ${file} .txt)" = "dataset" ]]; then
    #   continue
    # fi
    for i in "${dims[@]}"; do
      echo "dim = ${i}"
      echo " ./fasttext skipgram -input "${file}" -output "${result}_${i}" -dim ${i}"
      /fasttext skipgram -input "${file}" -output "${result}_${i}" -dim "${i}"
    done
  fi
  echo ""
done

echo "train with pretrained vecs per dimension"
echo ""

ptn_dim="([0-9]+)d"
for vecfile in "${__dir}"/wordvec/pretrained/*; do
  if [[ "${vecfile}" == *.vec ]]; then
    # for each dimensions, all and per company

    # regex to retrieve dimensions
    # echo "${file}"
    base="$(basename ${vecfile})"
    echo "${base}"
    if [[ "${base}" =~ ${ptn_dim} ]]; then
      dim=${BASH_REMATCH[1]}
      echo "dim = ${dim}"
    fi
      

    for tfile in "${__dir}"/preprocessed_tweets/slo/*; do
      if [[ "${tfile}" == *.txt ]]; then
        result="/data/wordvec/$(basename ${tfile} .txt)_$(basename ${base} .vec)"
        echo " ./fasttext skipgram -input "${tfile}" -output "${result}" -dim "${dim}" -pretrainedVectors "${vecfile}""
        /fasttext skipgram -input "${tfile}" -output "${result}" -dim "${dim}" -pretrainedVectors "${vecfile}"
      fi
    done
  fi
done

