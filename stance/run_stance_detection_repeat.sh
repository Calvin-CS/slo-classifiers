#!/usr/bin/env bash
#
# run_stance_detection_repeat.sh
# 
# This script runs autocoding_processor.py, tweet_preprocesor.py, and
# run_stance_detection.py for a certain number of iterations.
#
# Usage: bash run_stance_detection_repeat.sh TIMES_TO_RUN DATA_FP SLO_CLASSIFIER_FP
#   TIMES_TO_RUN: Number of iterations to run the stance detection process.
#   DATA_FP: Filepath to the data
#   SLO_CLASSIFIER_FP: Filepath to the slo-classifier project
#
# Author: Roy Adams
# Date: June 19, 2018
#
# Requirements: Must be running a python 3.6 virtual environment with the following modules:
#   -numpy, sklearn, keras, gensim, pandas, scipy, tensorflow, fire
#
#########################################################################################

# Create Parameters
DATASET="stance/datasets/dataset_20100101-20180510.csv"

AUTOCODED_SET="stance/coding/auto_20100101-20180510.csv"

MODEL="svm"
TRAINFP="coding/auto_20100101-20180510_tok.csv"
TESTFP="coding/gold_20180514_majority_fixed_tok.csv"
WVFP="wordvec/20100101-20180510/all-100.vec"
LOGGING=10

function run_repeat {
    # Retrieve Parameters
    TIMES_TO_RUN="${1:-1}"
    DATA_FP="${2}"
    SLO_CLASSIFIER_FP="${3}"
    AGAINST_MULT="${4:-1}"
    NEUTRAL_MULT="${5:-1}"
    LABEL="${6}"
    PROFILES="${7}"

    echo ${LABEL}

    # Start the iterations
    for i in $(seq 1 $TIMES_TO_RUN);
    do
        echo "==========================ITERATION $i=========================="
        python3.6 ${SLO_CLASSIFIER_FP}/stance/data/autocoding_processor.py \
            --dataset_filepath=${DATA_FP}/${DATASET} \
            --coding_path=${DATA_FP}/stance/coding \
            --against_multiplier=${AGAINST_MULT} \
            --neutral_multiplier=${NEUTRAL_MULT}

        python3.6 ${SLO_CLASSIFIER_FP}/stance/data/tweet_preprocessor.py \
            --fp=${DATA_FP}/${AUTOCODED_SET}

        if ${PROFILES} ; then
            python3.6 ${SLO_CLASSIFIER_FP}/stance/run_stance_detection.py \
                train \
                --model=${MODEL} \
                --path=${DATA_FP}/stance \
                --trainfp=${TRAINFP} \
                --testfp=${TESTFP} \
                --outfp=${DATA_FP}/svm-results/trial-results-temp.csv \
                --wvfp=${WVFP} \
                --logging_level=${LOGGING}
        else
            python3.6 ${SLO_CLASSIFIER_FP}/stance/run_stance_detection.py \
                train \
                --model=${MODEL} \
                --path=${DATA_FP}/stance \
                --trainfp=${TRAINFP} \
                --testfp=${TESTFP} \
                --outfp=${DATA_FP}/svm-results/trial-results-temp.csv \
                --wvfp=${WVFP} \
                --logging_level=${LOGGING} \
                --noprofile
        fi

        echo "Finished iteration $i"
    done

    python3.6 ${SLO_CLASSIFIER_FP}/stance/data/results_postprocessing.py \
        --input_filepath=${DATA_FP}/svm-results/trial-results-temp.csv \
        --output_filepath=${DATA_FP}/svm-results/results.csv \
        --label=${LABEL}

}

DATA="${1}"
SLO="${2}"

# Set this script to exit if a command returns an error
set -e

# Reset previous results files
cp ${DATA}/svm-results/results.csv ${DATA}/svm-results/results-backup.csv
cp ${DATA}/svm-results/results-template.csv ${DATA}/svm-results/results.csv

cp ${DATA}/svm-results/results-template.csv ${DATA}/svm-results/trial-results-temp.csv

run_repeat 10 ${DATA} ${SLO} 1 1 "1-1-No_Profiles" false

cp ${DATA}/svm-results/results-template.csv ${DATA}/svm-results/trial-results-temp.csv

run_repeat 10 ${DATA} ${SLO} 1 1 "1-1-Profiles" true

cp ${DATA}/svm-results/results-template.csv ${DATA}/svm-results/trial-results-temp.csv

run_repeat 10 ${DATA} ${SLO} 5 1 "5-1-Profiles" true

cp ${DATA}/svm-results/results-template.csv ${DATA}/svm-results/trial-results-temp.csv

run_repeat 10 ${DATA} ${SLO} 10 1 "10-1-Profiles" true

