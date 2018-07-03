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
# Requirements: Must be running a python 3.6 virtual environment with the following modules:
#   -numpy, sklearn, keras, gensim, pandas, scipy, tensorflow, fire
#
#########################################################################################

# Create Parameters
DATASET="stance/datasets/dataset_20100101-20180510.csv"

MODEL="svm"
TESTFP="coding/gold_20180514_majority_fixed_tok.csv"
WVFP="wordvec/20100101-20180510/all-100.vec"
LOGGING=10

function run_repeat {
    # Retrieve Parameters
    TIMES_TO_RUN="${1:-1}"
    DATA_FP="${2}"
    SLO_CLASSIFIER_FP="${3}"
    FOR_SAMPLE="${4:-1}"
    AGAINST_SAMPLE="${5:-1}"
    NEUTRAL_SAMPLE="${6:-1}"
    LABEL="${7}"
    COMPANY=${8:-false}

    echo ${LABEL}

    # Reset the output file to the template
    cp ${DATA}/svm-results/results-template.csv ${DATA}/svm-results/trial-results-temp.csv

    # Start the iterations
    for i in $(seq 1 $TIMES_TO_RUN);
    do
        echo "==========================ITERATION $i=========================="

        AUTOCODED_SET="stance/coding/auto_20100101-20180510_${i}.csv"


        # Build the auto-coded trainset, based on whether company tweets are to be used or not.
        if "$COMPANY" = true ; then
            python ${SLO_CLASSIFIER_FP}/stance/data/autocoding_processor.py \
                --dataset_filepath=${DATA_FP}/${DATASET} \
                --coding_filepath=${DATA_FP}/${AUTOCODED_SET} \
                --for_sample_size=${FOR_SAMPLE} \
                --against_sample_size=${AGAINST_SAMPLE} \
                --neutral_sample_size=${NEUTRAL_SAMPLE} \
                --companytweets
        else
            python ${SLO_CLASSIFIER_FP}/stance/data/autocoding_processor.py \
                --dataset_filepath=${DATA_FP}/${DATASET} \
                --coding_filepath=${DATA_FP}/${AUTOCODED_SET} \
                --for_sample_size=${FOR_SAMPLE} \
                --against_sample_size=${AGAINST_MULT} \
                --neutral_sample_size=${NEUTRAL_MULT} \
                --nocompanytweets
        fi

        # Tokenize the newly created trainset.
        python ${SLO_CLASSIFIER_FP}/stance/data/tweet_preprocessor.py \
            --fp=${DATA_FP}/${AUTOCODED_SET}

    done


    # Run run_stance_detection.py on the new trainsets.
    # This first loop runs it with no profile text
    for i in $(seq 1 $TIMES_TO_RUN);
    do
        TRAINFP="coding/auto_20100101-20180510_${i}_tok.csv"
        python ${SLO_CLASSIFIER_FP}/stance/run_stance_detection.py \
            train \
            --model=${MODEL} \
            --path=${DATA_FP}/stance \
            --trainfp=${TRAINFP} \
            --testfp=${TESTFP} \
            --outfp=${DATA_FP}/svm-results/trial-results-temp.csv \
            --wvfp=${WVFP} \
            --logging_level=${LOGGING} \
            --noprofile
    done

    # Add the data to the output CSV.
    python ${SLO_CLASSIFIER_FP}/stance/data/results_postprocessing.py \
        --input_filepath=${DATA_FP}/svm-results/trial-results-temp.csv \
        --output_filepath=${DATA_FP}/svm-results/test-results.csv \
        --label="${LABEL}-NoProfile"

    cp ${DATA}/svm-results/results-template.csv ${DATA}/svm-results/trial-results-temp.csv

    # This second loop runs it with profile text added.
    for i in $(seq 1 $TIMES_TO_RUN);
    do
        TRAINFP="coding/auto_20100101-20180510_${i}_tok.csv"
        python ${SLO_CLASSIFIER_FP}/stance/run_stance_detection.py \
            train \
            --model=${MODEL} \
            --path=${DATA_FP}/stance \
            --trainfp=${TRAINFP} \
            --testfp=${TESTFP} \
            --outfp=${DATA_FP}/svm-results/trial-results-temp.csv \
            --wvfp=${WVFP} \
            --logging_level=${LOGGING}
    done

    # Add the data to the output CSV.
    python ${SLO_CLASSIFIER_FP}/stance/data/results_postprocessing.py \
        --input_filepath=${DATA_FP}/svm-results/trial-results-temp.csv \
        --output_filepath=${DATA_FP}/svm-results/test-results.csv \
        --label="${LABEL}-Profile"

    rm ${DATA_FP}/svm-results/trial-results-temp.csv

    echo "Finished ${LABEL}"

}

DATA="${1}"
SLO="${2}"
ENV="${3}"

# Set this script to exit if a command returns an error
set -e

# Reset the output CSV file.
if [ -f ${DATA}/svm-results/test-results.csv ]; then
    rm ${DATA}/svm-results/test-results.csv
fi
touch ${DATA}/svm-results/test-results.csv

source ${ENV}

run_repeat 2 ${DATA} ${SLO} 806 806 806 "806-806-806-Company" true

run_repeat 2 ${DATA} ${SLO} 607 607 607 "607-607-607-NoCompany" false

run_repeat 2 ${DATA} ${SLO} 607 607 607 "607-607-607-Company" true

deactivate