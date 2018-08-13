#!/usr/bin/env bash
#
# run_stance_detection_repeat.sh
# 
# This script runs autocoding_processor.py, tweet_preprocesor.py, and
# run_stance_detection.py for a certain number of iterations.
#
# NOTE: This script is somewhat dated. It was designed to run on only one company's trainset.
# The new autocoding_processor builds trainsets for all the companies. It is recommended to
# use run_stance_detection_combined.sh instead.
#
# Usage: bash run_stance_detection_repeat.sh TIMES_TO_RUN DATA_FP SLO_CLASSIFIER_FP
#   DATA_FP: Filepath to the data
#   SLO_CLASSIFIER_FP: Filepath to the slo-classifier project
#   ENV: Python Virtual Environment to run in
#   OUT_NAME: CSV file name to output results to.
#   DATASET: What dataset you wish to run on.
#   RUN_ID: A Unique identifier for each run to allow multiple runs to occur simultaneously
#       with no thread of trainset read/write collision.
#   WITH_COMPANY: The number of tweets to use including company tweets.
#   WITHOUT_COMPANY: The number of tweets to use not including company tweets.
#
# Requirements: Must be running a python 3.6 virtual environment with the following modules:
#   -numpy, sklearn, keras, gensim, pandas, scipy, tensorflow, fire
#   There should be a folder named svm-results inside your data directory. This is where all
#   results will be stored.
#
#########################################################################################

# Create Parameters
DATA_FP="${1}"
SLO_CLASSIFIER_FP="${2}"
ENV="${3}"
OUT_NAME="${4}"
DATASET="${5}"
RUN_ID="${6}"
WITH_COMPANY=${7}
WITHOUT_COMPANY=${8}

MODEL="svm"
TESTFP="coding/gold_20180514_majority_fixed_tok.csv"
WVFP="wordvec/20100101-20180510/all-100.vec"
LOGGING=10

function run_repeat {
    # Retrieve Parameters
    TIMES_TO_RUN="${1:-1}"
    FOR_SAMPLE="${2:-1}"
    AGAINST_SAMPLE="${3:-1}"
    NEUTRAL_SAMPLE="${4:-1}"
    LABEL="${5}"
    COMPANY=${6:-false}

    echo ${LABEL}

    # Reset the output file to the template
    cp ${DATA_FP}/svm-results/results-template.csv ${DATA_FP}/svm-results/temp-${OUT_NAME}

    # Start the iterations
    for i in $(seq 1 $TIMES_TO_RUN);
    do
        echo "==================Building Auto-Coded Set ${i}=================="

        AUTOCODED_SET="stance/coding/auto_${RUN_ID}_${i}.csv"

        # Build the auto-coded trainset, based on whether company tweets are to be used or not.
        if "$COMPANY" = true ; then
            python ${SLO_CLASSIFIER_FP}/stance/data/autocoding_processor.py \
                --dataset_filepath=${DATASET} \
                --coding_filepath=${DATA_FP}/${AUTOCODED_SET} \
                --testset_filepath=${DATA_FP}/stance/${TESTFP} \
                --for_sample_size=${FOR_SAMPLE} \
                --against_sample_size=${AGAINST_SAMPLE} \
                --neutral_sample_size=${NEUTRAL_SAMPLE} \
                --companytweets
        else
            python ${SLO_CLASSIFIER_FP}/stance/data/autocoding_processor.py \
                --dataset_filepath=${DATASET} \
                --coding_filepath=${DATA_FP}/${AUTOCODED_SET} \
                --testset_filepath=${DATA_FP}/stance/${TESTFP} \
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
        echo "===========Running w/o Profile on Auto-Coded Set ${i}==========="
        TRAINFP="coding/auto_${RUN_ID}_${i}_tok.csv"
        python ${SLO_CLASSIFIER_FP}/stance/run_stance_detection.py \
            train \
            --model=${MODEL} \
            --path=${DATA_FP}/stance \
            --trainfp=${TRAINFP} \
            --testfp=${TESTFP} \
            --outfp=${DATA_FP}/svm-results/temp-${OUT_NAME} \
            --wvfp=${WVFP} \
            --logging_level=${LOGGING} \
            --noprofile
    done

    echo "Printing to ${DATA_FP}/svm-results/results.csv"

    # Add the data to the output CSV.
    python ${SLO_CLASSIFIER_FP}/stance/data/results_postprocessing.py \
        --input_filepath=${DATA_FP}/svm-results/temp-${OUT_NAME} \
        --output_filepath=${DATA_FP}/svm-results/${OUT_NAME} \
        --label="${LABEL}-NoProfile"

    cp ${DATA_FP}/svm-results/results-template.csv ${DATA_FP}/svm-results/temp-${OUT_NAME}

    # This second loop runs it with profile text added.
    for i in $(seq 1 $TIMES_TO_RUN);
    do
        echo "===========Running w/ Profile on Auto-Coded Set ${i}============"
        TRAINFP="coding/auto_${RUN_ID}_${i}_tok.csv"
        python ${SLO_CLASSIFIER_FP}/stance/run_stance_detection.py \
            train \
            --model=${MODEL} \
            --path=${DATA_FP}/stance \
            --trainfp=${TRAINFP} \
            --testfp=${TESTFP} \
            --outfp=${DATA_FP}/svm-results/temp-${OUT_NAME} \
            --wvfp=${WVFP} \
            --logging_level=${LOGGING}
    done

    echo "Printing to ${DATA_FP}/svm-results/results.csv"

    # Add the data to the output CSV.
    python ${SLO_CLASSIFIER_FP}/stance/data/results_postprocessing.py \
        --input_filepath=${DATA_FP}/svm-results/temp-${OUT_NAME} \
        --output_filepath=${DATA_FP}/svm-results/${OUT_NAME} \
        --label="${LABEL}-Profile"

    rm ${DATA_FP}/svm-results/temp-${OUT_NAME}

    # Remove the trainsets to save space.
    for i in $(seq 1 $TIMES_TO_RUN);
    do
        rm ${DATA_FP}/stance/coding/auto_${RUN_ID}_${i}.csv
        rm ${DATA_FP}/stance/coding/auto_${RUN_ID}_${i}_adani_tok.csv
        rm ${DATA_FP}/stance/coding/auto_${RUN_ID}_${i}_adani.csv
    done

    echo "Finished ${LABEL}"

}

# Set this script to exit if a command returns an error
set -e

# Reset the output CSV file.
if [ -f ${DATA_FP}/svm-results/${OUT_NAME} ]; then
    rm ${DATA_FP}/svm-results/${OUT_NAME}
fi
touch ${DATA_FP}/svm-results/${OUT_NAME}

source ${ENV}

run_repeat 2 ${WITHOUT_COMPANY} ${WITHOUT_COMPANY} ${WITHOUT_COMPANY} "${WITHOUT_COMPANY}-${WITHOUT_COMPANY}-${WITHOUT_COMPANY}-NoCompany" false

#run_repeat 25 ${WITHOUT_COMPANY} ${WITHOUT_COMPANY} ${WITHOUT_COMPANY} "${WITHOUT_COMPANY}-${WITHOUT_COMPANY}-${WITHOUT_COMPANY}-Company" true

run_repeat 2 ${WITH_COMPANY} ${WITH_COMPANY} ${WITH_COMPANY} "${WITH_COMPANY}-${WITH_COMPANY}-${WITH_COMPANY}-Company" true

#run_repeat 25 ${WITHOUT_COMPANY} $((5*WITHOUT_COMPANY)) ${WITHOUT_COMPANY} "${WITHOUT_COMPANY}-$((5*WITHOUT_COMPANY))-${WITHOUT_COMPANY}-NoCompany" false

#run_repeat 25 ${WITHOUT_COMPANY} $((5*WITHOUT_COMPANY)) ${WITHOUT_COMPANY} "${WITHOUT_COMPANY}-$((5*WITHOUT_COMPANY))-${WITHOUT_COMPANY}-Company" true

#run_repeat 25 ${WITH_COMPANY} $((5*WITH_COMPANY)) ${WITH_COMPANY} "${WITH_COMPANY}-$((5*WITH_COMPANY))-${WITH_COMPANY}-Company" true

#run_repeat 25 ${WITHOUT_COMPANY} $((10*WITHOUT_COMPANY)) ${WITHOUT_COMPANY} "${WITHOUT_COMPANY}-$((10*WITHOUT_COMPANY))-${WITHOUT_COMPANY}-NoCompany" false

#run_repeat 25 ${WITHOUT_COMPANY} $((10*WITHOUT_COMPANY)) ${WITHOUT_COMPANY} "${WITHOUT_COMPANY}-$((10*WITHOUT_COMPANY))-${WITHOUT_COMPANY}-Company" true

#run_repeat 25 ${WITH_COMPANY} $((5*WITH_COMPANY)) ${WITH_COMPANY} "${WITH_COMPANY}-$((10*WITH_COMPANY))-${WITH_COMPANY}-Company" true

deactivate
