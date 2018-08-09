#!/usr/bin/env bash
#
# run_stance_detection_combined.sh
# 
# This script runs autocoding_processor.py, tweet_preprocesor.py,
# run_stance_detection.py, and results_postprocessing.py for a certain number of iterations.
#
# NOTE: This script does not allow specific ratios of for/against/neutral tweets. If you want to
# build a training set with specified stance tag ratios, see run_stance_detection_repeat.py
# This script just builds traininsets with the minimum number of tweets to balance for/against/
# neutral.
#
# Usage: bash run_stance_detection_combined.sh
#   data_fp: Filepath to the data
#   slo_fp: Filepath to the slo-classifier project
#   dataset_fp: What dataset you wish to run on.
#   output_fp: CSV filepath to output results to.
#
# Requirements: Must be running in a python 3.6 virtual environment with the following modules:
#   numpy, sklearn, keras, gensim, pandas, scipy, tensorflow, fire
#
###################################################################################################

# Create Parameters
data_fp="${1}"
slo_fp="${2}"
dataset_fp="${3}"
output_fp="${4}"

set -e

run_id=$RANDOM
declare -a company_list=("adani" "bhp" "santos" "riotinto" "fortescue")
testset_fp="coding/gold_20180514_majority_fixed_tok.csv"
wvfp="wordvec/20100101-20180510/all-100.vec"
logging=10

# This function builds all the training sets using autocoding_processor.py
function build_trainsets {
    local times_to_run=${1:-1}
    local include_company="${2}"

    for i in $(seq 1 ${times_to_run}); do
        echo "===============================Building Trainset ${i}==============================="

        local autocoding_filename=$(echo "${3}" | sed "s/.csv/_${i}.csv/")
        local args=(--dataset_filepath=${dataset_fp} \
            --testset_filepath=${data_fp}/stance/${testset_fp}
            --coding_filepath=${data_fp}/stance/coding/${autocoding_filename})
        if [ ${include_company} = true ]; then
            args+=( '--companytweets' )
        fi
        python ${slo_fp}/stance/data/autocoding_processor.py "${args[@]}"

        tokenize_trainsets ${autocoding_filename}
    done
}

# This function is used by build_trainsets to tokenize all the trainsets it produces
function tokenize_trainsets {
    local autocoded_complete="${1}"
    python ${slo_fp}/stance/data/tweet_preprocessor.py \
        --fp=${data_fp}/stance/coding/${autocoded_complete}
    for company in "${company_list[@]}"; do
        local autocoded_filename=$(echo "${autocoded_complete}" | sed "s/.csv/_${company}.csv/")
        python ${slo_fp}/stance/data/tweet_preprocessor.py \
            --fp=${data_fp}/stance/coding/${autocoded_filename}
    done
}

# This function builds and tests a model using a specified tokenized trainset.
function run_stance_detection {
    local times_to_run="${1}"
    local autocoded_filename_tok="${2}"
    local company="${3}"
    local model="${4}"
    local combined="${5}"

    local company_label="${model}-${company}"

    local output_dirname=$(dirname $output_fp)
    local output_filename=$(basename $output_fp)

    cp ${data_fp}/stance/coding/autocoding_template.csv ${output_dirname}/temp-${output_filename}

    for i in $(seq 1 ${times_to_run}); do
        echo "==============================Running on ${company} Trainset ${i}=============================="
        local args=(train \
            --model=${model} \
            --path=${data_fp}/stance \
            --testfp=${testset_fp} \
            --outfp=${output_dirname}/temp-${output_filename} \
            --wvfp=${wvfp} \
            --logging_level=${logging} \
            --profile)
        if [ ${combined} = true ]; then
            local trainset_fp=$(echo "${autocoded_filename_tok}" | sed "s/tok/${i}_tok/")
            args+=( --combined )
        else
            local trainset_fp=$(echo "${autocoded_filename_tok}" | sed "s/tok/${i}_${company}_tok/")
        fi
        args+=( --trainfp=coding/${trainset_fp} )
        python ${slo_fp}/stance/run_stance_detection.py "${args[@]}"        
    done

    python ${slo_fp}/stance/data/results_postprocessing.py \
        --input_filepath=${output_dirname}/temp-${output_filename} \
        --output_filepath=${output_fp} \
        --label="${company_label}"
}

function run {
    local times_to_run="${1}"
    local autocoded_filename_tok="${2}"
    local model="${3}"
    for company in "${company_list[@]}"; do
        run_stance_detection ${times_to_run} ${autocoded_filename_tok} ${company} ${model} false
    done
    run_stance_detection ${times_to_run} ${autocoded_filename_tok} "combined" ${model} true
}

function cleanup {
    echo "Running Cleanup"
    rm ${data_fp}/stance/coding/*${run_id}*
}

# This is a central function that calls all the other functions.
function run_stance_detection_repeat {
    local times_to_run="${1:-1}"
    local model="${2}"
    local autocoding_filename=$(basename ${dataset_fp} | sed "s/dataset/auto_${run_id}/")
    build_trainsets ${times_to_run} true ${autocoding_filename}
    local autocoded_filename_tok=$(echo "${autocoding_filename}" | sed "s/.csv/_tok.csv/")
    run ${times_to_run} ${autocoded_filename_tok} ${model}
    cleanup
}

# Set this script to exit if a command returns an error
set -e

# Reset the output CSV file.
if [ -f ${output_fp} ]; then
    rm ${output_fp}
fi
touch ${output_fp}

# RUN
run_stance_detection_repeat 25 "svm"
run_stance_detection_repeat 25 "cn"
run_stance_detection_repeat 25 "mn"
run_stance_detection_repeat 25 "tf"

