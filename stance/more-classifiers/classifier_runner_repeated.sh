#!/usr/bin/env bash
#
# classifier_runner_repeated.sh
# 
# This script runs autocoding_processor.py, tweet_preprocesor.py,
# classifier_runner.py for a certain number of iterations.
#
# Usage: bash classifier_runner_repeated.sh
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
}

# This function builds and tests a model using a specified tokenized trainset.
function run_classifiers {
    local times_to_run="${1}"
    local autocoded_filename_tok="${2}"

    for i in $(seq 1 ${times_to_run}); do
    	local trainset_fp=$(echo "${autocoded_filename_tok}" | sed "s/tok/${i}_tok/")
		python classifier_runner.py \
			--trainset_fp=${data_fp}/stance/coding/${trainset_fp} \
			--testset_fp=${data_fp}/stance/${testset_fp} \
			--output_fp=${output_fp}
    done
}

function cleanup {
    echo "Running Cleanup"
    rm ${data_fp}/stance/coding/*${run_id}*
}

# This is a central function that calls all the other functions.
function classifier_runner {
    local times_to_run="${1:-1}"
    local autocoding_filename=$(basename ${dataset_fp} | sed "s/dataset/auto_${run_id}/")
    build_trainsets ${times_to_run} true ${autocoding_filename}
    local autocoded_filename_tok=$(echo "${autocoding_filename}" | sed "s/.csv/_tok.csv/")
    run_classifiers ${times_to_run} ${autocoded_filename_tok}
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
classifier_runner 5

#sorted_fp=$(echo $output_fp | sed "s/.csv/-sorted.csv/")
#echo $(sort ${output_fp} > ${sorted_fp}) > ${output_fp}
#rm ${sorted_fp}

python classifier_postprocessor.py ${output_fp}

# TODO: CREATE PLOTS

