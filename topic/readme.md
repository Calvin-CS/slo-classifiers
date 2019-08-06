# Topic Classifier

The topic classifier is a legacy system.

Early work on SLO developed a "value-tree" for topics, including governance, 
social, economic and environmental, but the distinctions were too challenging 
to tag and classify accurately so the task was scaled back to use the 
Triple-Bottom Line (TBL), which is a more forgiving task.

# Topic Modeling (topic extraction)

We have finished preliminary data analysis on the SLO Twitter dataset. Current work consists of topic extraction using 
Latent Dirichlet Allocation, Hierarchical Latent Dirichlet Allocation, Hierarchical Dirichlet Process,
Author-Topic Model, Biterm Topic Model, and Non-Negative Matrix Factorization.  We may implement more baseline topic
models in the future.

## Project Trello Board:

https://trello.com/b/74Bqfgac/slo-topic

## Links to Other Related GitHub Repositories:

* Joseph Jinn Summer Research 2019 GitHub Repository:

https://github.com/J-Jinn/Summer-Research-2019

* Joseph Jinn SLO-TBL Topic Classification Repository:

https://github.com/J-Jinn/cs344/tree/master/Project

## Directory Hierarchy:

"data" - SLO Twitter Dataset Processing and Analysis

"models" - SLO Twitter Dataset Topic Modeling Algorithms

"TBL" - SLO Twitter Dataset Topic Classification Algorithms (not part of master branch - only a unmerged feature branch)

### Data Directory:

- "images" directory - stores .png and .jpeg files used in Jupyter Notebooks

- "notebooks" directory - stores Jupyter Notebooks linked to by the data analysis table of contents Jupyter Notebook

&nbsp;

- slo_twitter_data_analysis.py
    - codebase for the Twitter dataset analysis.
    
- slo_twitter_data_analysis_utility_functions.py
    - utility functions for Twitter dataset analysis.

- slo-twitter-data-analysis-table-of-contents.ipynb
    - Jupyter Notebook table of contents file linking to each of the data analysis sub-categories.

- topic_dataset_processor.py
    - Python file that creates our CSV dataset from the raw JSON file.

### Models Directory:

- "images" directory - stores .png and .jpeg files used in Jupyter Notebooks

- "notebooks" directory - stores Jupyter Notebooks linked to by the topic analysis table of contents Jupyter Notebook

- "topic-modeling-code-examples" directory - stores example code running topic modeling algorithms on a sample dataset 
(not our Twitter dataset)

&nbsp;

- author-topic-model.py
    - Gensim Author-Topic Model Algorithm.
    
- biterm-topic-model.py
    - Biterm Topic Model Algorithm.
    
- hierarchical-dirichlet-process-model.py
    - Gensim HDP Topic Model Algorithm.
    
- hierarchical-latent-dirichlet-allocation-model.py
    - HLDA Topic Model Algorithm.

- latent_dirichlet_allocation-model.py
    - Scikit-Learn LDA Topic Model Algorithm.

- non-negative-matrix-factorization-model.py
    - Scikit-Learn NMF Topic Model Algorithm.
    
&nbsp;

- topic-modeling-algorithms-table-of-contents.ipynb
    - Jupyter Notebook table of contentsn file linking to each of the topic analysis sub-categories.

- topic_extraction_utility_functions.py
    - utility functions for Twitter topic extraction.

&nbsp;

- lda_visualization.html
    - pyldAVIS visualization of the Scikit-Learn LDA model topic extraction results with company names included in the
    Tweet text.
    
- lda_visualization-no-company-words.html
    - pyldAVIS visualization of the Scikit-Learn LDA model topic extraction results with company names excluded from the
    Tweet text.

### TBL Directory:

Note: Refer to the README.md for more information on the files included in this feature branch that is not part of the
SLO-classifiers Repository's master branch.

