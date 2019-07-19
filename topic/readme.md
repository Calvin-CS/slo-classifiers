# Topic Classifier

The topic classifier is a legacy system.

Early work on SLO developed a "value-tree" for topics, including governance, 
social, economic and environmental, but the distinctions were too challenging 
to tag and classify accurately so the task was scaled back to use the 
Triple-Bottom Line (TBL), which is a more forgiving task.

# Topic Modeling (topic extraction)

We have finished preliminary data analysis on the SLO Twitter dataset. Current work consists of topic extraction using 
Latent Dirichlet Allocation, Hierarchical Latent Dirichlet Allocation, Hierarchical Dirichlet Process,
Author-Topic Model, Biterm Topic Model, and Non-Negative Matrix Factorization.  We may implement more topic
models in the future.

## Project Trello Board:

https://trello.com/b/74Bqfgac/slo-topic

## Links to Other Related GitHub Repositories:

* Joseph Jinn Summer Research 2019 GitHub Repository:

https://github.com/J-Jinn/Summer-Research-2019

* Joseph Jinn SLO-TBL Topic Classification Repository:

https://github.com/J-Jinn/cs344/tree/master/Project

## Directory Hierarchy:

"Data" - SLO Twitter Dataset Processing and Analysis

"Model" - SLO Twitter Dataset Topic Modeling Algorithms

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

### Model Directory:

- "images" directory - stores .png and .jpeg files used in Jupyter Notebooks

- "notebooks" directory - stores Jupyter Notebooks linked to by the topic analysis table of contents Jupyter Notebook

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
    
&nbsp;

- topic-modeling-algorithms-table-of-contents.ipynb
    - Jupyter Notebook table of contentsn file linking to each of the topic analysis sub-categories.

- topic_extraction_utility_functions.py
    - utility functions for Twitter topic extraction.

- topic-modeling-code-examples/
    - example code running topic modeling algorithms on a sample dataset (not our Twitter dataset)

