# Roy and Brent's Senior Project

1. SSH into borg (which has singularity)
2. Clone this repository.
3. Run the following command to grab the container:
```
singularity build INSERT_CONTAINER_NAME.simg shub://brentritzema/senior-project:latest
```
4. Change directories so that you are in slo-classifiers/stance/more-classifiers (where this readme is)
5. Run slurm script as follows:
```
sbatch run_classifier_runner_repeated.script root/location/of/slo-classifiers/repo path/to/container.simg
```
To change the number of runs (currently set at 25 per classifier) go into classifier_runner_repeated.sh	and at the bottom change the number after classifier_runner to the desired number of runs.
To change the number of nodes/cores/memory to use on Borg, look at https://borg.calvin.edu/resources-slurm.html under Job Submission. The parameters are currently set in run_classifier_runner_repeated.script
