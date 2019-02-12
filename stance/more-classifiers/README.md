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
