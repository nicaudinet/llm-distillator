# Running a job

The `jobscript` is what I use to submit a job to the job queue (`sbatch`). It
is defined as a shell script, and you need to add a bunch of options for
`sbatch` at the start.

# Building an environment

This can be done either with modules or containers. Modules are faster and
easier to use but much less flexible (if Alvis doesn't have what you need you
will have to use modules).

To use containers, we can use `apptainer`. We first define a recipe
(`recipe.def`) which tells the system how to build the container. Then we need
to build it with:

```
apptainer build container.sif recipe.def
```

where `container.sif` is the filepath the container will be outputted to. For
some reason the python Alpine docker container bases don't seem to work (I got
some kind of fakeroot error) but using the Debian bookworm containers worked.

To execute a command within the container you can use the following:

```
apptainer exec container.sif COMMAND
```

To drop into a shell within the container:

```
apptainer shell container.sif
```

You can do either of these things within a jobscript.

# Downloading a model 

To download a model we first have to enable git lfs (Git Large File Storage),
then clone the model from huggingface to Mimer. Something like this:

```
module purge
module load git-lfs
cd /mimer/NOBACKUP/groups/ci-nlp-alvis/models/
git lfs install
git clone MODEL\_URL
```

The home directory only has ~30G of storage. I have an extra 500G in Mimer. You
can use the OnDemand page to see how much resources I'm using.

# Running Mistral

I've downloaded the Mistral 7B inference model into Mimer. Now I'm going to try
to make the container with the proper Python dependencies and run the code to
run inference on the model. 
