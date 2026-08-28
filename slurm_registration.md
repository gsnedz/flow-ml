# A Guide to Accessing Slurm on the Campus Cluster for Research
### By Garrett Snedden, last updated August 2026

## Register for an account.
Go to https://campuscluster.illinois.edu/ > Access > Request Access for Research.  Fill out the form.  The QGP queue is for Jaki’s group, not Sickles.  I selected the physics queue.  When I registered, I was also granted access to the secondary and scavenger queues.

## Logging in.
Once your account is approved, you may log in.  Open a terminal and enter the command

	ssh [net id]@cc-login.campuscluster.illinois.edu 
which will log into the campus cluster.  Note that when entering password on the cluster, it will not show the characters that you are typing.

## Moving files from local device to cluster and back.
If using the NCSA Jupyter notebook site (jupyter.ncsa.illinois.edu), you may enter the command 
  
  ```!ln -s /u/[net id] ${HOME}/home```
  
in a Jupyter notebook, which will create a shortcut in the file navigator GUI on NCSA to the folder on the Cluster.  Thus through the Jupyter notebook site you can easily upload, download, move, view, and edit files in the Cluster.
	Otherwise, to move files directly between the cluster and your local device, use the following terminal command when logged into your local device to move a file from your device to the cluster:
	
  ```scp [options] [home file path on local device] [net id]@cc-login.campuscluster.illinois.edu:/[destination file path on cluster]```
and this command to move a file from the cluster to your local device:
	
  ```scp [options] [net id]@cc-login.campuscluster.edu:/~[home file path on cluster] ./[destination file path on local device]```
	
Running code on the cluster.
	When on the cluster, all the usual Linux file commands (cd, ls, nano, etc.) work.  With access to Slurm, you can also use all of the Slurm commands, outlined here https://slurm.schedmd.com/quickstart.html.  Important to note are
	
```sbatch [file].slurm``` to run a .slurm file,

```scancel [job #]``` to cancel a job, and

```squeue -u [net id]``` to view jobs currently running.

A template  .slurm file is below (this runs the file testing.py).
```
#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --partition=secondary        # or: general, batch, etc.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00        # HH:MM:SS
#SBATCH --mem=16G
#SBATCH --output=testing.out
#SBATCH --error=testing.err
#SBATCH --exclude=ccc0089,ccc0090

# Load modules
module load python/3.10
pip install torch==2.4.1 torchvision torchaudio

# Optional: activate virtual environment
# source ~/venvs/myenv/bin/activate

# Run your program 
python testing_run.py
```


