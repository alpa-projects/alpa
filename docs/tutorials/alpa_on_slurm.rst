Using Alpa on Slurm
##########################

This page provides instructions to run Alpa on clusters managed by Slurm.
This guide is modified from `Deploy Ray on Slurm <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html>`_.

Prerequisites
**************

Alpa requires CUDA and runs on GPUs. Therefore, you need access to CUDA-compatible GPUs in the cluster to run Alpa.

If you have a virtual environment with required dependencies for Alpa installed, please jump to the section `Create sbatch Script`_.

If you don't have one already, please follow the steps below to create a Python virtual environment for Alpa.

Create Environment for Alpa
===========================

Here we give a sample workflow to create an environment for Alpa and verfy the environment as well.

We assume that there is Python environment management systems available on the Slurm cluster, e.g. `conda <https://docs.conda.io/en/latest/>`_.

The following steps can make sure you set up the environment needed by Alpa correctly. To illustrate this process we use conda, but the workflow also applies to other Python virtual environment management systems.

Enter Interactive Mode 
----------------------

The first step is to create an interactive mode on Slurm so that you can run commands and check results in real-time.

    .. code:: bash

        interact -p GPU-shared -N 1 --gres=gpu:v100-16:1 -t 10:00

    .. note::

        1. This command starts an interactive job on the partition of GPU-shared (:code:`-p GPU-shared`), on one GPU node (:code:`-N 1`) with one v100-16GB GPU assigned (:code:`--gres=gpu:v100-16:1`) for a time of 10 minutes (:code:`-t 10:00`).

        2. The minimum command to start an interactive job is by running :code:`interact` directly. This will ask the cluster to assign all resources by default. As we want to setup and test the environment, we need GPU to test and hence we need to specify all these options.

        3. The name of the partition, the option for assigning GPU (in some clusters, :code:`--gpus=` is used instead of :code:`--gres=`), and the name for GPU depend on the cluster you work on. Please use these values specified by your cluster.

Then the cluster will try to assign the resources you asked for.

Once you got these resources, the command :code:`interact` returns and you are now in an interactive session. Your commandline will show that you are now at the node of the cluster like :code:`user@v001:~/$` where :code:`v001` is the name of the node you are assigned to.

Load Required Software 
-----------------------

In some clusters, containers of common tools are packed and can be loaded through :code:`module load <target>`.

For example, to run Alpa, we need CUDA, so we run the following to get CUDA:

    .. code:: bash

        module load cuda

    .. note::

        Some clusters have multiple versions of CUDA for you to use, you can check all versions using:

            .. code:: bash

                module avail cuda

        Then you can specify a version to be used. Like Alpa requires CUDA >= 11.1.

            .. code:: bash

                module load cuda/11.1

        Please refer to the manager of the cluster to install a specific versions of CUDA or other softwares if not available.

Similarly, we can load CuDNN and NVCC:

    .. code:: bash

        module load cudnn
        module load nvhpc

    .. note::

        1. Note here :code:`nvhpc` is loaded as it contains :code:`nvcc` in the cluster. :code:`nvcc` might need to be loaded differently in other clusters. Please check with your cluster on how to load :code:`nvcc`.

        2. A common issue with CuDNN on clusters is the version provided by the cluster is older than the version required by Alpa (>= 8.0.5). To solve this, one can install a compatible version of CuDNN inside the Python virtual environment like the installation of other dependencies covered in `Install and Check Dependencies`_.

Install and Check Dependencies
------------------------------

Before you install dependencies of Alpa, create a virtual environment with the version of Python you use:

    .. code:: bash

        conda create -n alpa_environment python=3.9

    .. note::

        Please make sure the Python version meets the requirement by Alpa of >= 3.7.

Then enter this Python virtual environment:

    .. code:: bash

        conda activate alpa_environment

Then your commandline should show as :code:`(alpa_environment)user@v001:~/$`.

    .. note::
        Check the environment is entered by checking the version of python:

            .. code:: bash

                python3 --version

Then please follow the installation guide of Alpa in `Install Alpa <https://alpa.ai/install.html#install-alpa>`_.
All the commands mentioned in the installation guide works for you and can make sure the environment :code:`alpa_environment` works with Alpa.

Exit Virtual Environment
------------------------

Once you have finished installation and testing, exit the environment:

    .. code:: bash

        conda deactivate

Next time you want to activate this environment, use the following command:

    .. code:: bash

        conda activate alpa_environment

Exit Interactive Session
------------------------

To exit interactive session, press :code:`Ctrl+D`.

Create :code:`sbatch` Script
****************************

Usually large jobs like Alpa is run through sbatch on Slurm using a :code:`sbatch` script. :code:`sbatch` scripts are bash scripts with :code:`sbatch` options specified using the syntax of :code:`#SBATCH <options>`.
The Slurm cluster takes sbatch scripts submitted using command :code:`sbatch`` then queues the job specified by the script for execution.
When Slurm executes the script, the script works exactly the same as a shell script.

    .. note::

        The shell script commands are run on each of the nodes assigned for your job. To specify running a command at one node, use command :code:`srun`'s option of :code:`--nodes=1`. 
        Available options for :code:`srun` can be found in `SRUN <https://slurm.schedmd.com/srun.html>`_. :code:`srun` is to run a job for execution in real time while :code:`sbatch` allows you to submit a job for later execution without blocking.
        :code:`srun` is also compatible with the script we create.

A :code:`sbatch` script to run Alpa can be roughly summarized as four parts: resources setup, load dependencies, Ray startup, and run Alpa.

The first step is to create a :code:`sbatch` script in your directory, usually named as a :code:`.sh` file.
Here, this guide asumes the script is included in a file :code:`run_alpa_on_slurm.sh`.
Just like a shell script, the :code:`sbatch` script starts with a line specifying the path to interpreter:

    .. code:: bash

        #!/bin/bash

Resources Setup
================

The first lines in your sbatch script is used to specify resources for the job like all the options you specify when running srun or sbatch.
These usually includes the name of the job, partition the job should go to, CPU per task, memory per CPU, number of nodes, number of tasks per node, and time limit for the job.

    .. code:: bash

        #SBATCH --job-name=textgen_multinode_test
        #SBATCH --partition=GPU
        #SBATCH --nodes=2
        #SBATCH --tasks-per-node=1
        #SBATCH --cpus-per-task=1
        #SBATCH --mem-per-cpu=1GB
        #SBATCH --gpus-per-node=v100-16:8
        #SBATCH --time=00:10:00

    .. note::

        1. Setting the resources needed in the sbatch script is equivalent to setting them submitting the job to Slurm running :code:`sbatch <options> <sbatch script>`.
        
        2. Here, :code:`--tasks-per-node=1 --cpus-per-task=1` are specified to allow Ray (which uses CPU) to run on the nodes.

        3. The option :node:`--gpus-per-node=v100-16:8` is specified with GPU type and number. Please refer to your cluster on how to set this field.

Load Dependencies
=================

The next step is to setup the environment with Alpa's dependencies installed.
In some Slurm clusters, CUDA, NVCC, and CuDNN are packed in containers that can be loaded directly. Here, we provide an example that loads a combination of available container and user-defined environment from package management systems.
To load directly from available containers, use :code:`module load <module>`:

    .. code:: bash

        module load cuda
        module load cudnn
        module load nvhpc

    .. note::

        1. To check available softwares, run:

            .. code:: bash

                module avail
                module avail cuda

        When there is no required software, please ask manager of the cluster to install.

        When multiple versions available, one can specify the version to be used:

            .. code:: bash

                module load cuda/11.1.1
                module load cudnn/8.0.5

        You can check the module needed is used with:

            .. code:: bash

                module list cuda

        1. The load of pre-installed software can be different in different clusters, please use the way your cluster uses.

To activate an environment using package management systems like conda, add the following line:

    .. code:: bash

        conda activate alpa_environment

In summary, this step adds a chunk in the script like below:

    .. code:: bash

        # load containers
        module purge    # optional
        module load cuda
        module load cudnn
        module load nvhpc
        # activate conda environment
        conda activate alpa_environment

After this step, all the dependencies, including packages and softwares needed for Alpa is loaded and can be used.

Running within one node in the cluster, you can jump to use `Single Node Script`_.

Ray Startup
===========

Then it's time for Ray to run.
The first step is to grab the nodes assigned in the cluster to this job and name one node to be head node in the topology of a Ray cluster:

    .. code:: bash

        # Get names of nodes assigned
        nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
        nodes_array=($nodes)

        # By default, set the first node to be head_node on which we run HEAD of Ray
        head_node=${nodes_array[0]}
        head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    .. note::

        In the case of a cluster uses ipv6 addresses, one can use the following script after we get head node ip to change it to ipv4:

        .. code:: bash
            
            # Convert ipv6 address to ipv4 address
            if [[ "$head_node_ip" == *" "* ]]; then
            IFS=' ' read -ra ADDR <<<"$head_node_ip"
            if [[ ${#ADDR[0]} -gt 16 ]]; then
            head_node_ip=${ADDR[1]}
            else
            head_node_ip=${ADDR[0]}
            fi
            echo "Found IPV6 address, split the IPV4 address as $head_node_ip"
            fi

After we have a head node, we spawn HEAD on head node:

    .. code:: bash

        # Setup port and variables needed
        gpus_per_node=8
        port=6789
        ip_head=$head_node_ip:$port
        export ip_head
        # Start HEAD in background of head node
        srun --nodes=1 --ntasks=1 -w "$head_node" \
                ray start --head --node-ip-address="$head_node_ip" --port=$port \
                --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $gpus_per_node --block &

    .. note::

        Note here the argument :code:`gpus_per_node` should not exceed the number of GPU you have on each node.

Then we spawn worker nodes on other nodes we have and connect them to HEAD:

    .. code:: bash

        # Start worker nodes
        # Number of nodes other than the head node
        worker_num=$((SLURM_JOB_NUM_NODES - 1))
        # Iterate on each node other than head node, start ray worker and connect to HEAD
        for ((i = 1; i <= worker_num; i++)); do
            node_i=${nodes_array[$i]}
            echo "Starting WORKER $i at $node_i"
            srun --nodes=1 --ntasks=1 -w "$node_i" \
                ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" \
                --num-gpus $gpus_per_node --block &
            sleep 5
        done

[Optional] Check Ray is Running
===============================

You can check if Ray is started and all nodes connected by adding this line:

    .. code:: bash

        ray list nodes

In the output of the job, you are expected to see the same number of nodes you asked for listed by this command.

At this time, we have a running Ray instance and next we can run Alpa on it.

Run Alpa
===========

Just like running Alpa locally, the previous steps are equivalent of having run ray with :code:`ray start --head`.
Then it's time to run Alpa:

    .. code:: bash

        python3 alpa/examples/llm_serving/textgen.py --model alpa/bloom-560m --n-prompts 1

    .. note::

        To run text generation using Alpa, please first install llm_serving in your environment. The installation follws from `here <https://alpa.ai/tutorials/opt_serving.html#requirements>`_.

Submit Job
**********

To submit the job, run the following command:

    .. code:: bash

        sbatch run_alpa_on_slurm.sh

    .. note::

        After you submit the job, Slurm will tell you the job's number. You can check your submitted job's status using command squeue.
        To find all jobs you have, run:

        .. code:: bash

            squeue -u <your_user_name>

        To check all jobs running and queued in a partition, run:

        .. code:: bash

            squeue -p <partition_name>

        When you no longer see a job in the list, it means the job is finished.

    .. note::

        Another option is to run with :code:`srun`. 
        This will give an interactive session, meaning the output will show up in your terminal 
        while :code:`sbatch` collects output to a file.

        Run with :code:`srun` is exactly the same as :code:`sbatch`:
        
        .. code:: bash

            srun run_alpa_on_slurm.sh

Check Output
************

After a Slurm job is finished, the output will appear in your directory as a file (if you submitted the job through :code:`sbatch`).
On some Slurm clusters, the output file is named :code:`slurm-<job_number>.out`.
You can check the file for output the same way you read a text file.

Sample :code:`sbatch`` Scripts
******************************

Multi-node Script
=================

Putting things together, a sample sbatch script that runs Alpa is as follows:

    .. code:: bash

        #!/bin/bash
        #SBATCH --job-name=textgen_multinode_test
        #SBATCH --partition=GPU
        #SBATCH --nodes=2
        #SBATCH --tasks-per-node=1
        #SBATCH --cpus-per-task=1
        #SBATCH --mem-per-cpu=1GB
        #SBATCH --gpus-per-node=v100-16:8
        #SBATCH --time=00:10:00

        # load containers
        module purge
        module load cuda
        module load cudnn
        module load nvhpc
        # activate conda environment
        conda activate alpa_environment

        # Get names of nodes assigned
        nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
        nodes_array=($nodes)

        # By default, set the first node to be head_node on which we run HEAD of Ray
        head_node=${nodes_array[0]}
        head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

        # Setup port and variables needed
        gpus_per_node=8
        port=6789
        ip_head=$head_node_ip:$port
        export ip_head
        # Start HEAD in background of head node
        srun --nodes=1 --ntasks=1 -w "$head_node" \
                ray start --head --node-ip-address="$head_node_ip" --port=$port \
                --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $gpus_per_node --block &

        # Optional, sometimes needed for old Ray versions
        sleep 10

        # Start worker nodes
        # Number of nodes other than the head node
        worker_num=$((SLURM_JOB_NUM_NODES - 1))
        # Iterate on each node other than head node, start ray worker and connect to HEAD
        for ((i = 1; i <= worker_num; i++)); do
            node_i=${nodes_array[$i]}
            echo "Starting WORKER $i at $node_i"
            srun --nodes=1 --ntasks=1 -w "$node_i" \
                ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" \
                --num-gpus $gpus_per_node --block &
            sleep 5
        done

        # Run Alpa textgen
        python3 alpa/examples/llm_serving/textgen.py --model alpa/bloom-560m --n-prompts 1

        # Optional. Slurm will terminate all processes automatically
        ray stop
        conda deactivate
        exit

Single Node Script
==================

For running Alpa on Slurm with only one node or shared node, the following script can be used:

    .. code:: bash

        #!/bin/bash
        #SBATCH --job-name=textgen_uninode_test
        #SBATCH -p GPU-shared
        #SBATCH -N 1
        #SBATCH --gpus=v100-16:1
        #SBATCH -t 0:10:00

        # load containers
        module purge
        module load cuda
        module load cudnn
        module load nvhpc
        # activate conda environment
        conda activate alpa_environment

        # Start Ray on HEAD
        ray start --head

        # Run Alpa textgen
        python3 alpa/examples/llm_serving/textgen.py --model alpa/bloom-560m --n-prompts 1

        # Optional. Slurm will terminate all processes automatically
        ray stop
        conda deactivate
        exit
