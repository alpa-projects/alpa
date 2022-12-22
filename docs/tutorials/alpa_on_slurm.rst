Using Alpa on Slurm
###########

This page provides instructions to run Alpa on clusters managed by Slurm.
This guide is modified from `Deploy Ray on Slurm <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html>`_.

Prerequisites
**********

Alpa requires CUDA and runs on GPU. Therefore you need access to CUDA-compatible GPU in the cluster to run Alpa.

Please make sure you have or can create an environment with required dependencies installed for Alpa.
These dependencies are described in `Install Alpa <https://alpa.ai/install.html>`_.

.. note::

    One easy way to create an environment with all dependencies installed is to create an environment using package management systems like `conda <https://docs.conda.io/en/latest/>` and setup the environment running an interactive job on Slurm.
    The interact job on Slurm can allow you to run and experiment in the exact way you do locally.

Create :code:`sbatch` Script
**********

Usually large jobs like Alpa is run through sbatch on Slurm using a :code:`sbatch` script. :code:`sbatch` scripts are bash scripts with :code:`sbatch` options specified using :code:`#SBATCH <options>`.
The Slurm cluster takes sbatch scripts submitted using command sbatch then queue the job specified by the script for execution.
When Slurm executes the script, the script works exactly the same as a shell script.
A :code:`sbatch` script to run Alpa can be roughly summarized as four parts: resources setup, load dependencies, Ray startup, and run Alpa.

The first step is to create a :code:`sbatch` script in your directory, usually named as a :code:`.sh`` file.

Just like a shell script, the :code:`sbatch` script starts with a line specifying the path to interpreter:

.. code:: bash

    #!/bin/bash

Resources Setup
===========

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

    Setting the resources needed in the sbatch script is equivalent to setting them submitting the job to Slurm running :code:`sbatch <options> <sbatch script>`.

Load Dependencies
===========

The next step is to setup the environment with Alpa's dependencies installed.
In some Slurm clusters, CUDA, NVCC, and CuDNN are packed in containers that can be loaded directly. Here, we provide an example that loads a combination of available container and user-defined environment from package management systems.
To load directly from available containers, use :code:`module load <module>`:

.. code:: bash

    module load cuda
    module load cudnn
    module load nvhpc

.. note::

    At this step, usually one can specify the version used if available like:

    .. code:: bash

        module load cuda/11.1.1
        module load cudnn/8.0.5

To activate a environment using package management systems like conda, add the following line:

.. code:: bash

    conda activate alpa_environment

In summary, this step adds a chunk in the script like below:

.. code:: bash

    # load containers
    module purge
    module load cuda
    module load cudnn
    module load nvhpc
    # activate conda environment
    conda activate alpa_environment

After this step, all the dependencies, including packages and softwares needed for Alpa is loaded and can be used.

Running within one node in the cluster, you can jump to use :ref:`Single Node Script`.

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

.. notes::

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

    Note here the argument gpus_per_node should not exceed the number of GPU you have on each node.

Then we spawn worker nodes on other nodes we have and connect then to HEAD:

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

        squeue -u <partition_name>

    When you no longer see a job in the list, it means the job is finished.

Check Output
**********

After a Slurm job is finished, the output will appear in your directory as a file.
On some Slurm clusters, the output file is named :code:`slurm-<job_number>.out`.
You can check the file for output the same way you read a text file.

Sample sbatch Scripts
**********

Multi-node Script
===========

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
===========

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