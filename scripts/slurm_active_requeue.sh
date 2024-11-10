#! /bin/bash

################################
# Even though I only use spot quota with lowest priority so that anyone can preempt my job, I'm still blamed for 
# "occupying too many nodes". This script is designed to avoid holding the bag :)
# This script will automatically "requeuehold" all my running jobs when there are waiting jobs in the `monitor partition`
###############################

monitor_partition=mediaa
regular_check_period=2m
requeuehold_period=30m

while true; do
    # find the number of spot jobs in the queue submitted by other people
    num_spot_task_by_others=$(squeue -p ${monitor_partition} | awk '{if (($5 != "zhulei1") && (($4 == "spot") || ($4 == "auto")) && ($11 == "(Resources)")) print $0}' | wc -l)
    now=$(date '+%m-%d-%H:%M:%S')
    if [ ${num_spot_task_by_others} -eq 0 ]; then
        # no spot task waiting, check after regular_check_period
        echo "[${now}] no waiting spot tasks from others on partition ${monitor_partition}"
        sleep ${regular_check_period} 
    else
        # waiting spot tasks found, requeuehold my running jobs, and then relese after requeuehold_period
        my_running_jobids=$(squeue -p ${monitor_partition} | awk '{if (($5 == "zhulei1") && ($4 == "spot")  && ($7 == "R")) print $1}')
        echo "[${now}] found ${num_spot_task_by_others} waiting spot jobs, requeuehold my own running jobs (${my_running_jobids})"
        echo $my_running_jobids | xargs -n 1 scontrol requeuehold
        sleep ${requeuehold_period}
        echo $my_running_jobids | xargs -n 1 scontrol release
        sleep ${regular_check_period} 
    fi
done


# find the number of spot jobs in the queue submitted by other people
# num_spot_task_by_others=$(squeue -p ${monitor_partition} | awk '{if (($5 != "zhulei1") && (($4 == "spot") || ($4 == "auto")) && ($11 == "(Resources)")) print $0}' | wc -l)
# echo $num_spot_task_by_others

# # find my running jobids
# my_running_jobids=$(squeue -p ${monitor_partition} | awk '{if (($5 == "zhulei1") && ($7 == "R")) print $1}')
# echo $my_running_jobids | xargs -n 1 scontrol show job

## find number of idele nodes
# num_idle_nodes=$(sinfo | grep idle | awk '{print $4}') # TODO: if empty, set 0
