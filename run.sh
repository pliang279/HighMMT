#!/bin/bash
echo "Running HighMMT training and evaluation"
TASKCHOICE=$1
MULTITASK=$2
if [ "$TASKCHOICE" == "env" ]; then 
	echo "setting up environment"
	conda env create -f env_HighMMT.yml
	exit 
elif [ "$TASKCHOICE" == "help" ]; then
	echo "printing instructions"
	echo "to setup environment, run conda env create -f 'env_HighMMT.yml' or './run.sh env'"
	echo "to run robotics task code and save the model, run command 'python private_test_scripts/perceivers/roboticstasks.py model.pt' or './run.sh tasks robotics' "
	echo "to run multitask training code on the default single task, run command 'python private_test_scripts/perceivers/singletask.py' or './run.sh tasks single' "
	echo "to run multitask training code on the default double multitask, run command 'python private_test_scripts/perceivers/twomultitask.py' or './run.sh tasks two_multi'"
	echo "to run multitask training code on the default three multitask, run command 'python private_test_scripts/perceivers/threemultitask.py' or './run.sh tasks three_multi'"
	echo "to run multitask training code on the default four multitask, run command 'python private_test_scripts/perceivers/fourmultitask.py' or './run.sh tasks four_multi'"
	echo "to run multitask training code on the default medium task, run command 'python python private_test_scripts/perceivers/medium_tasks.py' or './run.sh tasks medium'"
	echo "To run get the heterogeneity matrix between individual modalitiesa and pairs of modalities, run command 'python private_test_scripts/perceivers/tasksim.py' or './run.sh matrix'"
	echo "To install datasets, run ./run.sh datasets"
	exit
elif [ "$TASKCHOICE" == "tasks" ]; then 
	if [ "$MULTITASK" == "robotics" ]; then
		echo "Running task for robotics"
		python private_test_scripts/perceivers/roboticstasks.py model.pt
	elif [ "$MULTITASK" == "single" ]; then
		echo "Running single sample tasks"
		python private_test_scripts/perceivers/singletask.py
	elif [ "$MULTITASK" == "two_multi" ]; then
		echo "Running multitasks, number of tasks:2"
		python private_test_scripts/perceivers/twomultitask.py
	elif [ "$MULTITASK" == "three_multi" ]; then
		echo "Running multitasks, number of tasks: 3"
		python private_test_scripts/perceivers/threemultitask.py
	elif [ "$MULTITASK" == "medium" ]; then 
		echo "Running medium tasks"
		python python private_test_scripts/perceivers/medium_tasks.py
	elif [ "$MULTITASK" == "four_multi" ]; then
		echo "Running multitasks, number of tasks: 4"
		python private_test_scripts/perceivers/fourmultitask.py
	else 
		echo "error: Invalid command for running scripts, for detailed instructions run './run.sh help'"
		exit
	fi
elif [ "$TASKCHOICE" == "matrix" ]; then 
	echo "generating matrix"
	python private_test_scripts/perceivers/tasksim.py
elif [ "$TASKCHOICE" ==  "datasets" ]; then 
	echo "download datasets enrico, RTFM and robotics, run './run.sh datasets'"
	./download_datasets.sh RTFM
	./download_datasets.sh enrico
	./download_datasets.sh robotics
	echo "datasets downloaded"
	exit
else
	echo "Invalid argument,for detailed instructions run './run.sh help'"
	exit
fi



