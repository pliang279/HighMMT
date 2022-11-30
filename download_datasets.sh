#!/bin/bash
echo "Setting Up datasets"
DATASET=$1
if [ "$DATASET" == "RTFM" ]; then 
	echo "setting up dataset RTFM"
	cd datasets/RTFM
	pip install -e .
	cd ../..
	echo "setup complete"
	exit 
elif [ '$DATASET' == "robotics" ]; then
	echo "Setting up robotics dataset"
	cd datasets/robotics
	wget http://downloads.cs.stanford.edu/juno/triangle_real_data.zip -O _tmp.zip
	unzip _tmp.zip
	rm _tmp.zip
	cd ../..
	echo "setup complete"
	exit
elif [ "$DATASET" == "enrico" ]; then
	echo "Setting up enrico dataset"
	cd datasets/enrico
	mkdir -p dataset
	# download data
	wget https://raw.githubusercontent.com/luileito/enrico/master/design_topics.csv -P dataset
	wget http://userinterfaces.aalto.fi/enrico/resources/screenshots.zip -P dataset
	wget http://userinterfaces.aalto.fi/enrico/resources/wireframes.zip -P dataset
	wget http://userinterfaces.aalto.fi/enrico/resources/hierarchies.zip -P dataset
	wget http://userinterfaces.aalto.fi/enrico/resources/metadata.zip -P dataset
	# unzip data
	cd dataset
	unzip screenshots.zip
	unzip wireframes.zip
	unzip hierarchies.zip
	unzip metadata.zip
	# remove archive files
	rm screenshots.zip
	rm wireframes.zip
	rm hierarchies.zip
	rm metadata.zip
	cd ..
	cd ../..
	echo "setup complete"
	exit
elif [ "$DATASET" == "help" ]; then
	echo "printing instructions"
	echo "to download dataset RTFM, run ./download_datasets.sh RTFM"
	echo "to download dataset robotics, run ./download_datasets.sh robotics"
	echo "to download dataset enrico, run ./download_datasets.sh enrico"
	echo "other datasets needs to be downloaded manually, please run 'cd datasets/<datasetname>' to check download details"
	exit
else
	echo "Invalid argument,for detailed instructions run './download_datasets.sh help'"
	exit
fi
