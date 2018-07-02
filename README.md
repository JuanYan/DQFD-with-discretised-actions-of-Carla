# This is a deep Q learning from demonstrations programme based on Pytorch

## Requirements

- Carla
- python (Anaconda)
- tqdm (progress bar)

1. Install the Carla library into python path

	Cd to Carla/PythonClient folder, then type: 
	
	```
	pip install -e .
	```
	
	to install the Carla Python Client package

2. Install the progress bar package

	```
	pip install tqdm 
	```


## Carla configuration

- autonomous driving dememonstrations were collected using the auto-piloting function of the simulator.  

- Steps as follows

1. Call Carla by 

	```
	./CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=15 -windowed -ResX=800 -ResY=600 
	```

2. Run the DQfD_env.py file with python. 



To be continued...


