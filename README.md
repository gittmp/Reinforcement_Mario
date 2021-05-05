## run.py:
- This is the main file for the project
- Through this, the whole system can be run: **python3 run.py --N *n* [-t | -p --path *path*]**
- Where ***n*** is the number of episodes we wish to run
- The aforementioned command should either specify training (**-t**) or that the agent is pretrained (**-p --path *path***), where ***path*** is the path towards the directory containing the pretrained network parameters.
- The original or adapted neural network can be configured through **--nwk [0 | 1]**
- The memory version can be configured through **--mem [0 | 1 | 2]**
- The environment representation can be configured through **--env [0 | 1 | 2]**
- Any world and stage of Super Mario Bros can be selected: **--w [1 | ... | 8] --s [1 | ... | 4]**
- If you wish to see the behaviour of Mario in his environment, his interaction can be rendered (**-plot**)

## agent.py:
- Implements the DDQL agent

## environment.py:
- Implements the environment manipulation operations
- Frame-skipping
- State downsampling
- Action reductions
- Intrinsic reward system

## network.py
- Implements two convolutional neural networks
- The original architecture (from 'Playing Atari with Deep Reinforcement Learning')
- An adapted architecture

## memory.py
- Contains all configurations of the agent's memory system
- Basic experience replay method
- Prioritised experience replay, utilising a Sum Tree

## utilities.py
- Contains a spectrum of functions for certain operations within the system

## read_results.py
- Function for converting the metric results output by training into a table format for input into a pgfplots graph
