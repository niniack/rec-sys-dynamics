# Recommender System Dynamics

The aim of this repository is to host the code for research in exploring user dynamics within recommender systems conducted by the Traffic Flow Dynamics and Computations Lab @ New York University Abu Dhabi (NYUAD).

## Introduction

The project helps explore a few questions about the behavior of users as they interact with recommender systems.

* What is the connection with causal inference and modeling user behaviour with recommendation systems?
* How does algorithmic confounding in recommendation systems increases homogeneity and decreases utility?

In order to tackle these questions and explore user dynamics, this project conducts time-stepped simulations where each step updates: 
* A user-interaction matrix 
* A recommender system with 3 collaborative filtering algorithms to select from
* A post-processing pipeline to analyze user clusters

## Usage

Check out the `notebooks` directory to view some prototyping of clustering algorithms and recommender system models. Please note that the work in the notebooks directory may be incomplete.

To replicate the results in our paper:

1. Clone this repository
```
git clone https://github.com/niniack/rec-sys-dynamics.git
```

1. Within the repository, use a tool like [virtualenv](https://virtualenv.pypa.io/en/latest/) to set up a virtual environment with python version 3.8
```
virtualenv [env-name]
```

1. Activate the virtual environment
```
source [env-name]/bin/activate
```

1. Install all dependencies found in `requirements.txt`
```
pip install -r requirements.txt
```

1. Change directory to the `examples` directory and view available scripts to execute

```
cd examples && ls
```

1. Execute the desired script

```
python [script-file-name].py
```

1. To deactivate the virtual environment
```
deactivate
```

### Dependencies

All python packages needed to run the code can be found within `requirements.txt`

## Authors
Research conducted for the Traffic Flow Dynamics and Computations Lab @ New York University Abu Dhabi (NYUAD)
* Abdur Rehman (@ar4366)
* Prajna Soni (@soniprajna)
* Nishant Aswani (@niniack)
* Chuhan Yang (@ChuhanYang)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
