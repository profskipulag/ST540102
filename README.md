# ST540102 watch_and_fetch.py
A package for automatically updating and fetching environmental observations for DT-GEO WP5 DTC4. The package consists of `pywaf`, a package that contains some helper classes, including

 * `RawApi`, a class that provides all the functionality of the Umhverfisstonum web API, including
   * getCurrent - get current data
   * getDate - get data for a given date
   * getStations - get a list of all stations
   * getStation - get info for a given station
 * `API`, a class that wraps the basic functionality of RawApi, allowing filtering by geographic location, date, species, and manages local storage of data to prevent repeated requests.


The useage of `API` is illustarted in the Jupyter notebook `notebook.ipynb`, and the script `watch_and_fetch.py` is the application of the library for the purposes of DTC4. 

## To do
 * fetching manual plume height and gas flux observations that have to be entered manually somewhere.
 
## Package structure


    ST540202/
    ├── environment.yaml      - configures conda environment with required packages
    ├── LICENSE               - GPL 3
    ├── notebook.ipynb        - example use of the pywaf package
    ├── pywaf                 - python Watch and Fetch (pywaf)
    │   ├── __init__.py       - initialises the package
    │   └── source.py         - source code for various classes
    ├── README.md             - this file
    ├── .gitignore            - files to be ignored by git
    └── watch_and_fetch.py    - script that calls the package for DTC4


## To download the repository
Clone the repository to your machine

    git clone https://github.com/profskipulag/ST540102.git

You will be asked for your username and password. For the password github now requires a token:
- on github, click yur user icon in the top right corner
- settings -> developer settings -> personal access tokens -> Tokens (classic) -> Generate new token -> Generate new token (classic) 
- enter you authentifcation code
- under note give it a name, click "repo" to select al check boxes, then click generate token
- copy result enter it as password

## To run the jupyter notebook
Create a new conda environment from the environment.yaml file:

    conda env create -f environment.yaml

Activate the environment

    conda activate st540102
    
Launch the notebook server

    jupyter notebook
    
Navigate to the st540102 directory and click the file `notebook.ipynb` to launch it.
