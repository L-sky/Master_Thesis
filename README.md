## Supplementary materials for the Master Thesis "Predicting Properties of Crystals" 

Read thesis at http://er.ucu.edu.ua/handle/1/2242

For the version of repo at the time of writing click [here!](https://github.com/L-sky/Master_Thesis/tree/f7db8ef237325e55e3705691c0555ee8358cac87) 

## Installation

Create conda environment
```
conda create --name e3_layer -y python=3.6	# create conda environment
conda activate e3_layer				# actvate environment
```

Install pytorch 
```
conda install -y pytorch::pytorch==1.4 pytorch::torchvision cudatoolkit=10.1
conda config --env --add pinned_packages pytorch==1.4				# prevent conda from random downgrades on update
```

Install pymatgen
```
conda install -y -c conda-forge pymatgen		# intall pymatgen
```


## Data

Materials Project Database. Register at https://materialsproject.org. Then get api from https://materialsproject.org/open. Attach it to environmental varibles

```
sudo -H gedit /etc/environment
PMG_MAPI_KEY="your_api_key" # append this to the file, save and restart machine to make changes have effect
```

Run scripts consequently in MP folder. 
