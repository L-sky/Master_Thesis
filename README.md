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

Install other packages
```
conda install -y -c dglteam dgl-cuda10.1		# install DGL
conda install -y -c conda-forge pymatgen		# intall pymatgen
```


## Data

Materials Project Database. Register at place\_for\_url. Then get api from another\_url. Attach it to environmental varibles

```
sudo -H gedit /etc/environment
PMG\_MAPI\_KEY="your\_api\_key" # append this to the file, save and restart machine to make changes have effect
```

or pass as a second argument to the function. 

