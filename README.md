# Enhancing pests classification through deep learning

#### Pedro Lucas de Oliveira Costa(1), Thiago Matheus de Oliveira Costa(1), Leandro Furtado(1), João Fernando Mari(1)

##### (1) Instituto de Ciências Exatas e Tecnologicas - Universidade Federal de Vicosa - UFV, Rio Paranaıba, MG, Brazil
##### (2) Instituto de Ciências Agrarias - Universidade Federal de Viçosa - UFV, Rio Paranaıba, MG, Brazil

#### *{pedro.l.costa@ufv.br, pedromoises, larissa.f.rodrigues, pedro.god, joaof.mari}@ufv.br*
---

* Code for the paper published in the XVIII Workshop of Computer Vision 2023 - WVC'2023.
    * https://fei.edu.br/sites/wvc2023/
---

## The development environment

* Install Anaconda from the Website https://www.anaconda.com/.

### Creating the conda environment
```
    $ conda update -n base -c defaults conda

    
    $ conda create -n env-classification-py39 python=3.9
    $ conda activate env-classification-py39

    $ conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

    $ conda install notebook
    $ conda install matplotlib
    $ conda install scikit-learn
    $ conda install scikit-image
    $ conda install pandas

    $ conda install notebook matplotlib scikit-learn scikit-image pandas 

    $ conda install -c conda-forge timm # conda install timm
    $ conda install -c conda-forge grad-cam # conda install grad-cam # pip install grad-cam

    $ conda install -c conda-forge ipywidgets

    $ pip install wandb *

```

### Save environment

```
    $ conda env export > env-classification-py39.yml
```

### Load enviroment from yml file

```
    $ conda env create -f env-classification2-py39.yml 
```

## Converting ipynb to py
---

```
    $ jupyter nbconvert train_test.ipynb --to python
```



## Executing the experiments

### Hyperparameter experiments

```
    $ nohup python grid-search.py --arch alexnet --optimizer Adam &
    $ nohup python grid-search.py --arch efficientnet_b4 --optimizer Adam &
    $ nohup python grid-search.py --arch mobilenet_v3 --optimizer Adam &
    $ nohup python grid-search.py --arch resnet50 --optimizer Adam &
```

* Pick up the best hyperparameter values

### Main experiments

* Run each experiment throug train_test.py, or...
* Set the experiments thall will run in the train_test_batch.py

```
    $ nohup python train_test_batch.py
```

## Utilities:
---

* Memória da GPU não libera, mesmo nvidia-smi não mostrando nenhum processo:
    * https://stackoverflow.com/a/46597252

```
fuser -v /dev/nvidia*
                     USER        PID ACCESS COMMAND
/dev/nvidia0:        joao      44525 F...m python
/dev/nvidiactl:      joao      44525 F...m python
/dev/nvidia-uvm:     joao      44525 F...m python

kill -9 44525
```

* List Python processes
```
    $ ps -ef | grep python
```
