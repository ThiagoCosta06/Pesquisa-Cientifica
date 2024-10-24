# %% [markdown]
#
# ---

# %% [markdown]
# ## Importing the libraries
# ---

# %%
import os
import random
import time
import platform
import sys
import argparse
import shutil
import datetime
import glob
import pickle   

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import PIL
import sklearn
from sklearn import metrics, preprocessing, model_selection
from sklearn.svm import SVC
from PIL import Image

import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.optim import lr_scheduler 
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms, models, datasets, utils

import timm

# Explainable AI
#import pytorch_grad_cam
### from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
#from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
#from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
#from pytorch_grad_cam.utils.image import show_cam_on_image

# Local imports
from early_stopping import EarlyStopping
from models import create_model

# %% [markdown]
# ## Verificando de está rodando no Colab
# ---

# %%
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

# DEBUG
print(f'Running in Colab: {IN_COLAB}')

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')

# %% [markdown]
# ## Configuração da GPU
# ---

# %%
print('Configurando GPU...')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {DEVICE}')

# %% [markdown]
# ## Argument parsing
# ---

# %%
parser = argparse.ArgumentParser()

# Dataset name. ['BreakHis', 'USK-Coffee', 'deep-coffee', 'smallopticalsorter']

parser.add_argument('--ds', help='Dataset name.', type=str, default='AgriculturalPestsDataset')
# Model architecture [alexnet, vgg, resnet, densenet, squeezenet, (inception), ...]
parser.add_argument('--arch', help='CNN architecture', type=str, default='vgg16', )
parser.add_argument('--optim', help="Hyperparameter optmization: ['none', 'grid', 'random'].", type=str, default='none', )
parser.add_argument('--sm', help='Save the model?', default=True, action='store_true')

parser.add_argument('--seed', help='Seed for random number generator.', type=int, default=42)
parser.add_argument('--num_workers', help='Number of available cores.', type=int, default=2)
parser.add_argument('--debug', help="Is running in debug mode?", required=False, default=False, action='store_true')

# Hyperparameters 
# ---------------
parser.add_argument('--bs', help='Barch size.', type=int, default=64)
parser.add_argument('--lr', help='Learning rate.', type=float, default=0.0001)
parser.add_argument('--mm', help='Momentum.', type=float, default=0.9)
parser.add_argument('--ss', help='Step size.', type=int, default=5)
### parser.add_argument('--wd', help='Weight decay.', type=float, default=0.1)
parser.add_argument('--ep', help='Number of epochs', type=int, default=2) # 200

parser.add_argument('--optimizer', help="Optimizer. ['SGD', 'Adam'].", type=str, default='Adam')
parser.add_argument('--scheduler', help="Scheduler. ['steplr', 'cossine', 'plateau'].", type=str, default='plateau')

# Fine-tunning
parser.add_argument('--ft', help='Treinamento com fine-tuning.', default=True, action='store_true')
# Data augmentation stretegy. Ignorado quando otimização de hiperparametros
parser.add_argument('--da', help='Data augmentation stretegy. 0 = no data augmentation.',  type=int, default=0)
# Usa BCELoss em problemas com duas classes. Se False, usa CrossEntropyLoss para qualquer número de classes
parser.add_argument('--bce', help='Usa Binary Cross Entropy em problemas com duas classes.', default=True, action='store_true')
# Explainable AI
parser.add_argument('--xai', help='Perform eXplainable AI analysis.', default=False, action='store_true')

# Early stopping
parser.add_argument('--es', help='Use early stopping.', default=True, action='store_true')
parser.add_argument('--patience', help='Patience for early stopping.', type=int, default=21) # Use 21, if plateau
parser.add_argument('--delta', help='Delta for early stopping', type=float, default=0.0001)

parser.add_argument('--wandb', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Use wandb.')

parser.add_argument('--ec', help='Experiment counter. Used for hp optimization.', type=int, default=0)

# ***** IMPORTANTE!!! *****
# Comentar esta linha após gerar o arquivo .py!
# *************************
### sys.argv = ['-f']

# Processa os argumentos informados na linha de comando
args = parser.parse_args()

# ***** IMPORTANTE!!! *****
# Set DEBUG mode:
# *************************
args.debug = False

if args.debug:
    args.ep = 4

# %%
if str(DEVICE) != 'cuda':
    # Caso não tenha uma GPU compatível disponível, executar apenas para prototipação.
    ### args.ep = 2

    print('CUDA not availavble. Finishing the program...')
    print('\nDone!\n\n')
    sys.exit()

if args.optim != 'none':
    args.sm = False
    ### args.da = 0
    # if hp optimization, always ignore XAI.
    args.xai = False

# %%
args_str = ''
for arg in vars(args):
    args_str += f'\n{arg}: {getattr(args, arg)}'
    print(f'{arg}: {getattr(args, arg)}')

# %%
def get_versions():

    str = ''
    str += f'\nNumPy: {np.__version__}'
    str += f'\nMatplotlib: {matplotlib.__version__}'
    str += f'\nPandas: {pd.__version__}'
    str += f'\nPIL: {PIL.__version__}'
    str += f'\nScikit-learn: {sklearn.__version__}'
    str += f'\nPyTorch: {torch.__version__}'
    str += f'\nTorchvision: {torchvision.__version__}'

    return str

# %% [markdown]
# ## Customized Dataset
# ---

# %%
class AgriculturalPestsDataset(Dataset):

    def __init__(self, path_list, label_list, transforms=None):
        self.path_list = path_list
        self.label_list = label_list
        self.transforms = transforms

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        ### image = io.imread(self.path_list[idx])
        image = Image.open(self.path_list[idx]) 

        # O dataset "Agricultural Pests Image Dataset" possui algumas imagens em níveis de cinza e algumas com canal de transparência.
        # É necessário trata-las, convertendo para imagens RGB (3 canais).
        if np.array(image).ndim != 3:
            # Trata imagens em níveis de cinza (com apenas 1 canal)
            ### print(path)
            ### print(np.array(image).ndim, np.array(image).shape)
            image = image.convert(mode='RGB')
            ### print(np.array(image).ndim, np.array(image).shape)
        elif np.array(image).shape[2] > 3:
            # Trata imagens com canal de transparência (com 4 canais)
            ### print(path)
            ### print(np.array(image).ndim, np.array(image).shape)
            image = image.convert(mode='RGB')
            ### print(np.array(image).ndim, np.array(image).shape)

        label = self.label_list[idx]

        if self.transforms:
            image = self.transforms(image)

        return (image, label, path)   
    
# %% [markdown]
# ## Configurating datasets
# ---

# %%
DS_PATH_MAIN = '/home/pedrocosta/Dev/Datasets/'

if args.ds == 'AgriculturalPestsDataset':
    DS_PATH = os.path.join(DS_PATH_MAIN, args.ds, 'images')

# DEBUG
print(f'Dataset: {args.ds}')
print(f'Dataset Path: {DS_PATH}')

# %% [markdown]
# ## Gravação dos experimentos
# ---

# %%
# Pasta principal para armazenar os experimentos
EXP_PATH_MAIN = f'exp_{args.ds}'
if args.optim != 'none':
    EXP_PATH_MAIN = f'exp_hp_{args.ds}'

# Cria uma pasta para armazenar os experimentos, caso ainda não exista
if not os.path.isdir(EXP_PATH_MAIN):
    os.mkdir(EXP_PATH_MAIN)

mm_str = f'-mm_{args.mm}' if args.optimizer == 'SGD' else ''

ss_str = f'-ss_{args.ss}' if args.scheduler == 'steplr' else ''

# String contendo os valores dos hiperparametros deste experimento.
hp_str = f'-bs_{args.bs}-lr_{args.lr}-op_{args.optimizer}{mm_str}-sh_{args.scheduler}{ss_str}-epochs_{args.ep}'

# Ajusta para o nome da pasta
hp_optim = '' if args.optim == 'none' else f'-{args.optim}'

str_aux1 = '' if args.ds != 'BreakHis' else f'-mag_{args.magnification}-fold_{str(args.fold)}'

# Pasta que ira armazenar os resultados deste treinamento
EXP_PATH = os.path.join(EXP_PATH_MAIN, f'({args.ds})-{args.arch}{hp_optim}-da_{args.da}{hp_str}{str_aux1}')
print(f'Exp path: {EXP_PATH}')

# Check if EXP_PATH exists. If not, create it.
### if not glob.glob(EXP_PATH):
if not os.path.exists(EXP_PATH):
    os.mkdir(EXP_PATH)

else:
    # If the folder already exists, it is possible the experiment should (or shouldn't) be complete.
    # Nós verificamos, observando se o arquivo 'done.txt' está na pasta.
    # O arquivo 'done.txt' só é criado quando o experimento terminou por completo.
    ### if os.path.exists(os.path.join(EXP_PATH, 'done.txt')):
    if os.path.exists(os.path.join(EXP_PATH, 'done.txt')):
        # The folder exists and the experiment is done.
        print('Experiment already done. Finishing the program...')
        print('\nDone!\n\n')
        sys.exit()

# %%
with open(os.path.join(EXP_PATH, 'general_report.txt'), 'w') as model_file:
    model_file.write('\nArguments:')
    ### model_file.write(str(args.__str__()))
    model_file.write(args_str)
    model_file.write('\n\nPackage versions:')
    model_file.write(str(get_versions()))

# %% [markdown]
# ## Reprodutibility configurations
# ---

# %%
random.seed(args.seed)
np.random.seed(args.seed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.environ["PYTHONHASHSEED"] = str(args.seed)

# %% [markdown]
# ## Preparando o conjunto de dados
# ---

# %% [markdown]
# ### Definindo transformações para os dados (aumento de dados)

# %%
# Média e desvio padrão do ImageNet.
DS_MEAN = [0.485, 0.456, 0.406]
DS_STD =  [0.229, 0.224, 0.225]

# Data transforms 
# ---------------
if args.da == 0:  # Resize(224)
    # Treinamento
    data_transforms_train = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD)
    ])

    # Validacao
    data_transforms_val = transforms.Compose([
            transforms.Resize(size=(224, 224)), 
            transforms.ToTensor(), 
            transforms.Normalize(DS_MEAN, DS_STD)
    ])

    # Test
    data_transform_test = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize(DS_MEAN, DS_STD)
    ])

elif args.da == 1: # Resise + CentreCrop (Train = val = test)
    # Treinamento
    data_transforms_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD)
    ])

    # Define transformations for validation and test sets
    data_transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

    data_transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

elif args.da == 2: # RandomResizedCrop (224)
    # Treinamento
    data_transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD)
    ])

    # Define transformations for validation and test sets
    data_transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

    data_transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])
    
elif args.da == 3: # Data augmentation base
    # Training
    data_transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

    # Validation
    data_transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

    # Test
    data_transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

elif args.da == 4: # Data augmentation base, No HUE.
    # Training
    data_transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        ### transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

    # Validation
    data_transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

    # Test
    data_transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

elif args.da == 5: # Data augmentation base. No HUE. No RandomErasing.
    # Training
    data_transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        ### transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        ### transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

    # Validation
    data_transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

    # Test
    data_transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

# %% [markdown]
# # Datasets and dataloaders
# ---


# %%
# Treino
# image_dataset_train = datasets.ImageFolder(os.path.join(DS_PATH, 'train'), data_transforms_train)

#-------------------------#
#NOME E NUMEROS DA CLASSES#
#-------------------------#
class_names = os.listdir(os.path.join(DS_PATH, 'train'))
num_classes = len(class_names)

X_train_ = []
y_train_ = []

for class_ in class_names:
    #lista ordenada dos arquivos (imagens) em cada pasta
    path_list_ = os.listdir(os.path.join(DS_PATH, 'train', class_))
    path_list_.sort()

    for path_image in path_list_:
        file_path = os.path.join(DS_PATH, 'train', class_, path_image)
        X_train_.append(file_path)
        y_train_.append(class_)

X_test = []
y_test = []

path_list = []

for class_ in class_names:
    #lista ordenada dos arquivos (imagens) em cada pasta
    path_list_ = os.listdir(os.path.join(DS_PATH, 'test', class_))
    path_list_.sort()

    #itera ao longo dos arquivos na pasta atual (classe)
    for path_image in path_list_:
        file_path = os.path.join(DS_PATH, 'test', class_, path_image)
        X_test.append(file_path)
        y_test.append(class_)

VAL_SIZE  = 0.2
TRAIN_SIZE = 1. -VAL_SIZE

# train_size = int(0.8 * len(image_dataset_train))
# val_size = len(image_dataset_train) - train_size

#Divisão do conjunto de treino para validação

le = preprocessing.LabelEncoder()
le.fit(class_names)
y_train_idx = le.transform(y_train_)
y_test_idx = le.transform(y_test)



X_train, X_val, y_train_idx, y_val_idx = model_selection.train_test_split(X_train_, y_train_idx, test_size=VAL_SIZE, stratify=y_train_idx, random_state=42)

train_dataset = AgriculturalPestsDataset(X_train, y_train_idx, transforms=data_transforms_train)
val_dataset = AgriculturalPestsDataset(X_val, y_val_idx, transforms=data_transforms_val)
test_dataset = AgriculturalPestsDataset(X_test, y_test_idx, transforms=data_transform_test)

# train_dataset, val_dataset = torch.utils.data.random_split(image_dataset_train, [train_size, val_size])    # MUDAR AQUI PARA A DIVISAO DA VERSAO ANTERIOR 

# Validação
# image_dataset_val = datasets.ImageFolder(os.path.join(DS_PATH, 'val'), data_transforms_val)

# Tamanho dos conjuntos de treino e de validação (número de imagens).
train_size = len(train_dataset)
val_size = len(val_dataset)
test_size = len(test_dataset)



# %%
# Construindo os Dataloaders
dataloader_train = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.bs, 
                                               num_workers=args.num_workers,
                                               shuffle=True,
                                              )
dataloader_val = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.bs, 
                                             num_workers=args.num_workers,
                                             shuffle=True,
                                            )

# %% [markdown]
# ## Visualizando um lote de imagens
# ---

# %%
def save_batch(images_batch):
    """ Save one batch of images in a grid.

    References
    ----------
    * https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    * https://pub.towardsai.net/image-classification-using-deep-learning-pytorch-a-case-study-with-flower-image-data-80a18554df63
    """

    # Unnormalize all channels (ImageNet weights)
    for t, m, s in zip(images_batch, DS_MEAN, DS_STD):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)

    images_batch_np = images_batch.numpy()
    fig_obj = plt.imshow(np.transpose(images_batch_np, (1, 2, 0)))
    
    # Grava a figura em disco
    plt.savefig(os.path.join(EXP_PATH, 'sample_batch.png')) 
    plt.savefig(os.path.join(EXP_PATH, 'sample_batch.pdf')) 
    with open(os.path.join(EXP_PATH, 'sample_batch.pickle'),'wb') as file:
        pickle.dump(fig_obj, file)

items = iter(dataloader_train)
image, label, paths = next(items)

save_batch(utils.make_grid(image))

# %% [markdown]
# ## Inicializando o modelo
# ---
# 
# * Modelos prontos para uso:
#     * alexnet
#     * vgg - Vgg11
#     * resnet - Resnet18
#     * resnet50 - Resnet50
#     * squeezenet
#     * densenet - Densenet121
#     * timm_vit - vit_base_patch16_224
#     * timm_efficientnet - efficientnet_b0
#     * timm_efficientnet_b2 - efficientnet_b2
#     * timm_efficientnet_b3 - efficientnet_b3
#     * timm_mobilenet - convnext_small
#     * vgg - Vgg16

# %%
print('\n>> Inicializando o modelo...')

model, input_size, _ = create_model(args.arch, args.ft, num_classes, args.bce)

# Envia o modelo para a GPU
if DEVICE.type == 'cuda':
    model = model.cuda() # Cuda
    
# Imprime o modelo
print(str(model))

# Grava a modelo da rede em um arquivo .txt
with open(os.path.join(EXP_PATH, 'model.txt'), 'w') as model_file:
    model_file.write(str(model))

# %% [markdown]
# ## Loss function and optimizer
# ---

# %%
# Função de perda
if num_classes > 2 or args.bce == False:
    # Classificação com mais de duas classes.
    criterion = nn.CrossEntropyLoss()
    print('criterion = nn.CrossEntropyLoss()')
    
else:
    # Binary classification:
    # ----------------------
    # Do not use BCELoss. Instead use BCEWithLogitsLoss.
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    # https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/21
    ### criterion = nn.BCELoss()

    criterion = nn.BCEWithLogitsLoss()
    print('criterion = nn.BCEWithLogitsLoss()')

# Otimizador
if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mm)

elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

# %% [markdown]
# ### Scheduler

# %%
if args.scheduler == 'plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, )
    print(scheduler)

elif args.scheduler == 'cossine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=len(dataloader_train), 
                                                           eta_min=0,
                                                           last_epoch=-1)
    print(scheduler)

elif args.scheduler ==  'steplr':

    # Step size of the learning rate
    if args.ss != 0:
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#StepLR
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.ss)
        print(scheduler)

# %%
print(criterion)
print(optimizer)

# %%
if args.wandb:   
    wandb.watch(model)

# %% [markdown]
# ## Training the model
# ---

# %%
print('\n>> Training the model...')

# Tempo total do treinamento (treinamento e validação)
time_total_start = time.time()

# Lista das perdas (loss) e acurácias (accuracy) de trino para cada época.
train_loss_list = []
train_acc_list = []

# Lista das perdas (loss) e acurácias (accuracy) de validação para cada época.
val_loss_list = []
val_acc_list = []

lr_list = []

# Lista de vetores de caracteristicas para o dataset
svm_features_train = []
svm_features_val = []

# Lista de rotulos correspondentes para o dataset
svm_labels_train = []
svm_labels_val = []

if args.es:
    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)


# Extração de Caracteristicas para o classificador SVM
for i, data_batch in enumerate(dataloader_train):
    inputs, labels, *_ = data_batch
    if DEVICE.type == 'cuda':
        inputs = inputs.to(DEVICE) 
        labels = labels.to(DEVICE) 

        features = model(inputs)

        features = features.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

    print('teste1')
    print(features.shape)
    print(labels.shape)

    for index in range(len(labels)):
        feature = features[index]
        label = labels[index]

        svm_features_train.append(feature)
        svm_labels_train.append(label)

    print('test2')
    print(len(svm_features_train))
    print(len(svm_features_train[0]))
    print(len(svm_labels_train))
    print(svm_labels_train[0])
    
print(model)


# Extração de Caracteristicas para o classificador SVM
for i, data_batch in enumerate(dataloader_val):
    inputs, labels, *_ = data_batch
    if DEVICE.type == 'cuda':
        inputs = inputs.to(DEVICE) 
        labels = labels.to(DEVICE) 

        features = model(inputs)

        features = features.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

    for index in range(len(labels)):
        feature = features[index]
        label = labels[index]

        svm_features_val.append(feature)
        svm_labels_val.append(label)

######-------------------------######
##--------TREINAMENTO DO SVM-------##
######-------------------------######

begin = time.time()

# FEATURE_INDEX = 0
# LABEL_INDEX = 1

# train_features = np.array(train_dataset[FEATURE_INDEX])
# train_labels = np.array(train_dataset[FEATURE_INDEX])

# test_features = np.array(test_dataset[FEATURE_INDEX])
# test_labels = np.array(test_dataset[FEATURE_INDEX])

#criação do modelo
svm_model = SVC(gamma="auto")

#treinamento do modelo
svm_model.fit(svm_features_train, svm_labels_train)

#teste 
pred_labels = svm_model.predict(svm_features_val)

# Classification report - Scikit-learn
class_rep_svm= metrics.classification_report(svm_labels_val, pred_labels)

class_rep_path = os.path.join(EXP_PATH, 'classification_report_test_svm.txt')
file_rep = open(class_rep_path, 'w')

file_rep.write('\n\nTEST SET:')
file_rep.write('\n---------------')
file_rep.write('\nClassification report:\n')
file_rep.write(class_rep_svm)

file_rep.close()
elapsed_time = time.time() - begin
print(f"Modelo produzido em {(elapsed_time // 60):.0f}m {(elapsed_time % 60):.0f}s")


# for epoch in range(args.ep):
#     # =========================================================================
#     # TRAINING
#     # =========================================================================
#     # Inicia contagem de tempo deste época
#     time_epoch_start = time.time()

#     # Perdas (loss) nesta época
#     train_loss_epoch = 0.
#     # Número de amostras classificadas corretamente nesta época
#     train_num_hits_epoch = 0  

#     # Habilita o modelo para o modo de treino 
#     model.train() 

#     # Iterate along the batches of the TRAINING SET
#     # ---------------------------------------------
#     for i, (inputs, labels, paths) in enumerate(dataloader_train):

#         if DEVICE.type == 'cuda':
#             inputs = inputs.to(DEVICE) 
#             labels = labels.to(DEVICE) 

#         # Zera os parametros do gradiente
#         optimizer.zero_grad() 

#         # FORWARD
#         # ------>
#         # Habilita o cálculo do gradiente
#         torch.set_grad_enabled(True) 

#         # Gera a saida a partir da entrada
#         features = model(inputs)
#         features = features.cpu().detach().numpy() 

#         if num_classes == 2 and args.bce:
#             # Calculate probabilities
#             # https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/27
#             outputs_prob = torch.sigmoid(features) 
#             preds = (outputs_prob > 0.5).float().squeeze()

#             # Calcula a perda (loss)
#             loss = criterion(features.squeeze(), labels.float())

#         else:
#             # 'outputs' estão em porcentagens. Tomar os maximos como a respostas.
#             # Ex: batch=3 com 2 classes, entao preds = [1, 0, 1]
#             preds = torch.argmax(features, dim=1).float() 

#             # Calcula a perda (loss)
#             loss = criterion(features, labels)

#         # BACKWARD
#         # <-------
#         loss.backward() 

#         # Atualiza o gradiente 
#         optimizer.step()

#         # Atualiza o loss da época
#         train_loss_epoch += float(loss.item()) * inputs.size(0) 

#         # Atualiza o número de amostras classificadas corretamente nessa época.
#         train_num_hits_epoch += torch.sum(preds == labels.data) 

#     train_loss_epoch /= len(train_dataset)
#     train_acc_epoch = float(train_num_hits_epoch.double() / len(train_dataset))

#     # Store loss and accuracy in lists
#     train_loss_list.append(train_loss_epoch)
#     train_acc_list.append(train_acc_epoch)

#     # =========================================================================
#     # VALIDATION
#     # =========================================================================
#     model.eval() 

#     # Pego o numero de perda e o numero de acertos
#     val_loss_epoch = 0. # Atual perda
#     val_num_hits_epoch = 0 # Numero de itens corretos
    
#     # Iterate along the batches of the VALIDATION SET
#     # -----------------------------------------------
#     for i, (inputs, labels, paths) in enumerate(dataloader_val):

#         if DEVICE.type == 'cuda':
#             inputs = inputs.to(DEVICE)
#             labels = labels.to(DEVICE)

#         # Zero o gradiente antes do calculo do loss
#         optimizer.zero_grad() 

#         # Desabilita o gradiente, pois os parametros nao podem mudar durante etapa de validacao
#         torch.set_grad_enabled(False) 

#         # Gero um tensor cujas linhas representam o tamanho do "batch" do input
#         features = model(inputs) 

#         if num_classes == 2 and args.bce:
#             # Calculate probabilities
#             # https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/27
#             outputs_prob = torch.sigmoid(features) 
#             preds = ((outputs_prob > 0.5).float()).squeeze()

#             # Calcula a perda (loss)
#             loss = criterion(features.squeeze(), labels.float())

#         else:
#             # Retorna a maior predicao.
#             preds = torch.argmax(features, dim=1).float()

#             # Calcula a perda (loss)
#             loss = criterion(features, labels) 

#         # Loss acumulado na época
#         val_loss_epoch += float(loss.item()) * inputs.size(0)

#         # Acertos acumulados na época
#         val_num_hits_epoch += torch.sum(preds == labels.data)

#     # Ajusta o learning rate
#     if args.scheduler == 'steplr' and args.ss != 0:
#         scheduler.step() 
    
#     elif args.scheduler == 'cossine' and epoch >= 10:
#         scheduler.step()

#     elif args.scheduler == 'plateau':
#         scheduler.step(val_loss_epoch)

#     lr_epoch = optimizer.param_groups[0]['lr']
#     lr_list.append(lr_epoch)
        
#     # Calculo a perda e acuracia de todo o conjunto de validacao
#     val_loss_epoch /= len(val_dataset)
#     val_acc_epoch = float(val_num_hits_epoch.double() / len(val_dataset))

#     # Inserindo a perda e acuracia para os arrays 
#     val_loss_list.append(val_loss_epoch)
#     val_acc_list.append(val_acc_epoch)

#     if args.es:
#         early_stopping(val_loss_epoch, model, epoch)
        
#         if early_stopping.early_stop:
#             print(f'Early stopping in epoch {early_stopping.best_epoch}!')
#             break

#     # Calculo de tempo total da época
#     time_epoch = time.time() - time_epoch_start
    
#     # PRINTING
#     # --------
#     print(f'Epoch {epoch}/{args.ep - 1} - TRAIN Loss: {train_loss_epoch:.4f} VAL. Loss: {val_loss_epoch:.4f} - TRAIN Acc: {train_acc_epoch:.4f} VAL. Acc: {val_acc_epoch:.4f} ({time_epoch:.4f} seconds)')

#     if args.wandb:   
#         wandb.log({'epoch': epoch, 'train_loss': train_loss_epoch, 'val_loss': val_loss_epoch, 'train_acc': train_acc_epoch,'val_acc': val_acc_epoch,'epoch_time': time_epoch})

# # Calcula do tempo total do treinamento (treinamento e validação)
# time_total_train = time.time() - time_total_start

# # PRINTING
# print('Treinamento finalizado. ({0}m {1}s)'.format(int(time_total_train // 60), int(time_total_train % 60)))


# %%
if args.es:
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

# %%
# Saving the model
if args.sm:
    model_file = os.path.join(EXP_PATH, 'model.pth')
    torch.save(model, model_file)

# %% [markdown]
# ## Analisando o treinamento
# ---

# Contar os parâmetros treináveis
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)



with open(os.path.join(EXP_PATH, f'general_report.txt'), 'w') as model_file:
    model_file.write(f'\n\nTrainable parameters: {trainable_params}')

# %%
# Lista com os indices das épocas. [0, 1, ... num_epochs - 1]
epochs_list = []
for i in range(len(train_loss_list)):
    epochs_list.append(i)

# Plot - Loss 
# -----------
fig_obj = plt.figure()

plt.title('Loss')
plt.plot(epochs_list, train_loss_list, c='magenta', label='Train loss', fillstyle='none')
plt.plot(epochs_list, val_loss_list, c='green', label='Val. loss', fillstyle='none')
if args.es:
    plt.axvline(x=early_stopping.best_epoch, color='r', label='Early stopping')
    ### plt.text(early_stopping.best_epoch + 0.1, (-early_stopping.best_score) + .05, str(f'{-early_stopping.best_score:.4f}'), color = 'blue', fontweight = 'bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')

# Grava a figura em disco
plt.savefig(os.path.join(EXP_PATH, 'chart_loss.png'))
plt.savefig(os.path.join(EXP_PATH, 'chart_loss.pdf')) 
with open(os.path.join(EXP_PATH, 'chart_loss.pickle'),'wb') as file:
    pickle.dump(fig_obj, file)

# Plot - Accuracy
# ---------------
fig_obj = plt.figure()

plt.title('Accuracy')
plt.plot(epochs_list, train_acc_list, c='magenta', label='Train accuracy', fillstyle='none')
plt.plot(epochs_list, val_acc_list, c='green', label='Val. accuracy', fillstyle='none')
if args.es:
    plt.axvline(x=early_stopping.best_epoch, color='r', label='Early stopping')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')

# Grava a figura em disco
plt.savefig(os.path.join(EXP_PATH, 'chart_acc.png')) 
plt.savefig(os.path.join(EXP_PATH, 'chart_acc.pdf')) 
with open(os.path.join(EXP_PATH, 'chart_acc.pickle'),'wb') as file:
    pickle.dump(fig_obj, file)

# Plot LR
# ---------------
fig_obj = plt.figure()

plt.title('LR')
plt.plot(epochs_list, lr_list, c='magenta', label='LR', fillstyle='none')
if args.es:
    plt.axvline(x=early_stopping.best_epoch, color='r', label='Early stopping')
plt.xlabel('Epochs')
plt.ylabel('LR')
plt.legend(loc='best')

# Grava a figura em disco
plt.savefig(os.path.join(EXP_PATH, 'chart_lr.png')) 
plt.savefig(os.path.join(EXP_PATH, 'chart_lr.pdf')) 
with open(os.path.join(EXP_PATH, 'chart_lr.pickle'),'wb') as file:
    pickle.dump(fig_obj, file)

# %% [markdown]
# ### Saving trainning report

# %%
# Arquivo CSV que irá armazenar todos os losses e acuracias.
report_filename = os.path.join(EXP_PATH, 'training_report' + '.csv')

# Cria o arquivo CSV
report_file = open(report_filename, 'w')

header = 'Epoch\tTrain Loss\tVal. Loss\tTrain Acc.\tVal. Acc.\n'
report_file.write(header)

# Putting values from each epoch inside of archive
for i in range(0, len(train_loss_list)):
    text = str(i) + '\t' + str(train_loss_list[i]) + '\t' + str(val_loss_list[i]) + '\t' + str(train_acc_list[i]) + '\t' + str(val_acc_list[i]) + '\t' + str(lr_list[i]) 
    if args.es and i == early_stopping.best_epoch:
        text += '\t *\n'
    else:
        text += '\n'

    report_file.write(text)

if args.es:
    report_file.write(f'Early stopping: \t {early_stopping.best_epoch}')

# Closing
report_file.close()

# %% [markdown]
# ## Avaliando o modelo
# ---

# %%
# Datasets
# ---------
# img_dataset_test =  AgriculturalPestsDataset(os.path.join(DS_PATH, 'test'), data_transform_test)
# img_dataset_train = AgriculturalPestsDataset(os.path.join(DS_PATH, 'train'), data_transforms_train)

img_dataset_train = AgriculturalPestsDataset(X_train, y_train_idx, transforms=data_transforms_train)
img_dataset_val = AgriculturalPestsDataset(X_val, y_val_idx, transforms=data_transforms_val)
img_dataset_test = AgriculturalPestsDataset(X_test, y_test_idx, transforms=data_transform_test)


VAL_SIZE  = 0.2 
TRAIN_SIZE = 1. -VAL_SIZE

# index = torch.randperm(len(img_dataset_test)).tolist()        # MUDAR AQUI PARA A DIVISAO DA CNN ORIGINAL 

# valid_idx, test_idx = index[:split_size], index[split_size:]
# valid_dataset =  torch.utils.data.Subset(img_dataset_test, valid_idx)
# test_dataset = torch.utils.data.Subset(img_dataset_test, test_idx)

# img_dataset_val =  ImageFolderWithPaths(os.path.join(DS_PATH, 'val'), data_transform_test)

# DataLoaders
# -----------
dataloader_test = torch.utils.data.DataLoader(img_dataset_test, 
                                              batch_size=args.bs,  
                                              num_workers=args.num_workers,
                                              shuffle=False,
                                             )

dataloader_val  = torch.utils.data.DataLoader(img_dataset_val,
                                              batch_size=args.bs,
                                              num_workers=args.num_workers,
                                              shuffle=False, 
                                             )


# %% [markdown]
# ### Conjunto de testes

# %%
# Habilita o modelo para avaliação
model.eval()

# Listas contendo as classes reais (true_test), as classes preditas pelo modelo ('pred_test') e
# os caminhos para cada imagem. Apenas para o conjunto de testes.
true_test_list = []
pred_test_list = []
path_test_list = []

prob_test_list = []

# Inicia a contagem do tempo apenas para teste
time_test_start = time.time()

# Itera sobre o dataloader_test
# -----------------------------
for i, (img_list, true_list, path_list) in enumerate(dataloader_test):


    if DEVICE.type == 'cuda':
        # Cuda extension
        img_list = img_list.to(DEVICE)
        true_list = true_list.to(DEVICE)
    # Para que o gradiente nao se atualize!
    torch.set_grad_enabled(False)

    features = model(img_list)

    if num_classes == 2 and args.bce:
        # Calculate probabilities
        # https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/27
        outputs_prob = torch.sigmoid(features) 
        preds = (outputs_prob > 0.5).float().squeeze()

        prob_test_batch = np.asarray(outputs_prob.cpu())

        # Temos a probabilidade só da classe 1. 
        # Probability of class 0 is (1 - prob(c_1))
        prob_test_batch = np.c_[1. - prob_test_batch, prob_test_batch]

    else:
        ### _, preds = torch.max(output, dim=1)
        preds = torch.argmax(features, dim=1)

        # https://discuss.pytorch.org/t/obtain-probabilities-from-cross-entropy-loss/157259
        outputs_prob = nn.functional.softmax(features, dim=1)

        prob_test_batch = np.asarray(outputs_prob.cpu())

    if DEVICE.type == 'cuda':
        # Lista de labels com a resposta (batch with 128 samples)
        true_test_batch = np.asarray(true_list.cpu())
        # Lista de labels com a predicao (batch with 128 samples)
        pred_test_batch = np.asarray(preds.cpu())

    else:
        true_test_batch = np.asarray(true_list)
        pred_test_batch = np.asarray(preds)

    # Lista com os caminhos das imagens (batch with 128 samples)
    path_test_batch = list(path_list)

    # Consolidate das listas de predicao e resposta
    for i in range(0, len(pred_test_batch)):
        true_test_list.append(true_test_batch[i])
        pred_test_list.append(pred_test_batch[i])
        path_test_list.append(path_test_batch[i])

        prob_test_list.append(prob_test_batch[i])

# Calculo o tempo final 
finish_test = time.time()

# Calculo do tempo total de teste
time_total_test = finish_test - time_test_start

# %% [markdown]
# ### Conjunto de validação

# %%
# resposta_val
true_val_list = []
pred_val_list = []
path_val_list = []

prob_val_list = []

# Itera sob o dataloader_val
# --------------------------
for i, (img_list, labelList, path_list) in enumerate(dataloader_val):

    if DEVICE.type == 'cuda':
        # Cuda extension
        img_list = img_list.to(DEVICE)
        labelList = labelList.to(DEVICE)

    # Nao atualizar o gradiente
    torch.set_grad_enabled(False) 

    # >>>>> FORWARD
    features = model(img_list)

    if num_classes == 2 and args.bce:
        # Calculate probabilities
        # https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/27
        outputs_prob = torch.sigmoid(features) 
        preds = (outputs_prob > 0.5).float().squeeze()

        prob_val_batch = np.asarray(outputs_prob.cpu())

        prob_val_batch = np.c_[1. - prob_val_batch, prob_val_batch]

    else:
        ### _, preds = torch.max(output, 1)
        preds = torch.argmax(features, dim=1)

        # https://discuss.pytorch.org/t/obtain-probabilities-from-cross-entropy-loss/157259
        outputs_prob = nn.functional.softmax(features, dim=1)

        prob_val_batch = np.asarray(outputs_prob.cpu())

    # Obtém as classes reais (True) e classes preditas (pred) deste lote.
    if DEVICE.type == 'cuda':
        true_val_batch = np.asarray(labelList.cpu())
        pred_val_batch = np.asarray(preds.cpu())
    else:
        true_val_batch = np.asarray(labelList)
        pred_val_batch = np.asarray(preds)
        
    # Obtém os caminhos das imagens deste lote
    path_val_batch = list(path_list)

    # Itera sob cada item predito. (Esse FOR tem tamanho do batch_size)
    for i in range(0, len(pred_val_batch)):
        true_val_list.append(true_val_batch[i])
        pred_val_list.append(pred_val_batch[i])
        path_val_list.append(path_val_batch[i])

        prob_val_list.append(prob_val_batch[i])

# %% [markdown]
# ### Matriz de confusão e relatórios de classificação (Scikit-learn)

# %%
# TEST SET
# -------------------------------------------------------------------------

# Confusion matrix
conf_mat_test = metrics.confusion_matrix(true_test_list, pred_test_list)
# Classification report - Scikit-learn
class_rep_test = metrics.classification_report(true_test_list, pred_test_list, 
                                               target_names=class_names, digits=4)
# Accuracy
acc_test = metrics.accuracy_score(true_test_list, pred_test_list)

class_rep_path = os.path.join(EXP_PATH, 'classification_report_test.txt')
file_rep = open(class_rep_path, 'w')

file_rep.write('\n\nTEST SET:')
file_rep.write('\n---------------')
file_rep.write('\nConfusion matrix:\n')
file_rep.write(str(conf_mat_test))
file_rep.write('\n')
file_rep.write('\nClassification report:\n')
file_rep.write(class_rep_test)
file_rep.write('\n')
file_rep.write('\nAccuracy:\t' + str(acc_test))

file_rep.close()

# Ploting the confusion matrix
fig_obj = plt.figure()
metrics.ConfusionMatrixDisplay(conf_mat_test).plot()
# Save figure in disk
plt.savefig(os.path.join(EXP_PATH, 'conf_mat_test.png')) 
plt.savefig(os.path.join(EXP_PATH, 'conf_mat_test.pdf')) 
with open(os.path.join(EXP_PATH, 'conf_mat_test.pickle'),'wb') as file:
    pickle.dump(fig_obj, file)

print('TEST. Acc.: {:.4f}'.format(acc_test))

# %%
# VALIDATION SET
# -------------------------------------------------------------------------
# Confusion matrix
conf_mat_val = metrics.confusion_matrix(true_val_list, pred_val_list)
# Classification report - Scikit-learn
class_rep_val = metrics.classification_report(true_val_list, pred_val_list, 
                                              target_names=class_names, digits=4)
# Accuracy
acc_val = metrics.accuracy_score(true_val_list, pred_val_list)

class_rep_path = os.path.join(EXP_PATH, 'classification_report_val.txt')
file_rep = open(class_rep_path, 'w')

file_rep.write('VALIDATION SET:')
file_rep.write('\n---------------')
file_rep.write('\nConfusion matrix:\n')
file_rep.write(str(conf_mat_val))
file_rep.write('\n')
file_rep.write('\nClassification report:\n')
file_rep.write(class_rep_val)
file_rep.write('\n')
file_rep.write('\nAccuracy:\t' + str(acc_val))

file_rep.close()

# Ploting the confusion matrix
fig_obj = plt.figure()
metrics.ConfusionMatrixDisplay(conf_mat_val).plot()
# Save figure in disk
plt.savefig(os.path.join(EXP_PATH, 'conf_mat_val.png')) 
plt.savefig(os.path.join(EXP_PATH, 'conf_mat_val.pdf')) 
with open(os.path.join(EXP_PATH, 'conf_mat_val.pickle'),'wb') as file:
    pickle.dump(fig_obj, file)

print('VAL. Acc.: {:.4f}'.format(acc_val))

# %%
if args.wandb:   
        wandb.log({'eval_val_acc': acc_val, 'eval_test_acc': acc_test, 'best_epoch': early_stopping.best_epoch})

# %% [markdown]
# ### Classification report

# %%
# Usa o método __get_item__ da classe ... extendida da classe Dataset

# Conjunto de validação
file_details_path = os.path.join(EXP_PATH, 'classification_details_val.csv')
file_details = open(file_details_path, 'w')

file_details.write('VALIDATION SET')
file_details.write('\n#\tFile path\tTarget\tPrediction')

for class_name in class_names:
    file_details.write('\t' + str(class_name))

for i, (target, pred, probs) in enumerate(zip(true_val_list, pred_val_list, prob_val_list)):
    image_name = str(path_val_list[i])
    file_details.write('\n' + str(i) + '\t' + image_name + '\t' + str(target) + '\t' + str(pred))

    for prob in probs:
        file_details.write('\t' + str(prob))

file_details.close()

# %%
# Conjunto de testes
file_details_path = os.path.join(EXP_PATH, 'classification_details_test.csv')
file_details = open(file_details_path, 'w')

file_details.write('TEST SET')
file_details.write('\n#\tFile path\tTarget\tPrediction')

for class_name in class_names:
    file_details.write('\t' + str(class_name))

for i, (target, pred, probs) in enumerate(zip(true_test_list, pred_test_list, prob_test_list)):
    image_name = str(path_test_list[i])
    file_details.write('\n' + str(i) + '\t' + image_name + '\t' + str(target) + '\t' + str(pred))

    for prob in probs:
        file_details.write('\t' + str(prob))

file_details.close()

# %% [markdown]
# ### Hyperparameter optimization report

# %%
if args.optim != 'none':
    print('\n>> Relatório da otimização de hiperparametros...')
    # O nome de um arquivo CSV. Irá armazenar todos os losses e acuracias.
    hp_filename = os.path.join(EXP_PATH_MAIN, '(' + args.ds + ')-' + args.arch + '-' + args.optim + '.csv')
else:
    print('\n>> Relatório do conjunto de experimentos...')
    # O nome de um arquivo CSV. Irá armazenar todos os losses e acuracias.
    hp_filename = os.path.join(EXP_PATH_MAIN, '(' + args.ds + ')-' + args.optim + '.csv')


if args.ec == 0:
    # Cria o arquivo CSV
    hp_file = open(hp_filename, 'w')

    # Criar cabeçalho
    header = '#\tDS\tARCH\tHP\tFT\tBS\tLR\tMM\tSS\tEP\tES\tACC_VAL\tACC_TEST\tACC_TRAIN(*)\tACC_VAL(*)\tTIME\n'
    hp_file.write(header)

else:
    # Cria o arquivo CSV
    hp_file = open(hp_filename, 'a')

if args.es:
    info = f'{args.ec}\t{args.ds}\t{args.arch}\t{args.optim}\t{args.ft}\t{args.bs}\t{args.lr}\t{args.mm}\t{args.ss}\t{args.ep}\t{early_stopping.best_epoch}\t{acc_val}\t{acc_test}\t{train_acc_list[-1]}\t{val_acc_list[-1]}\t{str(datetime.timedelta(seconds=time_total_train))}\n'
else:
    info = f'{args.ec}\t{args.ds}\t{args.arch}\t{args.optim}\t{args.ft}\t{args.bs}\t{args.lr}\t{args.mm}\t{args.ss}\t{args.ep}\t{args.ep - 1}\t{acc_val}\t{acc_test}\t{train_acc_list[-1]}\t{val_acc_list[-1]}\t{str(datetime.timedelta(seconds=time_total_train))}\n'

hp_file.write(info)

hp_file.close()

# %% [markdown]
# ## Done!
# ---

# %%
wandb.finish()

# Se o arquivo "done.txt" estiver na pasta, o experimento foi finalizado com sucesso!
done_file = open(os.path.join(EXP_PATH, 'done.txt'), 'w')
done_file.close()

print('\nDone!\n\n')

# %% [markdown]
# ## References 
# ---
# 
# * https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# * Finetuning Torchvision Models
#     * https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/finetuning_torchvision_models_tutorial.ipynb 
# * https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial
# * https://huggingface.co/docs/transformers/training
# * torchvision models
#     * https://pytorch.org/vision/stable/models.html
# * TIMM Models
#     * https://paperswithcode.com/lib/timm


