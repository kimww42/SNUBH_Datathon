import torch
import timm

from torchvision import models
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_model(args):
    if args.model == "SVM":
        model = SVC(kernel='rbf', random_state = 1, gamma=0.10, C=10.0)
    elif args.model == "RandomForest":
        model = RandomForestClassifier(n_estimators=20, max_depth=5,random_state=1)

    return model