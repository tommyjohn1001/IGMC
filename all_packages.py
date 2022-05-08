from __future__ import print_function

import argparse
import itertools
import math
import multiprocessing as mp
import os
import os.path as osp
import random
import sys
import time
import warnings
from datetime import datetime, timedelta
from typing import Any

import pytorch_lightning as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
