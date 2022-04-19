import warnings
import os
import argparse
from dataclasses import dataclass

parser = argparse.ArgumentParser()
#parser.add_argument('--gpu_num',default=0,type=1,help='Graphics card number')

# system parameter
parser.add_argument("--time_step",default=16,type=int,help='Number of time step')
parser.add_argument("--data_dim",default=2,type=int,help="Real, Imaginary Domain")
parser.add_argument("--cons_map",default= ([-3-3j,-3-1j,-3+1j,-3+3j,
                     -1-3j,-1-1j,-1+1j,-1+3j,
                     +1-3j,+1-1j,+1+1j,+1+3j,
                     +3-3j,+3-1j,+3+1j,+3+3j]),type=list,help="Constellation Map")

# NN parameter
parser.add_argument("--num_cells",default=16,type=int,help="Number of LSTM cells")
parser.add_argument("--batch_size",default=10000,type=int,help="Batch Size")

args = parser.parse_args()

@dataclass
class system:
    time_step = args.time_step
    data_dim = args.data_dim
    cons_map = args.cons_map
sys = system
#-------------------------------
@dataclass
class neuralnet:
    num_cells = args.num_cells
    batch_size = args.batch_size
nn = neuralnet