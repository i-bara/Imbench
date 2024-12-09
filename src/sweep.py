import wandb
from argparse import Namespace

from args import parse_args


args, baseline = parse_args()

config = Namespace(
    project_name = 'wandb_demo',
    
    batch_size = 512,
    
    hidden_layer_width = 64,
    dropout_p = 0.1,
    
    lr = 1e-4,
    optim_type = 'Adam',
    
    epochs = 15,
    ckpt_path = 'checkpoint.pt'
)

sweep_config = {
    'method': 'random', 
    'metric': {
        'name': 'test_acc', 
        'goal': 'maximize', 
    }, 
    'parameters': {
        
    },
}

def train(config=config):
    baseline(config)

wandb.agent(sweep_id, train, count=5)
