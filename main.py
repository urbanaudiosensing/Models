import yaml
import os
import json
import sys
import ast

from data_utils.dataset import ASPEDv2Dataset, SR
from data_utils.datamodule import AspedDataModule
from models import ASPEDModel

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from torch import compile

try:
    from pytorch_lightning.utilities.seed import seed_everything
except:
    from torch import manual_seed as seed_everything

from logging_debug import manifestHandler, CONFIG, EVAL, MODEL_PATH

CONFIG_PATH = 'config/base.yml'

def deepcopy(d):
    print(d)
    return ast.literal_eval(json.dumps(d))

def main(config):
    params = deepcopy(config)
    config['model_params']['segment_length'] = config['data_params']['segment_length']
    config['data_params']['n_classes'] = config['model_params']['n_classes']

    seed_everything(config["seed"])

    TEST_DIR = config['data_params'].pop('data_path')
    print(TEST_DIR)
    print(config, params)
    X = ASPEDv2Dataset.from_dirs_v1(TEST_DIR, **config['data_params']) #v1
    data = AspedDataModule(X, **config['dataloader_params'])

    model = ASPEDModel(**config['model_params'])


    tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                            name=config['logging_params']['name'],)

    runner = Trainer(logger=tb_logger,
                    callbacks=[
                        LearningRateMonitor(),
                        ModelCheckpoint(save_top_k=1, 
                                        dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                        monitor= "val_loss")
                    ],
                    val_check_interval=0.2,
                    #strategy=DDPStrategy(find_unused_parameters=False),
                    **config['trainer_params'])
    runner.fit(model, data.train_dataloader(), data.val_dataloader())

    eval_dict = dict()
    params['train_thresh'] = ASPEDv2Dataset.min_val
    for tt in [1,2,3,4]:
        ASPEDv2Dataset.min_val = tt
        m = runner.test(ckpt_path=runner.checkpoint_callback.best_model_path, dataloaders=data.test_dataloader())
        #m = runner.test(model, dataloaders=data.test_dataloader())
        eval_dict[str(tt)] = m[0]
    print(m)
    print(params)
    return {EVAL:eval_dict, CONFIG: params, MODEL_PATH: runner.checkpoint_callback.best_model_path}

if __name__ == '__main__':
    if len(sys.argv) > 1:
        CONFIG_PATH = sys.argv[1]

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    try:
        os.mkdir(config['logging_params']['manifest_path'])
    except FileExistsError:
        pass
    
    print(f'-----USING {CONFIG_PATH} AS CONFIG-----')
    #run training instances for each radius 
    results = main(config)

    print(results)

    manifest = manifestHandler(save_path=config['logging_params']['manifest_path'],
                                **results)
    manifest.save()


    
    
    