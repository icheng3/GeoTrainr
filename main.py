import argparse
from parameter import get_args_parser
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.experiment == "country":
        from train_scripts.trainer_cls import Trainer
        trainer = Trainer(args)
        trainer.train()

    if args.experiment == "dis":
        from train_scripts.trainer_dis import Trainer
        trainer = Trainer(args)
        trainer.train()
    
    if args.experiment == "euclidean":
        from train_scripts.trainer_dis_euc import Trainer
        trainer = Trainer(args)
        trainer.train()
        
    if args.experiment == "latlng":
        from train_scripts.trainer_latlng import Trainer
        trainer = Trainer(args)
        trainer.train()