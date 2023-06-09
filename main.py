import argparse
from parameter import get_args_parser
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(args.experiment)
    print(args.model)
    print(args.output_dir)
    
    if args.experiment == "country":
        from train_scripts.trainer_cls import Trainer
        trainer = Trainer(args)

    if args.experiment == "dis":
        from train_scripts.trainer_dis_debug import Trainer
        trainer = Trainer(args)
    
    if args.experiment == "dis_freeze":
        from train_scripts.trainer_dis_freeze import Trainer
        trainer = Trainer(args)
        
    if args.experiment == "ivnf":
        from train_scripts.trainer_ivnf import Trainer
        trainer = Trainer(args)

    if args.experiment == "euclidean":
        from train_scripts.trainer_dis_euc import Trainer
        trainer = Trainer(args)
        
    if args.experiment == "latlng":
        from train_scripts.trainer_latlng import Trainer
        trainer = Trainer(args)
    

    if args.eval:
        trainer.evaluate(trainer.data_loader_val)
    else:
        trainer.train()
        
# ! wget https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
# ! wget https://github.com/icheng3/GeoTrainr/releases/download/weights/checkpoint-best.pth