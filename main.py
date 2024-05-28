""" Import the dependencies"""
import torch
import argparse
import os
import wandb
from torch.distributed import init_process_group, destroy_process_group
from train import train, dataload, setup
from predict import predict, setup as setup_predict, dataload as dataload_predict

def main(file_path,in_channels,out_channels,lr,wd,world_size,rank,local_rank,epochs,batch_size,mode,task,n_workers=2,type=1,model_path=None):

    if not torch.cuda.is_available():
        print("Error: Distrbuted training is not supported without GPU")

    if mode == 'train':
        if rank == 0: # init wandb only if master process
            wandb.init(project="Video Frame Interpolation", config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "weight_decay": wd,
            }) 
        # init the process group for DDL
        init_process_group(backend='nccl',rank=rank,world_size=world_size)
        torch.cuda.set_device(local_rank)

        device = torch.device('cuda')
        data_loader, test_loader = dataload(file_path,batch_size,n_workers,task) # load the data
        (model,criteria,optim) = setup(
                                    lr,wd,in_channels,out_channels,
                                    n_layers=len(in_channels),bn_layers=2,
                                    model_type=type,
                                    model_path=model_path,
                                    task=task
                                ) # setup the model and the hyperparameters
        model = model.to(device)
        train(data_loader,test_loader,model,epochs,device,criteria,optim,local_rank,rank,task)

        if rank == 0:
            wandb.finish()

        destroy_process_group()
    
    elif mode == 'predict':
        device = torch.device('cuda')
        if model_path == None:
            raise Exception("Model Path not given")
    
        test_loader = dataload_predict(file_path,batch_size,n_workers,task) # load the data
        model = setup_predict(
                    in_channels,out_channels,
                    n_layers=len(in_channels),bn_layers=2,
                    model_type=type,
                    model_path=model_path,
                    task=task
                )
        model = model.to(device)
        predict(test_loader,model,device,task)


    else:
        raise Exception("Invalid mode, use --mode=train or --mode=predict")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Image colorization training")
    parser.add_argument('--batch_size', default=16, type=int, help='per GPU')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', default=0, type=int)
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--workers',default=1,type=int)
    parser.add_argument('--lr',default=1e-4,type=float)
    parser.add_argument('--wd',default=1e-3,type=float)
    parser.add_argument('--type',default=1,type=int)
    parser.add_argument('--path',default=None,type=str)
    parser.add_argument('--mode', default='train',type=str)
    parser.add_argument('--task', default='vfi',type=str)

    args = parser.parse_args()

    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])

    batch_size = args.batch_size
    epochs = args.epochs
    
    if args.task == "vfi":
        in_channels = [5,32,128,256,512]
        out_channels = [32,128,256,512,1024]
    elif args.task == "optFlow":
        in_channels = [3,32,64,128]
        out_channels = [32,64,128,256]

    main(args.data_path,in_channels,out_channels,args.lr,args.wd,args.world_size,
            args.rank,args.local_rank,epochs,batch_size,args.mode,args.task,n_workers=args.workers
            ,type=args.type,model_path=args.path)