wandb: Currently logged in as: somya23005. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.18.5 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.13.2
wandb: Run data is saved locally in /home/somya/thesis/fedgmm/sp_decentralized_mnist_lr_example/wandb/run-20241021_174725-lf2ou68k
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run comm_round-350_local_2_linear_fedAvg
wandb: ⭐️ View project at https://wandb.ai/somya23005/FEDGMM
wandb: 🚀 View run at https://wandb.ai/somya23005/FEDGMM/runs/lf2ou68k
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [__init__.py:163:init] args.rank = 0, args.worker_num = 10
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [ml_engine_adapter.py:147:get_torch_device] args = <fedml.arguments.Arguments object at 0x7fab13b18890>, using_gpu = True, device_id = 1, device_type = gpu
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [device.py:48:get_device] device = cuda:1
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [data_loader.py:259:load_synthetic_data] load_data. dataset_name = zoo
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:116:info] test
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:116:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]   x:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] ndarray
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] (float64): 
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 20000x1
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]   z:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] ndarray
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] (float64): 
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 20000x2
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]   y:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] ndarray
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] (float64): 
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 20000x1
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]   g:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] ndarray
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] (float64): 
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 20000x1
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]   w:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] ndarray
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] (float64): 
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 20000x1
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:116:info] train
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:116:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]   x:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] ndarray
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] (float64): 
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 20000x1
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info]       min: -6.13
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info] , max: 5.79
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]   z:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] ndarray
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] (float64): 
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 20000x2
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info]       min: -3.00
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info] , max: 3.00
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]   y:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] ndarray
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] (float64): 
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 20000x1
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info]       min: -3.79
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info] , max: 3.78
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]   g:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] ndarray
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] (float64): 
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 20000x1
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info]       min: -1.75
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info] , max: 1.68
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]   w:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] ndarray
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] (float64): 
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 20000x1
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info]       min: -6.13
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info] , max: 5.79
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:33:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:116:info] dev
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:116:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]   x:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] ndarray
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] (float64): 
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 20000x1
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]   z:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] ndarray
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] (float64): 
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 20000x2
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]   y:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] ndarray
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] (float64): 
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 20000x1
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]   g:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] ndarray
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] (float64): 
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 20000x1
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]   w:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] ndarray
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] (float64): 
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 20000x1
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [abstract_scenario.py:31:info] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:32] [INFO] [model_hub.py:24:create] create_model. model_name = lr, output_dim = None
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:33] [INFO] [fedavg_api.py:78:__init__] model = [[MLPModel(
  (model): Sequential(
    (0): Linear(in_features=1, out_features=20, bias=True)
    (1): LeakyReLU(negative_slope=0.01)
    (2): Linear(in_features=20, out_features=3, bias=True)
    (3): LeakyReLU(negative_slope=0.01)
    (4): Linear(in_features=3, out_features=1, bias=True)
  )
)], [MLPModel(
  (model): Sequential(
    (0): Linear(in_features=2, out_features=20, bias=True)
    (1): LeakyReLU(negative_slope=0.01)
    (2): Linear(in_features=20, out_features=1, bias=True)
  )
)], [MLPModel(
  (model): Sequential(
    (0): Linear(in_features=1, out_features=20, bias=True)
    (1): LeakyReLU(negative_slope=0.01)
    (2): Linear(in_features=20, out_features=3, bias=True)
    (3): LeakyReLU(negative_slope=0.01)
    (4): Linear(in_features=3, out_features=1, bias=True)
  )
)]]
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:33] [INFO] [model_selection_class.py:38:do_model_selection] starting learning args eval 0
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:47:33] [INFO] [model_selection_class.py:38:do_model_selection] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:53:26] [INFO] [model_selection_class.py:38:do_model_selection] starting learning args eval 1
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:53:26] [INFO] [model_selection_class.py:38:do_model_selection] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:58:58] [INFO] [model_selection_class.py:38:do_model_selection] starting learning args eval 2
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 17:58:58] [INFO] [model_selection_class.py:38:do_model_selection] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:27] [INFO] [model_selection_class.py:68:do_model_selection] learning eval:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:27] [INFO] [model_selection_class.py:68:do_model_selection]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:27] [INFO] [model_selection_class.py:68:do_model_selection] -0.06297157554662766
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:27] [INFO] [model_selection_class.py:68:do_model_selection]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:27] [INFO] [model_selection_class.py:68:do_model_selection] OptimalObjective::lambda_1=0.250000
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:27] [INFO] [model_selection_class.py:68:do_model_selection]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:27] [INFO] [model_selection_class.py:68:do_model_selection] CustomSGD:::'lr'=0.01:'momentum'=0.9
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:27] [INFO] [model_selection_class.py:68:do_model_selection]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:27] [INFO] [model_selection_class.py:68:do_model_selection] CustomSGD:::'lr'=0.05:'momentum'=0.9
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:27] [INFO] [model_selection_class.py:68:do_model_selection] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:29] [INFO] [model_selection_class.py:68:do_model_selection] learning eval:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:29] [INFO] [model_selection_class.py:68:do_model_selection]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:29] [INFO] [model_selection_class.py:68:do_model_selection] 0.002800603618362641
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:29] [INFO] [model_selection_class.py:68:do_model_selection]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:29] [INFO] [model_selection_class.py:68:do_model_selection] OptimalObjective::lambda_1=0.250000
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:29] [INFO] [model_selection_class.py:68:do_model_selection]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:29] [INFO] [model_selection_class.py:68:do_model_selection] CustomSGD:::'lr'=0.001:'momentum'=0.9
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:29] [INFO] [model_selection_class.py:68:do_model_selection]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:29] [INFO] [model_selection_class.py:68:do_model_selection] CustomSGD:::'lr'=0.005:'momentum'=0.9
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:29] [INFO] [model_selection_class.py:68:do_model_selection] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:32] [INFO] [model_selection_class.py:68:do_model_selection] learning eval:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:32] [INFO] [model_selection_class.py:68:do_model_selection]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:32] [INFO] [model_selection_class.py:68:do_model_selection] -0.0015743836422999445
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:32] [INFO] [model_selection_class.py:68:do_model_selection]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:32] [INFO] [model_selection_class.py:68:do_model_selection] OptimalObjective::lambda_1=0.250000
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:32] [INFO] [model_selection_class.py:68:do_model_selection]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:32] [INFO] [model_selection_class.py:68:do_model_selection] CustomSGD:::'lr'=0.0001:'momentum'=0.9
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:32] [INFO] [model_selection_class.py:68:do_model_selection]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:32] [INFO] [model_selection_class.py:68:do_model_selection] CustomSGD:::'lr'=0.0005:'momentum'=0.9
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:32] [INFO] [model_selection_class.py:68:do_model_selection] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:32] [INFO] [model_selection_class.py:82:do_model_selection] size of f_z collection:
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:32] [INFO] [model_selection_class.py:82:do_model_selection]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:32] [INFO] [model_selection_class.py:82:do_model_selection] 300
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:32] [INFO] [model_selection_class.py:82:do_model_selection] 

[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:32] [INFO] [fedavg_api.py:177:_setup_clients] ############setup_clients (START)#############
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:03:32] [INFO] [fedavg_api.py:194:_setup_clients] ############setup_clients (END)#############
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:04:55] [INFO] [fedavg_api.py:341:train] MSE on test ------------------------------>>>>>>>>>>>>>>>>>>
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:04:55] [INFO] [fedavg_api.py:341:train]  
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:04:55] [INFO] [fedavg_api.py:341:train] 0.3255299733246835
[FedML-Client @device-id-0] [Mon, 21 Oct 2024 18:04:55] [INFO] [fedavg_api.py:341:train] 

wandb: Waiting for W&B process to finish... (success).
wandb: - 0.001 MB of 0.001 MB uploaded (0.000 MB deduped)wandb: \ 0.001 MB of 0.001 MB uploaded (0.000 MB deduped)wandb: | 0.001 MB of 0.001 MB uploaded (0.000 MB deduped)wandb: / 0.001 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: - 0.001 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: \ 0.001 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: | 0.001 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: / 0.001 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: - 0.001 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: \ 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: | 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: / 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: - 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: \ 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: | 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: / 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: - 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: \ 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: | 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: / 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: - 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: \ 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: | 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: / 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: - 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: \ 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: | 0.021 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: / 0.042 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: - 0.042 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: \ 0.042 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: | 0.042 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: / 0.042 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: - 0.042 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: \ 0.042 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: | 0.042 MB of 0.042 MB uploaded (0.000 MB deduped)wandb: / 0.042 MB of 0.042 MB uploaded (0.000 MB deduped)wandb:                                                                                
wandb: 
wandb: Run history:
wandb:             max_eval       ▁▁▂▂▂▂▂     ▆████      ▇▇▇▇▇▇▇▇▇▇█
wandb: model_selection_step ▁▂▂▃▃▄▄▅▆▆▆▇█▁▁▂▃▃▄▄▅▅▆▆▇█▁▁▂▃▃▃▄▅▅▆▆▇▇█
wandb:                  psi ▁▆▇▅▁▇▅▅▇▇▃▇▇█▆█▅▆█▅▆▅▅▇▇███████████████
wandb:  selection_iteration ▁▁▂▂▃▃▃▄▄▅▅▆▆▁▂▂▃▃▃▄▄▅▅▁▁▂▂▃▃▄▄▅▅▅▆▆▇▇▇█
wandb: 
wandb: Run summary:
wandb:             max_eval -0.00166
wandb: model_selection_step 2999
wandb:                  psi -0.00157
wandb:  selection_iteration 149
wandb: 
wandb: Synced comm_round-350_local_2_linear_fedAvg: https://wandb.ai/somya23005/FEDGMM/runs/lf2ou68k
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20241021_174725-lf2ou68k/logs
