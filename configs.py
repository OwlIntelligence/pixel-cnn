def config_args(args, config):
    if "data_dir" in config:
        args.data_dir = config["data_dir"]
    if "save_dir" in config:
        args.save_dir = config["save_dir"]
    if "nr_filters" in config:
        args.nr_filters = config["nr_filters"]
    if "nr_resnet" in config:
        args.nr_resnet = config["nr_resnet"]
    if "data_set" in config:
        args.data_set = config["data_set"]
    if "batch_size" in config:
        args.batch_size = config["batch_size"]
    if "init_batch_size" in config:
        args.init_batch_size = config["init_batch_size"]
    if "spatial_conditional" in config:
        args.spatial_conditional = config['spatial_conditional']
    if "global_conditional" in config:
        args.global_conditional = config['global_conditional']
    if "save_interval" in config:
        args.save_interval = config['save_interval']
    if "map_sampling" in config:
        args.map_sampling = config['map_sampling']
    if "nr_gpu" in config:
        args.nr_gpu = config['nr_gpu']
    if "context_conditioning" in config:
        args.context_conditioning = config['context_conditioning']
    if "input_size" in config:
        args.input_size = config['input_size']
    if "global_latent_dim" in config:
        args.global_latent_dim = config['global_latent_dim']
    if "spatial_latent_num_channel" in config:
        args.spatial_latent_num_channel = config['spatial_latent_num_channel']
    if 'use_coordinates' in config:
        args.use_coordinates = config['use_coordinates']
    if 'deconv_z' in config:
        args.deconv_z = config['deconv_z']





configs = {}

configs['imagenet'] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/imagenet",
    "save_dir": "/data/ziz/jxu/models/imagenet-test",
    "nr_filters": 160,
    "nr_resnet": 5,
    "data_set": "imagenet",
    "batch_size": 8,
    "init_batch_size": 8,
    "spatial_conditional": True,
    "save_interval": 10,
    "map_sampling": True,
    "nr_gpu": 2,
}

configs['cifar'] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/cifar",
    "save_dir": "/data/ziz/jxu/models/cifar-test",
    "nr_filters": 160,
    "nr_resnet": 5,
    "data_set": "cifar",
    "batch_size": 8,
    "init_batch_size": 8,
    "spatial_conditional": True,
    "save_interval": 10,
    "map_sampling": False,
    "nr_gpu": 2,
}

configs['celeba64'] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/models/celeba-test",
    "nr_filters": 120,
    "nr_resnet": 4,
    "data_set": "celeba64",
    "batch_size": 8,
    "init_batch_size": 8,
    "spatial_conditional": True,
    "save_interval": 5,
    "map_sampling": False,
    "nr_gpu": 2,
}

configs['celeba128'] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/models/celeba128-model",
    "nr_filters": 50,
    "nr_resnet": 3,
    "data_set": "celeba128",
    "batch_size": 4,
    "init_batch_size": 4,
    "spatial_conditional": True,
    "save_interval": 5,
    "map_sampling": False,
    "nr_gpu": 4,
}

configs['celeba64-cond'] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/models/celeba64-cond",
    "nr_filters": 100,
    "nr_resnet": 4,
    "data_set": "celeba64",
    "batch_size": 4,
    "init_batch_size": 4,
    "spatial_conditional": True,
    'global_conditional': False,
    "save_interval": 5,
    "map_sampling": False,
    "nr_gpu": 6,
    'context_conditioning': True,
    'use_coordinates': False,
    "input_size": 64,
    #"global_latent_dim": 640, #100,
    "spatial_latent_num_channel": 4,
}

configs['celeba128-patch'] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/models/celeba128-patch-4",
    "nr_filters": 100,
    "nr_resnet": 4,
    "data_set": "celeba128",
    "batch_size": 16,
    "init_batch_size": 16,
    "spatial_conditional": True,
    'global_conditional': True,
    "save_interval": 5,
    "map_sampling": False,
    "nr_gpu": 4,
    'context_conditioning': True,
    'use_coordinates': True,
    "input_size": 32,
    "global_latent_dim": 640, #100,
    "spatial_latent_num_channel": 4,
}

configs['celeba128-patch-deconv'] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/models/celeba128-patch-deconv",
    "nr_filters": 100,
    "nr_resnet": 4,
    "data_set": "celeba128",
    "batch_size": 4,
    "init_batch_size": 4,
    "spatial_conditional": True,
    'global_conditional': False,
    "save_interval": 5,
    "map_sampling": False,
    "nr_gpu": 4,
    'context_conditioning': True,
    'use_coordinates': False,
    "input_size": 32,
    #"global_latent_dim": 640, #100,
    "spatial_latent_num_channel": 4,
    "deconv_z": True,
}

# configs['celeba128-patch'] = {
#     "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
#     "save_dir": "/data/ziz/jxu/models/celeba128-patch-1",
#     "nr_filters": 100,
#     "nr_resnet": 4,
#     "data_set": "celeba128",
#     "batch_size": 16,
#     "init_batch_size": 16,
#     "spatial_conditional": True,
#     'global_conditional': True,
#     "save_interval": 5,
#     "map_sampling": False,
#     "nr_gpu": 4,
#     #'context_conditioning': True,
#     "input_size": 32,
#     "global_latent_dim": 100,
#     "spatial_latent_num_channel": 2,
# }


configs['celeba128-full'] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/models/celeba128-full",
    "nr_filters": 50,
    "nr_resnet": 6,
    "data_set": "celeba128",
    "batch_size": 4,
    "init_batch_size": 4,
    "spatial_conditional": False,
    'global_conditional': True,
    "save_interval": 5,
    "map_sampling": False,
    "nr_gpu": 4,
    #'context_conditioning': True,
    "input_size": 128,
    "global_latent_dim": 100,
    #"spatial_latent_num_channel": 2,
}
