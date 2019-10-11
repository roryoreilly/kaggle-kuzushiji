import pandas as pd
import os
import torch

from fbmaskrcnn.maskrcnn_benchmark.config import cfg
from fbmaskrcnn.maskrcnn_benchmark.utils.comm import get_world_size
from .predictor import COCODemo


config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

cfg.merge_from_file(config_file)
COCODemo.CATEGORIES = pd.read_csv(
        os.path.join('/kaggle/input/kuzushiji-recognition', 'unicode_translation.csv')
    )['Unicode'].values.tolist()

demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

base_model = demo.model

torch.save(base_model, "base_model.pth")

# %% [code]

# def build_data_loader(cfg, dataset, is_train=True, is_distributed=False, start_iter=0):
#     num_gpus = get_world_size()
#     if is_train:
#         images_per_batch = cfg.SOLVER.IMS_PER_BATCH
#         assert (
#                 images_per_batch % num_gpus == 0
#         ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
#             images_per_batch, num_gpus)
#         images_per_gpu = images_per_batch // num_gpus
#         shuffle = True
#         num_iters = cfg.SOLVER.MAX_ITER
#     else:
#         images_per_batch = cfg.TEST.IMS_PER_BATCH
#         assert (
#                 images_per_batch % num_gpus == 0
#         ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
#             images_per_batch, num_gpus)
#         images_per_gpu = images_per_batch // num_gpus
#         shuffle = False if not is_distributed else True
#         num_iters = None
#         start_iter = 0
#
#     # group images which have similar aspect ratio. In this case, we only
#     # group in two cases: those with width / height > 1, and the other way around,
#     # but the code supports more general grouping strategy
#     aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
#
#     paths_catalog = import_file(
#         "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
#     )
#     DatasetCatalog = paths_catalog.DatasetCatalog
#     dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
#
#     # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
#     transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)
#
#     dataset.transforms = transforms
#     datasets = [dataset]
#
#     data_loaders = []
#     for dataset in datasets:
#         sampler = make_data_sampler(dataset, shuffle, is_distributed)
#         batch_sampler = make_batch_data_sampler(
#             dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
#         )
#         collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
#             BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
#         num_workers = cfg.DATALOADER.NUM_WORKERS
#         data_loader = torch.utils.data.DataLoader(
#             dataset,
#             num_workers=num_workers,
#             batch_sampler=batch_sampler,
#             collate_fn=collator,
#         )
#         data_loaders.append(data_loader)
#     if is_train:
#         # during training, a single (possibly concatenated) data_loader is returned
#         assert len(data_loaders) == 1
#         return data_loaders[0]
#     return data_loaders