_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0] # nusc
point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0] # nuplan
voxel_size = [0.15, 0.15, 4]

num_cams = 8 # 8 for nuplan

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone', 
    'generic_object', 'czone_sign', 'vehicle', 
]
col_class = ['vehicle', 'car', 'truck', 'construction_vehicle', 'bus'] # collision classes, result in loss term

num_classes = len(class_names)

# map_classes = ['lane_divider', 'road_edge', 'crosswalk', 'centerline']
map_classes = ['divider', 'ped_crossing', 'boundary']

plan_fut_mode = 4096
plan_anchors_path = "data/traj_clusters/4096/cluster_centers_ori.npy"

data_root = "data/sample_data/nuplan/dataset"
# local_path: Optional, local path for images to speed up loading.
# load images from local_path+img_info['filename'] if local_path is not None.
# local_path = "/data/ceph/"

# sample data########################
local_path = "data/sample_data/nuplan/dataset/"
ann_file_train = "data/sample_data/sample_ann.pkl"
ann_file_test = "data/sample_data/sample_ann.pkl"
ann_file_map = "data/sample_data/eval_map.json"
#####################################

# load_from = 'v116_datav18_q12_15000.pth' #'samplingnotlanefollow_iter_40000.pth' #'default_ckpt.pth'
# resume_from = 'work_dirs/VADv2_config_voca4096_h800_1205/latest.pth'

no_validate = True
find_unused_parameters = True

# fixed_ptsnum_per_line = 20
# map_classes = ['divider',]
map_num_vec = 100
map_fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 4 # each sequence contains `queue_length` frames.
total_epochs = 50
# total_epochs = 12

# camera view order
# view_names = ['rgb_front', 'rgb_front_right', 'rgb_front_left',
#                 'rgb_rear', 'rgb_rear_left', 'rgb_rear_right']
view_names = ['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0']

# dataset_type = 'v3ADTRCustomCarlaDataset'
dataset_type = 'VADCustomNuScenesDataset'

file_client_args = dict(backend='disk')

# # open-loop test param
# # route_id = 6
# test_data_root = 'data/carladata/v2/pkl/'
# test_pkl = 'town05_long_new.pkl' #'carladata_v18_selected.pkl'  #'town05_long_new.pkl' # 'carladata_town05long.pkl' #'carladata_town05long.pkl' #'carla_minival_v3.pkl' 
# gt_anno_file = 'test_record/carla/v2'
# map_ann_file = gt_anno_file + '/test_map.json'
# agent_ann_file = gt_anno_file + '/test_agent.json'
# eval_detection_configs_path = 'projects/mmdet3d_plugin/datasets/carladata_eval_detection_configs.json'

model = dict(
    type='VADv2',
    use_grid_mask=True,
    video_test_mode=True,
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='v116ADTRHead',
        mot_map_thresh=0.5,
        mot_dis_thresh=0.2,
        pe_normalization=True,
        plan_fut_mode=256, #1024
        plan_fut_mode_testing=4096,
        tot_epoch=total_epochs,
        ego_query_thresh=0.0,
        query_use_fix_pad=False,
        ego_lcf_feat_idx=[0,1,4],
        valid_fut_ts=6,
        plan_anchors_path=plan_anchors_path,
    ego_pv_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        ego_agent_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        ego_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        cf_backbone_ckpt='ckpts/resnet50-0676ba61.pth',
        cf_backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch'),
        mot_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        mot_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        interaction_pe_type='sine_mlp',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=300,
        num_classes=num_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        map_num_vec=map_num_vec,
        map_num_classes=map_num_classes,
        map_num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
        map_num_pts_per_gt_vec=map_fixed_ptsnum_per_gt_line,
        map_query_embed_type='instance_pts',
        map_transform_method='minmax',
        map_gt_shift_pts_pattern='v2',
        map_dir_interval=1,
        map_code_size=2,
        map_code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='VADPerceptionTransformer',
            num_cams=num_cams,
            map_num_vec=map_num_vec,
            map_num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            num_cams=num_cams,
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            map_decoder=dict(
                type='MapDetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='V2CustomNMSFreeCoder',
            post_center_range=[-20, -35, -10.0, 20, 35, 10.0],
            pc_range=point_cloud_range,
            max_num=100,
            voxel_size=voxel_size,
            num_classes=num_classes),
        map_bbox_coder=dict(
            type='MapNMSFreeCoder',
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=map_num_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.8),
        loss_bbox=dict(type='L1Loss', loss_weight=0.1),
        loss_mot_reg=dict(type='L1Loss', loss_weight=0.1),
        loss_mot_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.2),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.8),
        loss_map_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_map_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=0.4),
        loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
        loss_plan_cls_col=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.0),
        loss_plan_cls_bd=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.0),
        loss_plan_cls_cl=dict(type='L1Loss', loss_weight=0.0),
        # loss_plan_cls_cl=dict(
        #     type='FocalLoss',
        #     use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=2.0),
        loss_plan_cls_expert=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=250.),
        # loss_plan_reg=dict(type='L1Loss', loss_weight=0.0),
        # loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.0, lane_bound_cls_idx=0),
        # loss_plan_agent_dis=dict(type='PlanAgentDisLoss', loss_weight=0.0),
        # loss_plan_map_theta=dict(type='PlanMapThetaLoss', loss_weight=0.0, lane_div_cls_idx=0),  # fake idx
        # loss_tl_status_cls=dict(
        #     type='CrossEntropyLoss',
        #     use_sigmoid=False,
        #     loss_weight=0.8,
        #     class_weight=None),
        # loss_tl_trigger_cls=dict(
        #     type='CrossEntropyLoss',
        #     use_sigmoid=True,
        #     loss_weight=4.,
        #     class_weight=None),
        # loss_stopsign_trigger_cls=dict(
        #     type='CrossEntropyLoss',
        #     use_sigmoid=True,
        #     loss_weight=1.0,
        #     class_weight=None)
        ),
    
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=0.8),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.1),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range),
        map_assigner=dict(
            type='MapHungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=0.8),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            # reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            # iou_cost=dict(type='IoUCost', weight=1.0), # Fake cost. This is just to make it compatible with DETR head.
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', weight=0.4),
            pc_range=point_cloud_range))))

train_pipeline = [
    # dict(type='LoadMultiViewImageFromCarla', to_float32=True),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, local_path=local_path),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='VADObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='VADObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
    dict(type='PadMultiViewImage', size_divisor=32),
    # dict(type='CustomDefaultFormatBundle3DFromCarla', class_names=class_names, with_ego=True),
    dict(type='VADFormatBundle3D', class_names=class_names, with_ego=True),
    dict(type='CustomCollect3D',\
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
               'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
               'ego_lcf_feat', 'gt_attr_labels', 
               # added
               'command_id', 'target_point', 'traffic_signal', 'stop_sign_signal'])

]

test_pipeline = [
    # dict(type='LoadMultiViewImageFromCarla', to_float32=True),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, local_path=local_path),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    # dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='VADObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='VADObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(416, 320),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
            dict(type='PadMultiViewImage', size_divisor=32),
            # dict(type='CustomDefaultFormatBundle3DFromCarla',
            #      class_names=class_names, with_label=False, with_ego=True),
            dict(type='VADFormatBundle3D', class_names=class_names, with_ego=True),
            dict(type='CustomCollect3D',\
                 keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'fut_valid_flag',
                       'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
                       'ego_lcf_feat', 'gt_attr_labels', 'command_id', 'target_point', 'traffic_signal', 'stop_sign_signal'])])  #  'traffic_signal', 'stop_sign_signal'
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=10,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file= ann_file_train, # data_root + '2021.10.06.17.43.07_veh-28_00508_00877.pkl', #'vad_nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        queue_length=queue_length,
        map_classes=map_classes,
        map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
        map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        custom_eval_version='vad_nusc_detection_cvpr_2019'),
    val=dict(type=dataset_type,
             data_root=data_root,
             pc_range=point_cloud_range,
             ann_file=ann_file_test,
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1,
             map_classes=map_classes,
             map_ann_file=ann_file_map,
             map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
             map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
             use_pkl_result=True,
             custom_eval_version='vad_nusc_detection_cvpr_2019'),
    test=dict(type=dataset_type,
              data_root=data_root,
              pc_range=point_cloud_range,
              ann_file=ann_file_test,
              pipeline=test_pipeline,
              use_valid_flag=True,
              bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality, samples_per_gpu=1,
              map_classes=map_classes,
              map_ann_file=ann_file_map,
              map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
              map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
              use_pkl_result=True,
              custom_eval_version='vad_nusc_detection_cvpr_2019'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)


optimizer = dict(
    type='AdamW',
    lr=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-2)

# evaluation = dict(interval=total_epochs, pipeline=test_pipeline, metric='bbox', map_metric='chamfer')

# runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# fp16 = dict(loss_scale=512.)
# find_unused_parameters = True
checkpoint_config = dict(interval=2000, by_epoch=False, max_keep_ckpts=10)


custom_hooks = [dict(type='CustomSetEpochInfoHook')]