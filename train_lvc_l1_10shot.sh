# python -m tools.train_net \
#     --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_base.yaml \
#     --num-gpus 2 --resume

# python -m tools.ckpt_surgery \
#     --method remove \
#     --coco \
#     --src1 /backup/guanmandan/checkpoints/coco/faster_rcnn/baseline_base_60000_8/model_baseline_batch8_60000.pth \
#     --save-dir /backup/guanmandan/checkpoints/coco/faster_rcnn/baseline_base_60000_8/model_baseline_batch8_60000.pth 

# python -m tools.train_net \
#     --config-file configs/COCO-detection/base_loss.yaml \
#     --num-gpus 2 

# python -m tools.ckpt_surgery \
#     --method remove \
#     --coco \
#     --src1 /backup/guanmandan/checkpoints/coco/faster_rcnn/base_loss2/model_final.pth \
#     --save-dir /backup/guanmandan/checkpoints/coco/faster_rcnn/base_loss2/

# python -m tools.train_net \
#     --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_novel_30shot_l1.yaml \
#     --num-gpus 4

# # python -m tools.ckpt_surgery \
# #     --method combine \
# #     --coco \
# #     --src1 /backup/guanmandan/checkpoints/coco/faster_rcnn/base_loss2/model_final.pth \
# #     --src2 checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_novel_30shot_l1/model_final.pth \
# #     --save-dir /backup/guanmandan/checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_novel_30shot_l1

# python -m tools.train_net \
#     --config-file configs/COCO-detection/l1_faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout.yaml \
#     --num-gpus 2



##10shot

# python -m tools.train_net \
#     --config-file configs/COCO-detection/base_loss.yaml \
#     --num-gpus 2 

# python -m tools.ckpt_surgery \
#     --method remove \
#     --coco \
#     --src1 /backup/guanmandan/checkpoints/coco/faster_rcnn/base_loss2/model_final.pth \
#     --save-dir /backup/guanmandan/checkpoints/coco/faster_rcnn/base_loss2/

python -m tools.train_net \
    --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_novel_10shot_l1.yaml \
    --num-gpus 4

python -m tools.ckpt_surgery \
    --method combine \
    --coco \
    --src1 /backup/guanmandan/checkpoints/coco/faster_rcnn/base_loss2/model_final.pth \
    --src2 /backup/guanmandan/checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_novel_10shot_l1/model_final.pth \
    --save-dir /backup/guanmandan/checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_novel_10shot_l1

python -m tools.train_net \
    --config-file configs/COCO-detection/l1_faster_rcnn_R_50_FPN_ft_all_10shot_aug_ftmore_dropout.yaml \
    --num-gpus 4