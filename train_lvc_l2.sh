# python -m tools.train_net \
#     --config-file configs/COCO-detection/loss2.yaml \
#     --num-gpus 2 \
#     --resume

# python -m tools.ckpt_surgery \
#     --method remove \
#     --coco \
#     --src1 /backup/guanmandan/checkpoints/coco/faster_rcnn/loss2/model_final.pth \
#     --save-dir /backup/guanmandan/checkpoints/coco/faster_rcnn/loss2/

# python -m tools.train_net \
#     --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_novel_30shot_loss2.yaml \
#     --num-gpus 2

# python -m tools.ckpt_surgery \
#     --method combine \
#     --coco \
#     --src1 /backup/guanmandan/checkpoints/coco/faster_rcnn/loss2/model_final.pth \
#     --src2 /backup/guanmandan/checkpoints/coco/faster_rcnn/ft_novel_30shot_loss2/model_final.pth \
#     --save-dir /backup/guanmandan/checkpoints/coco/faster_rcnn/ft_novel_30shot_loss2

python -m tools.train_net \
    --config-file configs/COCO-detection/aug_ftmore_dropout_loss2.yaml \
    --num-gpus 4