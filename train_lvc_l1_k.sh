python -m tools.train_net --config-file configs/COCO-detection/loss12_k7_15k.yaml --num-gpus 4 
python -m tools.train_net --config-file configs/COCO-detection/loss12_k8_15k.yaml --num-gpus 4 

python -m tools.train_net --config-file configs/COCO-detection/loss12_k2.yaml --num-gpus 4 
python -m tools.train_net --config-file configs/COCO-detection/loss12_k3.yaml --num-gpus 4 
python -m tools.train_net --config-file configs/COCO-detection/loss12_k4.yaml --num-gpus 4 
python -m tools.train_net --config-file configs/COCO-detection/loss12_k5.yaml --num-gpus 4 
python -m tools.train_net --config-file configs/COCO-detection/loss12_k6.yaml --num-gpus 4 
python -m tools.train_net --config-file configs/COCO-detection/loss12_k7.yaml --num-gpus 4 
python -m tools.train_net --config-file configs/COCO-detection/loss12_k8.yaml --num-gpus 4 

# python -m tools.ckpt_surgery \
#     --method remove \
#     --coco \
#     --src1 /backup/guanmandan/checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_base_k_6/model_0000019.pth \
#     --save-dir /backup/guanmandan/checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_base_k_6/ \
#     --k 6

# # python -m tools.train_net \
# #     --config-file configs/COCO-detection/base_loss.yaml \
# #     --num-gpus 2 

# python -m tools.ckpt_surgery \
#     --method remove \
#     --coco \
#     --src1 /backup/guanmandan/checkpoints/coco/faster_rcnn/base_loss2/model_final.pth \
#     --save-dir /backup/guanmandan/checkpoints/coco/faster_rcnn/base_loss2/ \
#     --k 6

# python -m tools.train_net \
#     --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_novel_30shot_k_6.yaml \
#     --num-gpus 4

# python -m tools.ckpt_surgery \
#     --method combine \
#     --coco \
#     --src1 /backup/guanmandan/checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_base_k_6/model_0000019.pth \
#     --src2 /backup/guanmandan/checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_fs_k_6/model_final.pth \
#     --save-dir /backup/guanmandan/checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_novel_30shot_k_6/ \

# python -m tools.train_net \
#     --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout_k_6.yaml \
#     --num-gpus 4



##10shot

# python -m tools.train_net \
#     --config-file configs/COCO-detection/base_loss.yaml \
#     --num-gpus 2 

# python -m tools.ckpt_surgery \
#     --method remove \
#     --coco \
#     --src1 /backup/guanmandan/checkpoints/coco/faster_rcnn/base_loss2/model_final.pth \
#     --save-dir /backup/guanmandan/checkpoints/coco/faster_rcnn/base_loss2/

# CUDA_VISIBLE_DEVICES=2,3 python -m tools.train_net \
#     --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_novel_10shot_l1.yaml \
#     --num-gpus 2

# CUDA_VISIBLE_DEVICES=2,3 python -m tools.ckpt_surgery \
#     --method combine \
#     --coco \
#     --src1 /backup/guanmandan/checkpoints/coco/faster_rcnn/base_loss2/model_final.pth \
#     --src2 checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_novel_10shot_l1/model_final.pth \
#     --save-dir /backup/guanmandan/checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_novel_10shot_l1

# CUDA_VISIBLE_DEVICES=2,3 python -m tools.train_net \
#     --config-file configs/COCO-detection/l1_faster_rcnn_R_50_FPN_ft_all_10shot_aug_ftmore_dropout.yaml \
#     --num-gpus 2