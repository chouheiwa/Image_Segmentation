mkdir -p pretrained_models/vit_checkpoint/imagenet21k && cd pretrained_models/vit_checkpoint/imagenet21k

support_list=('ViT-B_16' 'ViT-B_32' 'ViT-L_16' 'R50+ViT-B_16' 'R50+ViT-L_16')
for model in "${support_list[@]}"; do
    file="${model}.npz"

    # 检查文件是否存在
    if [ ! -f "$file" ]; then
        echo "File $file does not exist, downloading from google storage."
        wget https://storage.googleapis.com/vit_models/imagenet21k/"$file"
    else
        echo "Fil $file exists, no need to download."
    fi
done
