learning_rate = 0.001
batch_size = 1024
num_classes = 10
epochs = 100
patch_size = 8
flatten_size = 3*patch_size*patch_size
Hidden_dim_base = 768 # 3x16x16
Hidden_dim_large = 1024
Hidden_dim_huge = 1280
Encoder_iter = 6
num_heads = 8
eta_min = 0.00001
MLP_scale = 2
Dropout_rate = 0.1
step_size = 40
CIFAR10_img_size = 32
CIFAR100_img_size = 32
ImageNet_img_size = 224
model_path = './model/ViT_CIFAR10.pth'