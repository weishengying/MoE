使用的镜像： gpu-image-cn-shanghai.cr.volces.com/gpu-train/skyllm2:v0.1

# 拉去第三方依赖
git submodule init

git submodule update

# 编译
python setup.py develop

# 运行单测
cd moe

python moe_test.py