使用的镜像： 使用 trtllm 官方推荐的镜像

# 拉去第三方依赖
git submodule init

git submodule update

# 编译
python setup.py develop

# 运行单测
cd moe

python moe_test.py
