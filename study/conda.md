# conda 换个源
```
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ 
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/win-64/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/win-64/
conda config --set show_channel_urls yes
```
# 动态库问题
```
conda update --all
```
# jupyter notebook 嵌入环境
```
conda activate xxx
conda install ipykernel
python -m ipykernel install --user --name xxx --display-name "写个名字"
```

# GPU
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3
```

# Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```
删除当前虚拟环境下的 Library\Bin\libiomp5md.dll  
```