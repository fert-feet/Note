# win10安装深度学习环境

## 简介

需要安装anaconda+pytorch+cuda+cunnd+pycharm，版本要严格对应

## 具体步骤

### 1. cuda安装

N 卡想用 GPU 跑深度学习要安装 cuda 和 cudnn，cuda 是 并行计算平台和编程模型，后者是深度神经网络加速库。

1. 首先我们找到自己机子支持的 CUDA 版本，从命令行输入以下命令查看，图中的 CUDA Version 代表机器支持的最高 CUDA版本。

   ```bash
   nvidia-smi
   ```

   ![image-20240409233900130](https://s2.loli.net/2024/04/09/BWMqZwYvCEDugkj.png)

2. 找到支持版本后，到官网下载对应版本的 CUDA 安装包，CUDA 历史版本下载地址：[CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive)，找到对应版本点击选择系统版本并点击下载（略大）。

   <img src="https://s2.loli.net/2024/04/09/H1EILtQfP64FT8O.png" alt="image-20240409234609780" style="zoom: 50%;" />

3. 下载完成后打开安装包，注意这里的地址实际上选择的是解压地址，建议放在另外的文件夹，和安装文件夹分开。

![image-20240409235456991](https://s2.loli.net/2024/04/09/OHxQn83CdmIrE4B.png)

4. 提取完成后跳出安装程序，点击自定义（高级），跳出安装选项后下一步，然后就到了最关键的安装位置（使用别人的图），这里需要记住这几个安装位置，也可以换成自己的路径，不过最好 Program Files 之后的路径名称保持相同，之后配置环境变量要用，其中 Documentation 和 Development 的路径是一样的。

<img src="https://s2.loli.net/2024/04/09/gba4irVufcGz2nx.png" style="zoom: 67%;" />

5. 接下来配置环境变量，一共需要配置6个环境变量，后2个是在 path 中的变量（实操在配置 CUDA_PATH 和 CUDA_PATH_V11_6 后 path 中的两个变量会自己配置好）。第一个是 CUDA_PATH，值为 `xxx\NVIDIA GPU Computing Toolkit\CUDA\v11.6`，第二个是 CUDA_PATH_V<版本号，点用下划线代替>，值和前者一模一样，第三个是 NVCUDASAMPLES_ROOT，值为 `xxx\NVIDIA Corporation\CUDA Samples\v11.6`。第四个是 NVCUDASAMPLES11_6_ROOT，值和前者一模一样。

   ![image-20240410000125479](https://s2.loli.net/2024/04/10/PQCFxbanLJvUklt.png)

   然后是 path 中，分别新建地址，添加 `D:\Application\WorkApplication\CUDA\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin`，和 `D:\Application\WorkApplication\CUDA\NVIDIA GPU Computing Toolkit\CUDA\v11.6\libnvvp`

   ![image-20240410001343227](https://s2.loli.net/2024/04/10/Jk2IRBOod7tEvwh.png)

6. 最后在命令行中输入以下命令检测 cuda 是否安装成功，若出现图片中内容即为安装成功。

   ```BASH
   nvcc -V
   ```

   ![image-20240410003039204](https://s2.loli.net/2024/04/10/BKoJ8yjG4p1cxLX.png)

### 2. CUNND 安装

CUNND 是深度神经网络的加速库，是深度学习环境必备的工具库。

1. 打开官网 [cuDNN Archive | NVIDIA Developer](https://developer.nvidia.com/rdp/cudnn-archive) 选择一个和刚刚装的 CUDA 版本相匹配的 CUNND 版本，比如我的 CUDA 版本是11.6，那我就选择一个 for CUDA 11.x 的版本，注意要选择第一个下载压缩包。

   ![image-20240410003711519](https://s2.loli.net/2024/04/10/kmUqJb6AQP9Otaw.png)

2. 将cuDNN安装包里的三个文件夹里的内容分别复制到 `xxx\NVIDIA GPU Computing Toolkit\CUDA\v11.6` 路径下的同名文件夹（bin、include、lib）里，LICENSE 直接复制到 v11.6 文件夹里。

3. 接下来配置环境变量（图用的别人的），将这三个地址添加到 path 中。

   ![](https://img-blog.csdnimg.cn/f2a3f475c00c42cf884d5a0a4643996a.png)

4. 最后在命令行中输入以下命令检测是否安装 CUNND 成功，正确返回应如图中一致。

   ```BASH
   cd xxx\NVIDIA GPU Computing Toolkit\CUDA\v11.6\extras\demo_suite
   .\bandwidthTest.exe
   .\deviceQuery.exe
   ```

   ![image-20240410004720542](https://s2.loli.net/2024/04/10/6xBnYtVTovjbap5.png)

   ![image-20240410004755810](https://s2.loli.net/2024/04/10/9kXnMhzxoIEYjK3.png)

### 3. Anaconda 安装

这里要说明以下，我不知道不同版本的 anaconda 是否会有影响，我认为应该没有！那就装官网最新版就好。Anaconda 的安装教程网上多如牛毛，这里直接省略，也是最简单的部分了，注意配置环境变量就好了。这里只说装好后的操作

1. 打开 Anaconda Prompt 创建 pytorch 虚拟环境，输入以下命令。

   ```BASH
   conda create -n pytorch python=3.9（输入 Anaconda 对应 python 版本）
   ```

2. 激活环境，并注意，接下来所有操作都是在 pytorch 环境中进行

   ```bash
   conda activate pytorch #激活环境
   conda deactivate #关闭环境
   ```

   ![image-20240410010430994](https://s2.loli.net/2024/04/10/z1kOcDhJCe5iAp9.png)

### 4. Pytorch 安装

这部分是最麻烦的，最主要的麻烦是由于资源在国外，国内容易出错，因此推荐第二种方法。

#### 第一种方法（硬刚网络）

1. 打开官网下载对应 CUDA 版本的 pytorch，这里选择我的版本，其他版本自行对应，复制这个命令。

   ![image-20240410005749591](https://s2.loli.net/2024/04/10/IkKOZ9tYg34Tq1l.png)

2. 在 Anaconda 命令行中（pytorch 环境）粘贴该下载指令进行下载（不出意外会因为网络意外失败），一路输入 Y 继续下载即可。

   ```bash
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
   ```

#### 第二种方法（本地下载）

1. 打开网址 [download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html) 找到对应 CUDA 版本和 Python 版本的文件，注意 cu116 指的是 CUDA 版本是11.6，cp310 指的是 python 版本是 3.10.x，然后就是选择 win 系统的文件并下载。

   ![image-20240410011152355](https://s2.loli.net/2024/04/10/wU7mtCnYsRo8Jhg.png)

2. 下载后，在 Anaconda（Pytorch 环境）中定位到下载文件的文件夹中，并运行以下指令，等待安装完毕即可。

   ```bash
   pip install .\torch-1.12.1+cu116-cp310-cp310-win_amd64.whl
   ```

   ![image-20240410011716654](https://s2.loli.net/2024/04/10/7UbBm4CxcMh1qQa.png)

### 5. Pycharm 配置

假设已经安装好了 Pycharm。

1. 选择新建项目并选择以下配置

   <img src="https://s2.loli.net/2024/04/10/Z6qgiry4wuEbxfK.png" alt="image-20240410012834519"  />

2. 新建项目后，选择设置中的 project，并根据下2图点击

   ![image-20240410013417811](https://s2.loli.net/2024/04/10/h4cqLvmWQE2yGoO.png)

   ![image-20240410013707366](https://s2.loli.net/2024/04/10/IjL3sNcGiU2a17w.png)

3. 最后新建一个 .py 文件并键入以下代码，若输出与图中相同则全部完工！

   ```python
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   ```

   ![image-20240410014003240](https://s2.loli.net/2024/04/10/tTcjCPmGEOXLxlB.png)