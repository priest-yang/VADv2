# Environment Configuration Document


## CUDA

Confirm `cuda-11.8` is installed

```bash
ls /usr/local
```
If `cuda-11.8` is not found, go to the [NVIDIA Official CUDA Download Page](https://developer.nvidia.com/cuda-toolkit-archive) to download CUDA 11.8. Select the appropriate operating system and installation method (usually a `.run` file or `.deb` package). For example, with `.run`:

```bash
sudo chmod +x cuda_11.8.*_linux.run # Grant executable permission to the .run file
sudo ./cuda_11.8.*_linux.run # Run the installation program
```

### Configure Environment Variables

After installing in `/usr/local`, you need to add the CUDA-related paths to the system's environment variables.

1. Edit the `.bashrc` file (or `.zshrc` file if you use Zsh):

   ```bash
   nano ~/.bashrc
   ```
   
2. Add the following content at the end of the file:

   ```bash
   export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
   export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   ```

3. Save and exit the editor, then load the new environment variables using the following command:

   ```bash
   source ~/.bashrc
   ```

## Create a Virtual Environment

Assume install `virtualenvwrapper` first

```bash
mkvirtualenv vadv2  --python=python3.8
```

## Set Mirror Source for Best Download Speed

```
pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
```

## Install Dependency Libraries
**Note: Torch requires cuda-11.8**

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r preinstall_requirements.txt
```

## Configure Environment Variables

```bash
export PATH=/usr/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
```

## Install mmcv Library

Note: It is necessary to compile mmcv on the local machine with `cuda-11.8`, otherwise it will report an error `error in ms_deformable_im2col_cuda: no kernal image is available for execution on the device`

```bash
python -m pip install -v -e .
```

## Download Pre-trained Models

Note: If `ckpts` already exists, skip this step

```bash
cd ckpts
wget https://hf-mirror.com/rethinklab/Bench2DriveZoo/resolve/main/resnet50-19c8e357.pth
wget https://hf-mirror.com/rethinklab/Bench2DriveZoo/resolve/main/r101_dcn_fcos3d_pretrain.pth
```
