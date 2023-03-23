
## Dependencies

### AI Library

- [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [PyTorch 2.0.0 built with CUDA 11.8](https://pytorch.org/get-started/locally/)
- [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- requirements.txt

### Remote Control

- [PS4KMA](https://github.com/starshinata/PS4-Keyboard-and-Mouse-Adapter)
- [Official PS Remote Play 5.5.0](https://github.com/starshinata/PS4-Keyboard-and-Mouse-Adapter/blob/master/installers/remotePlay/RemotePlayInstaller-5.5.0.08250_Win32.zip) ([why?](https://github.com/starshinata/PS4-Keyboard-and-Mouse-Adapter/issues/85#issuecomment-1466988088))

> on remote play v6 it seems like Sony is blocking Vigem (it is unclear if Sony did it intentionally or not)
>
> ----
> If people have an alternative way of emulating DS4 controllers i am interested in hearing
> 
> Work around in previous post seems to work
>
> Warning that this means you would be running a modded Remote Play exe
YMMV
>
> steps for workaround
> 
> 1. Download remote play v5.5
>    eg from https://github.com/starshinata/PS4-Keyboard-and-Mouse-Adapter/blob/master/installers/remotePlay/RemotePlayInstaller-5.5.0.08250_Win32.zip
> 2. extract and run the RemotePlayInstaller_5.5.0.08250_Win32.msi file
> 3. Get "xeropresence/remoteplay-version-patcher" from https://github.com/xeropresence/remoteplay-version-patcher/releases/download/1.0.0/Release.zip
> 4. extract and run the remoteplay-version-patcher.exe file
> 5. run remote play via PS4KMA and it works
