## Build steps

- Install OpenCV [prebuilt binaries](https://github.com/opencv/opencv/releases) and set `OpenCV_DIR` according to the OpenCV installation location
- If building tests, set `LIBVIDSCRAMBLE_BUILD_TEST` to `ON` and specify  `Python3_ROOT_DIR`

Example cmake definitions:

```
-DOpenCV_DIR=E:/cpp_lib/opencv/opencv/build/ 
-DLIBVIDSCRAMBLE_BUILD_TEST=ON 
-DPython3_ROOT_DIR=E:/anaconda/envs/video-scramble
```

