# llama2-infer

学习 <https://github.com/zjhellofss/KuiperInfer>，从零实现一个深度学习推理框架的练手实验。

## 编译

```
mkdir build && cd build
cmake ..
make
```

## 模型转换

读取huggingface格式模型，直接生成tokenizer, param和bin文件，使用了ncnn的param格式，便于netron等工具查看。

```
python export_model.py models/stories42M stories42M
```

## 推理测试

以转换时最后一个参数为stories42M为例，转换后会在当前文件夹下生成stories42M.ncnn.param, stories42M.ncnn.bin, stories42M_tokenizer.bin这3个文件。推理命令如下

```
./inference stories42M "Once upon a time" 256 0.8
```

## 参考

- <https://github.com/zjhellofss/KuiperInfer>

- <https://github.com/lrw04/llama2.c-to-ncnn>

- <https://github.com/karpathy/llama2.c>
