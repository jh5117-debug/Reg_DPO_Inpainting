# Fix: DiffuEraser.forward() 签名 + Unified YAML 输出

## 问题 1: `run_BR.py` TypeError

**根因**: `diffueraser/diffueraser.py` 的 `DiffuEraser.forward()` 方法签名缺少 `prompt` 和 `n_prompt` 参数。
方法体第 330 行引用了未定义的 `prompt` 变量：`validation_prompt = prompt if prompt else ""`。
`run_BR.py` 第 546 行传递了 `prompt=prompt, n_prompt=n_prompt`，触发 `TypeError`。

**修复**:
- `forward()` 签名增加 `prompt=""`, `n_prompt=""` 参数
- 两处 pipeline 调用增加 `negative_prompt=validation_n_prompt` 传参

## 问题 2: `generate_captions.py` 统一 YAML 输出

**需求**: 批量生成 caption 时，所有视频结果合并为一个大 YAML 文件。

**修复**:
- 新增 `--unified_yaml` 参数，批处理完成后写入统一 YAML
- 默认路径: `{batch_output_dir}/all_captions.yaml`
- 仍保留 per-video YAML 以兼容旧流程

**`run_BR.py` 适配**:
- 新增 `--unified_prompt_yaml` 参数，支持从统一 YAML 读取 prompt
- 优先级: 统一 YAML > per-video YAML

## 涉及文件

| 文件 | 变更 |
|------|------|
| `diffueraser/diffueraser.py` | `forward()` 增加 `prompt`, `n_prompt` 参数 + `negative_prompt` 传参 |
| `generate_captions.py` | 新增 `--unified_yaml` + `write_unified_yaml()` |
| `run_BR.py` | 新增 `--unified_prompt_yaml` + 统一 YAML 加载逻辑 |

## 验证

- [x] 三个文件均通过 `py_compile` 语法检查
