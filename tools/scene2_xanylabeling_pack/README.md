# Scene2 X-AnyLabeling 标注工具包

本目录用于 Scene2 三类物体的本地标注。

## 1. 标注对象

只标注以下三个类别：

```text
screwdriver
pipe_clamp
pipe_fitting



---

## 7. 最终目录结构

执行完后，你会得到：

```text
tools/scene2_xanylabeling_pack/
    .venv_label/                         # 安装后生成
    configs/
        scene2_labels.txt                # 三类对象配置
    docs/
        scene2_annotation_rules.md       # 标注规则
    scripts/
        install_xanylabeling.sh          # 安装脚本
        start_scene2_labeling.sh         # 启动脚本
    export/                              # 可选：放导出数据
    README.md
