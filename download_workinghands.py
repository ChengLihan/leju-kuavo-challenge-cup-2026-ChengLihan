from roboflow import Roboflow

rf = Roboflow(api_key="Hosfbi2fHEqsMh5laMRw")

project = rf.workspace("mechanical-tools").project("workinghands")

# 页面显示 Dataset versions: 2，所以优先试 version(2)
dataset = project.version(4).download("yolov8")

print("Downloaded to:", dataset.location)
