#!/usr/bin/env python3
"""FK V2: V1 81K位置 × 27手腕组合 → 选最佳朝向 → 7关节全解。"""
import mujoco, numpy as np, os, time, math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(SCRIPT_DIR, "../../challenge_cup_simulator/models/biped_s52/xml/biped_s52.xml")
LEG_L=[0.006,0,-0.503,0.862,-0.36,-0.006]; LEG_R=[-0.007,0,-0.502,0.861,-0.36,0.007]
EE=np.array([0.,0.,-0.17])

def down_angle(R):
    """夹爪局部Z与世界-Z的夹角(度)。0°=正朝下, 90°=水平, 180°=朝上。"""
    downZ = -R[2,2]  # dot(local_Z, world_-Z)
    return math.acos(max(-1.0, min(1.0, downZ)))

model=mujoco.MjModel.from_xml_path(XML); data=mujoco.MjData(model)
for i,v in enumerate(LEG_L): data.qpos[7+i]=v
for i,v in enumerate(LEG_R): data.qpos[13+i]=v
data.qpos[19]=data.qpos[42]=0;data.qpos[43]=0.349
L7=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"zarm_l7_link")
R7=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"zarm_r7_link")

v1_l=np.load(os.path.join(SCRIPT_DIR,"fk_table_left.npy"))
v1_r=np.load(os.path.join(SCRIPT_DIR,"fk_table_right.npy"))
n=len(v1_l)
print(f"V1 rows: {n}")

# 手腕3×3×3=27组合
HY=[55,75,95]; HP=[-5,20,45]; HR=[-25,0,25]
NCOL=7+3+1  # 7joints + XYZ + down_angle_rad
rL=np.zeros((n,NCOL),dtype=np.float32); rR=np.zeros((n,NCOL),dtype=np.float32)
t0=time.time()

for i in range(n):
    pL, rollL, yawL, foreL = v1_l[i,:4].astype(float)
    pR, rollR, yawR, foreR = v1_r[i,:4].astype(float)
    # V1表左右存同值; 右臂需镜像roll和yaw
    bestL=9e9; bestR=9e9; bLj=None; bRj=None

    for hy in HY:
        for hp in HP:
            for hr in HR:
                # Left arm
                data.qpos[20]=np.deg2rad(pL);data.qpos[21]=np.deg2rad(rollL)
                data.qpos[22]=np.deg2rad(yawL);data.qpos[23]=np.deg2rad(foreL)
                data.qpos[24]=np.deg2rad(hy);data.qpos[25]=np.deg2rad(hp);data.qpos[26]=np.deg2rad(hr)
                # Right arm — roll和yaw镜像
                data.qpos[35]=np.deg2rad(pR);data.qpos[36]=np.deg2rad(-rollR)
                data.qpos[37]=np.deg2rad(-yawR);data.qpos[38]=np.deg2rad(foreR)
                data.qpos[39]=np.deg2rad(hy);data.qpos[40]=np.deg2rad(hp);data.qpos[41]=np.deg2rad(hr)

                mujoco.mj_forward(model,data)
                bp=data.xpos[1]

                Rl=data.xmat[L7].reshape(3,3);el=data.xpos[L7]+Rl@EE-bp
                oriL=down_angle(Rl)
                if oriL<bestL:
                    bestL=oriL
                    bLj=[pL,rollL,yawL,foreL,hy,hp,hr,el[0],el[1],el[2]]

                Rr=data.xmat[R7].reshape(3,3);er=data.xpos[R7]+Rr@EE-bp
                oriR=down_angle(Rr)
                if oriR<bestR:
                    bestR=oriR
                    bRj=[pR,-rollR,-yawR,foreR,hy,hp,hr,er[0],er[1],er[2]]

    rL[i]=bLj+[bestL]
    rR[i]=bRj+[bestR]
    if i%20000==0: print(f"  {i}/{n} ({time.time()-t0:.0f}s)")

dt=time.time()-t0
print(f"\nDone {dt:.0f}s ({n*27/dt:.0f} FK/s)")
np.save(os.path.join(SCRIPT_DIR,"fk_table_left_v2.npy"),rL)
np.save(os.path.join(SCRIPT_DIR,"fk_table_right_v2.npy"),rR)
Ldeg=dict(zip(['min','max'],[f'{v:.0f}' for v in np.rad2deg([rL[:,10].min(),rL[:,10].max()])]))
Rdeg=dict(zip(['min','max'],[f'{v:.0f}' for v in np.rad2deg([rR[:,10].min(),rR[:,10].max()])]))
print(f"Left  down_angle=[{Ldeg['min']},{Ldeg['max']}]° (0°=perfect downwards)")
print(f"Right down_angle=[{Rdeg['min']},{Rdeg['max']}]°")

# Test
tx,ty,tz=0.338,0.186,-0.084
d=np.sqrt(3*(rL[:,7]-tx)**2+3*(rL[:,8]-ty)**2+(rL[:,9]-tz)**2)
pb=np.argmin(d); cb=np.argmin(d+rL[:,10]*0.15)
print(f"\nPos best: ee=({rL[pb,7]:.3f},{rL[pb,8]:.3f},{rL[pb,9]:.3f}) down={np.rad2deg(rL[pb,10]):.0f}° dist={d[pb]:.3f} j={rL[pb,:7]}")
print(f"Combo:    ee=({rL[cb,7]:.3f},{rL[cb,8]:.3f},{rL[cb,9]:.3f}) down={np.rad2deg(rL[cb,10]):.0f}° dist={d[cb]:.3f} j={rL[cb,:7]}")
