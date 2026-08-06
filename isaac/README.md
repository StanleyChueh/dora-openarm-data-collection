# Cartesian teleop: dora → Isaac Lab DifferentialIK

VR 末端姿態送進 Isaac Lab，用 `DifferentialIKController` 解 IK 驅動 OpenArm v1。
與既有的 dora `ik`（mink）路徑**並行**，不影響 MuJoCo 與 recorder。

```
dora ─┬─ ik (mink) ─────────────────► mujoco-collect + recorder      [不動]
      │
      └─ ros2-bridge ─┬─► ROS 2 topics                    給 rviz / ros2 topic echo
                      └─► UDP :5007  JSON                 給 Isaac Lab
                                  │
                        isaaclab_teleop.py                Isaac Sim 5.1 Python 3.11
                        2 × DifferentialIKController      (right / left)
```

## 為什麼走 UDP 而不是 rclpy

`DifferentialIKController` 要從 PhysX 的 articulation view 拿 Jacobian，**只能跑在 Isaac Sim 的直譯器裡**——不像 cuRobo 可以整個搬出去。而 Isaac Sim 5.x 是 **Python 3.11**，ROS 2 Humble 的 `rclpy` 是 **3.10** 編的，`import rclpy` 直接失敗。

payload 只有 16 個 float、100 Hz，stdlib UDP socket 沒有任何依賴問題。ROS topic 照發，除錯用。

## 檔案

| | |
|---|---|
| `make_isaac_urdf.py` | v1_camera.urdf → Isaac 可匯入的 URDF（產生器，不要手改輸出） |
| `inspect_usd.py` | 轉檔後檢查 ArticulationRootAPI 落在哪個 prim |
| `isaaclab_teleop.py` | Isaac Lab 主程式：UDP → 2× DiffIK → articulation |
| `check_frames.py` | V1：量 `arm_origin`；V2：檢查 Isaac URDF 的 grasp frame |
| `test_bridge_helpers.py` | V3：bridge.py 純函式測試（不需 dora/ROS） |

## 一次性設定

```bash
# 1. 產生 URDF（輸出到 package 根目錄，讓 assets/... 相對路徑對 Isaac importer 成立）
python3 isaac/make_isaac_urdf.py \
    openarm_description-main/assets/robot/openarm_v1.0/urdf/example/v1_camera.urdf \
    openarm_description-main/v1_camera_isaac.urdf

# 2. URDF → USD（在 IsaacLab checkout 底下跑，路徑用絕對路徑）
cd ${ISAACLAB}
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    ~/Ivan_ws/dora-openarm-data-collection/openarm_description-main/v1_camera_isaac.urdf \
    ~/Ivan_ws/dora-openarm-data-collection/openarm_description-main/v1_camera_isaac.usd \
    --fix-base --joint-target-type position
```

⚠️ **不要傳 `--merge-joints`**。它是 `store_true`、預設就是 False，所以「不傳 = 不合併」——這正是我們要的。合併會把 `openarm_{side}_grasp` / `hand_tcp` / camera frame 都吃掉。

⚠️ **路徑用絕對路徑**。`isaaclab.sh` 的 CWD 是 IsaacLab checkout，相對路徑會相對那裡解析而不是這個 repo。

```bash
# 3. 確認 ArticulationRootAPI 落在 default prim 上
${ISAACLAB}/isaaclab.sh -p isaac/inspect_usd.py <絕對路徑>/v1_camera_isaac.usd
```

沒過的話會直接告訴你 API 在哪個 prim、以及要怎麼修。

> 產生器預設會**移除 URDF 根部的空 `world` link**。`v1_camera.urdf` 用 ROS 慣例把
> `world --fixed--> openarm_body_link0` 當作錨點，但 `--fix-base` 已經在做這件事，
> 而一個沒有慣量、沒有幾何的根 link 會讓轉檔器把 `ArticulationRootAPI` 貼到子 prim 上，
> Isaac Lab 就會報 `Failed to find an articulation when resolving '.../Robot'`。
> 移掉之後 `openarm_body_link0` 才是真正的 root。要保留就加 `--keep-world-link`。

## Bring-up 順序

每一步都能單獨失敗，不要疊在一起除錯。

**V0 環境**
```bash
.venv-ros2/bin/python3 -c "import numpy, pyarrow, dora, rclpy; print('ok')"
${ISAACLAB}/isaaclab.sh -p -c "import sys, isaaclab; print(sys.version)"   # 預期 3.11
```

**V1 量 `arm_origin`（在 dora 主機上跑，不需 Isaac）**
```bash
python3 isaac/check_frames.py --scene demo
```
它會印出 `==> set ARM_ORIGIN_TO_BASE_Z=<z>`。**這一步不能跳過** —— bridge.py 預設的 `0.698` 是推論（v1 和 v2 的手臂都掛在 `xyz="0 ±0.031 0.698"`），不是量測。dora 跑的是 v2、Isaac 跑的是 v1，這個常數在銜接兩個不同模型。

如果它同時警告 rotation 不是 identity，那 bridge 只做 z 平移的假設就不成立，要回頭改。

**V2 Isaac URDF**
```bash
python3 isaac/check_frames.py --urdf openarm_description-main/v1_camera_isaac.urdf
```
確認 mesh 解析得到、grasp frame 位置合理。

**V3 bridge 純函式**（不需 dora / ROS）
```bash
.venv-ros2/bin/python3 isaac/test_bridge_helpers.py
```

**V4 dora + bridge**（不開 Isaac）
```bash
export ROS_DOMAIN_ID=1 && source /opt/ros/humble/setup.bash && ros2 daemon stop
dora start dataflow-vr-ros2-only.yaml --attach
# 另一個 terminal
ros2 topic hz /openarm/eef_pose        # 約 100 Hz，不是 ~1000
ros2 topic echo --once /openarm/eef_pose
```
三個斷言：
- `frame_id == "base_link"`、`len(poses) == 2`
- **兩支手把握在參考姿態**時 → `x ≈ -0.085, y ≈ 0, z ≈ ARM_ORIGIN_TO_BASE_Z - 0.14`
  （`FRAME_OFFSET_NECK = [-0.085, 0, -0.14]`）。z 讀到 `-0.14` 表示 offset 沒生效
- **遮住一支手把** → `ros2 topic hz` 應完全沒有輸出（確認沒有 identity 填補）

UDP 那條也順便看一下：
```bash
python3 -c "
import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(('0.0.0.0',5007))
print(s.recv(4096).decode())"
```

**V5 rviz**（不開 Isaac）
Fixed Frame `base_link`，加 `RobotModel` 與兩個 `Pose` display 指向 `/openarm/eef_pose/right`、`/left`。**兩個座標軸三腳架應落在手的位置**並跟著手把動。這一步抓掉所有 frame / 左右手 / 手性錯誤，還沒開 Isaac。

**V6 Isaac Lab dry run**
```bash
${ISAACLAB}/isaaclab.sh -p isaac/isaaclab_teleop.py \
    --usd openarm_description-main/v1_camera_isaac.usd --dry-run
```
確認機器人載入、兩隻手臂的 joint/body index 都解析得到（啟動時會印），以及 UDP 有收到東西。

**V7 全開** — 拿掉 `--dry-run`。單手慢慢移動幾公分 → 雙手 → 正常遙操作。

## 已知風險

1. **`arm_origin` 的旋轉假設**。bridge 只做 z 平移、四元數原封不動。v1 的手臂掛載 joint 是 `rpy="∓1.5708 0 0"` 而 v2 是 `rpy="0 0 0"`，如果兩個模型的 link7 frame 朝向不同，VR 的四元數慣例（`quest_receiver.py:159` 的 `_R_FRAME * r_rel * R_FIX`，是對著 v2 調的）就不會直接沿用。**症狀是「位置對、姿態差 90°」**，V5 在 rviz 就看得出來。

2. **grasp frame 可能不是 USD body**。`openarm_{side}_grasp` 是無質量 fixed frame，匯入後不保證成為獨立 rigid body。所以預設 IK 驅動的是 `openarm_{side}_link7`（一定存在、有慣量），再用 `--tool-offset-z 0.165` 把 grasp 目標退回 link7 目標。若你的匯入確實保留了 grasp frame，`--ee-body-fmt 'openarm_{side}_grasp' --tool-offset-z 0` 等價且少一次轉換。

3. **DiffIK 完全沒有碰撞感知**。局部 Jacobian 方法，不會避開自碰撞，接近奇異點也可能卡住。這是選它換來設定極簡的代價。要碰撞避免就得回去用 cuRobo。

4. **TCP `z = 0.165` 是估計值**。從 collision mesh 量出來的：finger mount 在 0.1025、指尖在 0.183，0.165 取在指墊附近。上真機前要實測。

5. **兩套 IK 會給出不同解**。mink（dora）和 DiffIK（Isaac）都是 Jacobian 阻尼最小平方，同一類演算法，所以行為應該相近——但 dataset 錄的是 mink 的解。若要用 Isaac 畫面訓練，錄到的 action 跟 Isaac 的關節軌跡仍不會完全一致。

6. **`openarm_description-main/` 目前沒有被 git 追蹤**。產生的 URDF/USD 也在裡面，換機器要記得一起帶。

## bridge.py 環境變數

| 變數 | 預設 | 用途 |
|---|---|---|
| `ARM_ORIGIN_TO_BASE_Z` | `0.698` | **用 V1 量到的值覆蓋** |
| `EEF_FRAME_ID` | `base_link` | eef_pose 的 frame_id |
| `EEF_RATE_HZ` | `100` | 限流（0 = 不限） |
| `EEF_STALE_SEC` | `0.25` | 超過就停發 |
| `GRIPPER_MODE` | `v1_prismatic` | 或 `v2_radian` 切回舊行為 |
| `ISAAC_UDP_HOST` / `ISAAC_UDP_PORT` | `127.0.0.1` / `5007` | port 設 0 關閉 UDP |
| `EEF_QOS` | `reliable` | 或 `best_effort` |
