# VR → Isaac Sim → OpenArm v1：Repo 解析與橋接實作指南

> 對象：`D:\Dora_VR\dora-openarm-data-collection`（fork 自 `enactic/dora-openarm-data-collection`）
> 目標：把 Meta Quest 的 VR 控制器輸入，經由 dora-rs 資料流送進 **NVIDIA Isaac Sim**，驅動 **OpenArm v1** 雙臂模型。
> 撰寫日期：2026-08-05

---

## 0. 三句話結論

1. **這個 repo 本身不是 VR 專案，是「資料收集」專案**。它用 dora-rs 把「VR 輸入 → IK → 手臂 → 相機 → 錄檔」串成一條 pipeline，Isaac Sim 只是這條 pipeline 尾端「執行器 + 觀測來源」的一個候選實作。
2. **接 Isaac Sim 不需要動 VR / IK 端**。上游已經幫你把 VR 姿態轉成關節角了：`ik/position_left`、`ik/position_right` 各是 8 維 `float32`（7 軸 + 夾爪）。你要做的只是把這兩個 topic 餵進 Isaac Sim 的 Articulation。
3. **repo 裡已經有一個半成品答案**：`nodes/dora-openarm-ros2-bridge/bridge.py` 把 Dora 的關節命令轉成 ROS 2 `JointState` 發到 `/openarm/vr_joint_command`。**這就是接 Isaac Sim 最短的路徑**，但目前它有三個已知問題（見 §6.3），必須先修。

---

## 1. Repo 現況盤點

### 1.1 目錄結構

```
dora-openarm-data-collection/
├── dataflow_dummy.yaml            # 純假資料，CI 用
├── dataflow-ker.yaml              # KER 外骨骼 leader → 真實 OpenArm
├── dataflow-vr.yaml               # VR → IK → 真實 OpenArm（+ MuJoCo viewer 旁觀）
├── dataflow-vr-mujoco.yaml        # VR → IK → MuJoCo（純模擬，可錄資料）
├── dataflow-vr-mujoco-ros2.yaml   # VR → IK → MuJoCo + ROS 2 bridge   ← 你目前的主線
├── dataflow-vr-ros2-only.yaml     # VR → IK → ROS 2 bridge（不開 MuJoCo）← 接 Isaac Sim 的基底
├── metadata.yaml                  # 真機用 metadata（含 cell lifter）
├── metadata_mujoco.yaml           # 模擬用 metadata（無 lifter）
├── MUJOCO_LOG.TXT                 # MuJoCo 執行紀錄（含 GLFW 失敗訊息）
├── vr_mujoco_data/dataset/        # 已錄製的 episode 0
├── .venv-ros2/                    # 給 ROS 2 bridge 用的 venv（見 §6.2）
└── nodes/
    ├── dora-openarm-ros2-bridge/  # ★ 本地新增，非 submodule
    │   ├── bridge.py
    │   └── run_bridge.sh
    └── (其餘 13 個都是 git submodule)
```

### 1.2 ⚠️ Submodule 目前全部沒有 checkout

```
$ git submodule status
-d988dd24... nodes/dora-openarm
-a4a2be34... nodes/dora-openarm-vr
-357eb4f0... nodes/dora-openarm-kinematics
-2bdf35f5... nodes/dora-openarm-mujoco
...（13 個，開頭的 "-" 代表未初始化）
```

`nodes/` 下除了 `dora-openarm-ros2-bridge` 之外全是空目錄。在 Windows 這台機器上你**無法** `dora build`。要看原始碼或執行，先在 Linux 機器上：

```bash
git submodule update --init --recursive
```

本文中關於各 submodule 的介面說明，來自上游 GitHub README；**標註「需驗證」的部分請以你實機 checkout 的版本為準**（見 §6.3 的版本漂移問題）。

### 1.3 Git 狀態

| commit | 說明 |
|---|---|
| `d98ed13` | **MuJoCo first run**（本地，StanleyChueh，2026-08-04）— 新增 ROS 2 bridge、兩個新 dataflow、錄好的 dataset、`.venv-ros2` |
| `8fcdb6b` | 上游最後一筆 — Switch pose origin and accept pose recording (#23) |

`d98ed13` 除了新增檔案，還**回退**了三處上游改動：

- `udp-receiver` 的 `joystick_y_left` → 改回 `joystick_y`
- 移除了 `vr_receive_times`（VR 延遲視覺化）
- `actions/setup-python@v7` → `v6`

這代表**你本地 checkout 的 submodule 版本比 `.gitmodules` 指標所指的舊**。後面 §6.3 的資料格式風險與此直接相關。

---

## 2. dora-rs 架構速覽

dora 是一個以 YAML 描述的資料流框架：每個 `node` 是獨立行程，用 shared-memory + Apache Arrow 傳資料，靠 `inputs`/`outputs` 的字串名稱連線。

```yaml
- id: ik                                  # 節點名
  build: pip install -e nodes/xxx          # dora build 時執行
  path: dora-openarm-ik                    # 可執行檔名（或腳本路徑）
  args: "--mode bimanual ..."              # 傳給該可執行檔的參數
  inputs:
    target_right: udp-receiver/pose_right  # 訂閱 <node_id>/<output_name>
  outputs:
    - position_right                       # 宣告自己會發的 topic
```

關鍵觀念：

- **`dora/timer/millis/N`** 是內建計時器來源。
- **`queue_size`** 控制輸入佇列深度；影像設 1（丟舊留新），按鈕設 100（不能漏）。
- 節點是行程，所以 `path` 可以是任何可執行檔——**這就是 ROS 2 bridge 用 shell script 混入不同 Python 環境的手法**（§6.2）。

---

## 3. 節點逐一解說

### 3.1 `udp-receiver`（submodule: `dora-openarm-vr`）

- 可執行檔：`dora-openarm-quest-receiver`，參數 `--host 0.0.0.0 --port 5006`
- Quest 3 上跑一支 sideload 的 APK，透過 **UDP** 把控制器狀態打到 PC 的 5006 port。
- 輸出：

| output | 型別 | 說明 |
|---|---|---|
| `pose_right` / `pose_left` | `float32[7]` | `[x, y, z, qw, qx, qy, qz]`，相對於 origin frame |
| `trigger_right` / `trigger_left` | `float32[1]` | 板機類比值 → 夾爪開合 |
| `joystick_y` | `float32[1]` | 搖桿 Y → 升降柱（cell lifter） |
| `button_a` / `button_b` | `bool` | 錄製控制（成功 / 失敗）|
| `button_x` | `bool` | 重置模擬場景 |
| `status` | — | 節點狀態 |

Quest 端注意事項（來自上游 README）：
- 需先裝 Meta Quest Developer Hub、建開發者帳號、sideload APK。
- 每次啟動：在 APK 左側選單輸入 PC 的 IP 與 port `5006`。
- **把眼部感測器貼起來**避免頭盔休眠；操作時頭盔可掛在脖子上。
- **安全**：一開始請「緩慢輕拉板機」讓機械臂對位，再做大動作。

### 3.2 `ik`（submodule: `dora-openarm-kinematics`）

- 可執行檔：`dora-openarm-ik`
- 演算法：**mink**（QP-based differential IK）+ MuJoCo，雙臂在**同一個 QP 中一次解完**。
- repo 中使用的參數：

```
--mode bimanual --max-iters 10 --dt 0.1 --damping 0.1 --posture-cost 0.01 --lm-damping 0.01
```

| 參數 | 本 repo 值 | 上游預設 | 意義 |
|---|---|---|---|
| `--mode` | `bimanual` | `bimanual` | 左/右/雙臂 |
| `--max-iters` | 10 | 5 | 每次事件迭代次數（↑ 收斂好但慢）|
| `--dt` | 0.1 | 0.5 | 積分步長（↓ 平滑但跟隨慢）|
| `--damping` | 0.1 | 1e-3 | 全域 Tikhonov 正則（↑ 穩定但鈍）|
| `--posture-cost` | 0.01 | 0.0 | 姿態任務權重，抑制冗餘自由度亂飄 |
| `--lm-damping` | 0.01 | 1e-4 | 各任務 LM 阻尼 |

> 注意本 repo 的 damping 明顯**比上游預設保守 100 倍**，`dt` 也小 5 倍。這是為了 500 Hz teleop 下的穩定性；改動前請先理解代價（跟隨延遲）。

- 輸入：`tick`（500 Hz）、`target_right/left`（VR pose）、`trigger_right/left`（夾爪）
- 輸出：**`position_left` / `position_right`，`float32[8]` = joint1..7 + gripper**

**這 8 維就是整個 Isaac Sim 橋接的核心契約。**

### 3.3 `quittable-tick-leader` / `quittable-tick-camera`（submodule: `dora-openarm-quitter`）

把 `dora/timer` 轉發成 tick，但收到 UI 的 `quit` command 就停止發送 → 讓整條 pipeline 優雅收工。

- leader tick：VR 系列是 **2 ms（500 Hz）**；dummy/ker 是 4 ms（250 Hz）
- camera tick：**33 ms（30 fps）**

### 3.4 `mujoco-collect` / `mujoco-viewer`（submodule: `dora-openarm-mujoco`）

目前的模擬後端。**這正是 Isaac Sim 要取代（或並存）的節點。**

參數：`--scene demo --keyframe home --enable-collision --ctrl --viewer --render`

| 參數 | 意義 |
|---|---|
| `--scene` | 內建 MJCF 場景：`cell` / `demo` / `pedestal` / `bimanual`（`--xml` 可指定自訂檔）|
| `--keyframe home` | 啟動時載入的關鍵影格 |
| `--enable-collision` | 開接觸偵測（預設關，避免 teleop 時關節被卡死）|
| `--ctrl` | 寫 `data.ctrl` 並跑物理；不加則直接寫 `data.qpos`（純運動學）|
| `--viewer` | 開互動視窗（需 display）|
| `--render` | 開離屏渲染並發布 JPEG 影格 |
| `--debug-frames` | 把 VR 控制器姿態畫成彩色箭頭（僅 viewer）|
| `--origin-frame arm_origin` | VR pose 的參考座標系（預設 site 型別）|

輸入輸出：

| 方向 | 名稱 | 型別 |
|---|---|---|
| in | `position_left/right` | `float32[8]`，~500 Hz |
| in | `pose_left/right` | `float32[7]`，僅供除錯視覺化 |
| in | `button_x` | 邊緣觸發 → 非手臂物件（自由關節、抽屜門等）回到 keyframe |
| out | `arm_left/right_observation` | `float32[8]` 實際關節位置 |
| out | `camera_wrist_left/right`, `camera_head_left/right`, `camera_ceiling` | JPEG `uint8[N]`，~30 Hz |

> `MUJOCO_LOG.TXT` 裡反覆出現 `ERROR: could not initialize GLFW`——那是無頭環境下開 viewer 失敗。`dataflow-vr-mujoco.yaml` 用 `DISPLAY: ":0"` 解，若跑 Xvfb 要改成對應的 display 號（上游原本寫 `:1`）。

### 3.5 `follower-right` / `follower-left`（submodule: `dora-openarm`）

**真實硬體驅動**。CAN bus + Damiao 馬達。

- 參數：`--side {left,right} --align-trigger gripper`，另有 `--align-threshold`（預設 0.1 rad）、`--align-delta-limit`（預設 0.001）
- 輸入：`request_state`（只用事件 ID 觸發查詢）、`move_position`（`qpos` 陣列）、`command`
- 輸出：`state`（關節位置/速度/力矩/MOS 與轉子溫度）、`status`（`stopped` / `started` / `aligned`）

**安全機制**：`--align` 預設開啟，機械臂會先慢慢對齊到目標位置（每次最多動 `align-delta-limit`）才進入 `aligned` 狀態接受全速命令。做 Isaac Sim → 真機 sim-to-real 時，這個對位邏輯必須保留。

### 3.6 `recorder`（submodule: `dora-openarm-dataset-recorder`）

env：`METADATA_FILE`、`DIRECTORY`。輸出格式（實測 `vr_mujoco_data/`）：

```
vr_mujoco_data/dataset/
├── metadata.yaml
└── episodes/0/
    ├── action/arms/{left,right}/qpos.parquet     # IK 送出的命令
    ├── obs/arms/{left,right}/qpos.parquet        # 模擬/真機回報的觀測
    └── cameras/{ceiling,head_left,head_right,wrist_left,wrist_right}/<ns_timestamp>.jpeg
```

`dataset/metadata.yaml` 實測內容（OpenArmArmDataset **v0.3.0**）：

```yaml
version: 0.3.0
operation_type: teleop
episodes: [{id: '0', success: true, task_index: 0}]
frequencies:
  action: {arms: {left: 500.0, right: 500.0}}   # ← 500 Hz，符合 2 ms tick
  obs:    {arms: {}}                             # ← 空的（見下方警告）
  cameras: {}                                    # ← 空的
```

> ⚠️ **這次錄製的 obs 與 cameras 頻率是空的**。commit message 說 "MuJoCo first run"，代表 `--render` 的影像可能沒真正流進 recorder（GLFW / DISPLAY 問題）。做 Isaac Sim 版本前，先確認這條錄製鏈是完整的，否則會錄出一堆只有 action 沒有 observation 的資料。

### 3.7 其餘節點

| 節點 | 用途 |
|---|---|
| `ui`（`dora-openarm-data-collection-ui`）| Web UI，**HTTP :8000**，`POST /start`、`/success`、`/quit`（見 `.github/workflows/test.yaml`）。也接 VR 的 `button_a`/`button_b` 做無鍵盤操作 |
| `lifter`（`dora-openarm-cell-lifter`）| 搖桿 Y → 升降柱，lead screw、行程 300 mm |
| `camera-head-stereo-splitter`（`dora-opencv-image-splitter`）| 2560×720 雙目影像垂直切成兩張 |
| `leader`（`dora-openarm-ker`）| KER 外骨骼 leader 裝置（非 VR 路線）|
| `dora-openarm-dummy*` | CI 用假資料節點 |

---

## 4. ★ 資料契約總表（做 Isaac Sim 節點只要看這張）

| Topic | 產生者 | 型別 | 頻率 | 語意 |
|---|---|---|---|---|
| `pose_right` / `pose_left` | udp-receiver | `float32[7]` | ~VR frame rate | `[x,y,z,qw,qx,qy,qz]`，公尺 + 四元數（w 在前）|
| `trigger_right` / `trigger_left` | udp-receiver | `float32[1]` | 同上 | 0..1 |
| `joystick_y` | udp-receiver | `float32[1]` | 同上 | -1..1 |
| `button_a` / `b` / `x` | udp-receiver | `bool` | 事件 | 錄製成功 / 失敗 / 重置 |
| **`position_right` / `position_left`** | **ik** | **`float32[8]`** | **500 Hz** | **joint1..7 (rad) + gripper** |
| `arm_*_observation` | mujoco / follower | `float32[8]` | 同步於命令 | 實際關節狀態 |
| `camera_*` | mujoco / opencv | JPEG `uint8[N]` | 30 Hz | metadata 含 `encoding: jpeg` |

**單位**：關節角 **radian**；VR 位置 **公尺**；四元數 **`[w,x,y,z]` 順序（w 在前）**——這與 ROS 2 的 `[x,y,z,w]` 相反，做座標轉換時是最常見的踩雷點。

---

## 5. 五個 dataflow 對照

| 檔案 | VR | IK | 模擬 | 真機 | 相機 | 錄製 | ROS 2 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `dataflow_dummy.yaml` | — | — | — | dummy | dummy | ✅ | — |
| `dataflow-ker.yaml` | KER | — | — | ✅ | 實體 | ✅ | — |
| `dataflow-vr.yaml` | ✅ | ✅ | viewer 旁觀 | ✅ | 實體 | ✅ | — |
| `dataflow-vr-mujoco.yaml` | ✅ | ✅ | ✅ 主體 | — | MuJoCo 渲染 | ✅ | — |
| `dataflow-vr-mujoco-ros2.yaml` | ✅ | ✅ | ✅ 主體 | — | MuJoCo 渲染 | ✅ | ✅ |
| `dataflow-vr-ros2-only.yaml` | ✅ | ✅ | — | — | — | — | ✅ |

**`dataflow-vr-ros2-only.yaml` 就是你要拿來接 Isaac Sim 的基底**——它已經把 MuJoCo 和 recorder 都拿掉，只剩 VR → IK → ROS 2。

管線視覺化（ros2-only）：

```
Quest APK --UDP:5006--> udp-receiver --pose[7]--> ik --qpos[8]--> ros2-bridge --JointState--> /openarm/vr_joint_command
                             |                     ^
                             +--trigger------------+
   quittable-tick-leader (500 Hz) --tick--> udp-receiver, ik
   ui (:8000) --command--> quittable-tick-leader
```

---

## 6. 現有 ROS 2 Bridge 解析

### 6.1 `bridge.py` 做了什麼

[bridge.py](../nodes/dora-openarm-ros2-bridge/bridge.py) 同時是 dora node 和 ROS 2 node：

```python
ros_node = rclpy.create_node("dora_openarm_ros2_bridge")
publisher = ros_node.create_publisher(JointState, "/openarm/vr_joint_command", 1)
dora_node = DoraNode()

for event in dora_node:                    # 阻塞在 dora 事件迴圈
    values = arrow_to_float_list(event["value"])
    if event["id"] == "position_left":  latest_left = values;  left_updated = True
    elif event["id"] == "position_right": latest_right = values; right_updated = True

    if not (left_updated and right_updated): continue   # 等左右都更新才發

    msg.name     = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES   # 14 個
    msg.position = latest_left[:7] + latest_right[:7]             # 只取 7 軸，丟掉夾爪
    publisher.publish(msg)
    rclpy.spin_once(ros_node, timeout_sec=0.0)
```

關節命名（對應 `openarm_description` 的雙臂 URDF）：

```
openarm_left_joint1 .. openarm_left_joint7,  openarm_right_joint1 .. openarm_right_joint7
```

設計決策：
- **左右配對發布**：兩側都收到新資料才發一次 → 保證同一則 `JointState` 內左右同步，代價是把 500 Hz 降成有效約 500 Hz 的配對速率（左右各 500 Hz 交錯 → 仍約 500 Hz，但可能有 ≤2 ms 的時間偏移）。
- **`queue depth = 1`**：只留最新命令，適合即時控制。
- **夾爪被丟棄**：`[:7]` 明確捨去第 8 維，程式碼註解說「之後再確認兩指映射」。

### 6.2 執行環境的巧思

`run_bridge.sh`：

```bash
source /opt/ros/humble/setup.bash
exec /home/csl/Ivan_ws/.../.venv-ros2/bin/python -u .../bridge.py
```

`.venv-ros2/pyvenv.cfg` 有 **`include-system-site-packages = true`**，所以：
- `rclpy`（來自 `/opt/ros/humble`，系統 Python 3.10）看得到 ✅
- `dora-rs 0.5.0`、`pyarrow 25.0.0`（裝在 venv 裡）也看得到 ✅

這是把 ROS 2 和 dora 塞進同一個行程的正確做法。**Isaac Sim 版本要沿用同樣的思路**（只是 Isaac Sim 端另有自己的 Python）。

### 6.3 ⚠️ 三個必修問題

**(1) 路徑不一致 — 會直接啟動失敗**

| 檔案 | 路徑 |
|---|---|
| `run_bridge.sh` | `.venv-ros2/bin/python` |
| `dataflow-vr-ros2-only.yaml:74` | `.venv_ros2/bin/python3` ← **底線，不是連字號** |

`dataflow-vr-ros2-only.yaml` 的路徑不存在。而且兩處都寫死 `/home/csl/Ivan_ws/...` 絕對路徑，換機器就爛。

修法（統一走 `run_bridge.sh`，並改用相對路徑）：

```yaml
  - id: ros2-bridge
    path: ./nodes/dora-openarm-ros2-bridge/run_bridge.sh
    inputs:
      position_left: ik/position_left
      position_right: ik/position_right
```

```bash
#!/usr/bin/env bash
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source /opt/ros/humble/setup.bash
exec "$REPO/.venv-ros2/bin/python" -u "$HERE/bridge.py"
```

**(2) Arrow 解碼可能不匹配 — 需驗證**

`arrow_to_float_list()` 假設 `event["value"]` 是**扁平**的 float 陣列：

```python
return [float(item) for item in value.to_pylist()]
```

但上游 `dora-openarm-kinematics` 的 README 說輸出是 **struct 陣列** `[{"qpos": float32[8]}]`。若你 checkout 到的是新版，`to_pylist()` 會回 `[{'qpos': [...]}]`，`float(dict)` 直接丟 `TypeError`，**而且會在長度檢查之前爆掉**，所以你不會看到那句「應為 8 維」的友善警告。

先加一段自我診斷再決定：

```python
def arrow_to_float_list(value) -> list[float]:
    items = value.to_pylist()
    # struct 格式：[{"qpos": [...]}]
    if items and isinstance(items[0], dict):
        return [float(x) for x in items[0]["qpos"]]
    return [float(x) for x in items]
```

**(3) 夾爪整條斷掉**

`msg.position = latest_left[:7] + latest_right[:7]`。VR 板機的開合指令到此為止，Isaac Sim 那端永遠拿不到。做 pick-and-place 任務時這是致命的——見 §7.4 的補法。

### 6.4 其他觀察

- **`.venv-ros2/` 被 commit 進 git** 了（含 45 MB 的 `dora.abi3.so`）。`.gitignore` 只擋 `/dataset/` 和 `/out/`。建議加上 `.venv*/` 並 `git rm -r --cached .venv-ros2`。
- `ros2-bridge` 節點沒有 `build:` 欄位，所以 `dora build` 不會處理它——venv 要手動建。這是刻意的，但要寫進 README 否則別人 clone 下來會卡住。

---

## 7. 接 Isaac Sim：三個方案

### 7.1 方案比較

| | A. ROS 2 Bridge（推薦）| B. Dora 原生節點 | C. Isaac Lab 環境 |
|---|---|---|---|
| 做法 | 沿用 `bridge.py`，Isaac Sim 用 ROS 2 Bridge 訂閱 `JointState` | 在 Isaac Sim 的 Python 裡直接跑 dora node | 把 dora 命令餵進 `openarm_isaac_lab` 的 env |
| Isaac Sim 端工作量 | **低**（OmniGraph 拉幾個節點）| 中（要把 `dora-rs` 裝進 Isaac Sim Python 3.11）| 高 |
| 延遲 | +1~3 ms（DDS）| 最低 | 中 |
| 除錯 | 最好（`ros2 topic echo` 直接看）| 差 | 中 |
| 已有基礎 | ✅ `bridge.py` + `dataflow-vr-ros2-only.yaml` | ❌ | ❌ |
| 未來接真機 | ✅ 同一個 topic 可同時餵 sim 和 `follower-*` | 需重寫 | 需重寫 |

**建議走 A**。除非之後量測出 DDS 延遲真的成為瓶頸（500 Hz = 2 ms 週期，DDS 抖動確實可能吃掉一部分），再考慮 B。

### 7.2 方案 A 的完整架構

```
┌─ PC / Linux ────────────────────────────────────────────────┐
│                                                              │
│  Quest 3 ──UDP:5006──> [udp-receiver] ──pose[7]──> [ik]      │
│                                                      │        │
│                                                 qpos[8] ×2    │
│                                                      ▼        │
│                                            [ros2-bridge]      │
│                                                      │        │
│                       ROS 2 DDS  /openarm/vr_joint_command    │
│                                                      │        │
│  ┌───────────────────────────────────────────────────▼─────┐ │
│  │ Isaac Sim 5.1                                            │ │
│  │  ROS2 Subscribe JointState → Articulation Controller     │ │
│  │  OpenArm v1 bimanual USD (從 openarm_description 匯入)    │ │
│  │  ROS2 Publish JointState → /openarm/joint_states  ────┐  │ │
│  │  ROS2 Camera Helper → /openarm/camera/*            ┐  │  │ │
│  └────────────────────────────────────────────────────┼──┼──┘ │
│                                                       │  │    │
│  [isaac-obs-bridge] <─────────────────────────────────┴──┘    │
│         │ arm_*_observation, camera_*                          │
│         ▼                                                      │
│    [recorder] ──> dataset/                                     │
└──────────────────────────────────────────────────────────────┘
```

注意這是**雙向**的：命令下行給 Isaac Sim，觀測與影像上行回 dora 給 recorder。只做下行的話錄不到 observation（就是 §3.6 那個問題）。

### 7.3 實作步驟

#### Step 1 — 準備 OpenArm v1 的 USD

```bash
# 1) 產生 v1 雙臂 URDF
git clone https://github.com/enactic/openarm_description
# 依 https://docs.openarm.dev/software/description 產生 v1 bimanual URDF
```

在 Isaac Sim 裡用 **URDF Importer**（`Tools > Robotics > URDF Importer`）匯入，重點設定：

- ❌ **不要**勾 `Fix Base Link`（除非 OpenArm 是固定底座——雙臂 pedestal 的話要勾）
- ✅ 勾 `Import as Reference`，輸出成獨立 USD
- **Joint Drive Type: `Position`**
- **Stiffness ≫ Damping**（位置控制必要條件；Isaac Sim 文件明確要求）。起手值：`stiffness=1e5`、`damping=1e3`，之後依實際跟隨誤差調
- 匯入後在 Stage 裡**確認 joint 的 prim 名稱**是不是 `openarm_left_joint1` … 這決定了 `bridge.py` 的名稱能不能對上

#### Step 2 — Isaac Sim 端 OmniGraph

最快：`Tools > Robotics > ROS 2 OmniGraphs > JointStates`（會一併幫你加 Articulation Controller）。

手動接線的話，連成：

```
On Playback Tick ──exec──> ROS2 Subscribe JointState ──> Articulation Controller
                    │                                          ▲
                    ├──exec──> ROS2 Publish JointState          │
                    │                                    (targetPrim = OpenArm root)
Isaac Read Simulation Time ──> (Publish JointState).timestamp
```

- `ROS2 Subscribe JointState` 的 **topicName 改成 `/openarm/vr_joint_command`**
- `ROS2 Publish JointState` 的 topicName 設 `/openarm/joint_states`
- Articulation Controller 的 `targetPrim` 指到 OpenArm 的 articulation root

#### Step 3 — 環境變數與 DDS

Isaac Sim 和 dora bridge 必須在同一個 ROS 2 domain：

```bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

⚠️ **Isaac Sim 5.1 內建的是 ROS 2 Humble 的 DDS 函式庫**，而 `run_bridge.sh` source 的也是 Humble——版本一致，很好。若你的 Isaac Sim 用 Jazzy，兩邊 RMW 要對齊否則會「topic 看得到但收不到訊息」。

#### Step 4 — 驗證順序（一定要照這個順序）

```bash
# 4-1. 先確認 dora 端有發東西（不要開 Isaac Sim）
dora run dataflow-vr-ros2-only.yaml
```

```bash
# 4-2. 另一個 terminal
source /opt/ros/humble/setup.bash
ros2 topic hz /openarm/vr_joint_command      # 期待 ~500 Hz
```

```bash
ros2 topic echo /openarm/vr_joint_command --once   # 檢查 name 有 14 個、position 是合理弧度
```

```bash
# 4-3. 都對了再開 Isaac Sim，按 Play
```

**如果 4-2 沒有輸出**，問題一定在 §6.3 的 (1) 或 (2)。

### 7.4 補回夾爪（必做）

OpenArm 的夾爪在 URDF 裡通常是**兩個 finger joint**（左右對稱）。把 `bridge.py` 改成：

```python
LEFT_GRIPPER_JOINT_NAMES  = ["openarm_left_finger_joint1",  "openarm_left_finger_joint2"]
RIGHT_GRIPPER_JOINT_NAMES = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]

# gripper 值 → 兩指鏡像
lg, rg = latest_left[7], latest_right[7]
msg.name = (LEFT_ARM_JOINT_NAMES + LEFT_GRIPPER_JOINT_NAMES
            + RIGHT_ARM_JOINT_NAMES + RIGHT_GRIPPER_JOINT_NAMES)
msg.position = (latest_left[:7]  + [lg, -lg]      # 符號依 URDF 的 axis 定義調整
                + latest_right[:7] + [rg, -rg])
```

> `[lg, -lg]` 的符號、以及 IK 輸出的 gripper 值域（0..1？還是關節弧度？）**都需要實測確認**。做法：`ros2 topic echo` 看 IK 送出的第 8 維在你完全放開/完全拉滿板機時的兩個端點值，再對照 URDF 裡該 joint 的 `<limit lower= upper=>`。

### 7.5 觀測回流節點（要錄資料才需要）

新增一個 `isaac-obs-bridge`（結構跟 `bridge.py` 對稱，方向相反）：

```python
# 訂閱 ROS 2 /openarm/joint_states → 發 dora arm_left/right_observation (float32[8])
# 訂閱 ROS 2 /openarm/camera/* (sensor_msgs/CompressedImage) → 發 dora camera_* (JPEG uint8[N])
```

dataflow 裡接成：

```yaml
  - id: isaac-obs-bridge
    path: ./nodes/dora-openarm-isaac-obs/run.sh
    inputs:
      tick: quittable-tick-camera/tick
    outputs: [arm_left_observation, arm_right_observation,
              camera_wrist_left, camera_wrist_right,
              camera_head_left, camera_head_right, camera_ceiling]

  - id: recorder
    build: pip install -e nodes/dora-openarm-dataset-recorder
    path: dora-openarm-dataset-recorder
    env:
      METADATA_FILE: "metadata_isaac.yaml"
      DIRECTORY: "vr_isaac_data"
    inputs:
      command: ui/command
      arm_left_action:  ik/position_left
      arm_right_action: ik/position_right
      arm_left_observation:  isaac-obs-bridge/arm_left_observation
      arm_right_observation: isaac-obs-bridge/arm_right_observation
      camera_wrist_left:  isaac-obs-bridge/camera_wrist_left
      # ...其餘相機同理
```

這樣錄出來的 dataset 格式跟 MuJoCo 版本完全一致，訓練 pipeline 不用改。

---

## 8. OpenArm v1 vs v2 — 必須先釐清

**你說要控制 v1，但 repo 裡所有 metadata 都寫 `version: "2.0"`：**

```yaml
# metadata.yaml 與 metadata_mujoco.yaml 皆為
equipment:
  embodiments:
    arms:
      id: OpenArm
      version: "2.0"
```

這帶出三個必須確認的點：

1. **IK 用的模型是哪一版？** `dora-openarm-kinematics` 從 `openarm_control` 套件載入 FK/IK 模型。若那是 v2 的 MJCF/URDF，而 Isaac Sim 載的是 v1 USD，**連桿長度與關節極限會不一致**，IK 解出來的角度送進 v1 會產生位置誤差（末端幾公分等級）。
   - 檢查：`git submodule update --init` 後看 `dora-openarm-kinematics` 依賴的模型檔；或 `pip show openarm-control` 找到安裝路徑後檢視 URDF。
2. **關節命名是否相同？** `bridge.py` 寫死 `openarm_{left,right}_joint1..7`。v1 的 URDF 若用不同前綴（例如 `openarm_v1_...`），Isaac Sim 的 subscriber 會靜默丟棄整則訊息——**不會報錯，只是機械臂不動**。這是最難除錯的失敗模式。
3. **metadata 要改**。若真的錄 v1 資料，新建 `metadata_isaac.yaml` 並把 `version` 改成 `"1.0"`，否則資料集標註錯誤，日後訓練/發表都會有問題。

**建議**：如果 IK 模型是 v2 而硬體是 v1，最乾淨的解法是**在 Isaac Sim 裡也載 v2 模型**先把 pipeline 打通，再回頭處理 v1 的模型替換（可能要改 `openarm_control` 的模型路徑）。

---

## 9. 陷阱清單

| # | 陷阱 | 症狀 | 對策 |
|---|---|---|---|
| 1 | 四元數順序 | 機械臂姿態整個歪掉 | dora pose 是 `[w,x,y,z]`，ROS 是 `[x,y,z,w]`。目前 `bridge.py` 不傳 pose 所以沒事，但你若寫方案 B 一定會遇到 |
| 2 | 關節名稱不匹配 | **完全不動，無錯誤訊息** | 用 `ros2 topic echo /openarm/joint_states` 看 Isaac Sim 回報的名稱，逐字比對 |
| 3 | Stiffness/Damping 設錯 | 手臂軟趴趴或劇烈震盪 | 位置控制需 stiffness ≫ damping；速度控制需 stiffness = 0 |
| 4 | 500 Hz 對 Isaac Sim 太快 | Isaac Sim 掉幀、CPU 打滿 | Isaac Sim 物理通常 60~240 Hz。可在 bridge 裡做降頻（每 N 筆發一次），或另開一條 `dora/timer/millis/10` tick |
| 5 | GLFW / DISPLAY | MuJoCo 或 Isaac Sim 開不了視窗 | 見 `MUJOCO_LOG.TXT`；無頭環境用 Xvfb 並對齊 `DISPLAY` |
| 6 | `ROS_DOMAIN_ID` 不同 | topic list 是空的 | 兩邊都 export 同一個值 |
| 7 | Submodule 未 init | `dora build` 找不到套件 | `git submodule update --init --recursive` |
| 8 | 錄製的 obs/cameras 為空 | dataset 只有 action | 見 §3.6，先確認觀測回流通了 |
| 9 | 硬編碼絕對路徑 | 換機器就爛 | 見 §6.3 (1) |
| 10 | IK 的 `--damping 0.1` | 跟隨遲鈍 | 這是本 repo 刻意調保守的值，Isaac Sim 下可考慮調回接近上游預設 |

---

## 10. 環境準備與執行

### 10.1 平台

**必須是 Linux（Ubuntu 22.04）**。理由：ROS 2 Humble、`/dev/camera_*`、Isaac Sim 的 ROS 2 Bridge、`run_bridge.sh` 全都是 Linux-only。目前這份 checkout 在 Windows 上只能讀 code，不能跑。

### 10.2 首次設定

```bash
git submodule update --init --recursive
```

```bash
pip install dora-rs-cli
```

```bash
python3 -m venv --system-site-packages .venv-ros2 && .venv-ros2/bin/pip install dora-rs pyarrow pyyaml
```

```bash
dora build dataflow-vr-ros2-only.yaml
```

### 10.3 執行

```bash
dora run dataflow-vr-ros2-only.yaml
```

UI 在 http://127.0.0.1:8000，或用 HTTP 直接控制：

```bash
curl -X POST http://127.0.0.1:8000/start
```

```bash
curl -X POST http://127.0.0.1:8000/success
```

```bash
curl -X POST http://127.0.0.1:8000/quit
```

VR 控制器對應：板機 = 夾爪、A = 成功結束、B = 失敗結束、X = 重置場景、搖桿 Y = 升降柱。

---

## 11. 建議施工順序

1. **修 §6.3 的三個問題**（路徑、Arrow 解碼、夾爪）——這些不修，後面全部卡住。
2. **釐清 §8 的 v1/v2 模型問題**——這決定你要匯入哪個 USD。
3. **只做下行**：Isaac Sim 訂閱 `/openarm/vr_joint_command`，用 §7.3 Step 4 的順序驗證，先看到手臂會動。
4. **調 stiffness/damping** 讓跟隨平順，量測端到端延遲（VR 動作 → Isaac Sim 畫面）。
5. **加上行**：`isaac-obs-bridge` + Isaac Sim 相機發布 → 接回 recorder。
6. **建立 `metadata_isaac.yaml` 與 `dataflow-vr-isaac.yaml`**，錄一段 episode，驗證 `frequencies` 三個欄位（action / obs / cameras）都不是空的。
7. （選）比較 MuJoCo 與 Isaac Sim 的軌跡差異，作為 sim-to-sim 基準。

---

## 12. 參考資料

- [enactic/OpenArm](https://github.com/enactic/OpenArm) — 主專案
- [enactic/openarm_isaac_lab](https://github.com/enactic/openarm_isaac_lab) — Isaac Sim 5.1 / Isaac Lab 2.3.0，含 OpenArm USD 資產（teleoperation 介面標示為開發中）
- [openarm_description 文件](https://docs.openarm.dev/software/description) — URDF/xacro 產生方式
- [Isaac Sim URDF Importer](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/ext_isaacsim_asset_importer_urdf.html)
- [Isaac Sim ROS 2 Joint Control 教學](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/ros2_tutorials/tutorial_ros2_manipulation.html)
- [OgnROS2SubscribeJointState 節點文件](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.ros2.bridge/docs/ogn/OgnROS2SubscribeJointState.html)
- [OpenArm Simulation — Isaac Lab & MuJoCo](https://www.roboticscenter.ai/wiki/openarm/simulation)
- [dora-rs 官網](https://dora-rs.ai/)

---

## 附錄：資訊來源與可信度

| 內容 | 來源 | 可信度 |
|---|---|---|
| dataflow 結構、節點連線、參數 | 本 repo YAML 檔 | ✅ 直接讀取 |
| `bridge.py` 行為、路徑不一致 | 本 repo 原始碼 | ✅ 直接讀取 |
| dataset 格式、v0.3.0、frequencies 為空 | `vr_mujoco_data/dataset/metadata.yaml` | ✅ 直接讀取 |
| UI port 8000 與 HTTP 端點 | `.github/workflows/test.yaml` | ✅ 直接讀取 |
| venv 設定 | `.venv-ros2/pyvenv.cfg` | ✅ 直接讀取 |
| 各 submodule 的輸入輸出型別 | 上游 GitHub README | ⚠️ **需以實機 checkout 驗證**（submodule 未初始化，且本地版本可能較舊）|
| Isaac Sim 設定步驟 | NVIDIA 官方文件（4.5.0 版）| ⚠️ Isaac Sim 5.1 的 UI 路徑可能略有不同 |
| OpenArm v1 的關節命名 | 未經驗證的推測 | ❌ **必須實際檢視 v1 URDF 確認** |
