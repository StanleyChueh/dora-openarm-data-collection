# Ubuntu 22.04 + ROS 2 Humble 執行手冊

> 搭配 [vr-to-isaac-sim.md](vr-to-isaac-sim.md) 使用。本篇只講「怎麼把環境裝起來、怎麼跑」。

---

## 0. ⚠️ 先讀這段：Python 版本衝突

Ubuntu 22.04 內建 Python **3.10**，ROS 2 Humble 的 `rclpy` 也是編譯給 3.10 的。
但這個 pipeline 的兩個核心節點要求 **Python ≥ 3.11**：

| 套件 | `requires-python` | 用在 |
|---|---|---|
| `dora-openarm-vr` | **>= 3.11** | `udp-receiver`（VR 接收）|
| `dora-openarm-kinematics` | **>= 3.11** | `ik`（IK 求解）|
| `dora-openarm-mujoco` | >= 3.10 | `mujoco-collect` |
| `dora-openarm-data-collection-ui` | >= 3.10 | `ui` |
| `rclpy`（ROS 2 Humble）| **= 3.10**（系統編譯）| `ros2-bridge` |

**所以在 Ubuntu 22.04 上，你不可能用單一個 Python 環境跑完整條 pipeline。**

好消息是 **dora 的每個 node 都是獨立行程**，`path:` 可以指向任何可執行檔，所以可以雙環境並存：

```
┌─ venv-dora（Python 3.11+）────────────┐   ┌─ .venv-ros2（Python 3.10）──────┐
│  dora-rs-cli                          │   │  system-site-packages = true    │
│  ui / quitter / udp-receiver / ik     │   │  → 看得到 /opt/ros/humble 的     │
│  mujoco / recorder / cameras          │   │    rclpy                        │
│                                        │   │  + dora-rs, pyarrow             │
│  ← dora build / dora run 在這裡跑      │   │  ← 只有 bridge.py 在這裡跑       │
└────────────────────────────────────────┘   └─────────────────────────────────┘
                    │                                      ▲
                    └── path: ./nodes/.../run_bridge.sh ────┘
                        （script 內部切換到 3.10 直譯器）
```

repo 裡的 `.venv-ros2/pyvenv.cfg` 有 `include-system-site-packages = true`，就是為了這件事。**這個設計是對的，照著做即可。**

---

## 1. 系統前置

### 1.1 基本套件

```bash
sudo apt update && sudo apt install -y git curl build-essential python3-venv v4l-utils
```

### 1.2 ROS 2 Humble

若尚未安裝，依 [官方文件](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) 裝 `ros-humble-desktop`。驗證：

```bash
source /opt/ros/humble/setup.bash && python3 -c "import rclpy; print(rclpy.__file__)"
```

應印出 `/opt/ros/humble/lib/python3.10/site-packages/rclpy/__init__.py`。

> **不要**把 `source /opt/ros/humble/setup.bash` 寫進 `~/.bashrc`。它會污染 `PYTHONPATH`，讓 3.11 的 dora 環境撞到 3.10 的 ROS 套件。只在 `run_bridge.sh` 裡 source。

### 1.3 取得 submodule

```bash
cd ~/Ivan_ws/dora-openarm-data-collection && git submodule update --init --recursive
```

沒做這步，`nodes/` 下全是空目錄，`dora build` 必定失敗。

---

## 2. 建立主環境（Python 3.11+）

### 方案 A：uv（推薦）

uv 會自己下載 Python 3.11，不用碰系統 Python，也是上游 README 用的方式。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
```

```bash
cd ~/Ivan_ws/dora-openarm-data-collection && uv venv --python 3.11 .venv-dora
```

```bash
source .venv-dora/bin/activate && uv pip install "dora-rs-cli==0.5.0"
```

> **版本必須釘 0.5.0**。`dora-openarm-vr` 的 pyproject 寫死 `dora-rs-cli==0.5.0`、`dora-rs==0.5.0`，`dora-openarm-mujoco` 也是 `dora-rs==0.5.0`。
> PyPI 上的版本序列是 `… 0.3.12 → 0.3.13 → 0.5.0`（**沒有 0.4.x**），裝到 0.3.x 會因為 `--uv` 旗標不存在而在 `dora build` 階段失敗。

### 方案 B：deadsnakes PPA

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update && sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

```bash
cd ~/Ivan_ws/dora-openarm-data-collection && python3.11 -m venv .venv-dora && source .venv-dora/bin/activate && pip install "dora-rs-cli==0.5.0"
```

### 驗證

```bash
python -V && dora --version && which -a dora
```

必須是 Python 3.11.x + dora **0.5.0**，且 `which -a dora` 只有一個結果、指向 `.venv-dora/bin/dora`。

若 `which -a dora` 印出多筆（常見於 `~/.cargo/bin/dora` 或 `/usr/local/bin/dora`），舊版會搶先被執行 —— 先移除舊的或調整 PATH。

確認 `--uv` 存在（0.5.0 才有）：

```bash
dora build --help | grep -- --uv
```

**若 pip 裝不上 0.5.0**，代表你的平台沒有對應 wheel（最常見於 aarch64 / Jetson）。用 `uname -m` 確認架構；ARM 平台需改用 cargo 從源碼編 `dora-cli`。

---

## 3. 建立 ROS 2 Bridge 環境（Python 3.10）

**注意：這一步要開新的 terminal，或先 `deactivate`**，因為要用系統 Python 3.10。

```bash
cd ~/Ivan_ws/dora-openarm-data-collection && python3 -m venv --system-site-packages .venv-ros2
```

`--system-site-packages` 是關鍵，少了它就看不到 `rclpy`。

```bash
.venv-ros2/bin/pip install "dora-rs==0.5.0" pyarrow pyyaml
```

驗證（同時看得到 rclpy 和 dora 才算過）：

```bash
bash -c 'source /opt/ros/humble/setup.bash && .venv-ros2/bin/python -c "import rclpy, dora, pyarrow; print(\"OK\", rclpy.__file__)"'
```

---

## 4. 修掉會直接爆掉的路徑問題

### 4.1 `run_bridge.sh` 改成相對路徑

原檔寫死 `/home/csl/Ivan_ws/...`，換機器或改目錄就爛。改成：

```bash
#!/usr/bin/env bash
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source /opt/ros/humble/setup.bash
exec "$REPO/.venv-ros2/bin/python" -u "$HERE/bridge.py"
```

```bash
chmod +x nodes/dora-openarm-ros2-bridge/run_bridge.sh
```

### 4.2 `dataflow-vr-ros2-only.yaml` 的 bridge 節點

現在寫的是 `.venv_ros2`（底線）＋硬編絕對路徑，**這個路徑不存在**。改成跟 `dataflow-vr-mujoco-ros2.yaml` 一致：

```yaml
  - id: ros2-bridge
    path: ./nodes/dora-openarm-ros2-bridge/run_bridge.sh
    inputs:
      position_left: ik/position_left
      position_right: ik/position_right
```

### 4.3 `bridge.py` 的 Arrow 解碼

上游 IK 可能輸出 struct 陣列，現在的解碼會丟 `TypeError`。改成相容兩種：

```python
def arrow_to_float_list(value) -> list[float]:
    items = value.to_pylist()
    if items and isinstance(items[0], dict):        # [{"qpos": [...]}]
        return [float(x) for x in items[0]["qpos"]]
    return [float(x) for x in items]                # [f, f, f, ...]
```

---

## 5. Build

**一定要在 repo 根目錄執行**（`build:` 用的是 `nodes/xxx` 相對路徑）。

```bash
cd ~/Ivan_ws/dora-openarm-data-collection && source .venv-dora/bin/activate
```

方案 A（uv）：

```bash
dora build dataflow-vr-ros2-only.yaml --uv
```

方案 B（一般 venv）：

```bash
dora build dataflow-vr-ros2-only.yaml
```

> `--uv` 只影響 build 階段：它把 YAML 裡 `pip install ...` 的 build line 改用 `uv pip install` 執行。
> `ros2-bridge` 節點沒有 `build:` 欄位，`dora build` 會跳過它——這是刻意的，它的環境在 §3 手動建。

驗證裝好了：

```bash
which dora-openarm-ik dora-openarm-quest-receiver dora-openarm-data-collection-ui
```

三個都要指向 `.venv-dora/bin/`。

---

## 6. Run

### 6.1 dora 的兩種執行模式

| 指令 | 用途 |
|---|---|
| `dora run <file>` | **單機前景執行**，Ctrl-C 結束。日常開發用這個（CI 也是）|
| `dora up` + `dora start <file>` | daemon 模式，多機分散式才需要 |

以下一律用 `dora run`。

### 6.2 循序驗證（強烈建議照順序）

**Step 1 — 先跑 dummy，確認 dora 本身沒問題**（不需要 VR、不需要硬體）：

```bash
dora build dataflow_dummy.yaml && dora run dataflow_dummy.yaml
```

另開 terminal：

```bash
curl -X POST http://127.0.0.1:8000/start && sleep 2 && curl -X POST http://127.0.0.1:8000/success && curl -X POST http://127.0.0.1:8000/quit
```

跑完 `dataset/` 下應該出現 episode。這步過了代表 dora + UI + recorder 鏈路正常。

**Step 2 — VR → IK → ROS 2**（要 Quest，不需 Isaac Sim）：

```bash
dora run dataflow-vr-ros2-only.yaml
```

另開 terminal 檢查 ROS 2 topic：

```bash
source /opt/ros/humble/setup.bash && ros2 topic hz /openarm/vr_joint_command
```

期待約 500 Hz。看內容：

```bash
source /opt/ros/humble/setup.bash && ros2 topic echo /openarm/vr_joint_command --once
```

`name` 應有 14 個、`position` 應是合理弧度值。**這步不過就不要往下走**。

**Step 3 — 加上 MuJoCo 視覺化**：

```bash
dora run dataflow-vr-mujoco-ros2.yaml
```

**Step 4 — 接 Isaac Sim**：見 [vr-to-isaac-sim.md §7.3](vr-to-isaac-sim.md)。

### 6.3 控制方式

| 動作 | 鍵盤/HTTP | VR 控制器 |
|---|---|---|
| 開始錄製 | `curl -X POST :8000/start` | — |
| 標記成功結束 | `curl -X POST :8000/success` | **A** |
| 標記失敗結束 | — | **B** |
| 重置場景 | — | **X** |
| 結束 pipeline | `curl -X POST :8000/quit` | — |
| 夾爪 | — | 板機（類比）|
| 升降柱 | — | 搖桿 Y |

UI 網頁：http://127.0.0.1:8000

---

## 7. Quest 3 端設定

1. **一次性**：安裝 Meta Quest Developer Hub、註冊開發者帳號、sideload OpenArm 的 teleop APK（向 enactic 索取或見 `dora-openarm-vr` repo）。
2. **每次**：
   - 確認 Quest 與 PC 在**同一個區網**。
   - PC 端查 IP：`ip addr show | grep "inet "`
   - APK 左側 menu 鍵 → 輸入 PC 的 IP 與 port **5006**。
   - 把眼部感測器貼起來，避免頭盔誤判休眠。
   - 操作時可把頭盔掛在脖子上，手持控制器即可。
3. **防火牆**（若 `ros2 topic hz` 沒東西且確定 dora 有跑）：

```bash
sudo ufw allow 5006/udp
```

⚠️ **安全**：第一次連上時**緩慢輕拉板機**讓機械臂對位，確認方向正確再做大動作。

---

## 8. 真機才需要的設定

> 只跑模擬 / Isaac Sim 的話跳過整章。

### 8.1 相機 udev 規則

`dataflow-vr.yaml` 用固定路徑 `/dev/camera_wrist_right` 等。先找出各相機的序號：

```bash
udevadm info --name=/dev/video0 --attribute-walk | grep -E "ATTRS\{serial\}|ATTRS\{idVendor\}|ATTRS\{idProduct\}"
```

寫成 `/etc/udev/rules.d/99-openarm-cameras.rules`（每台相機一行，`ATTRS{serial}` 換成實際值）：

```
SUBSYSTEM=="video4linux", ATTRS{serial}=="XXXX", ATTR{index}=="0", SYMLINK+="camera_wrist_right"
SUBSYSTEM=="video4linux", ATTRS{serial}=="YYYY", ATTR{index}=="0", SYMLINK+="camera_wrist_left"
SUBSYSTEM=="video4linux", ATTRS{serial}=="ZZZZ", ATTR{index}=="0", SYMLINK+="camera_head_stereo"
SUBSYSTEM=="video4linux", ATTRS{serial}=="WWWW", ATTR{index}=="0", SYMLINK+="camera_ceiling"
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger && ls -l /dev/camera_*
```

`ATTR{index}=="0"` 不能少，否則 UVC 相機的 metadata 裝置也會被綁到同一個 symlink。

### 8.2 CAN bus

`dora-openarm` 走 SocketCAN 驅動 Damiao 馬達。介面名稱與 bitrate 請以 [OpenArm 官方文件](https://docs.openarm.dev/) 與 `nodes/dora-openarm/` 的設定檔（預設 `openarm_cell.yaml`）為準——**不要直接套用網路上的通用數值**，bitrate 設錯會完全通訊不上。

```bash
ip -details link show can0
```

### 8.3 首次上電安全

`dora-openarm` 預設開啟 `--align`：手臂會以每次最多 `--align-delta-limit`（預設 0.001 rad）的步進慢慢對齊到目標，`status` 變成 `aligned` 後才接受全速命令。**不要為了求快關掉 `--align`**。

---

## 9. 錯誤對照表

| 症狀 | 原因 | 解法 |
|---|---|---|
| `dora build` 抱怨參數（如 `expected at most one ...`、`unexpected argument '--uv'`）| dora CLI 是 0.3.x，`--uv` 是 0.5.0 才有 | §2 升級到 `dora-rs-cli==0.5.0`，並用 `which -a dora` 確認沒有舊版蓋掉 |
| `ERROR: Package requires a different Python: 3.10.x not in '>=3.11'` | 用系統 Python 跑 `dora build` | §2 建 3.11 環境並 activate |
| `dora build` 說找不到 `nodes/dora-openarm-vr` | submodule 沒 init | §1.3 |
| `No such file or directory: .venv_ros2/bin/python3` | YAML 底線 vs 連字號 | §4.2 |
| `TypeError: float() argument must be...` in bridge.py | Arrow struct 格式 | §4.3 |
| `ModuleNotFoundError: rclpy` | venv 沒加 `--system-site-packages`，或沒 source setup.bash | §3 |
| `ros2 topic list` 看不到 topic | `ROS_DOMAIN_ID` 不一致 | 兩邊都 `export ROS_DOMAIN_ID=0` |
| topic 有但 `hz` 是 0 | Quest 沒連上 / 防火牆 | §7 |
| `could not initialize GLFW` | 無 display 卻開 viewer | 設對 `DISPLAY`，或用 Xvfb，或拿掉 `--viewer` |
| MuJoCo `Attach conflict ... timestep` | 場景 timestep 不一致（parent 0.002 / child 0.001）| 上游已設為 warning，可忽略 |
| 手臂完全不動、無錯誤訊息 | joint 名稱對不上 | 見 [vr-to-isaac-sim.md §8](vr-to-isaac-sim.md) |
| `dora run` 卡住不結束 | quitter 沒收到 quit | `curl -X POST :8000/quit`，或 Ctrl-C |

### 除錯用指令

看某個 node 的即時輸出：

```bash
dora logs <dataflow_id> <node_id>
```

列出跑著的 dataflow：

```bash
dora list
```

強制清乾淨（node 殘留時）：

```bash
dora destroy
```

---

## 10. 一鍵啟動腳本（選用）

存成 `run.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATAFLOW="${1:-dataflow-vr-ros2-only.yaml}"

# 主環境用 3.11；ROS 2 只在 run_bridge.sh 內部 source，這裡不要 source
source .venv-dora/bin/activate

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

echo "[*] dataflow: $DATAFLOW"
dora run "$DATAFLOW"
```

```bash
chmod +x run.sh && ./run.sh dataflow-vr-ros2-only.yaml
```

---

## 附錄：資訊來源

| 內容 | 來源 | 可信度 |
|---|---|---|
| `requires-python` 與相依套件 | 各 submodule 上游 `pyproject.toml` | ✅ 直接讀取 |
| `dora run` / `--uv` 行為 | dora-rs 官方文件與上游 README 用法 | ✅ |
| UI port 8000 與 HTTP 端點 | `.github/workflows/test.yaml` | ✅ 直接讀取 |
| `.venv-ros2` 設定 | 本 repo `pyvenv.cfg` | ✅ 直接讀取 |
| GLFW / timestep 警告 | 本 repo `MUJOCO_LOG.TXT` | ✅ 直接讀取 |
| udev 規則寫法 | 通用 v4l2 做法 | ⚠️ 序號需自行填 |
| CAN bitrate / 介面名 | **未驗證** | ❌ 請查 OpenArm 官方文件 |
| `dora logs` / `dora destroy` 子指令 | dora CLI 慣例 | ⚠️ 請用 `dora --help` 確認你的 0.5.0 版本 |
