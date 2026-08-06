# Usage

Launch OpenARM VR 

```
cd dora-openarm-data-collection
uv venv .venv
source .venv/bin/activate
uv pip install dora-rs-cli
```

```
dora run dataflow-vr-mujoco-ros2.yaml  --uv
```

Launch Isaac Sim

```
cd ~/isaacsim
./isaac-sim.selector.sh
```

Drag USD into Isaac Sim

```
openarm_ros_joint_control.usd
```

<img width="1241" height="563" alt="image" src="https://github.com/user-attachments/assets/cf1876c1-2814-47e6-b5a6-0d06d8f36857" />
