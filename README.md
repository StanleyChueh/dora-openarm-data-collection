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
