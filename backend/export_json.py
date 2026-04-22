import json
from simulation import Sim

sim = Sim()

frames = []
signals = []

for step in range(200):
    t = step * 0.1
    field, s = sim.step(t)

    frames.append(field.tolist())
    signals.append(float(s))

output = {
    "frames": frames,
    "signal": signals,
    "meta": {"size": 120}
}

with open("output.json", "w") as f:
    json.dump(output, f)

print("✅ output.json created")
