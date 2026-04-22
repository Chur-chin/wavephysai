import imageio
import numpy as np
import matplotlib.pyplot as plt
import json

# ============================
# 데이터 로드
# ============================
with open("output.json") as f:
    data = json.load(f)

frames = data["frames"]
signals = data["signal"]

# geometry가 없으면 fallback
geometry = data.get("geometry", None)
if geometry is not None:
    geometry = np.array(geometry)


# ============================
# GIF 생성
# ============================
images = []

for i, frame in enumerate(frames):
    frame = np.array(frame)

    fig, axs = plt.subplots(1, 2, figsize=(8,4))

    # ------------------------
    # 1. Heatmap + geometry
    # ------------------------
    ax = axs[0]
    im = ax.imshow(frame, cmap="RdBu")

    if geometry is not None:
        geom = np.array(geometry)

        # geometry overlay (투명도)
        ax.imshow(geom, cmap="gray", alpha=0.2)

    ax.set_title(f"Wave Field (t={i})")
    ax.axis("off")

    # ------------------------
    # 2. Signal plot
    # ------------------------
    ax2 = axs[1]
    ax2.plot(signals[:i+1])
    ax2.set_title("Tip Signal")
    ax2.set_xlim(0, len(signals))
    ax2.set_ylim(min(signals), max(signals)+1e-6)

    # 현재 위치 표시
    ax2.axvline(i, linestyle="--")

    # ------------------------
    # 이미지 저장 (메모리)
    # ------------------------
    plt.tight_layout()
    plt.savefig("temp.png")
    plt.close()

    images.append(imageio.imread("temp.png"))

# ============================
# GIF 저장
# ============================
imageio.mimsave("demo_advanced.gif", images, duration=0.05)

print("✅ demo_advanced.gif 생성 완료")
