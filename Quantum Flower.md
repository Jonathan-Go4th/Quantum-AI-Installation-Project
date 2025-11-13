## Bloch Garden — 3D Quantum Flower Field 🌸

# 🌸 Bloch Garden — 3D Quantum Flower Field  
**Quantum State Visualization using Qiskit + Plotly**

This project transforms single-qubit quantum states into an interactive 3D "flower field".  
Each qubit becomes a flower whose stem, petals, and color correspond to its Bloch vector and quantum phase.

---

## 🔮 Features
- Single-qubit circuit generation using H, Rx, Ry, Rz
- Statevector extraction using Qiskit
- Bloch sphere coordinate mapping
- Quantum phase → RGB color encoding
- 3D Plotly visualization with:
  - Stems
  - Petals
  - Buds
  - Ghost Bloch Sphere

---

## 🧠 Real-World Applications
- Quantum Cryptography (BB84 QKD)
- Quantum Machine Learning feature maps
- Single-qubit gate calibration
- Noise/Decoherence visualization
- State tomography
- Quantum outreach & education
- Quantum-inspired procedural generation

---

## 🖥️ Code

This script generates a 3D “quantum flower garden” by mapping single-qubit quantum states to Bloch vectors and visualizing them with Plotly.  
Each qubit is turned into a stem, petals, and a glowing bud on top of a ghost Bloch sphere.

> **Requirements:** `qiskit`, `plotly`, `numpy`

```python
"""
Credits: Bhuvan V , Divyanshu Jha, Jonathan Goforth
Bloch Garden — 3D Quantum Flower Field
Each qubit => single-qubit circuit (no entanglement).
Statevector -> Bloch vector -> 3D "flower" on a ghost Bloch sphere.
"""

import math
import numpy as np
import plotly.graph_objects as go
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# -----------------------
# Helper: build single-qubit circuit from seed & index
# -----------------------
def build_single_qubit_circuit(index, seed_float=0.123, bias_strength=1.0):
    """
    Build a small circuit for one qubit.
    Parameters:
      index: int (used for per-qubit variation)
      seed_float: 0..1 to bias rotations (derived from text seed)
      bias_strength: multiply angles to vary strength
    Returns:
      QuantumCircuit with 1 qubit
    """
    qc = QuantumCircuit(1)

    # deterministic "random"-looking gates based on seed and index
    r1 = (seed_float * 12.34 + index * 0.91) % 1.0
    r2 = (seed_float * 7.11  + index * 1.37) % 1.0
    r3 = (seed_float * 3.77  + index * 2.21) % 1.0

    # convert to angles
    a = (r1 * 2.0 - 1.0) * math.pi * bias_strength / 2.2
    b = (r2 * 2.0 - 1.0) * math.pi * bias_strength / 2.7
    c = (r3 * 2.0 - 1.0) * math.pi * bias_strength / 3.1

    qc.h(0)
    qc.rz(a, 0)
    qc.ry(b, 0)
    qc.rx(c, 0)
    return qc

# -----------------------
# Bloch vector calculation for single-qubit statevector psi = [a, b]
# Bloch components: x = 2 Re(conj(a)*b), y = 2 Im(conj(a)*b), z = |a|^2 - |b|^2
# -----------------------
def bloch_from_statevector(statevector):
    a = statevector[0]
    b = statevector[1]
    rho0 = abs(a) ** 2
    rho1 = abs(b) ** 2
    x = 2.0 * (a.conjugate() * b).real
    y = 2.0 * (a.conjugate() * b).imag
    z = rho0 - rho1
    return np.array([x, y, z])

# -----------------------
# Map phase to color (hue)
# -----------------------
def phase_to_rgb(phase):
    # phase in radians -> map to HSV hue; convert to rgb
    h = (phase % (2 * np.pi)) / (2 * np.pi)  # 0..1
    s = 0.85
    v = 0.95
    return hsv_to_rgb(h, s, v)

def hsv_to_rgb(h, s, v):
    # h, s, v in [0,1]
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    # return as CSS rgb() string for Plotly
    return "rgb({:.0f},{:.0f},{:.0f})".format(r * 255, g * 255, b * 255)

# -----------------------
# Create bloom — simple petal cloud orthogonal to stem
# -----------------------
def create_petals(stem_vec, center, petal_count=12, petal_radius=0.08):
    # stem_vec: unit vector direction
    # center: tuple position
    u = np.array(stem_vec, dtype=float)
    if np.allclose(u, 0):
        u = np.array([0.0, 0.0, 1.0])
    u = u / np.linalg.norm(u)

    # find any vector not parallel to u
    if abs(u[0]) < 0.9:
        other = np.array([1.0, 0.0, 0.0])
    else:
        other = np.array([0.0, 1.0, 0.0])

    v1 = np.cross(u, other)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(u, v1)
    v2 = v2 / np.linalg.norm(v2)

    petals = []
    angles = np.linspace(0, 2 * np.pi, petal_count, endpoint=False)
    for th in angles:
        pos = np.array(center) + petal_radius * (np.cos(th) * v1 + np.sin(th) * v2)
        petals.append(pos.tolist())
    return petals

# -----------------------
# Generate Bloch Garden data
# -----------------------
def generate_bloch_garden(num_qubits=64, seed_text="Pixie",
                          bias_strength=1.0, stem_scale=1.0):
    # seed to float 0..1 from text
    seed_float = sum(ord(ch) for ch in seed_text) % 997 / 997.0

    stems = []   # list of (x0,y0,z0)->(x1,y1,z1)
    petal_points_x = []
    petal_points_y = []
    petal_points_z = []
    petal_colors = []
    bud_x = []
    bud_y = []
    bud_z = []
    bud_colors = []
    stem_colors = []

    for i in range(num_qubits):
        qc = build_single_qubit_circuit(i, seed_float=seed_float,
                                        bias_strength=bias_strength)
        sv = Statevector.from_instruction(qc).data  # complex array length 2
        bloch = bloch_from_statevector(sv)

        # normalize and scale stem
        norm = np.linalg.norm(bloch)
        if norm < 1e-9:
            bloch_unit = np.array([0.0, 0.0, 1.0])
        else:
            bloch_unit = bloch / norm

        # randomize stem length slightly for garden variety
        length = stem_scale * (
            0.5 + 0.75 * ((i % 7) / 7.0 + 0.05 * math.sin(i * 1.3))
        )
        tip = bloch_unit * length

        # small lateral offset so stems don't all originate exactly at origin
        angle = 2 * math.pi * (i / num_qubits)
        radius = 0.4 * (0.5 + 0.5 * ((i % 5) / 5.0))
        base = np.array(
            [radius * math.cos(angle),
             radius * math.sin(angle),
             -0.1]  # slight ground offset
        )

        start = base.tolist()
        end = (base + tip).tolist()
        stems.append((start, end))

        # define petal characteristics from the state's relative phase
        a, b = sv[0], sv[1]
        rel_phase = np.angle(b) - np.angle(a)
        color = phase_to_rgb(rel_phase)

        # bud (a glowing sphere at tip)
        bud_x.append(end[0])
        bud_y.append(end[1])
        bud_z.append(end[2])
        bud_colors.append(color)

        # petals cloud
        petals = create_petals(
            bloch_unit,
            end,
            petal_count=10 + (i % 8),
            petal_radius=0.06 + 0.02 * (i % 3),
        )
        for p in petals:
            petal_points_x.append(p[0])
            petal_points_y.append(p[1])
            petal_points_z.append(p[2])
            petal_colors.append(color)

        # stem color slightly darker (just reuse color)
        stem_colors.append(color)

    data = {
        "stems": stems,
        "petal_pts": (petal_points_x, petal_points_y, petal_points_z, petal_colors),
        "buds": (bud_x, bud_y, bud_z, bud_colors),
        "stem_colors": stem_colors,
    }
    return data

# -----------------------
# Plot with Plotly
# -----------------------
def plot_bloch_garden(data,
                      title="Bloch Garden — Quantum Flower Field",
                      show=True,
                      save_html=None):
    fig = go.Figure()

    # stems: plot as small line segments
    for (start, end), color in zip(data["stems"], data["stem_colors"]):
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]
        zs = [start[2], end[2]]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(width=4),
                marker=dict(color=color),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # petals: small markers
    px, py, pz, pcolors = data["petal_pts"]
    fig.add_trace(
        go.Scatter3d(
            x=px,
            y=py,
            z=pz,
            mode="markers",
            marker=dict(size=3, opacity=0.85, color=pcolors),
            showlegend=False,
            hoverinfo="none",
        )
    )

    # buds: larger glowing markers
    bx, by, bz, bcolors = data["buds"]
    fig.add_trace(
        go.Scatter3d(
            x=bx,
            y=by,
            z=bz,
            mode="markers",
            marker=dict(size=8, symbol="circle",
                        line=dict(width=1), color=bcolors),
            showlegend=False,
            hoverinfo="none",
        )
    )

    # add subtle ambient sphere (the Bloch sphere ghost)
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    sphere_x = []
    sphere_y = []
    sphere_z = []
    for uu in u:
        for vv in v:
            sphere_x.append(0.9 * np.cos(uu) * np.sin(vv))
            sphere_y.append(0.9 * np.sin(uu) * np.sin(vv))
            sphere_z.append(0.9 * np.cos(vv))
    fig.add_trace(
        go.Scatter3d(
            x=sphere_x,
            y=sphere_y,
            z=sphere_z,
            mode="markers",
            marker=dict(size=1, color="rgba(200,200,200,0.03)"),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # stylize scene
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="black",
        plot_bgcolor="black",
    )

    if save_html:
        fig.write_html(save_html)
        print(f"Saved interactive Bloch Garden to {save_html}")

    if show:
        fig.show()

# -----------------------
# Entry point
# -----------------------
def main():
    print("Generating Bloch Garden…")
    data = generate_bloch_garden(
        num_qubits=64,        # try 64 first; can increase later
        seed_text="Hello World",
        bias_strength=4.0,
        stem_scale=3.0,
    )
    plot_bloch_garden(data, title="Bloch Garden — seed: Hello World")

main()   # in a notebook, this will immediately render the garden
