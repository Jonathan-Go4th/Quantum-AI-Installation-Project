# Quantum AI Mini Project

This project uses **Qiskit** and **Python** to explore basic quantum circuits and simple quantum–classical experiments.

## Why Qiskit?

Qiskit (Quantum Information Software Kit) is an open-source, Python-based, high-performance software framework for quantum computing, originally developed by IBM Research and first released in 2017. It provides tools to build quantum programs (by defining quantum circuits and operations) and run them on real quantum hardware or on classical simulators.

Qiskit is widely used because current quantum computers are still in early development, noisy, and expensive to operate. Simulating quantum circuits on classical computers allows developers and researchers to design, test, and optimize quantum algorithms before executing them on actual quantum devices.


---


## 1. Install Jupyter Notebook

Follow **Method 2** from this guide to install Jupyter Notebook on Windows:

https://www.geeksforgeeks.org/installation-guide/install-jupyter-notebook-in-windows/

After installation, start Jupyter Notebook from your terminal (PowerShell or CMD) with:

```bash
jupyter notebook
```

<br>

---

## 2. Install Qiskit

Qiskit is the main library used for quantum computing.

Install it using:

```bash
pip install qiskit
```

<br>

---

## 3. Install pylatexenc

`pylatexenc` is needed for drawing quantum circuits using `qc.draw("mpl")`.

```bash
pip install pylatexenc
```

<br>

---

## 4. Test if Qiskit is working

Run this in Jupyter Notebook:

```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)

qc.draw("mpl")
```

If this shows a circuit diagram, Qiskit works.

<img width="1228" height="428" alt="image" src="https://github.com/user-attachments/assets/1f197357-ea06-490d-b3d3-880520e42e83" />


<br>

---

## 5. Install Qiskit Aer (Quantum Simulator)

```bash
pip install qiskit-aer
```

<br>

---

## 6. Test Qiskit Aer (Bell State Example)

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Bell state with measurements
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

sim = AerSimulator()
tqc = transpile(qc, sim)
result = sim.run(tqc, shots=2000).result()
print(result.get_counts())  # Expect mostly '00' and '11'
```

<img width="1187" height="321" alt="image" src="https://github.com/user-attachments/assets/a8e9a39e-7300-4560-9a2b-15824d3abbc4" />


<br>

---

## 7. Install NumPy, Pandas, and Scikit-Learn

These are required for Quantum + ML experiments.

```bash
pip install numpy pandas scikit-learn
```

---

You're now ready to run your Quantum AI mini-project 🚀
