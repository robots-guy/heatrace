**HeaTrace — Thermal Prediction Model:**

This folder contains the Python implementation of the HeaTrace lumped-capacitance
thermal model used to size embedded resistive yarn heaters and predict their
transient temperature response in silicone soft-robot bodies.

**What the model does:**

Given a user-specified thermal/electrical target, the model performs four steps:

+ **Heater sizing (target → geometry)** : Computes the required electrical
resistance R = V² / P and the corresponding conductive yarn length from the
yarn's resistance-per-unit-length. This length is the value passed to the
parametric Grasshopper pattern generator (see ../grasshopper/).

+ **Transient thermal prediction** : Estimates the temperature rise T(t) using a
first-order lumped-capacitance model:
T(t) = T_amb + (P·α / h_eff)·(1 − exp(−t/τ)),
where h_eff combines convective, radiative, and conductive losses and τ is
the thermal time constant.

+ **Safety gating** : Checks operating current, areal power density, and linear
power along the yarn against material and operational limits.
Model-validity check. Reports the Biot number to confirm the
lumped-capacitance assumption (Bi < 0.1) is justified for the chosen geometry.

**Requirements**
+ Python 3.8+
+ Jupyter Notebook or Google Colab (the model uses ipywidgets for the
interactive sliders and will not display the UI in a plain script run)

**Usage**
In Jupyter / Colab: open a notebook, then run the script as a cell, or:
+ %run heatrace_thermal_model.py
