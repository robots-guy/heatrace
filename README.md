# heatrace
Parametric design-to-fabrication workflow for embedded resistive heaters in soft robots. Thermal model and Rhino-GH heater generation scripts.

## The HeaTrace Workflow
*The HeaTrace workflow: a target thermal specification drives heater sizing,
which is routed into a serpentine or spiral pattern and predicted by the
lumped-capacitance thermal model.*

<img width="1592" height="952" alt="image" src="https://github.com/user-attachments/assets/f9acbc3d-5c8e-4315-ac39-75a566a9d128" />

*Parametric generation of serpentine and spiral heater patterns from a target yarn
length, with Wallacei-based optimization to fit the layout within the footprint.*
<img width="1597" height="1147" alt="image" src="https://github.com/user-attachments/assets/1886ae11-0034-4b6e-98a9-8f0ba99364b5" />

## Repository Structure
- `model/` — Python lumped-capacitance thermal model and residual analysis
- `Rhino-GH/` — Parametric heater generator (.gh) + required plugins
