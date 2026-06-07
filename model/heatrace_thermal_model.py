"""
HeaTrace — Thermal Prediction Model 
"""
# use the necessary libraries

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ipywidgets import (
    interactive_output, FloatSlider, IntSlider,
    HBox, VBox, Layout, Output
)
from IPython.display import display, HTML
import io, base64


# CORE PREDICTION 

def run_prediction(
    T_ambient, rho, c_p, L, W, thickness,
    h, target_power, V_input, I_max_rated,
    t_max, threshold, alpha,
):
    epsilon         = 0.9
    sigma           = 5.67e-8
    k_s             = 0.22
    RU              = 0.04
    P_density_limit = 1.8

    V_vol    = L * 1e-3 * W * 1e-3 * thickness * 1e-3
    A        = L * 1e-3 * W * 1e-3
    L_c      = V_vol / A
    area_cm2 = (L * 0.1) * (W * 0.1)

    Bi    = (h * L_c) / k_s
    bi_ok = Bi < 0.1

    R_required  = (V_input ** 2) / target_power
    yarn_length = R_required / RU

    I_actual       = V_input / R_required
    current_ok     = I_actual <= I_max_rated
    P_density      = target_power / area_cm2
    p_density_ok   = P_density <= P_density_limit
    q_linear       = target_power / yarn_length
    q_linear_limit = (I_max_rated ** 2) * RU
    q_linear_ok    = q_linear <= q_linear_limit
    overall_safe   = current_ok and p_density_ok and q_linear_ok

    h_conv = h * A
    h_rad  = 4 * epsilon * sigma * A * (T_ambient + 273.15) ** 3
    h_cond = (k_s * A) / L_c
    h_eff  = h_conv + h_rad + h_cond
    tau    = (rho * V_vol * c_p) / h_eff

    times, temps = [], []
    t_thresh = None
    for t in range(0, t_max + 1, 1):
        T = T_ambient + (target_power * alpha / h_eff) * (1 - np.exp(-t / tau))
        times.append(t)
        temps.append(T)
        if T >= threshold and t_thresh is None:
            t_thresh = t

    # Heater Geometry constant; Adjust this as per your need.
    if alpha <= 0.65:
        pattern = f"Spiral (α={alpha})"
    elif alpha >= 0.75:
        pattern = f"Serpentine (α={alpha})"
    else:
        pattern = f"Custom (α={alpha})"

    return dict(
        times=times, temps=temps, tau=tau, t_thresh=t_thresh,
        R_required=R_required, yarn_length=yarn_length,
        I_actual=I_actual, current_ok=current_ok,
        P_density=P_density, p_density_ok=p_density_ok,
        q_linear=q_linear, q_linear_limit=q_linear_limit, q_linear_ok=q_linear_ok,
        overall_safe=overall_safe, Bi=Bi, bi_ok=bi_ok, L_c=L_c,
        pattern=pattern, threshold=threshold,
        target_power=target_power, V_input=V_input,
        t_max=t_max, I_max_rated=I_max_rated,
        T_ambient=T_ambient,
        P_density_limit=P_density_limit,
    )



# PLOT

def render_plot(
    T_ambient, rho, c_p, L, W, thickness,
    h, target_power, V_input, I_max_rated,
    t_max, threshold, alpha,
):
    d = run_prediction(
        T_ambient, rho, c_p, L, W, thickness,
        h, target_power, V_input, I_max_rated,
        t_max, threshold, alpha,
    )

    fig, ax = plt.subplots(figsize=(5.8, 4.2))

    y_upper = max(threshold * 1.25, max(d['temps']) * 1.08) + 5
    ax.set_ylim(d['T_ambient'] - 5, y_upper)

    curve_color = '#B03A2E' if not d['overall_safe'] else '#D85A30'
    legend_str  = (
        f"P = {target_power:.1f} W  |  V = {V_input} V  |  "
        f"{d['pattern']}  |  L = {d['yarn_length']:.0f} mm"
    )
    ax.plot(d['times'], d['temps'], color=curve_color, linewidth=2, label=legend_str)
    ax.axhline(threshold, color='#444441', linestyle='--', linewidth=1.2,
               label=f"Activation threshold: {threshold} °C")

    status_txt   = "✓  Safe" if d['overall_safe'] else "⚠  Check safety"
    status_color = '#1A7A4A' if d['overall_safe'] else '#B03A2E'
    ax.text(0.02, 0.96, status_txt,
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            color=status_color, verticalalignment='top')

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Temperature (°C)', fontsize=11)
    ax.set_title('Temperature vs Time', fontsize=12, fontweight='bold', pad=8)
    ax.tick_params(labelsize=10)
    ax.minorticks_on()
    ax.grid(False)
    ax.legend(loc='lower right', fontsize=8.5)
    plt.tight_layout()
    plt.show()


# METRICS

def render_metrics(
    T_ambient, rho, c_p, L, W, thickness,
    h, target_power, V_input, I_max_rated,
    t_max, threshold, alpha,
):
    d = run_prediction(
        T_ambient, rho, c_p, L, W, thickness,
        h, target_power, V_input, I_max_rated,
        t_max, threshold, alpha,
    )

    def ok(f):     return "✓" if f else "✗"
    def st(f):     return "OK" if f else "⚠ UNSAFE"
    sep  = "─" * 44
    sep2 = "═" * 44

    lines = []
    lines.append(sep2)
    lines.append("  1. HEATER DESIGN METRICS")
    lines.append(sep2)
    lines.append(f"  Target power          {target_power:.2f} W")
    lines.append(f"  Input voltage         {V_input} V")
    lines.append(f"  Required resistance   {d['R_required']:.3f} Ω")
    #lines.append(f"  ► Yarn length         {d['yarn_length']:.1f} mm   ← GH/Wallacei")
    #lines.append(f"  ► \033[91mYarn length         {d['yarn_length']:.1f} mm\033[0m   ← GH/Wallacei")
    lines.append(f"  ► \033[38;2;214;51;131mYarn length         {d['yarn_length']:.1f} mm\033[0m   ← GH/Wallacei")

    lines.append(f"  Pattern               {d['pattern']}")
    lines.append(f"  Steel yarn RU         0.04 Ω/mm")
    lines.append("")
    lines.append(sep2)
    lines.append("  2. THERMAL PREDICTION")
    lines.append(sep2)
    lines.append(f"  Time constant         τ = {d['tau']:.1f} s")
    lines.append(f"  Temp at t_max         {d['temps'][-1]:.1f} °C  (t={t_max}s)")
    lines.append(f"  Activation threshold  {threshold} °C  (ref)")
    if d['t_thresh'] is not None:
        lines.append(f"  Time to threshold     {d['t_thresh']} s  ✓")
    else:
        lines.append(f"  Time to threshold     Not reached in {t_max} s")
        lines.append(f"    → ↑ power, ↑ t_max, or ↓ threshold")
    lines.append("")
    lines.append("  Calibration notes:")
    lines.append("  Serpentine α=0.8 → ~88°C @5W (exp ~86°C ✓)")
    lines.append("  Spiral     α=0.4 → ~57°C @5W (exp ~60°C ✓)")
    lines.append("")
    lines.append(sep2)
    lines.append("  3. SAFETY ENVELOPE")
    lines.append(sep2)
    lines.append(f"  {'Check':<26} {'Val':>7}  {'Lim':>7}  St")
    lines.append(f"  {'─'*26}  {'─'*7}  {'─'*7}  {'─'*7}")
    lines.append(
        f"  {'1. Current (I=V/R)':<26} {d['I_actual']:>6.2f}A  "
        f"{d['I_max_rated']:>5.1f}A  {ok(d['current_ok'])} {st(d['current_ok'])}"
    )
    lines.append(
        f"  {'2. Power density':<26} {d['P_density']:>5.3f}W/cm² "
        f"{d['P_density_limit']:>3.1f}W/cm²  {ok(d['p_density_ok'])} {st(d['p_density_ok'])}"
    )
    lines.append(
        f"  {'3. Linear power (P/L)':<26} {d['q_linear']:>6.4f}W/mm "
        f"{d['q_linear_limit']:>5.4f}W/mm  {ok(d['q_linear_ok'])} {st(d['q_linear_ok'])}"
    )
    if not d['current_ok']:
        lines.append(f"\n  ⚠ I={d['I_actual']:.2f}A > I_max → yarn may fuse")
        lines.append("    → ↑V, ↓P, or use higher-rated yarn.")
    if not d['p_density_ok']:
        lines.append(f"\n  ⚠ P_density {d['P_density']:.2f} W/cm² > 1.8 W/cm²")
        lines.append("    → ↓P or ↑ footprint area.")
    if not d['q_linear_ok']:
        lines.append(f"\n  ⚠ q_lin {d['q_linear']:.4f} W/mm > limit")
        lines.append("    → ↑V or ↓ target power.")
    if d['overall_safe']:
        lines.append("\n  ✓  All safety checks passed.")
    lines.append("")
    lines.append(sep2)
    lines.append("  4. BIOT NUMBER — MODEL VALIDITY")
    lines.append(sep2)
    lines.append(f"  Lc = {d['L_c']*1e3:.2f} mm   Bi = {d['Bi']:.4f}")
    if d['bi_ok']:
        lines.append("  ✓  Bi < 0.1 — lumped capacitance VALID")
    else:
        lines.append(f"  ⚠  Bi = {d['Bi']:.3f} ≥ 0.1 — MARGINAL")
        lines.append("     → ↓ thickness or verify with FEA.")
    lines.append(sep2)

    print("\n".join(lines))


# SLIDERS  (shared between both panels)

style  = {'description_width': '148px'}
sl_lay = Layout(width='320px')

T_amb_sl  = FloatSlider(min=20,   max=100,  step=1,   value=25,   description='T_amb (°C)',               style=style, layout=sl_lay)
rho_sl    = FloatSlider(min=1000, max=2200, step=10,  value=1800, description='ρ (kg/m³)',                style=style, layout=sl_lay)
cp_sl     = FloatSlider(min=1000, max=2000, step=10,  value=1500, description='Cp (J/kg°C)',              style=style, layout=sl_lay)
L_sl      = IntSlider  (min=10,   max=120,  step=1,   value=30,   description='L-footprint (mm)',         style=style, layout=sl_lay)
W_sl      = IntSlider  (min=10,   max=120,  step=1,   value=30,   description='W-footprint (mm)',         style=style, layout=sl_lay)
thk_sl    = FloatSlider(min=1,    max=10,   step=0.5, value=4,    description='Thickness (mm)',           style=style, layout=sl_lay)
h_sl      = FloatSlider(min=1,    max=50,   step=0.5, value=10,   description='h W/m²°C',                style=style, layout=sl_lay)
pwr_sl    = FloatSlider(min=0.5,  max=30,   step=0.5, value=5,    description='Target Power (W)',         style=style, layout=sl_lay)
V_sl      = IntSlider  (min=3,    max=24,   step=1,   value=5,    description='Voltage (V)',              style=style, layout=sl_lay)
Imax_sl   = FloatSlider(min=0.5,  max=5,    step=0.1, value=2.0,  description='I_max rated (A)',          style=style, layout=sl_lay)
tmax_sl   = IntSlider  (min=100,  max=1000, step=10,  value=300,  description='t_max (s)',                style=style, layout=sl_lay)
thresh_sl = IntSlider  (min=25,   max=250,  step=1,   value=40,   description='Threshold °C (ref)',       style=style, layout=sl_lay)
alpha_sl  = FloatSlider(min=0.1,  max=1.5,  step=0.1, value=0.8,  description='α (0.6=Spiral 0.8=Ser)',  style=style, layout=sl_lay)

slider_kwargs = dict(
    T_ambient   = T_amb_sl,
    rho         = rho_sl,
    c_p         = cp_sl,
    L           = L_sl,
    W           = W_sl,
    thickness   = thk_sl,
    h           = h_sl,
    target_power= pwr_sl,
    V_input     = V_sl,
    I_max_rated = Imax_sl,
    t_max       = tmax_sl,
    threshold   = thresh_sl,
    alpha       = alpha_sl,
)


# 3-PANEL LAYOUT

panel_border = '1px solid #ddd'
panel_radius = '6px'
panel_pad    = '8px'

# Panel 1: Sliders
panel1 = VBox(
    [T_amb_sl, rho_sl, cp_sl, L_sl, W_sl, thk_sl,
     h_sl, pwr_sl, V_sl, Imax_sl, tmax_sl, thresh_sl, alpha_sl],
    layout=Layout(
        width='348px',
        min_width='348px',
        padding=panel_pad,
        border=panel_border,
        border_radius=panel_radius,
        margin='0 12px 0 0',
    )
)

# Panel 2: Plot output
plot_out = interactive_output(render_plot, slider_kwargs)
panel2 = VBox(
    [plot_out],
    layout=Layout(
        width='460px',
        min_width='460px',
        padding=panel_pad,
        border=panel_border,
        border_radius=panel_radius,
        margin='0 12px 0 0',
        overflow='hidden',
    )
)

# Panel 3: Metrics output
metrics_out = interactive_output(render_metrics, slider_kwargs)
panel3 = VBox(
    [metrics_out],
    layout=Layout(
        width='440px',
        min_width='440px',
        padding=panel_pad,
        border=panel_border,
        border_radius=panel_radius,
        font_family='monospace',
        overflow_y='auto',
    )
)

display(
    HBox(
        [panel1, panel2, panel3],
        layout=Layout(align_items='flex-start', width='100%')
    )
)
