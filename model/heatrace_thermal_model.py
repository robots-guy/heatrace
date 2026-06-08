"""
####################################
HeaTrace - Thermal Prediction Model
####################################

Interactive lumped-capacitance thermal model for embedded resistive yarn heaters
in silicone soft-robot bodies. Given a user-specified thermal target (power,
voltage, geometry), the model:

  1. Sizes the heater  - computes the required electrical resistance and the
                         corresponding conductive yarn length.
  2. Predicts thermal response - estimates the transient temperature rise
                         T(t) using a first-order lumped-capacitance model.
  3. Gates safety      - checks current, power density, and linear power
                         against material/operational limits.
  4. Validates the model assumption - reports the Biot number to confirm the
                         lumped-capacitance approximation is justified.

The accompanying parametric (Grasshopper) workflow consumes the computed yarn
length and routes it into a printable serpentine or spiral pattern.

NOTE ON DIRECTION OF USE: the model is target-driven. The user specifies the
desired thermal/electrical target, and the required heater length is derived
from it (see `R_required` and `yarn_length` below). The heat distribution along
the embedded yarn is assumed uniform; this assumption holds well for serpentine
layouts and less well for tightly wound spirals, where inter-turn coupling
introduces deviations (see manuscript residual analysis).
"""

# Standard scientific + Jupyter-widget stack.
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ipywidgets import (
    interactive_output, FloatSlider, IntSlider,
    HBox, VBox, Layout, Output
)
from IPython.display import display, HTML
import io, base64


# ============================================================================
# CORE PREDICTION
# ----------------------------------------------------------------------------
# Single function that performs all four steps (sizing, thermal prediction,
# safety gating, model-validity check) and returns every computed quantity in
# a dictionary for downstream display.
# ============================================================================

def run_prediction(
    T_ambient, rho, c_p, L, W, thickness,
    h, target_power, V_input, I_max_rated,
    t_max, threshold, alpha,
):
    # --- Fixed physical constants and material/process parameters
    # These are held constant across the design space; adjust to match a
    # different substrate or conductor if reproducing the workflow.
    epsilon         = 0.9        # Surface emissivity of silicone (-)
    sigma           = 5.67e-8    # Stefan-Boltzmann constant (W/m^2 K^4)
    k_s             = 0.22       # Thermal conductivity of DS30 silicone (W/m K)
    RU              = 0.04       # Steel yarn resistance per unit length (Ohm/mm)
    P_density_limit = 1.8        # Max safe areal power density (W/cm^2)

    # --- Geometry: derive volume, area, characteristic length 
    # Inputs L, W, thickness are in mm; convert to SI (m) for thermal terms.
    V_vol    = L * 1e-3 * W * 1e-3 * thickness * 1e-3   # Body volume (m^3)
    A        = L * 1e-3 * W * 1e-3                       # Top surface area (m^2)
    L_c      = V_vol / A                                # Characteristic length (m) = V/A
    area_cm2 = (L * 0.1) * (W * 0.1)                    # Footprint area (cm^2) for power density

    # --- Biot number: justifies the lumped-capacitance assumption 
    # Bi = h*L_c / k_s. If Bi < 0.1, internal conduction is fast relative to
    # surface convection, so a single uniform body temperature is a valid
    # approximation (i.e. the lumped model is appropriate).
    Bi    = (h * L_c) / k_s
    bi_ok = Bi < 0.1

    # --- Heater sizing (TARGET -> GEOMETRY) 
    # From the electrical target: required resistance R = V^2 / P, then the
    # yarn length needed to realise that resistance given the yarn's Ohm/mm.
    # This is the value passed to the parametric pattern generator.
    R_required  = (V_input ** 2) / target_power   # Required heater resistance (Ohm)
    yarn_length = R_required / RU                 # Required yarn length (mm)

    # --- Safety envelope checks 
    I_actual       = V_input / R_required          # Operating current (A)
    current_ok     = I_actual <= I_max_rated       # Current must stay under yarn rating
    P_density      = target_power / area_cm2        # Areal power density (W/cm^2)
    p_density_ok   = P_density <= P_density_limit  # Avoid localised overheating
    q_linear       = target_power / yarn_length     # Linear power along yarn (W/mm)
    q_linear_limit = (I_max_rated ** 2) * RU       # Max linear power before fusing (W/mm)
    q_linear_ok    = q_linear <= q_linear_limit
    overall_safe   = current_ok and p_density_ok and q_linear_ok  # All must pass

    # --- Effective heat-loss coefficient (parallel pathways) 
    # Total heat loss combines convection, (linearised) radiation, and
    # conduction to surroundings. Summed as parallel conductances (W/K).
    h_conv = h * A                                                  # Convective loss
    h_rad  = 4 * epsilon * sigma * A * (T_ambient + 273.15) ** 3   # Linearised radiative loss
    h_cond = (k_s * A) / L_c                                        # Conductive loss
    h_eff  = h_conv + h_rad + h_cond                                # Effective total (W/K)

    # Thermal time constant tau = (rho * V * c_p) / h_eff  (seconds).
    # Governs how quickly the body approaches its steady temperature.
    tau    = (rho * V_vol * c_p) / h_eff

    # --- Transient temperature prediction
    # First-order lumped-capacitance solution:
    #   T(t) = T_amb + (P * alpha / h_eff) * (1 - exp(-t / tau))
    # alpha is an empirical efficiency/coupling factor (calibrated per pattern,
    # see calibration notes in the metrics panel). Records the first time the
    # temperature crosses the user's reference activation threshold.
    times, temps = [], []
    t_thresh = None
    for t in range(0, t_max + 1, 1):
        T = T_ambient + (target_power * alpha / h_eff) * (1 - np.exp(-t / tau))
        times.append(t)
        temps.append(T)
        if T >= threshold and t_thresh is None:
            t_thresh = t

    # --- Map calibration factor alpha to heater pattern ---------------------
    # alpha is empirically associated with a layout: lower alpha (~spiral),
    # higher alpha (~serpentine). Thresholds are tunable to the user's setup.
    if alpha <= 0.65:
        pattern = f"Spiral (alpha={alpha})"
    elif alpha >= 0.75:
        pattern = f"Serpentine (alpha={alpha})"
    else:
        pattern = f"Custom (alpha={alpha})"

    # Return all computed quantities for the plot/metrics panels.
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


# ============================================================================
# PLOT PANEL
# ----------------------------------------------------------------------------
# Renders the predicted temperature-vs-time curve, the activation threshold,
# and a colour-coded safety status. Colour shifts to a warning hue when any
# safety check fails.
# ============================================================================

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

    # Y-axis scaled to show both the threshold and the full predicted rise.
    y_upper = max(threshold * 1.25, max(d['temps']) * 1.08) + 5
    ax.set_ylim(d['T_ambient'] - 5, y_upper)

    # Curve colour encodes safety: warning red if any check failed.
    curve_color = '#B03A2E' if not d['overall_safe'] else '#D85A30'
    legend_str  = (
        f"P = {target_power:.1f} W  |  V = {V_input} V  |  "
        f"{d['pattern']}  |  L = {d['yarn_length']:.0f} mm"
    )
    ax.plot(d['times'], d['temps'], color=curve_color, linewidth=2, label=legend_str)

    # Reference line for the user-specified activation threshold.
    ax.axhline(threshold, color='#444441', linestyle='--', linewidth=1.2,
               label=f"Activation threshold: {threshold} C")

    # On-plot safety status badge.
    status_txt   = "Safe" if d['overall_safe'] else "Check safety"
    status_color = '#1A7A4A' if d['overall_safe'] else '#B03A2E'
    ax.text(0.02, 0.96, status_txt,
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            color=status_color, verticalalignment='top')

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Temperature (C)', fontsize=11)
    ax.set_title('Temperature vs Time', fontsize=12, fontweight='bold', pad=8)
    ax.tick_params(labelsize=10)
    ax.minorticks_on()
    ax.grid(False)
    ax.legend(loc='lower right', fontsize=8.5)
    plt.tight_layout()
    plt.show()


# ============================================================================
# METRICS PANEL
# ----------------------------------------------------------------------------
# Text report grouping the four output categories:
#   1. Heater design metrics   (resistance, derived yarn length, pattern)
#   2. Thermal prediction      (time constant, temperatures, threshold timing)
#   3. Safety envelope         (current / power density / linear power checks)
#   4. Biot number             (lumped-capacitance validity)
# The derived yarn length is highlighted as the key handoff to the parametric
# (Grasshopper/Wallacei) pattern generator.
# ============================================================================

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

    # Small helpers for formatting pass/fail markers in the text report.
    def ok(f):     return "[OK]" if f else "[X]"
    def st(f):     return "OK" if f else "UNSAFE"
    sep  = "-" * 44
    sep2 = "=" * 44

    lines = []

    # --- Section 1: heater design metrics -----------------------------------
    lines.append(sep2)
    lines.append("  1. HEATER DESIGN METRICS")
    lines.append(sep2)
    lines.append(f"  Target power          {target_power:.2f} W")
    lines.append(f"  Input voltage         {V_input} V")
    lines.append(f"  Required resistance   {d['R_required']:.3f} Ohm")
    # Derived yarn length: this is the value exported to the parametric
    # pattern generator (Grasshopper/Wallacei). Highlighted in colour.
    lines.append(f"  > \033[38;2;214;51;131mYarn length         {d['yarn_length']:.1f} mm\033[0m   <- GH/Wallacei")
    lines.append(f"  Pattern               {d['pattern']}")
    lines.append(f"  Steel yarn RU         0.04 Ohm/mm")
    lines.append("")

    # --- Section 2: thermal prediction --------------------------------------
    lines.append(sep2)
    lines.append("  2. THERMAL PREDICTION")
    lines.append(sep2)
    lines.append(f"  Time constant         tau = {d['tau']:.1f} s")
    lines.append(f"  Temp at t_max         {d['temps'][-1]:.1f} C  (t={t_max}s)")
    lines.append(f"  Activation threshold  {threshold} C  (ref)")
    if d['t_thresh'] is not None:
        lines.append(f"  Time to threshold     {d['t_thresh']} s  [OK]")
    else:
        lines.append(f"  Time to threshold     Not reached in {t_max} s")
        lines.append(f"    -> increase power, increase t_max, or lower threshold")
    lines.append("")
    # Empirical calibration anchors: predicted vs experimentally observed
    # surface temperatures for the two reference patterns.
    lines.append("  Calibration notes:")
    lines.append("  Serpentine alpha=0.8 -> ~88C @5W (exp ~86C OK)")
    lines.append("  Spiral     alpha=0.4 -> ~57C @5W (exp ~60C OK)")
    lines.append("")

    # --- Section 3: safety envelope -----------------------------------------
    lines.append(sep2)
    lines.append("  3. SAFETY ENVELOPE")
    lines.append(sep2)
    lines.append(f"  {'Check':<26} {'Val':>7}  {'Lim':>7}  St")
    lines.append(f"  {'-'*26}  {'-'*7}  {'-'*7}  {'-'*7}")
    # Check 1: operating current must not exceed the yarn's rated current.
    lines.append(
        f"  {'1. Current (I=V/R)':<26} {d['I_actual']:>6.2f}A  "
        f"{d['I_max_rated']:>5.1f}A  {ok(d['current_ok'])} {st(d['current_ok'])}"
    )
    # Check 2: areal power density must stay under the substrate limit.
    lines.append(
        f"  {'2. Power density':<26} {d['P_density']:>5.3f}W/cm2 "
        f"{d['P_density_limit']:>3.1f}W/cm2  {ok(d['p_density_ok'])} {st(d['p_density_ok'])}"
    )
    # Check 3: linear power along the yarn must stay under the fusing limit.
    lines.append(
        f"  {'3. Linear power (P/L)':<26} {d['q_linear']:>6.4f}W/mm "
        f"{d['q_linear_limit']:>5.4f}W/mm  {ok(d['q_linear_ok'])} {st(d['q_linear_ok'])}"
    )
    # Targeted remediation hints when a specific check fails.
    if not d['current_ok']:
        lines.append(f"\n  ! I={d['I_actual']:.2f}A > I_max -> yarn may fuse")
        lines.append("    -> increase V, decrease P, or use higher-rated yarn.")
    if not d['p_density_ok']:
        lines.append(f"\n  ! P_density {d['P_density']:.2f} W/cm2 > 1.8 W/cm2")
        lines.append("    -> decrease P or increase footprint area.")
    if not d['q_linear_ok']:
        lines.append(f"\n  ! q_lin {d['q_linear']:.4f} W/mm > limit")
        lines.append("    -> increase V or decrease target power.")
    if d['overall_safe']:
        lines.append("\n  [OK]  All safety checks passed.")
    lines.append("")

    # --- Section 4: Biot number (model validity) ----------------------------
    lines.append(sep2)
    lines.append("  4. BIOT NUMBER - MODEL VALIDITY")
    lines.append(sep2)
    lines.append(f"  Lc = {d['L_c']*1e3:.2f} mm   Bi = {d['Bi']:.4f}")
    if d['bi_ok']:
        lines.append("  [OK]  Bi < 0.1 - lumped capacitance VALID")
    else:
        lines.append(f"  !  Bi = {d['Bi']:.3f} >= 0.1 - MARGINAL")
        lines.append("     -> decrease thickness or verify with FEA.")
    lines.append(sep2)

    print("\n".join(lines))


# ============================================================================
# INTERACTIVE SLIDERS
# ----------------------------------------------------------------------------
# Shared input widgets driving both the plot and metrics panels. Grouped by
# role: ambient/material properties, geometry, electrical inputs, and the
# calibration factor alpha.
# ============================================================================

style  = {'description_width': '148px'}
sl_lay = Layout(width='320px')

# Ambient and material thermal properties.
T_amb_sl  = FloatSlider(min=20,   max=100,  step=1,   value=25,   description='T_amb (C)',                style=style, layout=sl_lay)
rho_sl    = FloatSlider(min=1000, max=2200, step=10,  value=1800, description='rho (kg/m3)',              style=style, layout=sl_lay)
cp_sl     = FloatSlider(min=1000, max=2000, step=10,  value=1500, description='Cp (J/kgC)',               style=style, layout=sl_lay)

# Body geometry (footprint and thickness).
L_sl      = IntSlider  (min=10,   max=120,  step=1,   value=30,   description='L-footprint (mm)',         style=style, layout=sl_lay)
W_sl      = IntSlider  (min=10,   max=120,  step=1,   value=30,   description='W-footprint (mm)',         style=style, layout=sl_lay)
thk_sl    = FloatSlider(min=1,    max=10,   step=0.5, value=4,    description='Thickness (mm)',           style=style, layout=sl_lay)

# Heat-transfer and electrical inputs.
h_sl      = FloatSlider(min=1,    max=50,   step=0.5, value=10,   description='h W/m2C',                  style=style, layout=sl_lay)
pwr_sl    = FloatSlider(min=0.5,  max=30,   step=0.5, value=5,    description='Target Power (W)',         style=style, layout=sl_lay)
V_sl      = IntSlider  (min=3,    max=24,   step=1,   value=5,    description='Voltage (V)',              style=style, layout=sl_lay)
Imax_sl   = FloatSlider(min=0.5,  max=5,    step=0.1, value=2.0,  description='I_max rated (A)',          style=style, layout=sl_lay)

# Simulation window, reference threshold, and pattern calibration factor.
tmax_sl   = IntSlider  (min=100,  max=1000, step=10,  value=300,  description='t_max (s)',                style=style, layout=sl_lay)
thresh_sl = IntSlider  (min=25,   max=250,  step=1,   value=40,   description='Threshold C (ref)',        style=style, layout=sl_lay)
alpha_sl  = FloatSlider(min=0.1,  max=1.5,  step=0.1, value=0.8,  description='alpha (0.6=Spiral 0.8=Ser)', style=style, layout=sl_lay)

# Map slider widgets to the keyword arguments of run_prediction.
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


# ============================================================================
# 3-PANEL LAYOUT
# ----------------------------------------------------------------------------
# Panel 1: input sliders | Panel 2: live temperature plot | Panel 3: metrics.
# Both output panels recompute automatically when any slider changes.
# ============================================================================

panel_border = '1px solid #ddd'
panel_radius = '6px'
panel_pad    = '8px'

# Panel 1: slider stack (inputs).
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

# Panel 2: reactive temperature-vs-time plot.
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

# Panel 3: reactive text metrics report.
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

# Render the three panels side by side.
display(
    HBox(
        [panel1, panel2, panel3],
        layout=Layout(align_items='flex-start', width='100%')
    )
)
