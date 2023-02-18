import pandas as pd
import numpy as np
from math import sin, cos, tan, asin, acos, atan, radians, degrees, sqrt, log10

import plotly.graph_objs as go
import plotly.express as px

# round off to nearest 5
def myround(x, base=5):
    return base * round(x/base)

# Apparent Dip Calculation
def apparent_dip(dip, dipdir, slope_aspect):
    # dip = row[0]
    # dipdir = row[1]
    app_dip = degrees(atan(tan(radians(dip))*sin(radians(90-(slope_aspect-dipdir)))))
    return round(app_dip,2)
# Dip Direction and slope aspect relationship
def dipdir_cond(dipdir, slope_aspect):
    # dipdir = row[1]
    cond_30deg = sin(radians(90-(slope_aspect - dipdir)))
    if cond_30deg >= sin(radians(60)):
        return True
    else:
        return False

# Friction Condition Check
def friction_cond(dd_cond, app_dip, friction_angle, phir):
    # dd_cond = row['DipDirCond']
    # app_dip = row['AppDip']
    if friction_angle is not None:
        check_friction = friction_angle
    else:
        check_friction = phir

    if dd_cond and app_dip >= check_friction:
        return True
    else:
        return False

# Categorisation of Class 1, 2, 3, 4
def slope_class(app_dip, fric_cond, ira, bfa):
    if app_dip <= 0:
        return 0
    elif not fric_cond and app_dip > 0:
        return 1
    elif ira <= app_dip < bfa:
        return 3
    elif app_dip >= bfa:
        return 4
    else:
        return 2

def factor_safety(slope_class, fric_cond, app_dip, sh, bw, bfa, ira, dens, input_type, c, phi, jrc, jcs, phir):
    app_dip = radians(app_dip)
    bfa = radians(bfa)

    if slope_class in [2,3]:
        # z = sh*(1 - sqrt(((tan(bfa))**(-1))*tan(app_dip)))
        # a1 = (sh - z)*(sin(app_dip))**(-1)
        # a2 is the diagonal line of apparent  dip failure surface
        a2 = sh/(sin(app_dip))
        # w1 = (0.5*dens*sh**2)*((1-(z/sh)**2)*((tan(app_dip))**(-1)))
        # w2 is the weight of upper area of failure mass: w2 = Upper Area x density
        w2 = (0.5*dens*sh**2)*(((tan(app_dip))**(-1))-(tan(bfa))**(-1))
        if input_type == "Mohr Coulomb":
            phi = radians(phi)
            m2t = c*a2/9.81 + (w2*cos(app_dip))*tan(phi)
        else:
            m2t = (w2*cos(app_dip))*tan(radians(phir + jrc * log10(jcs * 1000/(w2*cos(app_dip)))))
        m2b = w2*sin(app_dip)
        return round(m2t/m2b,2)
    else:
        return np.nan

def runout(slope_class, fric_cond, app_dip, sh, bw, bfa, ira, repose_ang, swell_f):
    app_dip = radians(app_dip)
    bfa = radians(bfa)
    repose_ang = radians(repose_ang)
    if slope_class in [2,3]:
        z = sh*(1 - sqrt(((tan(bfa))**(-1))*tan(app_dip)))
        a1 = (sh - z)*(sin(app_dip))**(-1)
        a2 = sh*(sin(app_dip))**(-1)
        down_face_l = bw * sin(app_dip)/sin(bfa - app_dip)
        obtuse_angle = 180 - degrees(bfa)
        max_fail_plane_l = sqrt(bw**2 + down_face_l**2 - 2*bw*down_face_l*cos(radians(obtuse_angle)))
        if slope_class == 2:
            affected_bh = max_fail_plane_l*sin(app_dip)
            affected_pct = affected_bh/sh*100
            fail_plane_l = max_fail_plane_l * affected_pct/100
        else:
            affected_bh = sh
            affected_pct = 100
            fail_plane_l = a2 * affected_pct/100
        x = (1/tan(app_dip))-(1/tan(bfa))
        y = (1/tan(repose_ang)) - (1/tan(bfa))
        if x >= 0 and y >= 0:
            runout = fail_plane_l*sin(app_dip)*sqrt(x*y*swell_f)
            return round(runout,2)
        else:
            return np.nan
    else:
        return np.nan

def get_polar(dtv, bfa, slope_aspect):
    bfa_lst = [bfa]
    sa_lst = [slope_aspect]

    if not dtv.empty:
        cols = dtv.columns
        r = dtv[cols[0]]
        theta = dtv[cols[1]]

    else:
        r = []
        theta = []

    # Graph of dip and dip direction
    fig = go.Figure(data=go.Scatterpolar(
        r = r, theta = theta, mode = 'markers', name='Structure',
        marker=dict(size=8, opacity=0.3)))

    fig.add_trace(
        go.Scatterpolar(
            r = bfa_lst, theta = sa_lst, mode='markers', name='Slope',
            marker=dict(size=12, color='Orange',
                line=dict(
                    color='Black',
                    width=2)
                )))

    fig.update_layout(
        title = dict(
            text="Polar Plot of TV data",
            y=0.9, # new
            x=0.5,
            xanchor='center',
            yanchor='top' # new
            ),
        # showlegend = False,
        polar = dict(
            radialaxis_tickfont_size = 10,
            angularaxis = dict(
                tickfont_size=10,
                rotation=90, # start position of angular axis
                direction="clockwise")),)
    return fig

def get_coord(dtv, ceast, cnorth, selected):
    colors_selected = ["rgba(255,160,122,0.8)", "rgba(99,110,250,0.2)"]
    # if not dtv.empty:
    #     cols = dtv.columns
    #     x = dtv[cols[2]]
    #     y = dtv[cols[3]]

    # else:
    #     x = []
    #     y = []

    # find selected group and unselected group
    groups = dtv[selected].unique()
    groups = sorted(groups, reverse=True)
    # add each group to Figure()
    fig_coord = go.Figure()
    i = 0
    for group in groups:
        df_group = dtv[dtv[selected] == group]
        if group == True:
            name = 'Selected'
        else:
            name = 'Unselected'
        trace = go.Scatter(
            x=df_group[ceast], y=df_group[cnorth], mode='markers', name=name,
            marker=dict(color=colors_selected[i], size=9))
        fig_coord.add_trace(trace)
        i += 1
    fig_coord.update_layout(
        title = dict(
            text="Coordinates",
            y=0.9, # new
            x=0.5,
            xanchor='center',
            yanchor='top' # new
            ),
        showlegend=True,
        dragmode='lasso')
    fig_coord.update_xaxes(
        title_text = "x",
        title_standoff = 10)
    fig_coord.update_yaxes(
        title_text = "y",
        title_standoff = 10)
    fig_coord.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1.0, y1=1.0,
        line=dict(color="black", width=2))

    return fig_coord
