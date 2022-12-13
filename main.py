import streamlit as st
import pandas as pd
import numpy as np
from math import sin, cos, tan, asin, acos, atan, radians, degrees, sqrt

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events

from myfunctions import apparent_dip, dipdir_cond, friction_cond, slope_class
from myfunctions import factor_safety, runout, get_polar, get_coord

from collections import namedtuple
# ----------------- CONFIG -------------------------------------------
st.set_page_config(page_title='Kinematics', page_icon=None, layout="wide")

hide_table_row_index = """
    <style>
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """

st.markdown(hide_table_row_index, unsafe_allow_html=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Keyword columns
key_dip = ['dip']
key_dipdir = ['dip dir', 'dip direction', 'dipdir']
key_east = ['easting']
key_north = ['northing']
key_elev = ['elevation']

# Column names
cdip = 'Dip'
cdipdir = 'Dip Direction'
ceast = 'Easting'
cnorth = 'Northing'
celev = 'Elevation'
ccnt = 'Count'
cpof = "PoF (%)"
cbfa = "Batter Face Angle"
cbw = "Berm Width"
csh = "Batter Height"
cslasp = 'Slope Aspect'
cira = "Inter Ramp Angle"
c80run = '80% Runout'
c95run = '95% Runout'

Inputs = namedtuple('Inputs',
    ["dipcolumn", "dipdircolumn", "slopeaspect", "ira", "bfa",
    "sh", "bw", 'density', 'mattype', 'cohesion', 'friction',
    'jrc', 'jcs', 'phir', 'reposeangle', 'swellingfactor'])

# ---------------- SIDEBAR: INPUT ---------------------------------------
jrc = None
jcs = None
phir = None
cohesion = None
friction = None
# dip and dip dir file
uploaded_files = st.sidebar.file_uploader("Dip and Dip Direction file", accept_multiple_files=True)
if not uploaded_files:
    dtv = None
else:
    dtv = pd.DataFrame()
    dfn = pd.DataFrame()
    for uploaded_file in uploaded_files:
        # Can be used wherever a "file-like" object is accepted:
        df1 = pd.read_csv(uploaded_file)
        for col in df1.columns:
            if any(substring == col.lower() for substring in key_dip):
                dfn[cdip] = df1[col]
            if any(substring == col.lower() for substring in key_dipdir):
                dfn[cdipdir] = df1[col]
            if any(substring == col.lower() for substring in key_east):
                dfn[ceast] = df1[col]
            if any(substring == col.lower() for substring in key_north):
                dfn[cnorth] = df1[col]
            if any(substring == col.lower() for substring in key_elev):
                dfn[celev] = df1[col]
        # Concat the files
        dtv = pd.concat([dtv,dfn], axis=0, ignore_index=True)
# slope configurations
st.sidebar.markdown("**Slope Configurations**")
slope_aspect = st.sidebar.number_input(cslasp, key='slope_aspect', value=180,
    min_value=0, max_value=360, step=5)
bfa = st.sidebar.number_input(cbfa, key='bfa', value=60,
    min_value=0, max_value=90, step=5)
bw = st.sidebar.number_input(cbw, key='bw', value=8, min_value=0, step=1)
sh = st.sidebar.number_input(csh, key='sh', value=8, min_value=0, step=1)
# Inter ramp angle (IRA)
ira = degrees(atan(sh/(bw + (sh/tan(radians(bfa))))))
ira = round(ira,1)
st.sidebar.markdown(f"Inter Ramp Angle (IRA): {ira}°")
st.sidebar.markdown("----------")
# material parameters
st.sidebar.markdown("**Material Parameters**")
mat_type_selection = ['Mohr Coulomb', 'Barton Bandis']
mat_type = st.sidebar.radio("Type", mat_type_selection, horizontal=True)
density = st.sidebar.number_input('Density', key='density', value=2.5, min_value=0.0, step=0.1)
if mat_type == mat_type_selection[0]:
    cohesion = st.sidebar.number_input('Cohesion', key='cohesion', value=5, min_value=0, step=1)
    friction = st.sidebar.number_input('Friction Angle', key='friction', value=25, min_value=0, max_value=90, step=1)
if mat_type == mat_type_selection[1]:
    phir = st.sidebar.number_input('Phir', key='phir', value=25, min_value=0, max_value=90, step=1)
    jrc = st.sidebar.number_input('JRC', key='jrc', value=2.5, min_value=0.0, step=0.1)
    jcs = st.sidebar.number_input('JCS', key='jcs', value=5.0, min_value=0.0, step=0.1)
st.sidebar.markdown("----------")
# Runout parameters
st.sidebar.markdown("**Runout Parameters**")
repose_ang = st.sidebar.number_input('Angle of Repose', key='repoe_ang', value=35, min_value=0, max_value=90, step=1)
swell_f = st.sidebar.number_input('Swell Factor', key='swell_f', value=1.3, min_value=0.0, step=0.1)
st.sidebar.markdown("----------")
# Window Filer
window_filter_selection = ["Off", 'On']
window_filter = st.sidebar.radio("Window Filter", window_filter_selection, key='window_filter', horizontal=True)
wf_angle = st.sidebar.number_input('Window Filter Angle', key='wf_angle', value=30, min_value=0, max_value=90, step=5)
st.sidebar.markdown("----------")
# Elevation Filter
elevation_filter_selection = ["Off", 'On']
elevation_filter = st.sidebar.radio("Elevation Filter", elevation_filter_selection, key='elevation_filter', horizontal=True)
slope_elevation = st.sidebar.number_input('Slope Elevation', key='slope_elevation', value=500, min_value=0, step=10)
elevation_range = st.sidebar.number_input('Elevation Range', key='elevation_range', value=20, min_value=0, step=5)
st.sidebar.markdown("----------")


# ------------------ MAIN PAGE INPUT and INITIALISATION -------------------------------------
columns=[cbfa, csh, cira, cpof, c80run, c95run]
dpof = pd.DataFrame(columns=columns)
dm = pd.DataFrame()
xhist = []
xpof = []
ypof = []
xcoord = []
ycoord = []
bfa_original = bfa

col1, col2, col3 = st.columns([1,1,1], gap="medium")
with col1:
    min_bfa = st.number_input('Minimum BFA', key='min_bfa', value=40, min_value=0, max_value=90, step=5)
with col2:
    max_bfa = st.number_input('Maximum BFA', key='max_bfa', value=80, min_value=0, max_value=90, step=5)
with col3:
    inc_bfa = st.number_input('Increment', key='inc_bfa', value=5, min_value=0, max_value=90, step=5)

if None not in (min_bfa, max_bfa, inc_bfa):
    bfalst = np.arange(min_bfa, max_bfa+1, inc_bfa).tolist()

# ------------------ FILTER -----------------------------------------------
## Window Filter ##
if dtv is not None:
    if window_filter == window_filter_selection[1]:
        if wf_angle is not None:
            top_slope_aspect = slope_aspect
            bottom_slope_aspect = slope_aspect
            top_window_angle = slope_aspect + wf_angle
            if top_window_angle > 360:
                top_window_angle = top_window_angle - 360
                top_slope_aspect = 0
                bottom_slope_aspect = 360
            bottom_window_angle = slope_aspect - wf_angle
            if bottom_window_angle < 0:
                top_slope_aspect = 0
                bottom_window_angle = 360 + bottom_window_angle # bottom angle is negative value
                bottom_slope_aspect = 360
            dtv = dtv[((dtv[cdipdir]<=top_window_angle) & (dtv[cdipdir]>=top_slope_aspect)) |
                ((dtv[cdipdir]>=bottom_window_angle) & (dtv[cdipdir]<=bottom_slope_aspect))]

### TV dataframe before elevation filter
if dtv is not None:
    dtv_elev = dtv.copy()
    dpof_part = pd.DataFrame(columns=[celev, ccnt, cpof])
    xelpof = []
    yelpof = []
    numelpof = []
    ## Elevation Filter ##
    if elevation_filter == elevation_filter_selection[1]:
        if None not in (slope_elevation, elevation_range):
            top_elevation = slope_elevation + elevation_range
            bottom_elevation = slope_elevation - elevation_range
            dtv = dtv[(dtv[celev]<=top_elevation) & (dtv[celev]>=bottom_elevation)]
            # print(dtv[celev].unique())

# ----------------------- MAIN --------------------------------------------------------
def fos_runout(dtv, inp):
    dtv['AppDip'] = np.vectorize(apparent_dip)(dtv[inp.dipcolumn], dtv[inp.dipdircolumn],
        inp.slopeaspect)
    dtv['DipDirCond'] = np.vectorize(dipdir_cond)(dtv[inp.dipdircolumn], inp.slopeaspect)
    dtv['FrictionCond'] = np.vectorize(friction_cond)(dtv["DipDirCond"], dtv["AppDip"],
        inp.friction)
    dtv['SlopeClass'] = np.vectorize(slope_class)(dtv['AppDip'], dtv['FrictionCond'],
        inp.ira, inp.bfa)
    dtv['FOS'] = np.vectorize(factor_safety)(dtv['SlopeClass'], dtv['FrictionCond'], dtv['AppDip'],
        inp.sh, inp.bw, inp.bfa, inp.ira, inp.density, inp.mattype, inp.cohesion, inp.friction, inp.jrc, inp.jcs, inp.phir)
    dtv['RunOut'] = np.vectorize(runout)(dtv['SlopeClass'], dtv['FrictionCond'], dtv['AppDip'],
        inp.sh, inp.bw, inp.bfa, inp.ira, inp.reposeangle, inp.swellingfactor)
    return dtv

if dtv is not None:
    # ----------------- Graph coordinates ----------------------------
    # Initialise session state
    if "coord_query" not in st.session_state:
        st.session_state["coord_query"] = set()
    dtv['xycoord'] = (dtv[ceast].astype(str)+"-"+dtv[cnorth].astype(str))
    dtv['selected'] = True
    if st.session_state["coord_query"]:
        dtv.loc[~dtv["xycoord"].isin(st.session_state["coord_query"]), "selected"] = False
    df = dtv[dtv['selected']==True]

    # Coordinates graph
    fig_coord = get_coord(dtv, ceast, cnorth, 'selected')

    # ------------- Probability of Failure Calculation ----------------
    for bfa in bfalst:
        if ira != 0:
            bw2 = sh * (1/tan(radians(ira)) - 1/tan(radians(bfa)))
            bw2 = round(bw2, 1)
        else:
            bw2 = 0
        # FoS & Runout Calculation
        # inplist = [cdip, cdipdir, slope_aspect, ira, bfa, sh, bw2, density, mat_type, cohesion, friction, jrc, jcs, phir, repose_ang, swell_f]
        inp = Inputs(cdip, cdipdir, slope_aspect, ira, bfa, sh, bw2, density, mat_type,
            cohesion, friction, jrc, jcs, phir, repose_ang, swell_f)
        df = fos_runout(df, inp)

        # dtv['AppDip'] = np.vectorize(apparent_dip)(dtv[cdip], dtv[cdipdir], slope_aspect)
        # dtv['DipDirCond'] = np.vectorize(dipdir_cond)(dtv[cdipdir], slope_aspect)
        # dtv['FrictionCond'] = np.vectorize(friction_cond)(dtv["DipDirCond"], dtv["AppDip"], friction)
        # dtv['SlopeClass'] = np.vectorize(slope_class)(dtv['AppDip'], dtv['FrictionCond'], ira, bfa)
        # dtv['FOS'] = np.vectorize(factor_safety)(dtv['SlopeClass'], dtv['FrictionCond'], dtv['AppDip'], sh, bw2, bfa, ira, density, mat_type, cohesion, friction, jrc, jcs, phir)
        # dtv['RunOut'] = np.vectorize(runout)(dtv['SlopeClass'], dtv['FrictionCond'], dtv['AppDip'], sh, bw2, bfa, ira, repose_ang, swell_f)

        if bfa == bfa_original:
            dm = df.copy()
            xhist = dm['RunOut'].tolist()

        # Probability of Failure Calculation
        num_data = df.shape[0]
        num_fail = df[df['FOS'] < 1.0].count()[1]
        PoF = (num_fail/num_data) * 100
        PoF = round(PoF,1)

        runout_max = round(df['RunOut'].max(),1)
        runout_80 = round(df['RunOut'].quantile(0.8),1)
        runout_95 = round(df['RunOut'].quantile(0.95),1)

        to_append = [bfa, sh, ira, PoF, runout_80, runout_95]
        new_row = len(dpof)
        dpof.loc[new_row] = to_append

    # -------------- Probability of Failure per elevation --------------------------
    elev_min = int(dtv_elev[celev].min())
    elev_max = int(dtv_elev[celev].max())

    inp2 = Inputs(cdip, cdipdir, slope_aspect, ira, bfa_original, sh, bw, density, mat_type,
        cohesion, friction, jrc, jcs, phir, repose_ang, swell_f)
    for elv in range(elev_min, elev_max, sh):
        dtv_part = dtv_elev[(dtv_elev[celev]<=elv+sh) & (dtv_elev[celev]>=elv-sh)]
        dtv_part = fos_runout(dtv_part, inp2)
        # dtv_part['AppDip'] = np.vectorize(apparent_dip)(dtv_part[cdip], dtv_part[cdipdir], slope_aspect)
        # dtv_part['DipDirCond'] = np.vectorize(dipdir_cond)(dtv_part[cdipdir], slope_aspect)
        # dtv_part['FrictionCond'] = np.vectorize(friction_cond)(dtv_part["DipDirCond"], dtv_part["AppDip"], friction)
        # dtv_part['SlopeClass'] = np.vectorize(slope_class)(dtv_part['AppDip'], dtv_part['FrictionCond'], ira, bfa_original)
        # dtv_part['FOS'] = np.vectorize(factor_safety)(dtv_part['SlopeClass'], dtv_part['FrictionCond'], dtv_part['AppDip'],
        #     sh, bw, bfa_original, ira, density, mat_type, cohesion, friction, jrc, jcs, phir)
        num_data_part = dtv_part.shape[0]
        num_fail_part = dtv_part[dtv_part['FOS'] < 1.0].count()[1]
        PoF_part = (num_fail_part/num_data_part) * 100
        PoF_part = round(PoF_part,1)
        to_append_part = [elv, num_data_part, PoF_part]
        new_row_part = len(dpof_part)
        dpof_part.loc[new_row_part] = to_append_part
    xelpof = dpof_part[celev]
    yelpof = dpof_part[cpof]
    numelpof = dpof_part[ccnt]

    xpof = dpof[cbfa]
    ypof = dpof[cpof]

    xcoord = dtv[ceast]
    ycoord = dtv[cnorth]

    # ------------- Scatter plot of POF vs BFA -------------------------
    sca_pof = go.Figure(data=go.Scatter(x=xpof, y=ypof, mode='lines+markers'))
    sca_pof.update_layout(
        title = dict(
            text="PoF vs BFA",
            y=0.9, # new
            x=0.5,
            xanchor='center',
            yanchor='top'),
        showlegend=False)
    sca_pof.update_xaxes(
        title_text = "Batter Face Angle (°)",
        title_standoff = 10)
    sca_pof.update_yaxes(
        title_text = "Probability of Failure (%)",
        title_standoff = 10,
        range=[-5,105])

    # ------------ Graph of dip and dip direction ------------------------
    fig = get_polar(df, bfa_original, slope_aspect)

    # ------------ Histogram of runout distance --------------------------
    xhist_clean = [x for x in xhist if str(x) != 'nan']

    hist, bin_edges = np.histogram(xhist_clean, bins=20, density=True)
    cdf = np.cumsum(hist * np.diff(bin_edges))*100
    hist = hist*100
    # fig_hist = go.Figure(data=[
    #     go.Bar(x=bin_edges, y=hist, name='Histogram'),
    #     go.Scatter(x=bin_edges, y=cdf, name='CDF')])
    fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
    fig_hist.add_trace(
        go.Bar(x=bin_edges, y=hist, name='Histogram'),
        secondary_y=False,)
    fig_hist.add_trace(
        go.Scatter(x=bin_edges, y=cdf, name='CDF'),
        secondary_y=True,)

    fig_hist.update_layout(
        title = dict(
            text=f"Runout Distribution <br> BFA={bfa_original}°, BH={sh}, BW={bw}",
            y=0.9, # new
            x=0.5,
            xanchor='center',
            yanchor='top' # new
            ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
            ),
        )
    fig_hist.update_xaxes(
        title_text = "Runout",
        title_standoff = 10)
    fig_hist.update_yaxes(
        title_text = "Cummulative Distribution (%)",
        title_standoff = 10,
        secondary_y=True)
    fig_hist.update_yaxes(
        title_text = "Frequency (%)",
        title_standoff = 10,
        secondary_y=False,
        showgrid=False)

    # ------------------ FIGURE PLOTTING --------------------------------------
    col1, col2 = st.columns(2, gap='medium')
    with col1:
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    with col2:
        # st.plotly_chart(fig_coord, use_container_width=True, theme="streamlit")
        selected_points = plotly_events(fig_coord, select_event=True,  override_height=500)

    # Selected points
    current_query = {}
    current_query["coord_query"] = {f"{el['x']}-{el['y']}" for el in selected_points}
    # print(current_query['coord_query'])
    # print(dtv[dtv['xycoord'].str.contains('23051.87')].head())

    # Update session state
    rerun = False
    # print(current_query["coord_query"])
    # print(st.session_state["coord_query"])
    if current_query["coord_query"] - st.session_state["coord_query"]:
        st.session_state['coord_query'] = current_query["coord_query"]
        rerun = True
    if rerun:
        st.experimental_rerun()

    # Row 2
    col1, col2 = st.columns(2, gap='medium')
    with col1:
        st.plotly_chart(fig_hist, use_container_width=True, theme="streamlit")
    with col2:
        st.plotly_chart(sca_pof, use_container_width=True, theme="streamlit")

    # ------------------- Table of Summary ------------------------------------
    # style
    th_props = [
        ('font-size', '16px'),
        ('text-align', 'center'),
        ('font-weight', 'bold'),
        ('color', '#6d6d6d'),
        ('background-color', '#f7ffff')]

    td_props = [
        ('font-size', '16px'),
        ('text-align', 'center')]

    styles = [
        dict(selector="th", props=th_props),
        dict(selector="td", props=td_props)]

    df2=dpof.style.set_properties(**{'text-align': 'center'}).format({
        cbfa: '{:.0f}',
        csh: '{:.0f}',
        cira: '{:.1f}',
        cpof: '{:.1f}',
        c80run: '{:.1f}',
        c95run: '{:.1f}'}).set_table_styles(styles).hide_index()

    st.write(df2.to_html(), unsafe_allow_html=True)

    # ------------------- Bar Elevation Graph -----------------------------
    st.markdown("-------------")
    st.markdown("**Bar Elevation Plot**")
    threshold_count = st.number_input('Threshold Number of Data', key='threshold_count',
        value=10, min_value=0, step=1)
    ### Bar graph of Elevation vs PoF
    dtemp = pd.DataFrame([xelpof, yelpof, numelpof]).transpose()
    dtemp.columns = [celev, cpof, ccnt]
    dtemp['Threshold_Count'] = np.where(dtemp[ccnt]>threshold_count, "Above", "Below")
    color_discrete_map = {'Above': px.colors.qualitative.T10[3], 'Below': px.colors.qualitative.T10[2]}
    bar_elv_pof = px.bar(dtemp, x=cpof, y=celev, orientation='h',
        hover_data=[ccnt], color='Threshold_Count', color_discrete_map=color_discrete_map)
    bar_elv_pof.update_layout(
        title = dict(
            text=f"Elevation vs PoF: BFA={bfa_original}°, BH={sh}, BW={bw}",
            y=0.95, # new
            x=0.5,
            xanchor='center',
            yanchor='top' # new
            ),)
    bar_elv_pof.update_coloraxes(showscale=False)

    st.plotly_chart(bar_elv_pof, use_container_width=True, theme="streamlit")
