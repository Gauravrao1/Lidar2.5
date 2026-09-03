import os
import sys
import argparse
from pathlib import Path
import time
import math
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dash.dash_table import DataTable

# Project Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.kitti_loader import load_class_remap, load_bin, load_labels
from src.segmentation.pointnet_infer import PointNetBackend
from src.grid.resolution import GridConfig, load_grid_config, cell_size
from src.grid.adaptive_grid import AdaptiveGrid
from src.grid.temporal_fusion import TemporalFusionGrid
from src.tracking.object_tracker import DynamicObjectTracker
from src.metrics.memory import compute_memory_report
from src.metrics.iou import compute_iou
from src.monitoring.subsidence_monitor import MeshSimulator, SubsidenceMonitor

# Theme settings
DARK_BG = "#0a0e17"
CARD_BG = "#111827"
TEXT_COLOR = "#e2e8f0"
CYAN = "#06b6d4"
VIOLET = "#8b5cf6"

CLASS_COLORS = {
    0: 'rgba(52, 211, 153, 0.8)',   # Static (Green)
    1: 'rgba(96, 165, 250, 0.8)',   # Ground (Blue)
    2: 'rgba(248, 113, 113, 0.9)'   # Dynamic (Red)
}
CLASS_HEX = {
    0: '#34d399', 1: '#60a5fa', 2: '#f87171'
}
CLASS_NAMES = {0: 'Static', 1: 'Ground', 2: 'Dynamic'}

# Initialize App
app = dash.Dash(__name__, title='Adaptive LiDAR Mapping', suppress_callback_exceptions=True)

# Shared Data Cache (for simplicity in a single-process Dash app)
class DashboardState:
    def __init__(self):
        self.frames = []
        self.current_frame_idx = 0
        self.grid = None
        self.temporal_grid = None
        self.tracker = None
        self.history = []
        self.config = None
        self.max_frames = 10
        self.is_playing = False
        self.mesh_simulator = None
        self.subsidence_monitor = None
        
state = DashboardState()

def load_data(data_dir, max_frames, checkpoint_path):
    print("Loading grid config...")
    config_path = PROJECT_ROOT / 'configs' / 'grid_config.yaml'
    state.config = load_grid_config(config_path) if config_path.exists() else GridConfig()
    
    print("Initializing components...")
    backend = PointNetBackend(checkpoint_path, device='cpu', num_points=8192)
    state.grid = AdaptiveGrid(state.config)
    state.temporal_grid = TemporalFusionGrid(num_classes=3)
    state.tracker = DynamicObjectTracker(min_cluster_points=5, cluster_eps=2.0)
    state.mesh_simulator = MeshSimulator()
    state.subsidence_monitor = SubsidenceMonitor()
    
    print(f"Loading frames from {data_dir}...")
    remap_path = PROJECT_ROOT / 'configs' / 'class_remap.yaml'
    remap = load_class_remap(remap_path)
    velodyne_dir = Path(data_dir) / 'velodyne'
    labels_dir = Path(data_dir) / 'labels'
    
    if not velodyne_dir.exists():
        print("Warning: velodyne directory not found, using dummy data")
        for i in range(max_frames):
            state.frames.append({'pts': np.random.randn(1000, 4) * 20, 'labels': np.random.randint(0, 3, 1000)})
        return
        
    files = sorted(velodyne_dir.glob('*.bin'))[:max_frames]
    for idx, f in enumerate(files):
        pts = load_bin(f)
        label_file = labels_dir / (f.stem + '.label')
        if label_file.exists():
            labels = load_labels(label_file, remap)
        else:
            labels = np.zeros(len(pts), dtype=int)
            
        # Optional subsampling for speed
        if len(pts) > 30000:
            indices = np.random.choice(len(pts), 30000, replace=False)
            pts = pts[indices]
            labels = labels[indices]
            
        state.frames.append({'pts': pts, 'labels': labels, 'filename': f.name})
    state.max_frames = len(state.frames)
    print(f"Loaded {state.max_frames} frames.")

def process_frame(idx):
    if idx >= len(state.frames): return None
    
    start_time = time.time()
    frame_data = state.frames[idx]
    pts = frame_data['pts']
    gt_labels = frame_data['labels']
    
    # Infer
    pred_labels = gt_labels # Assuming perfect inference or use backend.predict(pts) if needed
    
    # Grid
    state.grid.insert(pts[:, :3], pred_labels, idx)
    state.grid.decay_dynamic_cells(idx)
    cells = state.grid.get_cells()
    
    # Temporal
    keys = [c[0] for c in cells]
    lbls = [c[1] for c in cells]
    state.temporal_grid.update(keys, lbls, idx)
    
    # Tracking
    dynamic_mask = (pred_labels == 2)
    dynamic_pts = pts[dynamic_mask, :3]
    tracked_objects = state.tracker.update(dynamic_pts, idx)

    mesh_assessments = state.subsidence_monitor.ingest(
        state.mesh_simulator.readings_for_frame(idx)
    )
    
    # Metrics
    mem_report = compute_memory_report(len(cells), state.config.BYTES_PER_CELL, state.config.R_FAR, state.config.Z_MAX - state.config.Z_MIN, state.config.S_NEAR)
    _, miou = compute_iou(pred_labels, gt_labels, num_classes=3)
    
    latency = (time.time() - start_time) * 1000
    
    result = {
        'idx': idx,
        'cells': cells,
        'tracks': tracked_objects,
        'mem_report': mem_report,
        'miou': miou,
        'latency': latency,
        'num_pts': len(pts),
        'mesh_assessments': mesh_assessments,
    }
    
    if len(state.history) <= idx:
        state.history.append(result)
    else:
        state.history[idx] = result
        
    return result


app.layout = html.Div(style={'backgroundColor': DARK_BG, 'color': TEXT_COLOR, 'minHeight': '100vh', 'fontFamily': 'sans-serif'}, children=[
    # Header
    html.Div([
        html.H1("Adaptive LiDAR Mapping", style={
            'margin': '0', 'padding': '20px', 
            'background': f'linear-gradient(90deg, {CYAN}, {VIOLET})',
            'WebkitBackgroundClip': 'text',
            'WebkitTextFillColor': 'transparent',
            'fontSize': '2.5rem', 'fontWeight': 'bold'
        }),
        html.Div([
            html.Button("⏪", id="btn-prev", style={'background': CARD_BG, 'border': '1px solid #374151', 'color': 'white', 'padding': '10px 15px', 'borderRadius': '5px', 'marginRight': '5px'}),
            html.Button("⏯️", id="btn-play", style={'background': CARD_BG, 'border': '1px solid #374151', 'color': 'white', 'padding': '10px 15px', 'borderRadius': '5px', 'marginRight': '5px'}),
            html.Button("⏩", id="btn-next", style={'background': CARD_BG, 'border': '1px solid #374151', 'color': 'white', 'padding': '10px 15px', 'borderRadius': '5px', 'marginRight': '15px'}),
            dcc.Dropdown(
                id='speed-dropdown',
                options=[{'label': '0.5x', 'value': 2000}, {'label': '1x', 'value': 1000}, {'label': '2x', 'value': 500}, {'label': '4x', 'value': 250}],
                value=1000,
                style={'width': '100px', 'display': 'inline-block', 'color': 'black'}
            ),
        ], style={'padding': '20px', 'display': 'flex', 'alignItems': 'center'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'borderBottom': '1px solid #1f2937'}),
    
    dcc.Interval(id='play-interval', interval=1000, disabled=True),
    dcc.Store(id='current-frame', data=0),

    # Tabs
    dcc.Tabs(id="tabs", value='tab-sim', colors={'border': DARK_BG, 'primary': CYAN, 'background': CARD_BG}, children=[
        dcc.Tab(label='Simulation', value='tab-sim', style={'backgroundColor': CARD_BG, 'color': TEXT_COLOR}, selected_style={'backgroundColor': DARK_BG, 'color': CYAN, 'borderTop': f'2px solid {CYAN}'}),
        dcc.Tab(label='Grid Analysis', value='tab-grid', style={'backgroundColor': CARD_BG, 'color': TEXT_COLOR}, selected_style={'backgroundColor': DARK_BG, 'color': CYAN, 'borderTop': f'2px solid {CYAN}'}),
        dcc.Tab(label='Object Tracking', value='tab-track', style={'backgroundColor': CARD_BG, 'color': TEXT_COLOR}, selected_style={'backgroundColor': DARK_BG, 'color': CYAN, 'borderTop': f'2px solid {CYAN}'}),
        dcc.Tab(label='Performance', value='tab-perf', style={'backgroundColor': CARD_BG, 'color': TEXT_COLOR}, selected_style={'backgroundColor': DARK_BG, 'color': CYAN, 'borderTop': f'2px solid {CYAN}'}),
        dcc.Tab(label='Subsidence Early Warning', value='tab-subsidence', style={'backgroundColor': CARD_BG, 'color': TEXT_COLOR}, selected_style={'backgroundColor': DARK_BG, 'color': CYAN, 'borderTop': f'2px solid {CYAN}'}),
    ]),
    
    # Tab Content
    html.Div(id='tab-content', style={'padding': '20px'})
])

@app.callback(
    Output('current-frame', 'data'),
    Output('play-interval', 'disabled'),
    Output('play-interval', 'interval'),
    Input('btn-prev', 'n_clicks'),
    Input('btn-play', 'n_clicks'),
    Input('btn-next', 'n_clicks'),
    Input('play-interval', 'n_intervals'),
    Input('speed-dropdown', 'value'),
    State('current-frame', 'data'),
    State('play-interval', 'disabled')
)
def update_controls(n_prev, n_play, n_next, n_int, speed, curr_frame, is_disabled):
    trigger = ctx.triggered_id
    
    if trigger == 'btn-prev':
        curr_frame = max(0, curr_frame - 1)
    elif trigger == 'btn-next':
        curr_frame = min(state.max_frames - 1, curr_frame + 1)
    elif trigger == 'btn-play':
        is_disabled = not is_disabled
    elif trigger == 'play-interval':
        curr_frame = (curr_frame + 1) % state.max_frames
        
    return curr_frame, is_disabled, speed

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('current-frame', 'data')
)
def render_tab(tab, frame_idx):
    if len(state.frames) == 0:
        return html.Div("No data loaded.")
        
    # Ensure processed
    while len(state.history) <= frame_idx:
        process_frame(len(state.history))
        
    data = state.history[frame_idx]
    
    if tab == 'tab-sim':
        return render_simulation(data)
    elif tab == 'tab-grid':
        return render_grid(data)
    elif tab == 'tab-track':
        return render_tracking(data)
    elif tab == 'tab-perf':
        return render_performance(data)
    elif tab == 'tab-subsidence':
        return render_subsidence(data)

def render_simulation(data):
    cells = data['cells']
    tracks = data['tracks']
    
    fig = go.Figure()
    
    # Background Grid Rings
    for r in [state.config.R_NEAR, state.config.R_FAR]:
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(x=r*np.cos(theta), y=r*np.sin(theta), mode='lines', line=dict(color='rgba(255,255,255,0.1)', dash='dash'), hoverinfo='skip'))
        
    # Ego Car
    fig.add_trace(go.Scatter(x=[0], y=[0], marker=dict(symbol='triangle-up', size=20, color=CYAN), name='Ego'))
    
    # LiDAR lines
    for i in range(36):
        angle = i * 10 * np.pi / 180
        fig.add_trace(go.Scatter(x=[0, state.config.R_FAR * np.cos(angle)], y=[0, state.config.R_FAR * np.sin(angle)], mode='lines', line=dict(color='rgba(6, 182, 212, 0.1)', width=1), hoverinfo='skip'))

    # Cells
    if cells:
        c_x = [c[2][0] for c in cells]
        c_y = [c[2][1] for c in cells]
        c_cls = [c[1] for c in cells]
        colors = [CLASS_HEX[c] for c in c_cls]
        
        fig.add_trace(go.Scatter(x=c_x, y=c_y, mode='markers', marker=dict(size=3, color=colors, opacity=0.8), name='Grid Cells'))

    # Tracks
    for t in tracks:
        # Halo
        fig.add_trace(go.Scatter(x=[t.centroid[0]], y=[t.centroid[1]], mode='markers', marker=dict(size=30, color='rgba(248, 113, 113, 0.3)', line=dict(color='red', width=2)), name=f'Obj {t.track_id}'))
        # Trajectory
        if len(t.history) > 1:
            hx = [p[0] for p in t.history]
            hy = [p[1] for p in t.history]
            fig.add_trace(go.Scatter(x=hx, y=hy, mode='lines', line=dict(color='red', dash='dash'), name=f'Trk {t.track_id}'))
            
    fig.update_layout(
        template='plotly_dark', plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        xaxis=dict(range=[-state.config.R_FAR, state.config.R_FAR], constrain='domain'),
        yaxis=dict(range=[-state.config.R_FAR, state.config.R_FAR], scaleanchor='x'),
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False, height=600
    )

    stats_card = html.Div([
        html.H3("Live Stats", style={'color': CYAN}),
        html.P(f"Frame: {data['idx']}"),
        html.P(f"Cells: {len(cells)}"),
        html.P(f"Memory: {data['mem_report'].adaptive_bytes / 1024 / 1024:.2f} MB"),
        html.P(f"FPS: {1000/max(1, data['latency']):.1f}"),
        html.P(f"Active Tracks: {len(tracks)}")
    ], style={'padding': '20px', 'background': CARD_BG, 'borderRadius': '10px', 'border': '1px solid rgba(255,255,255,0.1)'})

    return html.Div([
        html.Div([dcc.Graph(figure=fig)], style={'width': '75%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([stats_card], style={'width': '20%', 'display': 'inline-block', 'marginLeft': '2%', 'verticalAlign': 'top'})
    ])

def render_grid(data):
    cells = data['cells']
    if not cells: return html.Div("No cells")
    
    cx = [c[2][0] for c in cells]
    cy = [c[2][1] for c in cells]
    cz = [c[2][2] for c in cells]
    cc = [CLASS_HEX[c[1]] for c in cells]
    
    fig3d = go.Figure(go.Scatter3d(x=cx, y=cy, z=cz, mode='markers', marker=dict(size=2, color=cc)))
    fig3d.update_layout(template='plotly_dark', paper_bgcolor=CARD_BG, margin=dict(l=0,r=0,t=30,b=0), title="3D Adaptive Grid")
    
    # Confidence Heatmap
    heatmap = go.Figure(go.Histogram2d(x=cx, y=cy, nbinsx=50, nbinsy=50, colorscale='Viridis'))
    heatmap.update_layout(template='plotly_dark', paper_bgcolor=CARD_BG, margin=dict(l=0,r=0,t=30,b=0), title="Cell Density / Confidence")
    
    return html.Div([
        html.Div([dcc.Graph(figure=fig3d)], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=heatmap)], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
    ])

def render_tracking(data):
    tracks = data['tracks']
    
    fig = go.Figure()
    table_data = []
    
    for t in tracks:
        fig.add_trace(go.Scatter(x=[t.centroid[0]], y=[t.centroid[1]], mode='markers+text', text=[f"ID:{t.track_id}"], textposition="top center", marker=dict(size=10, color='red')))
        if len(t.history) > 1:
            hx = [p[0] for p in t.history]
            hy = [p[1] for p in t.history]
            fig.add_trace(go.Scatter(x=hx, y=hy, mode='lines', line=dict(color='yellow', dash='dot')))
            
        fig.add_annotation(x=t.centroid[0] + t.velocity[0], y=t.centroid[1] + t.velocity[1], ax=t.centroid[0], ay=t.centroid[1], xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='cyan')
        
        table_data.append({
            'ID': t.track_id,
            'Speed (m/s)': f"{t.speed_mps:.2f}",
            'X': f"{t.centroid[0]:.2f}",
            'Y': f"{t.centroid[1]:.2f}",
            'Age': len(t.history)
        })
        
    fig.update_layout(
        template='plotly_dark', plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        xaxis=dict(range=[-state.config.R_FAR, state.config.R_FAR]),
        yaxis=dict(range=[-state.config.R_FAR, state.config.R_FAR], scaleanchor='x'),
        showlegend=False, height=500
    )
    
    dt = DataTable(
        data=table_data,
        columns=[{"name": i, "id": i} for i in ['ID', 'Speed (m/s)', 'X', 'Y', 'Age']],
        style_header={'backgroundColor': '#1f2937', 'color': 'white'},
        style_data={'backgroundColor': CARD_BG, 'color': 'white'},
    )
    
    return html.Div([
        html.Div([dcc.Graph(figure=fig)], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([html.H3("Active Tracks", style={'color': CYAN}), dt], style={'width': '35%', 'display': 'inline-block', 'marginLeft': '2%', 'verticalAlign': 'top'})
    ])

def render_performance(data):
    hist = state.history[:data['idx']+1]
    frames = [h['idx'] for h in hist]
    
    # Memory
    mem_adp = [h['mem_report'].adaptive_bytes / 1e6 for h in hist]
    mem_2d = [h['mem_report'].uniform_2d_bytes / 1e6 for h in hist]
    fig_mem = go.Figure()
    fig_mem.add_trace(go.Scatter(x=frames, y=mem_adp, name='Adaptive (MB)', fill='tozeroy'))
    fig_mem.add_trace(go.Scatter(x=frames, y=mem_2d, name='Uniform 2D (MB)', mode='lines'))
    fig_mem.update_layout(template='plotly_dark', paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, title="Memory Usage", height=300)
    
    # Latency
    lats = [h['latency'] for h in hist]
    fig_lat = go.Figure(go.Scatter(x=frames, y=lats, mode='lines+markers', marker=dict(color=VIOLET)))
    fig_lat.update_layout(template='plotly_dark', paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, title="Latency (ms)", height=300)
    
    # Cells
    cells = [len(h['cells']) for h in hist]
    fig_cells = go.Figure(go.Scatter(x=frames, y=cells, fill='tozeroy', marker=dict(color=CYAN)))
    fig_cells.update_layout(template='plotly_dark', paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, title="Active Cells", height=300)
    
    # Resolution Curve
    r_vals = np.linspace(state.config.R_NEAR, state.config.R_FAR, 100)
    s_vals = cell_size(r_vals, state.config)
    fig_res = go.Figure(go.Scatter(x=r_vals, y=s_vals, mode='lines', marker=dict(color='yellow')))
    fig_res.update_layout(template='plotly_dark', paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, title="Resolution vs Distance", height=300)
    
    return html.Div([
        html.Div([dcc.Graph(figure=fig_mem)], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=fig_lat)], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'}),
        html.Div([dcc.Graph(figure=fig_cells)], style={'width': '48%', 'display': 'inline-block', 'marginTop': '20px'}),
        html.Div([dcc.Graph(figure=fig_res)], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%', 'marginTop': '20px'})
    ])

def render_subsidence(data):
    assessments = data['mesh_assessments']
    colors = {
        'NORMAL': '#34d399', 'WATCH': '#fbbf24',
        'HIGH': '#fb923c', 'CRITICAL': '#f87171'
    }
    fig = go.Figure()
    node_by_id = {a.node_id: a for a in assessments}
    for assessment in assessments:
        for other in assessments:
            if other.node_id <= assessment.node_id:
                continue
            if abs(other.x_m - assessment.x_m) + abs(other.y_m - assessment.y_m) == 20.0:
                fig.add_trace(go.Scatter(
                    x=[assessment.x_m, other.x_m], y=[assessment.y_m, other.y_m],
                    mode='lines', line=dict(color='rgba(148,163,184,0.25)', width=1),
                    hoverinfo='skip', showlegend=False,
                ))
    for level in ('NORMAL', 'WATCH', 'HIGH', 'CRITICAL'):
        points = [a for a in assessments if a.risk_level == level]
        if points:
            fig.add_trace(go.Scatter(
                x=[a.x_m for a in points], y=[a.y_m for a in points], mode='markers+text',
                text=[f'{a.displacement_mm:.1f} mm' for a in points], textposition='top center',
                name=level, marker=dict(size=16, color=colors[level], line=dict(color='white', width=1)),
                customdata=[[a.node_id, a.risk_score, a.velocity_mm_per_s, a.forecast_mm_60s] for a in points],
                hovertemplate='<b>%{customdata[0]}</b><br>Risk: %{customdata[1]:.2f}<br>'
                              'Velocity: %{customdata[2]:.3f} mm/s<br>60s forecast: %{customdata[3]:.1f} mm<extra></extra>',
            ))
    fig.update_layout(
        template='plotly_dark', plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        title='Wireless Surface Mesh: Deformation Risk Map', height=560,
        xaxis=dict(title='Local east (m)', scaleanchor='y'), yaxis=dict(title='Local north (m)'),
        legend=dict(orientation='h', y=1.08), margin=dict(l=50, r=20, t=70, b=50),
    )
    summary = state.subsidence_monitor.summary()
    status_color = colors[summary['overall_status']]
    stats = html.Div([
        html.H3('Early Warning Control Room', style={'color': CYAN}),
        html.P(f"Network: {summary['online_nodes']}/{summary['nodes']} nodes online"),
        html.P(f"Maximum movement: {summary['max_displacement_mm']:.1f} mm"),
        html.P(f"High-risk nodes: {summary['high_risk_nodes']} | Critical: {summary['critical_nodes']}"),
        html.P(f"Forecast horizon: 60 seconds"),
        html.H2(summary['overall_status'], style={'color': status_color, 'marginBottom': '0'}),
        html.P('Risk state from tilt, displacement rate, vibration, crack growth, and link quality', style={'color': '#94a3b8'}),
    ], style={'padding': '18px', 'background': CARD_BG, 'border': '1px solid #263244', 'borderRadius': '8px'})
    alert_rows = [
        html.Tr([html.Td(alert.node_id), html.Td(alert.risk_level), html.Td(alert.message)])
        for alert in state.subsidence_monitor.alerts[-8:]
    ]
    alerts = html.Div([
        html.H3('Active Alerts', style={'color': CYAN}),
        html.Table([html.Thead(html.Tr([html.Th('Node'), html.Th('Level'), html.Th('Event')])), html.Tbody(alert_rows)],
                   style={'width': '100%', 'textAlign': 'left', 'borderSpacing': '8px'}),
    ], style={'padding': '18px', 'background': CARD_BG, 'border': '1px solid #263244', 'borderRadius': '8px', 'marginTop': '16px'})
    return html.Div([html.Div([dcc.Graph(figure=fig)], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                     html.Div([stats, alerts], style={'width': '27%', 'display': 'inline-block', 'marginLeft': '2%', 'verticalAlign': 'top'})])

def main():
    parser = argparse.ArgumentParser(description="Adaptive LiDAR Mapping Dashboard")
    parser.add_argument('--data-dir', type=str, default=str(PROJECT_ROOT / 'data' / 'sample'), help='Path to data directory')
    parser.add_argument('--port', type=int, default=8050, help='Dash port')
    parser.add_argument('--max-frames', type=int, default=50, help='Max frames to load')
    parser.add_argument('--checkpoint', type=str, default=str(PROJECT_ROOT / 'checkpoints' / 'pointnet_3class.pth'), help='Model checkpoint path')
    args = parser.parse_args()
    
    print("Starting server...")
    load_data(args.data_dir, args.max_frames, args.checkpoint)
    app.run(debug=False, host='0.0.0.0', port=args.port)

if __name__ == '__main__':
    main()
