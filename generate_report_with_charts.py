"""
generate_report_with_charts.py — Rapport HTML avec graphiques SVG
Pas besoin de matplotlib
"""

import json
import math
from pathlib import Path
from datetime import datetime

def load_metrics(metrics_file: str) -> dict:
    """Charge les métriques."""
    with open(metrics_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_svg_pie_chart(data_dict, title="", width=400, height=400):
    """Crée un graphique camembert en SVG."""
    colors = ['#E63946', '#F77F00', '#06A77D', '#4A90E2', '#FFB4A2']
    
    total = sum(data_dict.values())
    if total == 0:
        return "<svg></svg>"
    
    svg = f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
        <style>
            .pie-slice {{ cursor: pointer; }}
            .pie-slice:hover {{ opacity: 0.8; }}
            .pie-label {{ font-size: 14px; font-weight: bold; fill: white; pointer-events: none; }}
            .pie-title {{ font-size: 16px; font-weight: bold; fill: #333; }}
        </style>
        <text x="{width/2}" y="20" text-anchor="middle" class="pie-title">{title}</text>'''
    
    cx, cy, radius = width/2, height/2 + 20, 120
    start_angle = 0
    
    for idx, (label, value) in enumerate(data_dict.items()):
        slice_angle = (value / total) * 360
        end_angle = start_angle + slice_angle
        
        # Convertir en radians
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)
        
        # Points du slice
        x1 = cx + radius * math.cos(start_rad)
        y1 = cy + radius * math.sin(start_rad)
        x2 = cx + radius * math.cos(end_rad)
        y2 = cy + radius * math.sin(end_rad)
        
        # Large arc flag
        large_arc = 1 if slice_angle > 180 else 0
        
        # Path du slice
        path = f"M {cx} {cy} L {x1} {y1} A {radius} {radius} 0 {large_arc} 1 {x2} {y2} Z"
        
        color = colors[idx % len(colors)]
        svg += f'<path d="{path}" fill="{color}" class="pie-slice"/>'
        
        # Label (pourcentage)
        mid_angle = start_angle + slice_angle / 2
        mid_rad = math.radians(mid_angle)
        label_x = cx + (radius * 0.6) * math.cos(mid_rad)
        label_y = cy + (radius * 0.6) * math.sin(mid_rad)
        
        percentage = (value / total) * 100
        svg += f'<text x="{label_x}" y="{label_y}" text-anchor="middle" dominant-baseline="middle" class="pie-label">{percentage:.0f}%</text>'
        
        start_angle = end_angle
    
    svg += "</svg>"
    return svg


def create_svg_bar_chart(values, labels, title="", width=600, height=400):
    """Crée un graphique en barres en SVG."""
    if not values:
        return "<svg></svg>"
    
    max_val = max(values)
    bar_width = (width - 100) / len(values)
    chart_height = height - 80
    
    svg = f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
        <style>
            .bar {{ fill: #667eea; opacity: 0.8; }}
            .bar:hover {{ opacity: 1; }}
            .bar-label {{ font-size: 12px; text-anchor: middle; fill: #333; }}
            .bar-value {{ font-size: 14px; font-weight: bold; text-anchor: middle; fill: #333; }}
            .chart-title {{ font-size: 16px; font-weight: bold; fill: #333; }}
            .axis-label {{ font-size: 12px; fill: #666; }}
        </style>
        
        <text x="{width/2}" y="20" text-anchor="middle" class="chart-title">{title}</text>
        
        <!-- Y-axis -->
        <line x1="50" y1="30" x2="50" y2="{30 + chart_height}" stroke="#666" stroke-width="2"/>
        
        <!-- X-axis -->
        <line x1="50" y1="{30 + chart_height}" x2="{width-20}" y2="{30 + chart_height}" stroke="#666" stroke-width="2"/>'''
    
    for idx, (value, label) in enumerate(zip(values, labels)):
        x = 50 + (idx + 0.5) * bar_width
        bar_height = (value / max_val) * chart_height
        
        y = 30 + chart_height - bar_height
        
        svg += f'<rect x="{x - bar_width*0.35}" y="{y}" width="{bar_width*0.7}" height="{bar_height}" class="bar"/>'
        svg += f'<text x="{x}" y="{y - 5}" class="bar-value">{value:.2f}</text>'
        svg += f'<text x="{x}" y="{30 + chart_height + 20}" class="bar-label">{label}</text>'
    
    svg += "</svg>"
    return svg


def create_svg_comparison_chart(before_value, after_value, before_label, after_label, title="", width=700, height=420):
    """Crée un graphique de comparaison avant/après en SVG."""
    values = [before_value, after_value]
    labels = [before_label, after_label]
    colors = ['#E63946', '#06A77D']
    max_val = max(values) if max(values) > 0 else 1

    svg = f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
        <style>
            .bar {{ opacity: 0.9; rx: 10; ry: 10; }}
            .bar-label {{ font-size: 13px; font-weight: bold; text-anchor: middle; fill: #333; }}
            .bar-value {{ font-size: 14px; font-weight: bold; text-anchor: middle; fill: #333; }}
            .chart-title {{ font-size: 18px; font-weight: bold; fill: #333; }}
            .chart-subtitle {{ font-size: 12px; fill: #666; }}
        </style>
        <text x="{width/2}" y="24" text-anchor="middle" class="chart-title">{title}</text>
        <text x="{width/2}" y="44" text-anchor="middle" class="chart-subtitle">Taux d'alertes / faux positifs en pourcentage</text>
        <line x1="70" y1="360" x2="{width-30}" y2="360" stroke="#666" stroke-width="2"/>
        <line x1="70" y1="70" x2="70" y2="360" stroke="#666" stroke-width="2"/>'''

    chart_width = width - 140
    bar_width = chart_width / 2
    chart_height = 290

    for idx, (value, label) in enumerate(zip(values, labels)):
        x_center = 70 + bar_width * (idx + 0.5)
        bar_height = (value / max_val) * chart_height
        y = 360 - bar_height
        color = colors[idx]
        safe_label = label.replace('\n', ' ')
        svg += f'<rect x="{x_center - 70}" y="{y}" width="140" height="{bar_height}" fill="{color}" class="bar"/>'
        svg += f'<text x="{x_center}" y="{y - 8}" class="bar-value">{value:.2f}%</text>'
        svg += f'<text x="{x_center}" y="382" class="bar-label">{safe_label}</text>'

    svg += "</svg>"
    return svg


def render_image_card(image_path: str, title: str, description: str) -> str:
    """Retourne une carte HTML avec image locale ou message de remplacement."""
    path = Path(image_path)
    if path.exists():
        return f'''
        <div class="image-card">
            <h3>{title}</h3>
            <img src="{image_path.replace('\\', '/')}" alt="{title}" />
            <p>{description}</p>
        </div>'''

    return f'''
    <div class="image-card missing">
        <h3>{title}</h3>
        <div class="missing-box">Image introuvable: {image_path}</div>
        <p>{description}</p>
    </div>'''


def generate_html_report(metrics_data, output_dir="reports"):
    """Génère le rapport HTML avec graphiques SVG."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    stats = metrics_data.get("statistics", {})
    
    # Créer les graphiques SVG
    alerts_by_type = stats.get('alerts_by_type', {})
    pie_chart = create_svg_pie_chart(alerts_by_type, "Répartition des Alertes")
    
    bar_chart = create_svg_bar_chart(
        [stats.get('fps_average', 0), stats.get('alert_rate', 0)*100],
        ['FPS Moyen', 'Taux Alertes (%)'],
        "Performance: FPS et Ratio Alertes"
    )

    comparison_chart = create_svg_comparison_chart(
        before_value=15.0,
        after_value=stats.get('alert_rate', 0) * 100,
        before_label="Avant corrections\n(ratio seul)",
        after_label="Après triple validation",
        title="Comparaison Avant / Après corrections"
    )

    t1_results_image = render_image_card(
        "../T1/fine_tuning2/runs/train/results.png",
        "Courbe d'entraînement T1",
        "Courbe des pertes et des métriques du modèle T1 pendant l'entraînement."
    )
    t1_confusion_image = render_image_card(
        "../T1/fine_tuning2/runs/train/confusion_matrix.png",
        "Confusion Matrix T1",
        "Matrice de confusion Supine vs Not_Supine pour le modèle T1."
    )
    
    # Créer le rapport HTML
    latency_data = stats.get('latency_ms', {})
    
    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport PFE - Système de Surveillance Intelligente</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            border-left: 5px solid #667eea;
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }}
        
        .metric-card.good {{
            border-left-color: #06a77d;
        }}
        
        .metric-card.warning {{
            border-left-color: #f77f00;
        }}
        
        .metric-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }}
        
        .metric-card .label {{
            font-size: 1.1em;
            color: #666;
        }}
        
        .metric-card .unit {{
            font-size: 0.9em;
            color: #999;
        }}
        
        .chart-container {{
            display: flex;
            justify-content: center;
            margin: 30px 0;
            background: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
        }}
        
        .chart-container svg {{
            max-width: 100%;
            height: auto;
        }}

        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 24px;
            margin-top: 20px;
        }}

        .image-card {{
            background: #f9f9f9;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }}

        .image-card h3 {{
            font-size: 1.1em;
            color: #333;
            margin-bottom: 12px;
        }}

        .image-card img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid #e5e5e5;
            background: white;
        }}

        .image-card p {{
            margin-top: 10px;
            color: #666;
            font-size: 0.95em;
        }}

        .image-card.missing .missing-box {{
            padding: 18px;
            border-radius: 8px;
            background: #fff3cd;
            color: #856404;
            border: 1px dashed #f0ad4e;
            font-size: 0.95em;
        }}
        
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin: 30px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }}
        
        .table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e5e5e5;
        }}
        
        .table tr:last-child td {{
            border-bottom: none;
        }}
        
        .table tr:hover {{
            background: #f9f9f9;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .footer {{
            background: #f5f5f5;
            padding: 20px;
            text-align: center;
            color: #999;
            font-size: 0.9em;
        }}
        
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            margin-bottom: 40px;
        }}
        
        @media (max-width: 1024px) {{
            .two-col {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Rapport PFE</h1>
            <p>Système de Surveillance Intelligente — Métriques de Performance</p>
        </div>
        
        <div class="content">
            <!-- Métriques principales -->
            <div class="section">
                <h2>📈 Métriques Clés</h2>
                <div class="metrics-grid">
                    <div class="metric-card good">
                        <div class="label">Frames Traitées</div>
                        <div class="value">{stats.get('frames_processed', 0)}</div>
                        <div class="unit">frames</div>
                    </div>
                    <div class="metric-card good">
                        <div class="label">FPS Moyen</div>
                        <div class="value">{stats.get('fps_average', 0):.2f}</div>
                        <div class="unit">fps</div>
                    </div>
                    <div class="metric-card warning">
                        <div class="label">Latence Moyenne</div>
                        <div class="value">{latency_data.get('avg', 0):.2f}</div>
                        <div class="unit">ms</div>
                    </div>
                    <div class="metric-card warning">
                        <div class="label">Alertes Générées</div>
                        <div class="value">{stats.get('alerts_generated', 0)}</div>
                        <div class="unit">alertes</div>
                    </div>
                </div>
            </div>
            
            <!-- Graphiques -->
            <div class="section">
                <h2>📊 Graphiques</h2>
                <div class="two-col">
                    <div class="chart-container">
                        {pie_chart}
                    </div>
                    <div class="chart-container">
                        {bar_chart}
                    </div>
                </div>
                <div class="chart-container">
                    {comparison_chart}
                </div>
            </div>

            <!-- Modèle T1 -->
            <div class="section">
                <h2>🧠 Modèle T1 — Entraînement et Validation</h2>
                <p style="color: #666; margin-bottom: 16px;">Images directement intégrées depuis les exports d'entraînement du modèle T1.</p>
                <div class="image-grid">
                    {t1_results_image}
                    {t1_confusion_image}
                </div>
            </div>
            
            <!-- Performance -->
            <div class="section">
                <h2>⏱️ Performance</h2>
                <table class="table">
                    <tr>
                        <th>Métrique</th>
                        <th>Valeur</th>
                    </tr>
                    <tr>
                        <td>Durée totale</td>
                        <td>{stats.get('total_time_seconds', 0):.2f} secondes ({stats.get('total_time_seconds', 0)/60:.1f} minutes)</td>
                    </tr>
                    <tr>
                        <td>Frames traitées</td>
                        <td>{stats.get('frames_processed', 0)}</td>
                    </tr>
                    <tr>
                        <td>FPS moyen</td>
                        <td>{stats.get('fps_average', 0):.2f} fps</td>
                    </tr>
                    <tr>
                        <td>Latence minimale</td>
                        <td>{latency_data.get('min', 0):.2f} ms</td>
                    </tr>
                    <tr>
                        <td>Latence moyenne</td>
                        <td>{latency_data.get('avg', 0):.2f} ms</td>
                    </tr>
                    <tr>
                        <td>Latence maximale</td>
                        <td>{latency_data.get('max', 0):.2f} ms</td>
                    </tr>
                </table>
            </div>
            
            <!-- Alertes -->
            <div class="section">
                <h2>🚨 Alertes par Type</h2>
                <table class="table">
                    <tr>
                        <th>Type d'Alerte</th>
                        <th>Nombre</th>
                        <th>Confiance Moyenne</th>
                        <th>Pourcentage</th>
                    </tr>
"""
    
    alerts_by_type = stats.get('alerts_by_type', {})
    total_alerts = sum(alerts_by_type.values())
    
    for alert_type, count in alerts_by_type.items():
        avg_conf = stats.get('avg_confidence_by_type', {}).get(alert_type, 0)
        percentage = (count / total_alerts * 100) if total_alerts > 0 else 0
        
        html_content += f"""                    <tr>
                        <td>{alert_type}</td>
                        <td>{count}</td>
                        <td>{avg_conf:.3f}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>
"""
    
    html_content += """                </table>
            </div>
        </div>
        
        <div class="footer">
            <p>Rapport généré le """ + datetime.now().strftime('%d/%m/%Y à %H:%M:%S') + """</p>
            <p>Système de Surveillance Intelligente — ENSA Béni Mellal</p>
        </div>
    </div>
</body>
</html>"""
    
    # Sauvegarder
    report_file = output_path / "rapport_pfe.html"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Rapport HTML généré : {report_file}")


def main():
    """Génère le rapport."""
    metrics_file = "reports/metrics.json"
    
    if not Path(metrics_file).exists():
        print(f"❌ Fichier de métriques non trouvé : {metrics_file}")
        return False
    
    print(f"📖 Chargement des métriques...")
    metrics_data = load_metrics(metrics_file)
    
    print(f"📄 Génération du rapport HTML avec graphiques SVG...")
    generate_html_report(metrics_data)
    
    print(f"\n✅ Rapport généré avec succès!")
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
