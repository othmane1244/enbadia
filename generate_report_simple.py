"""
generate_report_simple.py — Rapport HTML simple sans matplotlib
Utilise uniquement JSON et HTML
"""

import json
from pathlib import Path
from datetime import datetime

def load_metrics(metrics_file: str) -> dict:
    """Charge les métriques depuis le fichier JSON."""
    with open(metrics_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_html_report(metrics_data, output_dir="reports"):
    """Génère le rapport HTML simple sans graphiques."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    stats = metrics_data.get("statistics", {})
    
    # Créer le rapport HTML
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
            max-width: 1200px;
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
        
        .metric-card.critical {{
            border-left-color: #e63946;
        }}
        
        .metric-card.value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }}
        
        .metric-card.label {{
            font-size: 1.1em;
            color: #666;
        }}
        
        .metric-card.unit {{
            font-size: 0.9em;
            color: #999;
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
                        <div class="metric-card.label">Frames Traitées</div>
                        <div class="metric-card.value">{stats.get('frames_processed', 0)}</div>
                        <div class="metric-card.unit">frames</div>
                    </div>
                    <div class="metric-card good">
                        <div class="metric-card.label">FPS Moyen</div>
                        <div class="metric-card.value">{stats.get('fps_average', 0):.2f}</div>
                        <div class="metric-card.unit">fps</div>
                    </div>
                    <div class="metric-card warning">
                        <div class="metric-card.label">Latence Moyenne</div>
                        <div class="metric-card.value">{stats.get('latency_ms', {}).get('avg', 0):.2f}</div>
                        <div class="metric-card.unit">ms</div>
                    </div>
                    <div class="metric-card warning">
                        <div class="metric-card.label">Alertes Générées</div>
                        <div class="metric-card.value">{stats.get('alerts_generated', 0)}</div>
                        <div class="metric-card.unit">alertes</div>
                    </div>
                </div>
            </div>
            
            <!-- Durée et performance -->
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
                        <td>{stats.get('latency_ms', {}).get('min', 0):.2f} ms</td>
                    </tr>
                    <tr>
                        <td>Latence moyenne</td>
                        <td>{stats.get('latency_ms', {}).get('avg', 0):.2f} ms</td>
                    </tr>
                    <tr>
                        <td>Latence maximale</td>
                        <td>{stats.get('latency_ms', {}).get('max', 0):.2f} ms</td>
                    </tr>
                </table>
            </div>
            
            <!-- Alertes par type -->
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
            
            <!-- Statistiques supplémentaires -->
            <div class="section">
                <h2>📊 Statistiques Supplémentaires</h2>
                <table class="table">
                    <tr>
                        <th>Métrique</th>
                        <th>Valeur</th>
                    </tr>
"""
    
    html_content += f"""                    <tr>
                        <td>Taux d'alertes</td>
                        <td>{stats.get('alert_rate', 0)*100:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Timestamp de génération</td>
                        <td>{metrics_data.get('timestamp', 'N/A')}</td>
                    </tr>
                </table>
            </div>
        </div>
        
        <div class="footer">
            <p>Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</p>
            <p>Système de Surveillance Intelligente — ENSA Béni Mellal</p>
        </div>
    </div>
</body>
</html>"""
    
    # Sauvegarder le rapport
    report_file = output_path / "rapport_pfe.html"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Rapport HTML généré : {report_file}")
    return str(report_file)


def main():
    """Génère le rapport à partir des métriques existantes."""
    metrics_file = "reports/metrics.json"
    
    if not Path(metrics_file).exists():
        print(f"❌ Fichier de métriques non trouvé : {metrics_file}")
        return False
    
    print(f"📖 Chargement des métriques...")
    metrics_data = load_metrics(metrics_file)
    
    print(f"📄 Génération du rapport HTML...")
    generate_html_report(metrics_data)
    
    print(f"\n✅ Rapport généré avec succès!")
    print(f"   Ouvrir : reports/rapport_pfe.html")
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
