"""
generate_report.py — Génération des graphiques et rapport PFE
Système de Surveillance Intelligente — ENSA Béni Mellal
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_metrics(metrics_file: str = "reports/metrics.json") -> dict:
    """Charger les métriques depuis le fichier JSON."""
    try:
        with open(metrics_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ Fichier {metrics_file} non trouvé")
        return {}


def generate_performance_graphs(metrics_data: dict, output_dir: str = "reports"):
    """Générer les graphiques de performance."""
    if not metrics_data.get("statistics"):
        logger.error("❌ Pas de statistiques disponibles")
        return
    
    stats = metrics_data["statistics"]
    frames_data = metrics_data.get("frames", [])
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. COURBE LATENCE DANS LE TEMPS
    if frames_data:
        plt.figure(figsize=(12, 6))
        frame_ids = [f["frame_id"] for f in frames_data]
        latencies = [f["latency_ms"] for f in frames_data]
        
        plt.plot(frame_ids, latencies, linewidth=1, color='#2E86AB', alpha=0.8)
        avg_latency = stats.get("latency_ms", {}).get("avg", 0)
        plt.axhline(y=avg_latency, color='red', linestyle='--', label=f'Moyenne: {avg_latency:.2f}ms')
        
        plt.xlabel('Frame ID', fontsize=12, fontweight='bold')
        plt.ylabel('Latence (ms)', fontsize=12, fontweight='bold')
        plt.title('Latence Frame → Alerte dans le Temps', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/01_latence_timeline.png", dpi=300)
        logger.info(f"✅ Graphique sauvegardé: 01_latence_timeline.png")
        plt.close()
    
    # 2. DISTRIBUTION LATENCE (HISTOGRAMME)
    if frames_data:
        plt.figure(figsize=(12, 6))
        latencies = [f["latency_ms"] for f in frames_data]
        
        plt.hist(latencies, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
        plt.xlabel('Latence (ms)', fontsize=12, fontweight='bold')
        plt.ylabel('Fréquence', fontsize=12, fontweight='bold')
        plt.title('Distribution Latence API', fontsize=14, fontweight='bold')
        plt.axvline(np.mean(latencies), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {np.mean(latencies):.2f}ms')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/02_latence_distribution.png", dpi=300)
        logger.info(f"✅ Graphique sauvegardé: 02_latence_distribution.png")
        plt.close()
    
    # 3. ALERTES PAR TYPE (CAMEMBERT)
    alerts_by_type = stats.get("alerts_by_type", {})
    if alerts_by_type:
        plt.figure(figsize=(10, 8))
        colors = ['#E63946', '#F77F00', '#06A77D', '#4A90E2']
        types = list(alerts_by_type.keys())
        counts = list(alerts_by_type.values())
        
        wedges, texts, autotexts = plt.pie(
            counts, 
            labels=types, 
            autopct='%1.1f%%',
            colors=colors[:len(types)],
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        
        plt.title('Distribution Alertes par Type', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/03_alertes_par_type.png", dpi=300)
        logger.info(f"✅ Graphique sauvegardé: 03_alertes_par_type.png")
        plt.close()
    
    # 4. FPS ET PERFORMANCE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # FPS
    fps = stats.get("fps_average", 0)
    ax1.bar(["FPS Moyen"], [fps], color='#06A77D', width=0.5, edgecolor='black', linewidth=2)
    ax1.set_ylabel('FPS', fontsize=12, fontweight='bold')
    ax1.set_title('Performance FPS', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(fps * 1.2, 30))
    ax1.text(0, fps + 1, f'{fps:.1f}', ha='center', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Ratio alertes
    frames = stats.get("frames_processed", 1)
    alerts = stats.get("alerts_generated", 0)
    alert_rate = stats.get("alert_rate", 0)
    
    ax2.bar(
        ["Frames", "Alertes"],
        [frames, alerts],
        color=['#4A90E2', '#E63946'],
        edgecolor='black',
        linewidth=2
    )
    ax2.set_ylabel('Nombre', fontsize=12, fontweight='bold')
    ax2.set_title(f'Frames vs Alertes (Taux: {alert_rate:.1%})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_fps_et_alertes.png", dpi=300)
    logger.info(f"✅ Graphique sauvegardé: 04_fps_et_alertes.png")
    plt.close()
    
    # 5. BOÎTE À MOUSTACHES LATENCE
    if frames_data:
        plt.figure(figsize=(10, 6))
        latencies = [f["latency_ms"] for f in frames_data]
        
        plt.boxplot(latencies, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#A23B72', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2),
                   whiskerprops=dict(linewidth=1.5),
                   capprops=dict(linewidth=1.5))
        
        plt.ylabel('Latence (ms)', fontsize=12, fontweight='bold')
        plt.title('Analyse Latence (Q1, Médiane, Q3)', fontsize=14, fontweight='bold')
        plt.xticks([1], ['Latence API'])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/05_latence_boxplot.png", dpi=300)
        logger.info(f"✅ Graphique sauvegardé: 05_latence_boxplot.png")
        plt.close()


def generate_html_report(metrics_data: dict, output_dir: str = "reports"):
    """Générer un rapport HTML avec les résultats."""
    stats = metrics_data.get("statistics", {})
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rapport PFE - Système de Surveillance</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2E86AB;
                text-align: center;
                border-bottom: 3px solid #2E86AB;
                padding-bottom: 20px;
            }}
            h2 {{
                color: #A23B72;
                margin-top: 30px;
                border-left: 4px solid #A23B72;
                padding-left: 15px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .metric-box {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .metric-box.good {{
                background: linear-gradient(135deg, #06A77D 0%, #218c7c 100%);
            }}
            .metric-box.warning {{
                background: linear-gradient(135deg, #F77F00 0%, #d46e00 100%);
            }}
            .metric-box h3 {{
                margin: 0;
                font-size: 14px;
                opacity: 0.9;
            }}
            .metric-box .value {{
                font-size: 28px;
                font-weight: bold;
                margin: 10px 0 0 0;
            }}
            .metric-box .unit {{
                font-size: 12px;
                opacity: 0.8;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th {{
                background-color: #2E86AB;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 10px 12px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background-color: #f0f0f0;
            }}
            .image-section {{
                margin: 30px 0;
                text-align: center;
            }}
            .image-section img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin: 10px 0;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #666;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Rapport PFE - Système de Surveillance Intelligente</h1>
            <p style="text-align: center; color: #666; margin-top: -10px;">
                ENSA Béni Mellal | Généré le {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
            </p>
            
            <h2>🎯 Résumé des Performances</h2>
            <div class="metrics-grid">
                <div class="metric-box good">
                    <h3>Frames Traitées</h3>
                    <div class="value">{stats.get('frames_processed', 0)}</div>
                    <div class="unit">frames</div>
                </div>
                <div class="metric-box">
                    <h3>Alertes Générées</h3>
                    <div class="value">{stats.get('alerts_generated', 0)}</div>
                    <div class="unit">alertes</div>
                </div>
                <div class="metric-box good">
                    <h3>FPS Moyen</h3>
                    <div class="value">{stats.get('fps_average', 0):.1f}</div>
                    <div class="unit">fps</div>
                </div>
                <div class="metric-box warning">
                    <h3>Latence Moyenne</h3>
                    <div class="value">{stats.get('latency_ms', {}).get('avg', 0):.1f}</div>
                    <div class="unit">ms</div>
                </div>
            </div>
            
            <h2>📈 Statistiques Détaillées</h2>
            <table>
                <tr>
                    <th>Métrique</th>
                    <th>Valeur</th>
                </tr>
                <tr>
                    <td>Durée totale</td>
                    <td>{stats.get('total_time_seconds', 0):.2f} secondes</td>
                </tr>
                <tr>
                    <td>Latence Min</td>
                    <td>{stats.get('latency_ms', {}).get('min', 0):.2f} ms</td>
                </tr>
                <tr>
                    <td>Latence Max</td>
                    <td>{stats.get('latency_ms', {}).get('max', 0):.2f} ms</td>
                </tr>
                <tr>
                    <td>Taux alertes/frames</td>
                    <td>{stats.get('alert_rate', 0):.2%}</td>
                </tr>
            </table>
            
            <h2>🚨 Alertes par Type</h2>
            <table>
                <tr>
                    <th>Type d'Alerte</th>
                    <th>Nombre</th>
                    <th>Confiance Moyenne</th>
                </tr>
    """
    
    for alert_type, count in stats.get('alerts_by_type', {}).items():
        avg_conf = stats.get('avg_confidence_by_type', {}).get(alert_type, 0)
        html_content += f"""
                <tr>
                    <td>{alert_type}</td>
                    <td>{count}</td>
                    <td>{avg_conf:.3f}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>📊 Graphiques Générés</h2>
            <div class="image-section">
                <h3>1. Latence dans le Temps</h3>
                <img src="01_latence_timeline.png" alt="Latence Timeline">
            </div>
            <div class="image-section">
                <h3>2. Distribution Latence</h3>
                <img src="02_latence_distribution.png" alt="Latence Distribution">
            </div>
            <div class="image-section">
                <h3>3. Alertes par Type</h3>
                <img src="03_alertes_par_type.png" alt="Alertes par Type">
            </div>
            <div class="image-section">
                <h3>4. FPS et Alertes</h3>
                <img src="04_fps_et_alertes.png" alt="FPS et Alertes">
            </div>
            <div class="image-section">
                <h3>5. Analyse Latence</h3>
                <img src="05_latence_boxplot.png" alt="Latence Boxplot">
            </div>
            
            <div class="footer">
                <p>© 2026 ENSA Béni Mellal - Système de Surveillance Intelligente</p>
                <p>Ce rapport a été généré automatiquement par le système de test.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    report_file = f"{output_dir}/rapport_pfe.html"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"✅ Rapport HTML généré: {report_file}")


def main():
    """Générer tous les rapports et graphiques."""
    logger.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("\n" + "="*70)
    print("🔄 GÉNÉRATION DES RAPPORTS ET GRAPHIQUES")
    print("="*70 + "\n")
    
    # Charger les métriques
    metrics_data = load_metrics()
    if not metrics_data:
        print("❌ Aucune donnée de métrique disponible")
        return
    
    # Afficher le résumé
    from metrics_collector import MetricsCollector
    collector = MetricsCollector()
    collector.frames = metrics_data.get("frames", [])
    collector.alerts = metrics_data.get("alerts", [])
    collector.frame_count = len(collector.frames)
    collector.alert_count = len(collector.alerts)
    collector.start_time = 0
    collector.end_time = collector.frame_count / 30  # Estimation basée sur 30 fps
    collector.print_summary()
    
    # Générer les graphiques
    print("📊 Génération des graphiques...")
    generate_performance_graphs(metrics_data)
    
    # Générer le rapport HTML
    print("📄 Génération du rapport HTML...")
    generate_html_report(metrics_data)
    
    print("\n✅ Tous les rapports ont été générés avec succès!")
    print("📁 Fichiers créés dans le dossier 'reports/'")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
